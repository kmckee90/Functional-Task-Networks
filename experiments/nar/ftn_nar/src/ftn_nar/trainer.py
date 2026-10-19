"""Block-sequential continual training for all primary methods (design sections 10, 11)."""

from __future__ import annotations

import dataclasses
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import algorithms as A
from . import generator as G
from . import seeds as S
from . import support as SUP
from .batch import build_batch
from .config import ExperimentCfg, config_hash, save_config, to_dict
from .configurer import configure, configure_with_restarts
from .evaluator import EvalCache, build_support_batch, evaluate_program, infer_program
from .losses import model_training_loss
from .masks import mask_from_indices, medoid, random_topk_mask
from .model import DPModel, Program, count_parameters, oracle_masks, oracle_model_cfg
from .replay import ReplayBuffer, ReplayItem

FTN_CONFIGURED = {"FTN_FULL", "FTN_NO_REPLAY", "JOINT_MULTITASK_FTN"}
NO_REPLAY = {"FTN_NO_REPLAY", "SINGLE_TASK"}
UNIFORM_REPLAY = {"ORACLE_COMPOSITION", "DISJOINT_TASK_MODULES", "DENSE_ACTIVE_MATCHED"}


def run_name(cfg: ExperimentCfg) -> str:
    return (f"{cfg.method}__heldout-{A.ALGORITHM_NAMES[cfg.held_out]}"
            f"__seed-{cfg.seed}__confighash-{config_hash(cfg)}")


def _git_hash() -> str:
    for cwd in (Path(__file__).resolve().parent, Path.cwd()):
        try:
            return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                                  text=True, timeout=10).stdout.strip() or "unknown"
        except Exception:
            continue
    return "unknown"


def env_manifest() -> dict:
    m = {"torch": torch.__version__, "python": platform.python_version(),
         "platform": platform.platform(), "numpy": np.__version__}
    if torch.cuda.is_available():
        m["cuda"] = torch.version.cuda
        m["cudnn"] = torch.backends.cudnn.version()
        m["gpu"] = torch.cuda.get_device_name(0)
    return m


class Trainer:
    def __init__(self, cfg: ExperimentCfg, resume: bool = False, run_dir: str | None = None):
        self.cfg = cfg
        m = cfg.method
        assert m in {"FTN_FULL", "ORACLE_COMPOSITION", "FTN_RANDOM_MASK_TRAIN", "FTN_NO_REPLAY",
                     "DENSE_ACTIVE_MATCHED", "SOFT_MIXTURE", "DISJOINT_TASK_MODULES",
                     "JOINT_MULTITASK_FTN", "SINGLE_TASK", "FTN_PER_TASK_MASK"}, m
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        # run_dir may be given explicitly (resume/eval): the config hash can shift when
        # new defaulted fields are added to the config schema, which would otherwise
        # point resume at a nonexistent freshly-hashed directory.
        self.run_dir = Path(run_dir) if run_dir is not None else Path(cfg.out_dir) / run_name(cfg)
        if (self.run_dir.exists() and not resume
                and any((self.run_dir / "checkpoints").glob("checkpoint_*.pt"))):
            raise FileExistsError(f"run dir exists (no overwrites allowed): {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        # --- tasks ---
        if m == "SINGLE_TASK":
            self.tasks = [cfg.held_out]
        else:
            seen = [a for a in A.ALL_ALGORITHMS if a != cfg.held_out]
            order = S.rng(cfg.seed, S.STREAM_TASK_ORDER, cfg.held_out).permutation(len(seen))
            self.tasks = [seen[i] for i in order]
        self.joint = m == "JOINT_MULTITASK_FTN"
        self.n_blocks = 1 if m == "SINGLE_TASK" else len(self.tasks)

        # --- model ---
        mcfg = cfg.model
        kind = "ftn"
        if m in ("ORACLE_COMPOSITION", "SINGLE_TASK"):
            mcfg = oracle_model_cfg(mcfg)
        elif m == "DISJOINT_TASK_MODULES":
            k = mcfg.active_extender_atoms
            mcfg = dataclasses.replace(mcfg, extender_atoms=7 * k, reducer_atoms=7 * k)
        elif m == "DENSE_ACTIVE_MATCHED":
            kind = "dense"
        torch.manual_seed(S.torch_seed(cfg.seed, S.STREAM_MODEL_INIT, cfg.held_out, m))
        self.model = DPModel(mcfg, kind=kind).to(self.device)
        self.mcfg = mcfg

        self.replay = ReplayBuffer(cfg.replay.capacity,
                                   S.rng(cfg.seed, S.STREAM_REPLAY, cfg.held_out))
        self.mask_rng = S.rng(cfg.seed, S.STREAM_RANDOM_MASK, cfg.held_out)
        self.joint_rng = S.rng(cfg.seed, "joint_tasks", cfg.held_out)

        self.global_step = 0
        self.block_idx = 0
        self.step_in_block = 0
        self.opt = None
        self._block_prog = None
        self.boundary_records: list[dict] = []
        self.prototypes: dict[int, dict] = {}  # task -> {masks:[...], medoid_ext, medoid_red}
        self._support_cache: dict = {}
        self.eval_cache = EvalCache(cfg.seed, cfg)
        self.log_f = open(self.run_dir / "log.jsonl", "a")

        if resume:
            self._load_latest_checkpoint()
        else:
            save_config(cfg, self.run_dir / "config.yaml")
            (self.run_dir / "meta.json").write_text(json.dumps({
                "run_name": run_name(cfg), "config_hash": config_hash(cfg),
                "git": _git_hash(), "env": env_manifest(),
                "tasks": [A.ALGORITHM_NAMES[t] for t in self.tasks],
                "parameters": count_parameters(self.model)}, indent=1))
            self._save_checkpoint("init")

    # ------------------------------------------------------------------ utils

    def _log(self, rec: dict):
        rec["t"] = time.time()
        self.log_f.write(json.dumps(rec) + "\n")
        self.log_f.flush()

    def _lr(self, step_in_block: int, total: int) -> float:
        o = self.cfg.optimizer
        w = o.warmup_steps
        if step_in_block < w:
            return o.learning_rate * (step_in_block + 1) / w
        p = (step_in_block - w) / max(1, total - w)
        return o.minimum_learning_rate + 0.5 * (o.learning_rate - o.minimum_learning_rate) * (
            1 + math.cos(math.pi * min(1.0, p)))

    def _new_optimizer(self):
        o = self.cfg.optimizer
        return torch.optim.AdamW(self.model.parameters(), lr=o.learning_rate,
                                 betas=tuple(o.betas), eps=o.eps, weight_decay=o.weight_decay)

    def _edge_mode(self) -> str:
        w = self.mcfg.hard_attention_warmup_steps
        return "soft" if (self.block_idx == 0 and self.step_in_block < w) else "ste"

    def _disjoint_mask(self, block_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.mcfg.active_extender_atoms
        idx = range(block_idx * k, (block_idx + 1) * k)
        return (mask_from_indices(self.mcfg.extender_atoms, idx, self.device),
                mask_from_indices(self.mcfg.reducer_atoms, idx, self.device))

    def _train_batches(self, task: int):
        cfg = self.cfg
        gs = self.global_step
        sup = [G.generate("train", cfg.seed, "train_sup", cfg.held_out, gs, i)
               for i in range(cfg.data.support_size)]
        qry = [G.generate("train", cfg.seed, "train_qry", cfg.held_out, gs, i)
               for i in range(cfg.data.query_batch_size)]
        ss, qs = {g.seed for g in sup}, {g.seed for g in qry}
        assert not (ss & qs), "support/query seed overlap"
        sup_b = build_batch(sup, task, with_labels=True, with_trace=False).to(self.device)
        qry_labels = [A.execute(g, task) for g in qry]
        qry_b = build_batch(qry, task, with_labels=True, with_trace=True,
                            labels=qry_labels).to(self.device)
        return sup_b, qry, qry_labels, qry_b

    # ------------------------------------------------------------------ program acquisition

    def _acquire_program(self, task: int, sup_b, edge_mode: str) -> tuple[Program, dict]:
        m = self.cfg.method
        info = {}
        if m in ("ORACLE_COMPOSITION", "SINGLE_TASK"):
            me, mr = oracle_masks(task, self.device)
            return Program(gate_ext=me, gate_red=mr), info
        if m == "DISJOINT_TASK_MODULES":
            me, mr = self._disjoint_mask(self.block_idx if not self.joint else self.tasks.index(task))
            return Program(gate_ext=me, gate_red=mr), info
        if m == "FTN_RANDOM_MASK_TRAIN":
            me = random_topk_mask(self.mcfg.extender_atoms, self.mcfg.active_extender_atoms,
                                  self.mask_rng, self.device)
            mr = random_topk_mask(self.mcfg.reducer_atoms, self.mcfg.active_reducer_atoms,
                                  self.mask_rng, self.device)
            return Program(gate_ext=me, gate_red=mr), info
        if m == "FTN_PER_TASK_MASK":  # configure once at block start, reuse (design 12.6)
            if self._block_prog is None:
                res = configure(self.model, sup_b, self.cfg,
                                (self.cfg.seed, self.cfg.held_out, "train", self.global_step),
                                edge_mode=edge_mode, mode="ftn")
                self._block_prog = res.program
                info = {"config_init_loss": res.init_loss,
                        "config_final_loss": res.selection_loss}
            return self._block_prog, info
        mode = {"SOFT_MIXTURE": "soft", "DENSE_ACTIVE_MATCHED": "dense"}.get(m, "ftn")
        res = configure(self.model, sup_b, self.cfg,
                        (self.cfg.seed, self.cfg.held_out, "train", self.global_step),
                        edge_mode=edge_mode, mode=mode)
        info = {"config_init_loss": res.init_loss, "config_final_loss": res.selection_loss}
        # detached hard program; no backprop through inner loop in v1
        return res.program, info

    def _program_arrays(self, prog: Program) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        e = prog.gate_ext.detach().cpu().numpy() if prog.gate_ext is not None else None
        r = prog.gate_red.detach().cpu().numpy() if prog.gate_red is not None else None
        z = prog.z.detach().cpu().numpy() if prog.z is not None else None
        return e, r, z

    # ------------------------------------------------------------------ replay

    def _replay_batch(self, prog: Program):
        cfg = self.cfg
        if cfg.method in NO_REPLAY or len(self.replay) == 0:
            return None, []
        me, mr, _ = self._program_arrays(prog)
        uniform = cfg.method in UNIFORM_REPLAY
        if cfg.method == "SOFT_MIXTURE" and me is not None:
            k = self.mcfg.active_extender_atoms
            me = (me >= np.sort(me)[-k]).astype(np.float32)
            mr = (mr >= np.sort(mr)[-self.mcfg.active_reducer_atoms]).astype(np.float32)
        items = self.replay.sample(me, mr, self.mcfg.active_extender_atoms,
                                   self.mcfg.active_reducer_atoms, cfg.replay.overlap_low,
                                   cfg.replay.overlap_high, cfg.replay.max_batch,
                                   uniform=uniform)
        if not items:
            return None, []
        rb = build_batch([it.graph for it in items], None, with_labels=True, with_trace=True,
                         labels=[it.labels for it in items]).to(self.device)
        if self.cfg.method == "DENSE_ACTIVE_MATCHED":
            z = torch.tensor(np.stack([it.z for it in items]), dtype=torch.float32,
                             device=self.device)
            return (rb, Program(z=z)), items
        ge = torch.tensor(np.stack([it.mask_ext for it in items]), dtype=torch.float32,
                          device=self.device)
        gr = torch.tensor(np.stack([it.mask_red for it in items]), dtype=torch.float32,
                          device=self.device)
        return (rb, Program(gate_ext=ge, gate_red=gr)), items

    def _insert_replay(self, task: int, qry, qry_labels, prog: Program):
        cfg = self.cfg
        if cfg.method in NO_REPLAY:
            return
        me, mr, z = self._program_arrays(prog)
        if cfg.method == "SOFT_MIXTURE":
            k = self.mcfg.active_extender_atoms
            me = (me >= np.sort(me)[-k]).astype(np.float32)
            mr = (mr >= np.sort(mr)[-self.mcfg.active_reducer_atoms]).astype(np.float32)
        idx = self.replay.rng.choice(len(qry), size=min(cfg.replay.insertion_count, len(qry)),
                                     replace=False)
        for i in idx:
            self.replay.insert(ReplayItem(graph=qry[i], labels=qry_labels[i],
                                          mask_ext=me, mask_red=mr, z=z,
                                          algorithm_id=task, graph_seed=qry[i].seed,
                                          insertion_step=self.global_step))

    # ------------------------------------------------------------------ training

    def train(self):
        total_blocks = self.n_blocks
        while self.block_idx < total_blocks:
            task = None if self.joint else self.tasks[min(self.block_idx, len(self.tasks) - 1)]
            self._train_block(task)
            self.block_idx += 1
            self.step_in_block = 0
            self.opt = None  # AdamW state reset at every task boundary (design 10.1)
            self._block_prog = None
            self._save_checkpoint(f"task{self.block_idx - 1}")
        self._save_checkpoint("final")
        self.log_f.flush()

    def _train_block(self, task: int | None):
        cfg = self.cfg
        if self.opt is None:
            self.opt = self._new_optimizer()
        opt = self.opt
        steps = cfg.steps_per_task * (len(self.tasks) if self.joint else 1)
        t_block0 = time.time()
        while self.step_in_block < steps:
            step_task = int(self.joint_rng.choice(self.tasks)) if self.joint else task
            assert cfg.method == "SINGLE_TASK" or step_task != cfg.held_out, \
                "held-out algorithm reached the optimizer"
            lr = self._lr(self.step_in_block, steps)
            for gp in opt.param_groups:
                gp["lr"] = lr
            edge_mode = self._edge_mode()
            t0 = time.time()
            sup_b, qry, qry_labels, qry_b = self._train_batches(step_task)
            t_gen = time.time() - t0

            t0 = time.time()
            prog, cinfo = self._acquire_program(step_task, sup_b, edge_mode)
            t_cfg = time.time() - t0

            t0 = time.time()
            self.model.train()
            out = self.model(qry_b, prog, edge_mode=edge_mode,
                             return_trace=cfg.trace_supervision)
            qloss = model_training_loss(out, qry_b, cfg.loss, trace=cfg.trace_supervision)

            rb, ritems = self._replay_batch(prog)
            if rb is not None:
                rbatch, rprog = rb
                rout = self.model(rbatch, rprog, edge_mode=edge_mode,
                                  return_trace=cfg.trace_supervision)
                rloss = model_training_loss(rout, rbatch, cfg.loss, trace=cfg.trace_supervision)
                total = qloss["loss"] + cfg.replay.replay_weight * rloss["loss"]
            else:
                rloss = None
                total = qloss["loss"]

            opt.zero_grad(set_to_none=True)
            total.backward()
            if cfg.method == "DISJOINT_TASK_MODULES":
                self._mask_disjoint_grads(step_task)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optimizer.gradient_clip)
            opt.step()
            t_upd = time.time() - t0

            if self.global_step % cfg.replay.insertion_period == 0:
                self._insert_replay(step_task, qry, qry_labels, prog)

            if self.global_step % cfg.log_interval == 0:
                me, mr, _ = self._program_arrays(prog)
                rec = {"step": self.global_step, "block": self.block_idx,
                       "task": A.ALGORITHM_NAMES[step_task], "lr": lr,
                       "loss": float(total.detach()), "q_loss": float(qloss["loss"].detach()),
                       "q_final_value": float(qloss["final_value"].detach()),
                       "q_final_pointer": float(qloss["final_pointer"].detach()),
                       "r_loss": float(rloss["loss"].detach()) if rloss else None,
                       "replay_n": len(ritems), "grad_norm": float(gn),
                       "edge_mode": edge_mode, "t_gen": t_gen, "t_cfg": t_cfg, "t_upd": t_upd,
                       **cinfo}
                if me is not None:
                    rec["mask_ext_ids"] = np.flatnonzero(me > 0.5).tolist()[:16]
                    rec["mask_red_ids"] = np.flatnonzero(mr > 0.5).tolist()[:16]
                self._log(rec)

            self.global_step += 1
            self.step_in_block += 1
            if self.global_step % cfg.checkpoint_interval == 0:
                self._save_checkpoint(f"step{self.global_step}")
            if self.joint and self.step_in_block % cfg.steps_per_task == 0:
                self._boundary(pseudo_block=self.step_in_block // cfg.steps_per_task - 1)

        if not self.joint:
            self._boundary(pseudo_block=self.block_idx)
        self._log({"event": "block_done", "block": self.block_idx,
                   "wall_s": time.time() - t_block0})

    def _mask_disjoint_grads(self, task: int):
        """DISJOINT_TASK_MODULES: only the current block's module atoms receive gradients."""
        k = self.mcfg.active_extender_atoms
        b = self.block_idx if not self.joint else self.tasks.index(task)
        lo, hi = b * k, (b + 1) * k
        for bank in (self.model.ext_bank, self.model.red_bank):
            for p in (bank.w1, bank.b1, bank.w2, bank.b2):
                if p.grad is not None:
                    g = p.grad
                    g[:lo] = 0
                    g[hi:] = 0

    # ------------------------------------------------------------------ boundary evaluation

    def _seen_support(self, task: int, support_id: int):
        key = (task, support_id)
        if key not in self._support_cache:
            if self.cfg.method == "SINGLE_TASK":
                family = [task]
                graphs = SUP.random_support(self.cfg.seed, self.cfg.held_out, task, support_id,
                                            self.cfg.data.support_size)
            else:
                family = sorted(self.tasks)
                graphs = SUP.greedy_identifiable_support(
                    self.cfg.seed, self.cfg.held_out, task, family, support_id,
                    size=self.cfg.data.support_size,
                    pool_size=self.cfg.data.support_candidate_pool)
            self._support_cache[key] = graphs
        return self._support_cache[key]

    def _infer_for_task(self, task: int, support_id: int = 0, n_restarts: int | None = None):
        """Support-conditioned program for a seen task (no task identity to the model)."""
        m = self.cfg.method
        graphs = self._seen_support(task, support_id)
        sup_b = build_support_batch(graphs, task, self.device)
        if m in ("ORACLE_COMPOSITION", "SINGLE_TASK"):
            return infer_program(self.model, sup_b, self.cfg, (), "oracle", algorithm=task)[0]
        if m == "DISJOINT_TASK_MODULES":
            protos = [{"mask_ext": self._disjoint_mask(b)[0].cpu().numpy(),
                       "mask_red": self._disjoint_mask(b)[1].cpu().numpy(),
                       "task": A.ALGORITHM_NAMES[self.tasks[b]]}
                      for b in range(min(self.block_idx + 1, len(self.tasks)))]
            return infer_program(self.model, sup_b, self.cfg, (), "retrieval",
                                 prototypes=protos)[0]
        mode = {"SOFT_MIXTURE": "soft", "DENSE_ACTIVE_MATCHED": "dense"}.get(m, "ftn")
        key = (self.cfg.seed, self.cfg.held_out, "boundary", task, support_id, self.block_idx)
        prog, _, _ = infer_program(self.model, sup_b, self.cfg, key, mode,
                                   n_restarts=n_restarts)
        return prog

    def _boundary(self, pseudo_block: int):
        """R-matrix row + prototype masks for the finished task + checkpoint."""
        cfg = self.cfg
        t0 = time.time()
        if self.joint:
            finished = self.tasks if pseudo_block == len(self.tasks) - 1 else []
        else:
            finished = [self.tasks[pseudo_block]]
        if cfg.method not in ("ORACLE_COMPOSITION", "SINGLE_TASK", "DISJOINT_TASK_MODULES"):
            for task in finished:
                self._store_prototypes(task)

        row = {}
        trained = self.tasks if self.joint else self.tasks[:pseudo_block + 1]
        for task in (self.tasks if cfg.method != "SINGLE_TASK" else [cfg.held_out]):
            prog = self._infer_for_task(task)
            batches = self.eval_cache.batches(task, "id_val", cfg.boundary_eval_count)
            metrics = evaluate_program(self.model, prog, batches, cfg, self.device)
            row[A.ALGORITHM_NAMES[task]] = metrics["exact_graph_value_acc"]
            row[A.ALGORITHM_NAMES[task] + "/node_acc"] = metrics["node_acc"]
        rec = {"event": "boundary", "block": pseudo_block,
               "trained": [A.ALGORITHM_NAMES[t] for t in trained],
               "replay_hist": {str(k): v for k, v in self.replay.task_histogram().items()},
               "r_row": row, "wall_s": time.time() - t0}
        self.boundary_records.append(rec)
        (self.run_dir / "boundary.json").write_text(json.dumps(self.boundary_records, indent=1))
        self._log(rec)

    def _store_prototypes(self, task: int):
        """Ten prototype masks from ten independent support sets + Hamming medoid."""
        masks = []
        for sid in range(10):
            prog = self._infer_for_task(task, support_id=sid, n_restarts=1)
            me, mr, z = self._program_arrays(prog)
            if self.cfg.method == "SOFT_MIXTURE":
                k = self.mcfg.active_extender_atoms
                me = (me >= np.sort(me)[-k]).astype(np.float32)
                mr = (mr >= np.sort(mr)[-k]).astype(np.float32)
            if self.cfg.method == "DENSE_ACTIVE_MATCHED":
                masks.append({"z": z.tolist()})
            else:
                masks.append({"mask_ext": me.tolist(), "mask_red": mr.tolist()})
        entry = {"task": A.ALGORITHM_NAMES[task], "block": self.block_idx, "masks": masks}
        if "mask_ext" in masks[0]:
            entry["medoid_ext"] = medoid([np.array(m["mask_ext"]) for m in masks]).tolist()
            entry["medoid_red"] = medoid([np.array(m["mask_red"]) for m in masks]).tolist()
        elif "z" in masks[0]:
            zs = np.stack([np.array(m["z"]) for m in masks])
            entry["medoid_z"] = zs.mean(0).tolist()
        self.prototypes[task] = entry
        (self.run_dir / "prototypes.json").write_text(
            json.dumps({A.ALGORITHM_NAMES[k]: v for k, v in self.prototypes.items()}, indent=1))

    # ------------------------------------------------------------------ checkpointing

    def _save_checkpoint(self, tag: str):
        path = self.run_dir / "checkpoints" / f"checkpoint_{tag}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict() if self.opt is not None else None,
            "global_step": self.global_step, "block_idx": self.block_idx,
            "step_in_block": self.step_in_block,
            "tasks": self.tasks, "boundary_records": self.boundary_records,
            "prototypes": self.prototypes,
            "replay_items": self.replay.items, "replay_n_seen": self.replay.n_seen,
            "replay_rng": self.replay.rng.bit_generator.state,
            "mask_rng": self.mask_rng.bit_generator.state,
            "joint_rng": self.joint_rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "py_rng": random.getstate(),
            "config": to_dict(self.cfg), "config_hash": config_hash(self.cfg),
            "git": _git_hash(), "env": env_manifest(),
        }, path)
        latest = self.run_dir / "checkpoints" / "latest.txt"
        latest.write_text(str(path.name))

    def _load_latest_checkpoint(self):
        latest = self.run_dir / "checkpoints" / "latest.txt"
        path = self.run_dir / "checkpoints" / latest.read_text().strip()
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        if ck.get("optimizer") is not None:
            self.opt = self._new_optimizer()
            self.opt.load_state_dict(ck["optimizer"])
        self.global_step = ck["global_step"]
        self.block_idx = ck["block_idx"]
        self.step_in_block = ck["step_in_block"]
        self.tasks = ck["tasks"]
        self.boundary_records = ck["boundary_records"]
        self.prototypes = ck["prototypes"]
        self.replay.items = ck["replay_items"]
        self.replay.n_seen = ck["replay_n_seen"]
        self.replay.rng.bit_generator.state = ck["replay_rng"]
        self.mask_rng.bit_generator.state = ck["mask_rng"]
        self.joint_rng.bit_generator.state = ck["joint_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["cuda_rng"]])
        random.setstate(ck["py_rng"])
