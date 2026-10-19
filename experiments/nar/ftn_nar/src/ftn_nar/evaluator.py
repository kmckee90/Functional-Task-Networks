"""Evaluation: fixed seeded eval sets, exact-graph metrics, retention and induction
protocols (design sections 4.7, 14, 19.4)."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch

from . import algorithms as A
from . import generator as G
from . import ops
from . import support as SUP
from .batch import GraphBatch, build_batch
from .config import ExperimentCfg
from .configurer import configure_with_restarts, local_swap, params_frozen
from .losses import support_loss
from .model import Program, oracle_masks

# condition name -> (dist, kwargs)
CONDITIONS: dict[str, tuple[str, dict]] = {
    "id_val": ("train", {}),
    "id_test": ("train", {}),
    "ood_width_32": ("ood_width", {"n_target": 32}),
    "ood_width_64": ("ood_width", {"n_target": 64}),
    "ood_width_128": ("ood_width", {"n_target": 128}),
    "ood_depth_8": ("ood_depth", {"d_target": 8}),
    "ood_depth_16": ("ood_depth", {"d_target": 16}),
    "ood_depth_32": ("ood_depth", {"d_target": 32}),
    "ood_both_32": ("ood_both", {"n_target": 32}),
    "ood_both_64": ("ood_both", {"n_target": 64}),
    "ood_both_128": ("ood_both", {"n_target": 128}),
}


def condition_count(cond: str, cfg: ExperimentCfg) -> int:
    if cond == "id_val":
        return cfg.data.id_val_count
    if cond == "id_test":
        return cfg.data.id_test_count
    return cfg.data.ood_count


def eval_graphs(run_seed: int, algorithm: int, cond: str, count: int) -> list[G.Graph]:
    dist, kw = CONDITIONS[cond]
    stream = f"eval/{cond}/{algorithm}"
    return [G.generate(dist, run_seed, stream, i, **kw) for i in range(count)]


class EvalCache:
    """CPU-side cache of labeled, chunked GraphBatches per (algorithm, condition)."""

    def __init__(self, run_seed: int, cfg: ExperimentCfg):
        self.run_seed = run_seed
        self.cfg = cfg
        self._batches: dict[tuple, list[GraphBatch]] = {}
        self._seeds: dict[tuple, set] = {}

    def batches(self, algorithm: int, cond: str, count: int | None = None) -> list[GraphBatch]:
        count = count or condition_count(cond, self.cfg)
        key = (algorithm, cond, count)
        if key not in self._batches:
            graphs = eval_graphs(self.run_seed, algorithm, cond, count)
            self._seeds[key] = {g.seed for g in graphs}
            chunks, cur, nodes = [], [], 0
            for g in graphs:
                if cur and nodes + g.n > self.cfg.data.eval_node_chunk:
                    chunks.append(cur)
                    cur, nodes = [], 0
                cur.append(g)
                nodes += g.n
            if cur:
                chunks.append(cur)
            self._batches[key] = [
                build_batch(c, algorithm, with_labels=True, with_trace=False) for c in chunks]
        return self._batches[key]

    def seeds(self, algorithm: int, cond: str, count: int | None = None) -> set:
        count = count or condition_count(cond, self.cfg)
        self.batches(algorithm, cond, count)
        return self._seeds[(algorithm, cond, count)]


def manifest_entry(run_seed: int, algorithm: int, cond: str, count: int) -> dict:
    graphs = eval_graphs(run_seed, algorithm, cond, count)
    blob = b"".join(A.final_values(g, algorithm).tobytes() for g in graphs)
    return {"cond": cond, "algorithm": algorithm, "seed": run_seed, "count": count,
            "label_checksum": zlib.crc32(blob)}


def write_manifest(run_seed: int, algorithm: int, cfg: ExperimentCfg, path: str):
    entries = [manifest_entry(run_seed, algorithm, c, condition_count(c, cfg)) for c in CONDITIONS]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(entries, indent=1))


@torch.no_grad()
def evaluate_program(model, prog: Program, batches: list[GraphBatch], cfg: ExperimentCfg,
                     device) -> dict:
    """Exact-graph metrics for one fixed program on pre-built batches (hard execution)."""
    tot_nodes = tot_correct_nodes = 0
    tot_graphs = 0
    exact_val = exact_ptr = exact_both = 0
    abs_err_sum = 0.0
    max_abs = []
    ptr_correct = 0
    sup_losses = []

    model.eval()
    for cpu_batch in batches:
        batch = cpu_batch.to(device)
        out = model(batch, prog, edge_mode="hard", return_trace=False)
        sup_losses.append(float(support_loss(out, batch, cfg.loss)))

        gidx = batch.graph_index_per_node
        gcount = batch.num_graphs
        nonsrc = ~batch.source_mask
        target = batch.final_values
        pred_round = torch.round(out["final_values"])
        node_ok = (pred_round == target) | batch.source_mask  # source pinned exact
        err = (out["final_values"] - target).abs() * nonsrc.float()

        graph_val_ok = torch.ones(gcount, dtype=torch.long, device=device)
        graph_val_ok.scatter_reduce_(0, gidx, node_ok.long(), reduce="amin", include_self=True)
        graph_val_ok = graph_val_ok > 0
        gmax = torch.zeros(gcount, device=device)
        gmax.scatter_reduce_(0, gidx, err, reduce="amax", include_self=True)

        # pointer: argmax reducer score among valid incoming edges at final step
        p_hard = ops.segment_argmax_onehot(out["final_edge_scores"], batch.edge_index[1],
                                           batch.num_nodes, out["final_edge_valid"])
        chosen_ok_e = (p_hard > 0.5) & batch.valid_pred_edge_mask
        node_ptr_ok = ops.segment_any(chosen_ok_e, batch.edge_index[1], batch.num_nodes)
        node_ptr_ok = node_ptr_ok | batch.source_mask
        graph_ptr_ok = torch.ones(gcount, dtype=torch.long, device=device)
        graph_ptr_ok.scatter_reduce_(0, gidx, node_ptr_ok.long(), reduce="amin",
                                     include_self=True)
        graph_ptr_ok = graph_ptr_ok > 0

        tot_nodes += int(nonsrc.sum())
        tot_correct_nodes += int((node_ok & nonsrc).sum())
        abs_err_sum += float(err.sum())
        tot_graphs += gcount
        exact_val += int(graph_val_ok.sum())
        exact_ptr += int(graph_ptr_ok.sum())
        exact_both += int((graph_val_ok & graph_ptr_ok).sum())
        max_abs.extend(gmax.cpu().tolist())
        ptr_correct += int((node_ptr_ok & nonsrc).sum())

    return {
        "graphs": tot_graphs,
        "node_acc": tot_correct_nodes / max(1, tot_nodes),
        "node_mae": abs_err_sum / max(1, tot_nodes),
        "exact_graph_value_acc": exact_val / max(1, tot_graphs),
        "exact_graph_pointer_acc": exact_ptr / max(1, tot_graphs),
        "exact_graph_value_and_pointer_acc": exact_both / max(1, tot_graphs),
        "pointer_node_acc": ptr_correct / max(1, tot_nodes),
        "mean_max_abs_err": float(np.mean(max_abs)) if max_abs else 0.0,
        "query_support_loss": float(np.mean(sup_losses)),
    }


def build_support_batch(graphs: list[G.Graph], algorithm: int, device):
    return build_batch(graphs, algorithm, with_labels=True, with_trace=False).to(device)


def infer_program(model, support_batch, cfg: ExperimentCfg, key: tuple, method_mode: str,
                  prototypes: list[dict] | None = None, algorithm: int | None = None,
                  n_restarts: int | None = None, edge_mode: str = "ste"):
    """Evaluation-time program inference. Returns (Program, info, ConfigResult|None).

    method_mode: 'ftn' | 'soft' | 'dense' | 'retrieval' | 'oracle'.
    'retrieval' scores stored prototype masks on the support set and picks the best
    (FTN_RETRIEVAL, DISJOINT_TASK_MODULES). 'oracle' uses the true labeled modules.
    """
    dev = support_batch.node_features.device
    info: dict = {"mode": method_mode}
    if method_mode == "oracle":
        m_ext, m_red = oracle_masks(algorithm, dev)
        return Program(gate_ext=m_ext, gate_red=m_red), info, None
    if method_mode == "retrieval":
        assert prototypes, "retrieval needs stored prototypes"
        best, best_loss = None, float("inf")
        with torch.no_grad(), params_frozen(model):
            for proto in prototypes:
                prog = Program(
                    gate_ext=torch.as_tensor(proto["mask_ext"], dtype=torch.float32, device=dev),
                    gate_red=torch.as_tensor(proto["mask_red"], dtype=torch.float32, device=dev))
                out = model(support_batch, prog, edge_mode="hard", return_trace=False)
                l = float(support_loss(out, support_batch, cfg.loss))
                if l < best_loss:
                    best, best_loss = (prog, proto), l
        info.update({"selection_loss": best_loss, "retrieved_task": best[1].get("task")})
        return best[0], info, None
    n_r = n_restarts if n_restarts is not None else cfg.configurer.eval_restarts
    best, allr = configure_with_restarts(model, support_batch, cfg, key, n_r,
                                         edge_mode=edge_mode, mode=method_mode)
    info.update({"selection_loss": best.selection_loss,
                 "init_loss": best.init_loss,
                 "restart_losses": [r.selection_loss for r in allr]})
    return best.program, info, best


def held_out_evaluation(model, cfg: ExperimentCfg, run_seed: int, held_out: int,
                        method_mode: str, prototypes: list[dict] | None, device,
                        cache: EvalCache, n_support_sets: int | None = None,
                        identifiable: bool = True, conditions: list[str] | None = None,
                        do_local_swap: bool = False) -> dict:
    """Design 19.4: per support set, infer once, evaluate the same program everywhere."""
    n_sets = n_support_sets or cfg.n_eval_support_sets
    conditions = conditions or list(CONDITIONS)
    family = A.ALL_ALGORITHMS
    results = []
    for sid in range(n_sets):
        if identifiable:
            graphs = SUP.greedy_identifiable_support(run_seed, held_out, held_out, family,
                                                     support_id=sid,
                                                     size=cfg.data.support_size,
                                                     pool_size=cfg.data.support_candidate_pool)
        else:
            graphs = SUP.random_support(run_seed, held_out, held_out, sid,
                                        size=cfg.data.support_size)
        consistent = SUP.consistent_algorithms(graphs, held_out, family)
        sbatch = build_support_batch(graphs, held_out, device)
        prog, info, cres = infer_program(model, sbatch, cfg,
                                         (run_seed, held_out, "eval", sid),
                                         method_mode, prototypes=prototypes,
                                         algorithm=held_out)
        sup_seeds = {g.seed for g in graphs}
        entry = {"support_id": sid, "identifiable": identifiable,
                 "consistency_set": [A.ALGORITHM_NAMES[a] for a in consistent],
                 "consistency_size": len(consistent), "inference": info,
                 "conditions": {}}
        if prog.gate_ext is not None:
            entry["mask_ext"] = prog.gate_ext.detach().cpu().numpy().tolist()
            entry["mask_red"] = prog.gate_red.detach().cpu().numpy().tolist()
        for cond in conditions:
            batches = cache.batches(held_out, cond)
            for b in batches:
                assert not (set(b.graph_seeds) & sup_seeds), "support/query seed overlap"
            entry["conditions"][cond] = evaluate_program(model, prog, batches, cfg, device)
        if do_local_swap and method_mode == "ftn" and cres is not None:
            swapped = local_swap(model, sbatch, cfg, cres)
            sw = {"selection_loss": swapped.selection_loss, "conditions": {}}
            for cond in conditions:
                sw["conditions"][cond] = evaluate_program(
                    model, swapped.program, cache.batches(held_out, cond), cfg, device)
            entry["local_swap"] = sw
        results.append(entry)

    agg = {"held_out": A.ALGORITHM_NAMES[held_out], "mode": method_mode,
           "identifiable": identifiable, "support_sets": results}
    if results and "mask_ext" in results[0]:
        me = [np.array(r["mask_ext"]) for r in results]
        mr = [np.array(r["mask_red"]) for r in results]
        ham = [int(np.count_nonzero(me[i] != me[j])) + int(np.count_nonzero(mr[i] != mr[j]))
               for i in range(len(me)) for j in range(i + 1, len(me))]
        agg["mask_pairwise_hamming_mean"] = float(np.mean(ham)) if ham else 0.0
        agg["mask_identical_fraction"] = float(np.mean([h == 0 for h in ham])) if ham else 1.0
    for cond in conditions:
        vals = [r["conditions"][cond]["exact_graph_value_acc"] for r in results]
        agg[f"{cond}/exact_mean"] = float(np.mean(vals))
        agg[f"{cond}/exact_std"] = float(np.std(vals))
    return agg
