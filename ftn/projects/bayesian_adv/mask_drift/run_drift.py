"""Measure predictor drift under exact task masks; reuse the parent train loop.

The fixed query data, mask, and original predicted probabilities are retained
for every task. Later evaluations pass that same mask explicitly to forward(),
without any mask search. The `replay_best` mask is selected from masks actually
stored by the ordinary filtered-replay buffer, using separate task support.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))
import config  # noqa: E402
import metrics  # noqa: E402
import run_experiments as pipeline  # noqa: E402
from bayes_lora_ftn import BayesLoRAFTN  # noqa: E402
from ftn import BayesFTN, FTN  # noqa: E402
from lora_ftn import LoRAFTN  # noqa: E402


class PureDomainShift(config.DomainShift):
    CONCEPT_SHIFT = False


class ConceptShift(config.Default):
    pass


class DriftSettings:
    # Keep the parent architecture, training schedule, reconfiguration, and
    # filtered replay; remove the ordinary cold-search eval and visualizations.
    EVAL_BACKWARDS = False
    EVAL_PROSPECTIVE = False
    RENDER_STEPS = False
    ENTROPY_TESTS = False


ARCHITECTURES = {
    "Base": (config.Base, FTN),
    "LoRA": (config.LoRA, LoRAFTN),
    "Bayes": (config.Bayes, BayesFTN),
    "BayesLoRA": (config.BayesLoRA, BayesLoRAFTN),
}

FIELDS = [
    "block", "task", "mask_source", "acc", "ce", "kl_from_origin",
    "tv_from_origin", "class_flip_fraction", "mask_ones", "overlap_current_mask",
]


def fixed_probe_data(data_gen, seed, n_query_batches=2):
    # Reuse the exact task generator parameters but a distinct input RNG stream.
    # Restore its RNG afterwards so training receives the normal data stream.
    rng_state = data_gen.rng.get_state()
    data_gen.rng.manual_seed(seed + 100_000)
    support, query = {}, {}
    for task in range(data_gen.n_tasks):
        support[task] = data_gen(task)
        parts = [data_gen(task) for _ in range(n_query_batches)]
        query[task] = tuple(torch.cat([p[i] for p in parts]) for i in (0, 1))
    data_gen.rng.set_state(rng_state)
    return support, query


class ChangepointReplayBuffer(pipeline.ReplayBuffer):
    """Refresh or append outgoing mask/data at a detected changepoint."""

    def __init__(self, maxlen, batch_size, tail_batches, policy="replace",
                 append_batches=32):
        super().__init__(maxlen, batch_size)
        if policy not in ("replace", "append"):
            raise ValueError(policy)
        self.policy = policy
        self.append_batches = append_batches
        self.segment = 0
        self.segment_ids = deque(maxlen=maxlen)
        self.task_ids = deque(maxlen=maxlen)
        self.current_task = None
        self.protected_group = None
        self.tails = defaultdict(lambda: deque(maxlen=tail_batches))
        self.events = []

    def add(self, dat, m):
        super().add(dat, m)
        self.segment_ids.append(self.segment)
        self.task_ids.append(self.current_task)
        self.protected_group = self.segment

    def add_completed_batch(self, dat):
        self.tails[self.segment].append((dat, self.current_task))

    def snapshot_before_change(self, old_mask):
        old_segment = self.segment
        # A later false changepoint may occur after this task's first 100
        # storage batches. It then has no new slots. Keep refreshing the most
        # recently allocated slot group until the next task allocates its own.
        slots = ([i for i, sid in enumerate(self.segment_ids)
                  if sid == self.protected_group]
                 if self.protected_group is not None else [])
        recent = list(self.tails.get(old_segment, ()))
        event = {"segment": old_segment, "slot_group": self.protected_group,
                 "stored_slots": len(slots), "tail_batches": len(recent),
                 "policy": self.policy, "buffer_before": len(self)}
        if self.policy == "replace":
            n = min(len(slots), len(recent)) if old_mask.sum() > 0 else 0
            pairs = zip(slots[-n:], recent[-n:]) if n else ()
            for slot, ((x, y), task) in pairs:
                if self.batch_size is not None:
                    x, y = x[:self.batch_size], y[:self.batch_size]
                self.data[slot] = (x, y)
                self.masks[slot] = old_mask.detach().clone()
                self.task_ids[slot] = task
            event["replaced_slots"] = n
        else:
            n = min(self.append_batches, len(recent)) if old_mask.sum() > 0 else 0
            event["evicted_slots"] = max(0, len(self) + n - self.data.maxlen)
            for (x, y), task in (recent[-n:] if n else ()):
                if self.batch_size is not None:
                    x, y = x[:self.batch_size], y[:self.batch_size]
                self.data.append((x, y))
                self.masks.append(old_mask.detach().clone())
                self.segment_ids.append(old_segment)
                self.task_ids.append(task)
            if n:
                self.protected_group = old_segment
            event["appended_slots"] = n
        event["buffer_after"] = len(self)
        self.events.append(event)
        self.tails.pop(old_segment, None)
        self.segment += 1
        return event


class TrackedDataGen:
    """Give the snapshot hook only batches completed before the current one."""

    def __init__(self, data_gen, replay):
        self.data_gen = data_gen
        self.replay = replay
        self.pending = None

    def __call__(self, task):
        if self.pending is not None:
            self.replay.add_completed_batch(self.pending)
        self.replay.current_task = task
        self.pending = self.data_gen(task)
        return self.pending


class DriftCollector(metrics.MetricsCollector):
    def __init__(self, ftn, cfg, replay, support, query, out_dir):
        super().__init__()
        self.ftn = ftn
        self.cfg = cfg
        self.replay = replay
        self.support = support
        self.query = query
        self.out_dir = out_dir
        self.masks = {}
        self.origin_probs = {}
        self.probe_rows = []

    def _best_replay_mask(self, task):
        # Rolling snapshots can interleave tasks and evict early entries.
        # Select from the actual stored masks labeled with this task.
        if hasattr(self.replay, "task_ids"):
            candidates = [m for m, tid in zip(self.replay.masks,
                                               self.replay.task_ids) if tid == task]
        else:
            candidates = list(self.replay.masks)[-self.cfg.N_SAMPLE_PER_TASK:]
        if not candidates:
            raise RuntimeError(f"No replay mask remains for task {task}")
        indices = np.unique(np.linspace(0, len(candidates) - 1, 16).round().astype(int))
        x, y = self.support[task]
        was_training = self.ftn.training
        self.ftn.eval()
        with torch.no_grad():
            losses = [float(self.ftn.loss(self.ftn(x, m=candidates[i]), y))
                      for i in indices]
        self.ftn.train(was_training)
        return candidates[int(indices[int(np.argmin(losses))])].detach().clone()

    def _cold_mask(self, task):
        # Additional task-end oracle: search a full support batch once, freeze
        # the resulting exact hard mask, and never search it again.
        x, y = self.support[task]
        state = self.ftn.get_reconfig_state()
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            self.ftn.cold_reconfigure(
                x, y, steps=self.cfg.EVAL_RECONFIG_STEPS,
                lr=self.cfg.EVAL_RECONFIG_LR, sample_size=None,
                temperature=self.cfg.EVAL_RECONFIG_TEMPERATURE,
                restarts=self.cfg.EVAL_RECONFIG_RESTARTS, take_best=True,
            )
            mask = self.ftn.m.detach().clone()
        self.ftn.set_reconfig_state(state)
        return mask

    def _measure(self, block, task, source):
        mask = self.masks[(task, source)]
        x, y = self.query[task]
        with torch.no_grad():
            logits = self.ftn(x, m=mask)
            p = logits.softmax(-1)
            acc = pipeline.accuracy(logits, y)
            ce = float(self.ftn.loss(logits, y))
            origin = self.origin_probs.get((task, source))
            if origin is None:
                self.origin_probs[(task, source)] = p.detach().clone()
                kl = tv = flips = 0.0
            else:
                eps = 1e-8
                kl = float((origin * (origin.clamp_min(eps).log()
                                       - p.clamp_min(eps).log())).sum(-1).mean())
                tv = float(0.5 * (origin - p).abs().sum(-1).mean())
                flips = float((origin.argmax(-1) != p.argmax(-1)).float().mean())
        return {
            "block": block, "task": task, "mask_source": source,
            "acc": acc, "ce": ce, "kl_from_origin": kl,
            "tv_from_origin": tv, "class_flip_fraction": flips,
            "mask_ones": int(mask.sum()),
            "overlap_current_mask": float((mask * self.ftn.m).sum() / mask.sum()),
        }

    def record_block(self, order, task, train_acc, eval_accs, seen_tasks,
                     block_wall_s=None):
        super().record_block(order, task, train_acc, eval_accs, seen_tasks,
                             block_wall_s=block_wall_s)
        # Called by the unchanged parent loop after optimizer updates finish.
        self.masks[(task, "last_train")] = self.ftn.m.detach().clone()
        self.masks[(task, "replay_best")] = self._best_replay_mask(task)
        self.masks[(task, "cold_oracle")] = self._cold_mask(task)
        was_training = self.ftn.training
        self.ftn.eval()
        for prior_task in seen_tasks:
            for source in ("last_train", "replay_best", "cold_oracle"):
                self.probe_rows.append(self._measure(order, prior_task, source))
        self.ftn.train(was_training)
        self.write_csv()
        old = [r for r in self.probe_rows if r["block"] == order
               and r["task"] < task and r["mask_source"] == "replay_best"]
        print(f"  [exact-mask drift] after task {task}: "
              f"old replay-mask acc={np.mean([r['acc'] for r in old]):.4f} "
              f"KL={np.mean([r['kl_from_origin'] for r in old]):.4f}"
              if old else f"  [exact-mask drift] after task {task}: baseline saved",
              flush=True)

    def write_csv(self):
        with (self.out_dir / "trajectories.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.probe_rows)

    def save_figures(self):
        for source in ("last_train", "replay_best", "cold_oracle"):
            rows = [r for r in self.probe_rows if r["mask_source"] == source]
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
            for task in range(self.cfg.N_TASK):
                sub = [r for r in rows if r["task"] == task]
                blocks = [r["block"] for r in sub]
                for ax, key in zip(axes, ("acc", "ce", "kl_from_origin")):
                    ax.plot(blocks, [r[key] for r in sub], marker="o", label=f"task {task}")
            for ax, label in zip(axes, ("Accuracy", "Cross-entropy", "KL from task end")):
                ax.set(xlabel="Last trained task block", ylabel=label)
                ax.grid(alpha=0.3)
            axes[0].set_ylim(0.4, 1.0)
            axes[0].legend(fontsize=7, ncol=2)
            fig.suptitle(f"Exact frozen {source} masks; fixed query examples")
            fig.tight_layout()
            fig.savefig(self.out_dir / f"drift_{source}.png", dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--shift", choices=("domain", "concept"), required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-root", default="artifacts")
    parser.add_argument("--replay-samples", type=int, default=None,
                        help="Override replay-gradient batches per step; keep the buffer.")
    parser.add_argument("--snapshot-on-changepoint", action="store_true",
                        help="Store the outgoing mask and recent data at detected changes.")
    parser.add_argument("--snapshot-policy", choices=("replace", "append"),
                        default="replace")
    parser.add_argument("--cp-append-batches", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    arch_cfg, model_class = ARCHITECTURES[args.architecture]
    shift_cfg = PureDomainShift if args.shift == "domain" else ConceptShift
    overrides = type("RunOverrides", (DriftSettings,), {
        "SEED": args.seed,
        **({"N_REPLAY_SAMPLES": args.replay_samples}
           if args.replay_samples is not None else {}),
        **({"N_TASK": 3, "TRAIN_DURATION": 4, "N_SAMPLE_PER_TASK": 2,
            "SQUARE_DIM": 4, "K": 4, "LORA_LAYERS": 2,
            "LORA_WIDTH": 16, "LORA_ALPHA": 4.0,
            "EVAL_RECONFIG_STEPS": 2} if args.smoke else {}),
        **({"TRAIN_DURATION": 12, "N_SAMPLE_PER_TASK": 1,
            "RECONFIG_REACTIVE_SD": 0.0}
           if args.smoke and args.snapshot_on_changepoint else {}),
    })
    cfg = type(f"{args.architecture}_{args.shift}_drift",
               (overrides, shift_cfg, arch_cfg), {})()
    pipeline.Config = cfg
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    data_gen = pipeline.make_data_gen(args.seed + 100)
    support, query = fixed_probe_data(data_gen, args.seed)
    pars_by_arch = {
        "Base": dict(cfg.FTN_PARS),
        "LoRA": dict(cfg.LORA_PARS),
        "Bayes": {**cfg.FTN_PARS, **cfg.BAYES_FTN_PARS},
        "BayesLoRA": {**cfg.LORA_FTN_PARS, **cfg.BAYES_FTN_PARS},
    }
    ftn = model_class(
        input_size=data_gen.total_input_dim,
        output_size=data_gen.output_dim,
        hidden_size=cfg.HIDDEN_SIZE,
        **pars_by_arch[args.architecture],
    )
    optimizer = torch.optim.Adam(
        [p for p in ftn.parameters() if p.requires_grad], lr=cfg.TRAIN_LR)
    if args.snapshot_on_changepoint and not cfg.RECONFIG_REACTIVE:
        parser.error("changepoint snapshots require reactive reconfiguration")
    replay = (ChangepointReplayBuffer(cfg.REPLAY_MAXLEN, cfg.REPLAY_BATCH_SIZE,
                                     cfg.N_SAMPLE_PER_TASK,
                                     policy=args.snapshot_policy,
                                     append_batches=args.cp_append_batches)
              if args.snapshot_on_changepoint else
              pipeline.ReplayBuffer(cfg.REPLAY_MAXLEN, cfg.REPLAY_BATCH_SIZE))
    training_gen = data_gen
    if args.snapshot_on_changepoint:
        training_gen = TrackedDataGen(data_gen, replay)
        original_reconfigure = ftn.reconfigure_changepoint

        def reconfigure_with_snapshot(x, y):
            old_mask = ftn.m.detach().clone()
            event = replay.snapshot_before_change(old_mask)
            result = original_reconfigure(x, y)
            if old_mask.sum() > 0:
                event["overlap_after_change"] = float(
                    (old_mask * ftn.m).sum() / old_mask.sum())
                replay_overlap = float((old_mask * ftn.m).sum() / ftn.m.sum())
                event["snapshot_replay_eligible"] = 0.2 < replay_overlap <= 0.8
            print(f"  [changepoint snapshot] {event}", flush=True)
            return result

        ftn.reconfigure_changepoint = reconfigure_with_snapshot
    replay_tag = (f"_r{args.replay_samples}"
                  if args.replay_samples is not None else "")
    snapshot_tag = ("_cp" if args.snapshot_policy == "replace" else
                    f"_cp_append{args.cp_append_batches}") if args.snapshot_on_changepoint else ""
    out_dir = (Path(args.out_root) /
               f"{args.architecture}_{args.shift}_{args.seed}{replay_tag}{snapshot_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()
                and isinstance(getattr(cfg, k), (str, int, float, bool, type(None)))}
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    collector = DriftCollector(ftn, cfg, replay, support, query, out_dir)
    print(f"Running {args.architecture} {args.shift} seed={args.seed} "
          f"train_duration={cfg.TRAIN_DURATION}, tasks={cfg.N_TASK}, "
          f"replay={cfg.N_REPLAY_SAMPLES}, cp_snapshot={args.snapshot_on_changepoint}",
          flush=True)
    pipeline.train(
        ftn, training_gen, list(range(cfg.N_TASK)), cfg.N_EPOCHS,
        out_dir.name, optimizer, replay, out_dir=str(out_dir),
        collector=collector, save_viz=False,
    )
    collector.save_figures()
    (out_dir / "train_metrics.json").write_text(json.dumps(collector.summary(), indent=2))
    if args.snapshot_on_changepoint:
        (out_dir / "snapshot_events.json").write_text(json.dumps(replay.events, indent=2))
        counts = {str(task): list(replay.task_ids).count(task)
                  for task in range(cfg.N_TASK)}
        counts["unlabeled"] = list(replay.task_ids).count(None)
        (out_dir / "buffer_composition.json").write_text(json.dumps(counts, indent=2))
    print(f"Wrote {out_dir / 'trajectories.csv'}", flush=True)


if __name__ == "__main__":
    main()
