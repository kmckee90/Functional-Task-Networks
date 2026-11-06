"""
metrics.py - Scalar continual-learning + cost metrics for one FTN run.

Two halves, both architecture-agnostic (they touch only ftn.m, ftn._reconfig,
ftn(x) and ftn.parameters(), which every FTN variant in this repo exposes):

  - MetricsCollector: accumulates per-task-block training / eval accuracies as
    run_experiments.train() steps through the task stream (one row per backwards
    eval). summary() reduces those rows to the requested continual-learning
    scalars (recall, forward transfer, forward-transfer trend, current-task
    train vs. eval accuracy, and mean per-task-block training wall time -
    reconfig + forward + backward together).

  - Cost helpers: parameter counts, forward FLOPs, forward / reconfiguration
    wall time, and the average pairwise task-mask overlap (Jaccard), measured
    once after training on the trained model.

write_metrics() serializes a flat {key: value} dict to a parseable plaintext
file (KEY = value, NA for anything unavailable) so a later aggregation pass can
pool metrics across seeds / conditions. This module never imports config; the
caller passes in whatever it needs.
"""

import time

import numpy as np
import torch

# Key order for the emitted metrics file (extras are appended after these).
_KEY_ORDER = [
    "run_name",
    "config",
    "seed",
    "n_tasks",
    "n_params_trainable",
    "n_params_total",
    "flops_forward",
    "wall_time_forward_ms",
    "wall_time_reconfig_ms",
    "avg_task_block_wall_s",
    "avg_recall_eval_acc",
    "avg_forward_transfer_eval_acc",
    "avg_forward_transfer_change_per_block",
    "avg_train_acc_current_block",
    "avg_eval_acc_current_task",
    "avg_mask_overlap_pct",
]


def _mean_or_none(vals):
    return float(np.mean(vals)) if len(vals) else None


# =============================================================================
# Per-block accuracy collector (populated during training)
# =============================================================================


class MetricsCollector:
    """Records one row per completed task block (at its backwards eval).

    Each row carries the block's training accuracy, the backwards/prospective
    eval accuracies keyed by task, and the set of tasks seen so far (including
    the just-trained one). summary() derives the continual-learning scalars from
    these rows, so all filtering (what counts as recall vs. forward transfer)
    lives in one place.
    """

    def __init__(self):
        self.rows = []

    def record_block(self, order, task, train_acc, eval_accs, seen_tasks,
                     block_wall_s=None):
        """order: 0-based block index across the whole run (epochs included).
        task: task index trained in this block.
        train_acc: mean training accuracy over the block's tail (or None).
        eval_accs: {task_idx: eval_acc} from the backwards eval ({} if disabled).
        seen_tasks: iterable of tasks trained so far, INCLUDING `task`.
        block_wall_s: wall time (s) of the block's training loop - reconfig,
            forward, backward, replay, optimizer all together (or None).
        """
        self.rows.append(
            {
                "order": order,
                "task": task,
                "train_acc": train_acc,
                "eval_accs": dict(eval_accs),
                "seen": set(seen_tasks),
                "block_wall_s": block_wall_s,
            }
        )

    def summary(self):
        """Reduce the recorded rows to the requested continual-learning scalars.

        recall            = eval acc on tasks trained BEFORE the current block
                            (retention of old solutions), pooled over blocks.
        forward transfer  = eval acc on tasks NOT YET trained (prospective),
                            pooled over blocks.
        ft change / block = mean consecutive change in the per-block mean forward
                            transfer, i.e. how much better forward transfer gets
                            with each new task block.
        current train acc = mean per-block tail training accuracy.
        current eval acc  = eval acc on the task the block just trained
                            (recovery of the freshly trained solution).
        """
        recall_vals, ft_vals, cur_vals, train_vals, wall_vals = [], [], [], [], []
        ft_per_block = []
        for r in self.rows:
            if r["train_acc"] is not None:
                train_vals.append(r["train_acc"])
            if r.get("block_wall_s") is not None:
                wall_vals.append(r["block_wall_s"])
            ea, seen, task = r["eval_accs"], r["seen"], r["task"]
            if task in ea:
                cur_vals.append(ea[task])
            recall_vals.extend(a for t, a in ea.items() if t != task and t in seen)
            fts = [a for t, a in ea.items() if t not in seen]
            ft_vals.extend(fts)
            if fts:
                ft_per_block.append((r["order"], float(np.mean(fts))))

        ft_change = None
        if len(ft_per_block) >= 2:
            ft_per_block.sort()
            ft_change = float(np.mean(np.diff([v for _, v in ft_per_block])))

        return {
            "avg_task_block_wall_s": _mean_or_none(wall_vals),
            "avg_recall_eval_acc": _mean_or_none(recall_vals),
            "avg_forward_transfer_eval_acc": _mean_or_none(ft_vals),
            "avg_forward_transfer_change_per_block": ft_change,
            "avg_train_acc_current_block": _mean_or_none(train_vals),
            "avg_eval_acc_current_task": _mean_or_none(cur_vals),
        }


# =============================================================================
# Cost / structure metrics (measured after training, on the trained model)
# =============================================================================


def count_params(ftn):
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in ftn.parameters() if p.requires_grad)
    total = sum(p.numel() for p in ftn.parameters())
    return int(trainable), int(total)


def measure_flops(ftn, x):
    """FLOPs for a single forward pass under the current mask, or None if the
    FLOP counter is unavailable / errors. Uses torch's FlopCounterMode, which
    tallies matmul/bmm/conv/einsum (the dominant cost); mask values don't change
    the count. Runs in eval() + no_grad so it never touches ftn's grad state."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    was_training = ftn.training
    ftn.eval()
    total = None
    try:
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            ftn(x)
        total = counter.get_total_flops()
    except Exception:
        total = None
    if was_training:
        ftn.train()
    return int(total) if total else total


def _sync(cuda):
    if cuda:
        torch.cuda.synchronize()


def benchmark_timings(ftn, data_gen, task, n_warmup=5, n_iters=20):
    """Mean wall time (ms) of one forward pass and one reconfiguration.

    Forward is timed under the current mask (no_grad, eval). Reconfiguration is
    timed as ftn._reconfig(x, y) (it manages its own grad internally). Both are
    warmed up and CUDA-synchronized around the timed region. The mask and the
    train/eval flag are saved and restored so this never perturbs the model.
    """
    cuda = torch.cuda.is_available()
    was_training = ftn.training
    m_save = ftn.m.clone()
    x, y = data_gen(task)

    ftn.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            ftn(x)
        _sync(cuda)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            ftn(x)
        _sync(cuda)
    fwd_ms = (time.perf_counter() - t0) / n_iters * 1e3

    for _ in range(n_warmup):
        ftn._reconfig(x, y)
    _sync(cuda)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        ftn._reconfig(x, y)
    _sync(cuda)
    recon_ms = (time.perf_counter() - t0) / n_iters * 1e3

    ftn.m.copy_(m_save)
    if was_training:
        ftn.train()
    return fwd_ms, recon_ms


def mean_pairwise_mask_overlap(ftn, data_gen, task_indices):
    """Average Jaccard overlap |m_i & m_j| / |m_i | m_j| over all task pairs, as
    a percentage. Recovers each task's mask by reconfiguring on a fresh batch;
    restores the training mask and train/eval flag on exit. None if <2 tasks."""
    task_indices = list(task_indices)
    was_training = ftn.training
    m_save = ftn.m.clone()
    ftn.eval()
    masks = []
    for t in task_indices:
        ftn._reconfig(*data_gen(t))
        masks.append((ftn.m > 0.5).float().clone())
    ftn.m.copy_(m_save)
    if was_training:
        ftn.train()

    n = len(masks)
    if n < 2:
        return None
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = float((masks[i] * masks[j]).sum().item())
            union = float(((masks[i] + masks[j]) > 0.5).float().sum().item())
            vals.append(inter / union if union > 0 else 0.0)
    return float(np.mean(vals)) * 100.0


# =============================================================================
# Serialization
# =============================================================================


def _fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_metrics(path, fields):
    """Write fields to `path` as parseable `KEY = value` lines (NA = missing).

    Keys in _KEY_ORDER are emitted first (in that order), any extras after.
    """
    with open(path, "w") as f:
        f.write("# FTN run metrics (parseable: KEY = value; NA = unavailable)\n")
        for k in _KEY_ORDER:
            if k in fields:
                f.write(f"{k} = {_fmt(fields[k])}\n")
        for k in fields:
            if k not in _KEY_ORDER:
                f.write(f"{k} = {_fmt(fields[k])}\n")
