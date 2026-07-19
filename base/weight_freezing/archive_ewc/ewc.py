"""
ewc.py - Mask-gated Elastic Weight Consolidation for FTNs (replaces replay).

Why this shape
==============
In an FTN the backbone is a ParallelMLP: every hidden unit owns its own
parameter slice (weights are (hidden, in, out) per layer, biases (hidden, out),
and each readout column belongs to one unit). Under a binary mask, task
gradients only reach units with m = 1. Consequence: the ONLY way a new task
can damage an old one is through units shared between their masks - exactly
the population the replay rule targeted ("replay only memories whose masks
overlap the current one").

MaskedEWC is the direct translation of that rule into a weight anchor:

  - consolidate(ftn, task, batches): at end of a task, estimate a diagonal
    empirical Fisher of the task loss under freshly reconfigured masks, and
    store (fisher, anchor params, soft usage mask) for the task. Fisher is
    normalized to mean 1 over the entries of units the task actually used,
    so `lam` has the same meaning across tasks and scales.

  - penalty(ftn): during training, for every stored task whose usage mask
    overlaps the CURRENT mask ftn.m inside the replay gate
    (overlap_min < corr <= overlap_max), add

        0.5 * lam * sum_j  usage_t[j] * m[j] * F_t[theta_j] * (theta_j - theta*_t,j)^2

    i.e. the quadratic constraint is gated per unit to the intersection of
    the stored and current masks. Off-mask units of old tasks need no
    protection (no gradient reaches them); units of the current task that no
    old task used stay fully plastic. This is what keeps the population from
    being depleted: consolidation pressure exists only where interference
    can actually occur, and even there only along directions the Fisher says
    mattered.

The same gate as replay: corr = (usage_t * m).sum() / m.sum();
corr <= overlap_min  -> negligible interference, skip;
corr >  overlap_max  -> essentially the same solution/task, training it IS
                        rehearsing it, skip (don't fight refinement).
"""

import torch


def _unit_gate(name: str, shared: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-unit gate (hidden,) to a parameter's shape.

    parallel.weights.{l}: (K, in, out) -> (K, 1, 1)
    parallel.biases.{l}:  (K, out)     -> (K, 1)
    linear.weight:        (Y, K)       -> (1, K)
    """
    if name.startswith("parallel.weights"):
        return shared.view(-1, 1, 1)
    if name.startswith("parallel.biases"):
        return shared.view(-1, 1)
    if name == "linear.weight":
        return shared.view(1, -1)
    raise ValueError(f"Unexpected parameter for EWC: {name}")


def _ewc_params(ftn):
    """Trainable, per-unit-sliceable parameters (skips the frozen lateral convs)."""
    return {n: p for n, p in ftn.named_parameters() if p.requires_grad}


class MaskedEWC:
    """Per-task Fisher/anchor bank with mask-overlap gating.

    Args:
        lam:          Penalty strength (Fisher is mean-normalized, so lam ~ the
                      ratio of consolidation pressure to average curvature).
        overlap_min:  Skip stored tasks with mask correlation <= this (no risk).
        overlap_max:  Skip stored tasks with mask correlation > this (same task).
        reduce:       "sum" adds one full-strength penalty per gated record;
                      "mean" divides by the number of gated records, so units
                      claimed by many old tasks (generic shared features) are
                      not multiply-pulled and pressure stays bounded as the
                      bank grows.
    """

    def __init__(
        self,
        lam: float,
        overlap_min: float = 0.2,
        overlap_max: float = 0.8,
        reduce: str = "sum",
    ):
        self.lam = lam
        self.overlap_min = overlap_min
        self.overlap_max = overlap_max
        self.reduce = reduce
        self.bank = {}  # task_id -> {"fisher": {n: T}, "anchor": {n: T}, "usage": T}

    # ------------------------------------------------------------ consolidate

    @torch.enable_grad()
    def consolidate(self, ftn, task_id, batches):
        """Store (normalized Fisher, anchor, usage mask) for a task.

        Each batch is reconfigured on before the Fisher pass, mirroring how the
        task will be retrieved later; the usage mask is the mean of those
        reconfigured masks (soft where the mask population drifts).
        Re-consolidating a task replaces its entry.
        """
        params = _ewc_params(ftn)
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        usage = torch.zeros_like(ftn.m)

        was_training = ftn.training
        ftn.eval()  # deterministic features (no dropout) for the Fisher pass
        n_batches = 0
        for x, y in batches:
            ftn.reconfigure(x, y)
            usage += ftn.m
            pred = ftn(x)
            loss = ftn.loss(pred, y)
            grads = torch.autograd.grad(loss, list(params.values()))
            for (n, _), g in zip(params.items(), grads):
                fisher[n] += g.detach() ** 2
            n_batches += 1
        if was_training:
            ftn.train()
        if n_batches == 0:
            return

        usage /= n_batches
        for n in fisher:
            fisher[n] /= n_batches

        # Normalize Fisher to mean 1 over entries of units this task used,
        # so `lam` means the same thing for every task.
        total, count = 0.0, 0.0
        for n, p in params.items():
            gate = _unit_gate(n, usage, p).expand_as(p)
            total = total + (fisher[n] * gate).sum()
            count = count + gate.sum()
        scale = (total / count.clamp(min=1e-12)).clamp(min=1e-12)
        for n in fisher:
            fisher[n] /= scale

        self.bank[task_id] = {
            "fisher": fisher,
            "anchor": {n: p.detach().clone() for n, p in params.items()},
            "usage": usage,
        }

    # ---------------------------------------------------------------- penalty

    def penalty(self, ftn):
        """Quadratic anchor loss gated by mask overlap; None if nothing applies."""
        if self.lam == 0 or not self.bank:
            return None
        m = ftn.m
        m_sum = m.sum().clamp(min=1.0)
        params = _ewc_params(ftn)

        terms = []
        n_gated = 0
        for rec in self.bank.values():
            corr = (rec["usage"] * m).sum() / m_sum
            if not (self.overlap_min < corr.item() <= self.overlap_max):
                continue
            n_gated += 1
            shared = rec["usage"] * m  # per-unit gate on the intersection
            for n, p in params.items():
                gate = _unit_gate(n, shared, p)
                delta = p - rec["anchor"][n]
                terms.append((rec["fisher"][n] * gate * delta**2).sum())
        if not terms:
            return None
        total = 0.5 * self.lam * sum(terms)
        if self.reduce == "mean":
            total = total / n_gated
        return total

    def penalty_backward(self, ftn):
        """Accumulate the penalty gradient (drop-in for the old replay_step)."""
        pen = self.penalty(ftn)
        if pen is not None:
            pen.backward()

    # ------------------------------------------------------------- monitoring

    def coverage(self) -> float:
        """Fraction of the population under any consolidation pressure."""
        if not self.bank:
            return 0.0
        stacked = torch.stack([rec["usage"] for rec in self.bank.values()])
        return (stacked.max(0).values > 0).float().mean().item()

    def __len__(self):
        return len(self.bank)
