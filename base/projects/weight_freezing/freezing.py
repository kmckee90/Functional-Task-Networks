"""
freezing.py - Usage-density weight freezing for FTNs (replaces replay).

The whole mechanism in one sentence: ONLY TRAIN THE LEAST-GLOBALLY-USED
NEURONS IN THE CURRENT MASK.

State: a single (hidden_size,) vector of counts - how many training batches
each unit has been mask-active in ("batch involvement"). Counting is
independent of whether the unit's weights were frozen or free on that batch;
that is what makes seniority permanent: a reused (frozen) unit keeps pace
with the batch counter, so units recruited later can never overtake it. No
per-task records, no anchors, no stored masks, no decay - lifetime counts.
Decay (EMA) is deliberately absent: it would assume a task-recurrence
timescale, which is a-priori task knowledge.

Maturity (the one addition forced by data): per-batch reconfiguration churns
the mask heavily while a task settles, so batch involvement is a CONTINUOUS
heavy-tailed spectrum, not bimodal - after one task, count quantiles run
0/5/66/996 with 72% coverage. Instantaneous density therefore cannot
distinguish "high because I am the core of the task being learned right now"
from "high because I was trained in the past" - both are locally senior. The
fix is a LAGGED copy of the counts,

    counts_lag <- counts_lag + (counts - counts_lag) / lag,

which is NOT a decay: counts are lifetime-monotone and nothing is ever
forgotten; the lag only means seniority must be ~`lag` batches mature before
it confers protection. A task's own core has high instantaneous but low
lagged counts (no premature self-freeze; its core consolidates only near the
end of its own training block), while a previous task's core arrives with
mature lagged counts and is protected from the first step of the next task.

Freeze rule, evaluated per step on the current mask's units only: freeze
unit j iff

    counts_lag[j] - quantile_low(counts_lag over mask) > margin,

i.e. j's mature seniority exceeds the mask's junior reference by a discrete
margin (raw batches). When the mask is uniformly fresh (new task, no
overlap) or uniformly senior (pure revisit of an old solution), no unit
clears the margin over the reference and nothing freezes - which reproduces
both edges of the old replay gate (corr <= 0.2: nothing to protect;
corr > 0.8: training IS rehearsal) as emergent cases.

`min_plastic` guard: if freezing would leave fewer than this fraction of the
mask trainable, nothing freezes. Depletion guard: a workable plastic
population always remains.

Freezing is applied by zeroing the gradient slices of frozen units (their
per-unit MLP weights/biases and their readout column). With the runner's
per-task Adam state reset and WEIGHT_DECAY=0, a zeroed gradient means an
exactly zero update.
"""

import torch


class UsageFreezer:
    """Per-unit training-density tracker + senior-cohort freeze rule.

    Args:
        hidden_size:  Number of maskable units.
        margin:       Mature-seniority lead (in raw batches) over the mask's
                      junior reference that gets a unit frozen. None disables
                      freezing (baseline: base_rf with replay disabled).
        lag:          Maturity timescale (batches): how long accumulated
                      density takes to confer protection.
        min_plastic:  Minimum fraction of the current mask that must remain
                      trainable for a freeze to be applied.
        low_q:        Quantile of within-mask lagged counts used as the
                      junior reference (robust min).
    """

    def __init__(self, hidden_size: int, margin: float = None,
                 lag: float = 1000.0, min_plastic: float = 0.1,
                 low_q: float = 0.1):
        self.margin = margin
        self.lag = lag
        self.min_plastic = min_plastic
        self.low_q = low_q
        self.counts = torch.zeros(hidden_size)
        self.counts_lag = torch.zeros(hidden_size)

    # ------------------------------------------------------------------ state

    def record(self, m: torch.Tensor):
        """Count batch involvement for this training step (mask-active units)."""
        self.counts += (m > 0).float()
        self.counts_lag += (self.counts - self.counts_lag) / self.lag

    def density(self) -> torch.Tensor:
        """Globally min-max-normalized density in [0, 1] (for logging/viz)."""
        c = self.counts
        return (c - c.min()) / (c.max() - c.min() + 1e-12)

    # ------------------------------------------------------------------- rule

    def frozen(self, m: torch.Tensor) -> torch.Tensor:
        """Bool (hidden_size,): active units senior enough to freeze this step."""
        out = torch.zeros_like(m, dtype=torch.bool)
        if self.margin is None:
            return out
        active = m > 0
        c = self.counts_lag[active]
        if c.numel() < 2:
            return out
        ref = torch.quantile(c, self.low_q)  # the mask's junior reference
        frz = c - ref > self.margin
        # Uniformly fresh or uniformly senior masks produce no freeze (both
        # replay-gate edges); mixed masks freeze their mature-senior cohort.
        if (~frz).sum().item() < self.min_plastic * c.numel():
            return out  # keep a workable plastic population
        out[active] = frz
        return out

    # ------------------------------------------------------------ application

    @staticmethod
    def zero_frozen_grads(ftn, frz: torch.Tensor):
        """Zero gradient slices of frozen units (per-unit MLP + readout column)."""
        if not frz.any():
            return
        for w in ftn.parallel.weights:
            if w.grad is not None:
                w.grad[frz] = 0
        for b in ftn.parallel.biases:
            if b.grad is not None:
                b.grad[frz] = 0
        if ftn.linear.weight.grad is not None:
            ftn.linear.weight.grad[:, frz] = 0

    # ------------------------------------------------------------- monitoring

    def stats(self, frz: torch.Tensor) -> str:
        q = torch.quantile(
            self.counts, torch.tensor([0.25, 0.5, 0.75, 1.0])
        ).tolist()
        return (
            f"frozen={int(frz.sum().item())} "
            f"counts_q25/50/75/max={q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}/{q[3]:.0f} "
            f"coverage={(self.counts > 0).float().mean().item():.2f}"
        )
