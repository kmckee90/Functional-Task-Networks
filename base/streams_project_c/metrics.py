"""
metrics.py — Continual learning evaluation metrics for project_c.

Standard CL metrics computed from the performance matrix R, where
R[i][j] is the performance on task j immediately after training task i.
(R[i][j] is only defined for j <= i.)

References:
  Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory"
  Chaudhry et al. (2018) "Efficient Lifelong Learning with A-GEM"
"""

from __future__ import annotations
import numpy as np


def pad_perf_matrix(R: list[list[float]]) -> np.ndarray:
    """
    Convert a ragged lower-triangular list-of-lists to a square numpy matrix.
    Upper triangle (not yet observed) is filled with NaN.
    """
    n = len(R)
    mat = np.full((n, n), np.nan)
    for i, row in enumerate(R):
        for j, val in enumerate(row):
            mat[i][j] = val
    return mat


def average_accuracy(R: list[list[float]]) -> float:
    """
    Average accuracy across all tasks after training on the final task.

    ACC = (1/N) sum_j R[N-1][j]
    """
    final_row = R[-1]
    return float(np.mean(final_row))


def forgetting_measure(R: list[list[float]]) -> float:
    """
    Average backward forgetting (FM).

    FM = (1/(N-1)) sum_{j=0}^{N-2} (max_{i<=N-1} R[i][j] - R[N-1][j])

    Measures how much performance on each earlier task degraded from its
    peak to the final evaluation.  A value near 0 indicates no forgetting.
    """
    mat = pad_perf_matrix(R)
    n = len(R)
    if n < 2:
        return 0.0

    forgetting = []
    for j in range(n - 1):
        col = mat[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        peak  = float(np.max(valid))
        final = float(mat[-1, j])
        forgetting.append(peak - final)
    return float(np.mean(forgetting)) if forgetting else 0.0


def backward_transfer(R: list[list[float]]) -> float:
    """
    Average Backward Transfer (BWT).

    BWT = (1/(N-1)) sum_{j=0}^{N-2} (R[N-1][j] - R[j][j])

    Negative BWT indicates forgetting; positive indicates improvement
    of earlier tasks due to later training (positive transfer).
    """
    mat = pad_perf_matrix(R)
    n = len(R)
    if n < 2:
        return 0.0

    bwt = [mat[-1, j] - mat[j, j] for j in range(n - 1)]
    return float(np.mean(bwt))


def forward_transfer(R: list[list[float]], baseline_perf: list[float]) -> float:
    """
    Average Forward Transfer (FWT).

    FWT = (1/(N-1)) sum_{j=1}^{N-1} (R[j-1][j] - baseline_perf[j])

    Measures how well training on tasks 0..j-1 helps with task j,
    relative to a random-init baseline performance.

    Args:
        R:              Performance matrix.
        baseline_perf:  Per-task baseline performance at random init.
    """
    mat = pad_perf_matrix(R)
    n = len(R)
    if n < 2:
        return 0.0

    fwt = [mat[j-1, j] - baseline_perf[j] for j in range(1, n)]
    return float(np.mean(fwt))


def summarise(
    R: list[list[float]],
    baseline_perf: list[float] | None = None,
    label: str = "",
) -> dict:
    """
    Compute and return all CL metrics for a given performance matrix.

    Returns a dict with keys: ACC, FM, BWT, (FWT if baseline given).
    """
    results = {
        "ACC": average_accuracy(R),
        "FM":  forgetting_measure(R),
        "BWT": backward_transfer(R),
    }
    if baseline_perf is not None:
        results["FWT"] = forward_transfer(R, baseline_perf)
    if label:
        results["label"] = label
    return results


def print_table(summaries: list[dict], title: str = "") -> None:
    """Pretty-print a comparison table of CL metrics."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    header = f"{'Method':<20}  {'ACC':>7}  {'FM':>7}  {'BWT':>8}"
    if any("FWT" in s for s in summaries):
        header += f"  {'FWT':>7}"
    print(header)
    print("-" * len(header))

    for s in summaries:
        name = s.get("label", "")
        row = f"{name:<20}  {s['ACC']:>7.4f}  {s['FM']:>7.4f}  {s['BWT']:>8.4f}"
        if "FWT" in s:
            row += f"  {s['FWT']:>7.4f}"
        print(row)
    print()
