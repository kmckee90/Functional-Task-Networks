"""
analyze_mask_overlap.py — Compute pairwise mask-overlap statistics
from already-saved results, without rerunning anything.

For each run we look at model_sharing[i,j] = number of neurons shared
between tasks i and j (off-diagonal) or unique to task i (diagonal).
We normalise by k (the per-task mask budget) so values are fractions
in [0, 1]:

    frac_off       : mean over off-diagonal entries (j != i) of S[i,j] / k
    frac_consec    : mean over (i, i+1) pairs of S[i,j] / k
    frac_max_off   : per-run max off-diagonal entry / k

Each statistic is averaged over seeds within a condition. Output is
written to:
    results/mask_overlap_stats.json
    paper/neurips/tables/mask_overlap.tex
    paper/arxiv/tables/mask_overlap.tex
"""

from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


HIDDEN_SIZE = 8192


def stats_for_runs(runs):
    fracs_off, fracs_consec, fracs_max, ks = [], [], [], []
    for r in runs:
        S = np.array(r["model_sharing"], dtype=float)
        k = int(r.get("k", 512))
        ks.append(k)
        n = S.shape[0]
        off = ~np.eye(n, dtype=bool)
        fracs_off.append(S[off].mean() / k)
        consec = [S[i, i + 1] for i in range(n - 1)]
        fracs_consec.append(np.mean(consec) / k)
        fracs_max.append(S[off].max() / k)
    k_med = int(np.median(ks)) if ks else 512
    return {
        "frac_off_mean": float(np.mean(fracs_off)),
        "frac_off_std": float(np.std(fracs_off)),
        "frac_consec_mean": float(np.mean(fracs_consec)),
        "frac_consec_std": float(np.std(fracs_consec)),
        "frac_max_mean": float(np.mean(fracs_max)),
        "frac_max_std": float(np.std(fracs_max)),
        "n_seeds": len(runs),
        "k": k_med,
        "random_baseline": k_med / HIDDEN_SIZE,
    }


def fmt(s):
    return (
        f"n={s['n_seeds']:>2}  k={s['k']:<4}  "
        f"frac_off={s['frac_off_mean']:.3f}±{s['frac_off_std']:.3f}  "
        f"frac_consec={s['frac_consec_mean']:.3f}±{s['frac_consec_std']:.3f}  "
        f"frac_max={s['frac_max_mean']:.3f}±{s['frac_max_std']:.3f}  "
        f"chance={s['random_baseline']:.3f}"
    )


def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect():
    """Return ordered list of (group_label, condition_label, stats) tuples."""
    out = []

    # exp2: correlated vs orthogonal, 8 seeds
    d = load("results/exp2_synthetic.json")
    if d:
        for label, key in [("Correlated", False), ("Orthogonal", True)]:
            rs = [r for r in d["runs"].values() if r["orthogonal"] == key]
            if rs:
                out.append(("Task structure", label, stats_for_runs(rs)))

    # exp3: reconfig freq
    d = load("results/exp3_synthetic.json")
    if d:
        for label in ["batch", "task"]:
            rs = [r for r in d["runs"].values() if r["reconfig_freq"] == label]
            if rs:
                out.append(("Reconfig freq", f"per-{label}", stats_for_runs(rs)))

    # exp4: K sweep
    d = load("results/exp4_synthetic.json")
    if d:
        for kv in [512, 128, 32, 8]:
            rs = [r for r in d["runs"].values() if r["k"] == kv]
            if rs:
                out.append(("Budget K", f"K={kv}", stats_for_runs(rs)))

    # exp5: reconfig_n
    d = load("results/exp5_synthetic.json")
    if d:
        for nv in [256, 64, 16, 4]:
            rs = [r for r in d["runs"].values() if r["reconfig_n"] == nv]
            if rs:
                out.append(("Reconfig batch", f"n={nv}", stats_for_runs(rs)))

    # exp6: MNIST
    d = load("results/exp6_mnist_shared.json")
    if d:
        out.append(("MNIST", "default", stats_for_runs(list(d["runs"].values()))))

    # Baselines synthetic
    d = load("results/baselines_synthetic.json")
    if d:
        for method in ["NoMask", "FixedMask"]:
            rs = [r for r in d["runs"].values() if r["method"] == method]
            if rs:
                out.append(("Baselines (synth)", method, stats_for_runs(rs)))

    # Baselines MNIST
    d = load("results/baselines_mnist_shared.json")
    if d:
        for method in ["NoMask", "FixedMask"]:
            rs = [r for r in d["runs"].values() if r["method"] == method]
            if rs:
                out.append(("Baselines (MNIST)", method, stats_for_runs(rs)))

    return out


def write_json(rows, path):
    serial = []
    for grp, cond, st in rows:
        serial.append({"group": grp, "condition": cond, **st})
    with open(path, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"Wrote {path}")


def write_latex(rows, path):
    """
    Compact LaTeX table with grouped rows.
    Columns: condition | n | K | mean off-diag | max off-diag | chance (k/H)
    """
    lines = []
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"Condition & $n$ & $K$ & mean off-diag & max off-diag & chance ($K/H$) \\"
    )
    lines.append(r"\midrule")
    last_grp = None
    for grp, cond, st in rows:
        if grp != last_grp:
            if last_grp is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{grp}}}}} \\")
            last_grp = grp
        cond_esc = cond.replace("_", r"\_")
        lines.append(
            rf"\quad {cond_esc} & {st['n_seeds']} & {st['k']} "
            rf"& ${st['frac_off_mean']:.3f} \pm {st['frac_off_std']:.3f}$ "
            rf"& ${st['frac_max_mean']:.3f} \pm {st['frac_max_std']:.3f}$ "
            rf"& ${st['random_baseline']:.3f}$ \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    rows = collect()

    section("Mask overlap (row-normalised by K, fraction in [0,1])")
    last_grp = None
    for grp, cond, st in rows:
        if grp != last_grp:
            print(f"\n[{grp}]")
            last_grp = grp
        print(f"  {cond:<14}  {fmt(st)}")

    write_json(rows, "results/mask_overlap_stats.json")
    write_latex(rows, "paper/neurips/tables/mask_overlap.tex")
    write_latex(rows, "paper/arxiv/tables/mask_overlap.tex")

    print(
        "\nNote: model_sharing[i,j] is the count of neurons shared between\n"
        "task i and j; we report S[i,j]/K so values are fractions in [0,1].\n"
        "Random baseline for K-of-H subsets is K/H (=K/8192 here)."
    )


if __name__ == "__main__":
    main()
