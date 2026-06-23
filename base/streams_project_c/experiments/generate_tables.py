"""
generate_tables.py — Emit LaTeX tables from the project_c_4 results JSONs.

Run from project_c_4/:
    uv run python experiments/generate_tables.py

Produces paper/tables/exp{1,2,3,4}.tex.  Each table reports mean ± std
across seeds for ACC, FM, BWT under both evaluation protocols (stored /
reconfig).  For Experiment 1 the classification and regression variants
are placed side-by-side in a single table.
"""

import json
import os
import sys


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(m, s):
    return f"${m:.3f}\\pm{s:.3f}$"


def _rows(data, proto, methods_order=None):
    """Return list of (method, ACC, ACC_std, FM, FM_std, BWT, BWT_std)."""
    agg = data.get("aggregated", {})
    if not agg:
        return []
    methods = list(agg.keys())
    if methods_order is not None:
        methods = [m for m in methods_order if m in methods] + [
            m for m in methods if m not in methods_order
        ]
    out = []
    for m in methods:
        if proto not in agg[m]:
            continue
        a = agg[m][proto]
        out.append(
            (m, a["ACC"], a["ACC_std"], a["FM"], a["FM_std"], a["BWT"], a["BWT_std"])
        )
    return out


def _seed_count(data):
    return len(data.get("seeds", {}))


# ---------------------------------------------------------------------------
# Experiment 1 — synthetic (classification + regression side-by-side)
# ---------------------------------------------------------------------------

def table_exp1(d_clf, d_reg, out_path):
    if d_clf is None and d_reg is None:
        return
    proto = "reconfig"
    order = ["NoMask", "FixedMask", "KWTA", "FTN-Fast", "FTN-Slow", "EWC"]
    rows_clf = _rows(d_clf, proto, order) if d_clf else []
    rows_reg = _rows(d_reg, proto, order) if d_reg else []

    clf_by = {r[0]: r for r in rows_clf}
    reg_by = {r[0]: r for r in rows_reg}
    methods = [m for m in order if m in clf_by or m in reg_by]

    n_clf = _seed_count(d_clf) if d_clf else 0
    n_reg = _seed_count(d_reg) if d_reg else 0

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Experiment 1 (Synthetic CL, 3 tasks; mask-recovery "
        f"eval protocol).  Mean $\\pm$ std over {n_clf} seeds (clf) / "
        f"{n_reg} seeds (reg).  Regression columns report MSE "
        "(lower is better); the regression FM is the average (peak MSE "
        "$-$ final MSE) across earlier tasks (lower is better); BWT "
        "is the average (final MSE $-$ first-trained MSE) across "
        "earlier tasks (lower is better, since rising MSE signals "
        "forgetting).}"
    )
    lines.append("\\label{tab:exp1}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l rrr rrr}")
    lines.append("\\toprule")
    lines.append(
        " & \\multicolumn{3}{c}{Classification} & "
        "\\multicolumn{3}{c}{Regression} \\\\"
    )
    lines.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    lines.append(
        "Method & ACC$\\uparrow$ & FM$\\downarrow$ & BWT$\\uparrow$ & "
        "MSE$\\downarrow$ & FM$\\downarrow$ & BWT$\\downarrow$ \\\\"
    )
    lines.append("\\midrule")
    for m in methods:
        c = clf_by.get(m)
        r = reg_by.get(m)
        parts = [m]
        for row in (c, r):
            if row is None:
                parts += ["--", "--", "--"]
            else:
                _, acc, acc_s, fm, fm_s, bwt, bwt_s = row
                parts += [_fmt(acc, acc_s), _fmt(fm, fm_s), _fmt(bwt, bwt_s)]
        lines.append(" & ".join(parts) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Single-variant experiment tables (Exp 2, 3, 4)
# ---------------------------------------------------------------------------

def table_single(data, out_path, caption, label, order=None):
    if data is None:
        return
    proto = "reconfig"
    rows = _rows(data, proto, order)
    n = _seed_count(data)

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{{caption}  Mean $\\pm$ std over {n} seeds; "
        "mask-recovery eval protocol.}"
    )
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l rrr}")
    lines.append("\\toprule")
    lines.append("Method & ACC$\\uparrow$ & FM$\\downarrow$ & BWT \\\\")
    lines.append("\\midrule")
    for m, acc, acc_s, fm, fm_s, bwt, bwt_s in rows:
        lines.append(
            f"{m} & {_fmt(acc, acc_s)} & {_fmt(fm, fm_s)} & "
            f"{_fmt(bwt, bwt_s)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}")

    d1_clf = _load("results/exp1_synthetic_clf.json")
    d1_reg = _load("results/exp1_synthetic_reg.json")
    d2 = _load("results/exp2_mnist_shuffled.json")
    d3 = _load("results/exp3_permuted_mnist.json")
    d4 = _load("results/exp4_elec2.json")

    order_24 = ["NoMask", "FixedMask", "KWTA", "FTN-Fast", "FTN-Slow", "EWC"]
    order_3 = ["NoMask", "KWTA", "FTN-Fast", "FTN-Slow", "EWC"]

    table_exp1(d1_clf, d1_reg, "paper/tables/exp1.tex")
    table_single(
        d2,
        "paper/tables/exp2.tex",
        "Experiment 2: MNIST Shuffled Labels (concept shift, 5 tasks).",
        "tab:exp2",
        order_3,
    )
    table_single(
        d3,
        "paper/tables/exp3.tex",
        "Experiment 3: Permuted MNIST (domain shift, 10 tasks).",
        "tab:exp3",
        order_3,
    )
    table_single(
        d4,
        "paper/tables/exp4.tex",
        "Experiment 4: Electricity ELEC2 (real tabular concept drift, 4 time regimes).",
        "tab:exp4",
        order_24,
    )

    print("\nDone.")
