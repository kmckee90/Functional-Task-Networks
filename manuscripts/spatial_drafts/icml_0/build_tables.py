"""
Generate ICML-compact tables for the FTN paper into paper/icml/tables/.

Differences from the parent paper/tables/exp{1..4}.tex:
- Drops the BWT column (concise, two-column-friendly).
- Splits Experiment 1 into two separate tables (classification and regression),
  since the parent's side-by-side 6-column table does not fit in an ICML two-column layout.

Run from project_c_4/:
    uv run python paper/icml/build_tables.py
"""

import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.dirname(os.path.abspath(__file__)) + "/tables"
os.makedirs(OUT, exist_ok=True)


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _fmt(m, s):
    return f"${m:.3f}\\pm{s:.3f}$"


def _rows(d, proto, order):
    agg = d.get("aggregated", {})
    out = []
    methods = [m for m in order if m in agg] + [m for m in agg if m not in order]
    for m in methods:
        if proto not in agg[m]:
            continue
        a = agg[m][proto]
        out.append((m, a["ACC"], a["ACC_std"], a["FM"], a["FM_std"]))
    return out


def _seed_count(d):
    return len(d.get("seeds", {}))


def write_table_clf(d, path):
    rows = _rows(d, "reconfig", ["NoMask", "FixedMask", "KWTA", "FTN-Fast", "FTN-Slow", "EWC"])
    n = _seed_count(d)
    L = []
    L.append("\\begin{table}[t]")
    L.append("\\caption{Experiment 1 (synthetic), classification variant; "
             f"mean $\\pm$ std over {n} seeds, mask-recovery protocol.}}")
    L.append("\\label{tab:exp1clf}")
    L.append("\\vskip 0.05in\n\\begin{center}\\begin{small}\\begin{sc}")
    L.append("\\begin{tabular}{lrr}")
    L.append("\\toprule")
    L.append("Method & ACC$\\uparrow$ & FM$\\downarrow$ \\\\")
    L.append("\\midrule")
    for m, a, a_s, fm, fm_s in rows:
        L.append(f"{m} & {_fmt(a, a_s)} & {_fmt(fm, fm_s)} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{sc}\\end{small}\\end{center}\n\\vskip -0.1in\n\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  Wrote {path}")


def write_table_reg(d, path):
    rows = _rows(d, "reconfig", ["NoMask", "FixedMask", "KWTA", "FTN-Fast", "FTN-Slow", "EWC"])
    n = _seed_count(d)
    L = []
    L.append("\\begin{table}[t]")
    L.append("\\caption{Experiment 1 (synthetic), regression variant; "
             f"mean $\\pm$ std over {n} seeds, mask-recovery protocol. MSE: lower is better; "
             "FM is the average peak-MSE minus final-MSE on prior tasks.}")
    L.append("\\label{tab:exp1reg}")
    L.append("\\vskip 0.05in\n\\begin{center}\\begin{small}\\begin{sc}")
    L.append("\\begin{tabular}{lrr}")
    L.append("\\toprule")
    L.append("Method & MSE$\\downarrow$ & FM$\\downarrow$ \\\\")
    L.append("\\midrule")
    for m, a, a_s, fm, fm_s in rows:
        L.append(f"{m} & {_fmt(a, a_s)} & {_fmt(fm, fm_s)} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{sc}\\end{small}\\end{center}\n\\vskip -0.1in\n\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  Wrote {path}")


def write_table_single(d, path, caption, label, order):
    rows = _rows(d, "reconfig", order)
    n = _seed_count(d)
    L = []
    L.append("\\begin{table}[t]")
    L.append(f"\\caption{{{caption} Mean $\\pm$ std over {n} seeds, mask-recovery protocol.}}")
    L.append(f"\\label{{{label}}}")
    L.append("\\vskip 0.05in\n\\begin{center}\\begin{small}\\begin{sc}")
    L.append("\\begin{tabular}{lrr}")
    L.append("\\toprule")
    L.append("Method & ACC$\\uparrow$ & FM$\\downarrow$ \\\\")
    L.append("\\midrule")
    for m, a, a_s, fm, fm_s in rows:
        L.append(f"{m} & {_fmt(a, a_s)} & {_fmt(fm, fm_s)} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{sc}\\end{small}\\end{center}\n\\vskip -0.1in\n\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  Wrote {path}")


if __name__ == "__main__":
    d1c = _load("exp1_synthetic_clf.json")
    d1r = _load("exp1_synthetic_reg.json")
    d2  = _load("exp2_mnist_shuffled.json")
    d3  = _load("exp3_permuted_mnist.json")
    d4  = _load("exp4_elec2.json")

    order_14 = ["NoMask", "FixedMask", "KWTA", "FTN-Fast", "FTN-Slow", "EWC"]
    order_3  = ["NoMask", "KWTA", "FTN-Fast", "FTN-Slow", "EWC"]

    if d1c: write_table_clf(d1c, f"{OUT}/exp1_clf.tex")
    if d1r: write_table_reg(d1r, f"{OUT}/exp1_reg.tex")
    if d2:  write_table_single(d2, f"{OUT}/exp2.tex",
                              "Experiment 2: MNIST Shuffled Labels (concept shift, 5 tasks).",
                              "tab:exp2", order_3)
    if d3:  write_table_single(d3, f"{OUT}/exp3.tex",
                              "Experiment 3: Permuted MNIST (domain shift, 10 tasks).",
                              "tab:exp3", order_3)
    if d4:  write_table_single(d4, f"{OUT}/exp4.tex",
                              "Experiment 4: Electricity ELEC2 (real tabular concept drift, 4 time regimes).",
                              "tab:exp4", order_14)
    print("Done.")
