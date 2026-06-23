"""
build_tables.py — Generate ICML tables for the LoRA-FTN experiment from
results/lora_styles.json into icml/tables/.

Run from project_c_4/:
    uv run python experiments/lora/icml/build_tables.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results", "lora_styles.json"))
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)


def _f(v: float) -> str:
    sign = "" if v < 0 else "+"
    return f"${sign}{v:.3f}$" if v != 0 else "$0.000$"


def write_main_table(d, path):
    methods = ["LoRA-Naive", "LoRA-FixedMask", "LoRA-FTN"]
    L = []
    L.append("\\begin{table}[t]")
    L.append("\\caption{Block-sequential continual fine-tuning of {\\tt distilgpt2} on six "
             "mutually-exclusive style-transformation tasks (concept shift). "
             "Per-token cross-entropy on a held-out test split, mean over 8 random seeds; "
             "FM is the average rise in CE on prior tasks above the lowest level "
             "they reached during training (positive $\\Rightarrow$ forgetting). "
             "{\\sc stored} uses each task's training-time mask (oracle); "
             "{\\sc reconfig} cold-starts the SmoothKWTA configurer on a fresh "
             "support batch with no task identity.}")
    L.append("\\label{tab:lora_styles}")
    L.append("\\vskip 0.05in\n\\begin{center}\\begin{small}\\begin{sc}")
    L.append("\\begin{tabular}{lrrrr}")
    L.append("\\toprule")
    L.append(" & \\multicolumn{2}{c}{Stored} & \\multicolumn{2}{c}{Reconfig} \\\\")
    L.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    L.append("Method & CE$\\downarrow$ & FM$\\downarrow$ & CE$\\downarrow$ & FM$\\downarrow$ \\\\")
    L.append("\\midrule")
    for m in methods:
        if m not in d["methods"]:
            continue
        ms = d["methods"][m]["metrics_stored"]
        mr = d["methods"][m]["metrics_reconfig"]
        L.append(
            f"{m} & ${ms['avg_loss']:.3f}$ & {_f(ms['FM'])} "
            f"& ${mr['avg_loss']:.3f}$ & {_f(mr['FM'])} \\\\"
        )
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{sc}\\end{small}\\end{center}\n\\vskip -0.1in\n\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  Wrote {path}")


def write_overlap_table(d, path):
    """Pairwise mask overlap diagnostic for LoRA-FTN: |m_i \\cap m_j|/k."""
    if "LoRA-FTN" not in d["methods"]:
        return
    mux = d["methods"]["LoRA-FTN"]
    masks = mux["mask_store"]
    styles = d["styles"]
    n = len(styles)

    import numpy as np
    M = np.array([masks[str(i)] for i in range(n)])
    k = float(M[0].sum())

    L = []
    L.append("\\begin{table}[t]")
    L.append("\\caption{Pairwise overlap of LoRA-FTN training-time masks: "
             f"$|m_i\\cap m_j|/k$, where $k={int(k)}$ is the per-task active count. "
             "Diagonal is identically $1$ and omitted; off-diagonal $0$ means "
             "exact disjointness (Theorem~\\ref{thm:isolation} gives "
             "FM$=0$ for that pair).}")
    L.append("\\label{tab:lora_overlap}")
    L.append("\\vskip 0.05in\n\\begin{center}\\begin{small}")
    L.append("\\begin{tabular}{l" + "r" * n + "}")
    L.append("\\toprule")
    head = " & " + " & ".join(s.replace("_", "\\_")[:8] for s in styles) + " \\\\"
    L.append(head)
    L.append("\\midrule")
    for i, si in enumerate(styles):
        row = [si.replace("_", "\\_")[:12]]
        for j in range(n):
            if i == j:
                row.append("$\\cdot$")
            else:
                ov = float((M[i] * M[j]).sum() / max(k, 1))
                row.append(f"${ov:.2f}$")
        L.append(" & ".join(row) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{small}\\end{center}\n\\vskip -0.1in\n\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  Wrote {path}")


def main():
    if not os.path.exists(RESULTS):
        raise SystemExit(f"No results JSON at {RESULTS}. "
                         "Run experiments/lora/run.py first.")
    with open(RESULTS) as f:
        d = json.load(f)
    write_main_table(d, os.path.join(OUT, "lora_styles.tex"))
    write_overlap_table(d, os.path.join(OUT, "lora_overlap.tex"))
    print("Done.")


if __name__ == "__main__":
    main()
