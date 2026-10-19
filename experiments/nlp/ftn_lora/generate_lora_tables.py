"""Aggregate FTN-LoRA result JSONs into LaTeX tables for the NeurIPS paper.

Reads results/lora/trace_<method>_s<seed>.json and emits:
  paper/neurips/tables/lora_main.tex          — method × metric (mean over seeds)
  paper/neurips/tables/lora_headstart_per_task.tex — method × task head-start NLL

Metrics:
  ACC: mean accuracy on the lower triangle of the post-task evaluation matrix.
  FM (forgetting): mean over j of [max_{i>=j} R[i, j] - R[T, j]] using ACC.
  BWT: mean over j<T of [R[T, j] - R[j, j]] using ACC.
  HS (head-start NLL): mean head-start NLL across tasks (excluding task 0).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def metric_matrix(result, key="acc"):
    """Build R[i, j] from post_task_matrix where R[i, j] = `key` after training task i, evaluated on task j."""
    order = result["task_order"]
    T = len(order)
    R = [[None] * T for _ in range(T)]
    for row in result["post_task_matrix"]:
        i = order.index(row["train_task"])
        for j_name, m in row["evals"].items():
            j = order.index(j_name)
            R[i][j] = m.get(key, float("nan"))
    return R


def compute_summary(result):
    R_acc = metric_matrix(result, "acc")
    R_nll = metric_matrix(result, "nll")
    T = len(result["task_order"])
    # ACC = mean over lower triangle of final scores: actually it's mean across
    # j of R[T-1, j], i.e. final accuracies
    final = [R_acc[T - 1][j] for j in range(T) if R_acc[T - 1][j] is not None]
    acc = mean(final) if final else float("nan")

    # FM: max over i>=j of R[i,j] minus R[T-1,j], averaged
    fm_vals = []
    for j in range(T - 1):  # last task has no forgetting yet
        diags = [R_acc[i][j] for i in range(j, T) if R_acc[i][j] is not None]
        if not diags or R_acc[T - 1][j] is None:
            continue
        fm_vals.append(max(diags) - R_acc[T - 1][j])
    fm = mean(fm_vals) if fm_vals else 0.0

    # BWT: mean over j<T-1 of R[T-1,j] - R[j,j]
    bwt_vals = []
    for j in range(T - 1):
        if R_acc[T - 1][j] is None or R_acc[j][j] is None:
            continue
        bwt_vals.append(R_acc[T - 1][j] - R_acc[j][j])
    bwt = mean(bwt_vals) if bwt_vals else 0.0

    # Head-start NLL: skip task 0 (no prior tasks to transfer from)
    hs_vals = []
    order = result["task_order"]
    for ti, t in enumerate(order):
        if ti == 0:
            continue
        v = result["head_start"].get(t, {})
        if "nll" in v:
            hs_vals.append(v["nll"])
    hs_nll = mean(hs_vals) if hs_vals else float("nan")

    # Head-start ACC: same skip
    hs_acc_vals = []
    for ti, t in enumerate(order):
        if ti == 0:
            continue
        v = result["head_start"].get(t, {})
        if "acc" in v:
            hs_acc_vals.append(v["acc"])
    hs_acc = mean(hs_acc_vals) if hs_acc_vals else float("nan")

    return {"ACC": acc, "FM": fm, "BWT": bwt, "HS_NLL": hs_nll, "HS_ACC": hs_acc}


def aggregate(results_dir: Path, methods: list[str]):
    """Group results by method, average across seeds.

    The method key is reconstructed from the filename when the saved
    `method` field doesn't already capture an ablation suffix
    (e.g. files named `trace_FTN_s0_bnonzero.json` whose internal
    `method` field is just `FTN` get re-keyed to `FTN_bnonzero`).
    """
    by_method: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(results_dir.glob("trace_*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        m = r.get("method")
        if not m:
            continue
        # If filename has an extra suffix beyond _s<seed>, fold it into the method
        # key so ablations don't collide with the base run.
        stem = f.stem  # e.g. trace_FTN_s0_bnonzero
        prefix = f"trace_{m}_s"
        if stem.startswith(prefix):
            tail = stem[len(prefix):]  # e.g. "0_bnonzero" or "0"
            seed_str, *suffix_parts = tail.split("_", 1)
            if suffix_parts:
                m = f"{m}_{suffix_parts[0]}"
        if methods and m not in methods:
            continue
        r["method"] = m
        by_method[m].append(r)
    summary: dict[str, dict] = {}
    for m, runs in by_method.items():
        per_seed = [compute_summary(r) for r in runs]
        agg = {}
        for k in ("ACC", "FM", "BWT", "HS_NLL", "HS_ACC"):
            vs = [p[k] for p in per_seed if p[k] == p[k]]  # skip nan
            agg[k] = (mean(vs), stdev(vs) if len(vs) > 1 else 0.0, len(vs))
        # Per-task head-start NLL aggregated across seeds
        per_task = defaultdict(list)
        for r in runs:
            for t, v in r["head_start"].items():
                if "nll" in v:
                    per_task[t].append(v["nll"])
        per_task_agg = {t: (mean(vs), stdev(vs) if len(vs) > 1 else 0.0)
                        for t, vs in per_task.items()}
        summary[m] = {"agg": agg, "per_task_hs": per_task_agg,
                      "n_seeds": len(runs), "task_order": runs[0]["task_order"]}
    return summary


METHOD_DISPLAY = {
    "FTN":             r"FTN-LoRA, $B_0=0$",
    "FTN_router":      r"FTN-LoRA, $B_0=0$ + Router",
    "FTN_noreplay":    r"FTN-LoRA, $B_0=0$, no replay",
    "FTN_bnonzero":    r"FTN-LoRA",
    "FTN_RandMask_bnonzero": r"FTN-LoRA, random per-batch masks",
    "SeqLoRA":         r"SeqLoRA",
    "SeqLoRA_ER":      r"SeqLoRA + ER",
    "FixedMask":       r"Indep.\ LoRA (oracle)",
    "FixedMask_router":r"Indep.\ LoRA + Router",
    "RandMask_router": r"RandMask + Router",
}


# Body table: clean comparison plus the decisive routing-vs-random ablation.
METHOD_ORDER_BODY = [
    "SeqLoRA", "SeqLoRA_ER",
    "FixedMask", "FixedMask_router",
    "FTN_RandMask_bnonzero",
    "FTN_bnonzero",
]


# Appendix ablations table: full method list including diagnostic intermediate steps.
METHOD_ORDER = [
    "SeqLoRA", "SeqLoRA_ER", "RandMask_router",
    "FixedMask", "FixedMask_router",
    "FTN_noreplay", "FTN_router", "FTN",
    "FTN_RandMask_bnonzero",
    "FTN_bnonzero",
]


def fmt_mean_std(mean_val: float, std_val: float, n: int, fmt="{:.3f}") -> str:
    if mean_val != mean_val:
        return "--"
    if n <= 1 or std_val == 0:
        return fmt.format(mean_val)
    return f"{fmt.format(mean_val)}\\,\\scriptsize$\\pm${fmt.format(std_val)}"


def _emit_table(summary, out_path: Path, method_order, label, caption):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & ACC $\uparrow$ & FM $\downarrow$ & BWT $\uparrow$ & HS NLL $\downarrow$ \\",
        r"\midrule",
    ]
    for m in method_order:
        if m not in summary:
            continue
        s = summary[m]["agg"]
        row = (
            f"{METHOD_DISPLAY.get(m, m)} & "
            f"{fmt_mean_std(*s['ACC'])} & "
            f"{fmt_mean_std(*s['FM'])} & "
            f"{fmt_mean_std(*s['BWT'])} & "
            f"{fmt_mean_std(*s['HS_NLL'], fmt='{:.2f}')} \\\\"
        )
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")


def emit_main_table(summary, out_path: Path):
    _emit_table(
        summary, out_path, METHOD_ORDER_BODY,
        label="tab:lora_main",
        caption=(
            "Qwen2.5 LoRA-bank experiment on an assembled six-task "
            "continual-classification stream (C-STANCE-A, Financial "
            "PhraseBank, TREC, dair-ai/emotion, AG News, DBpedia~14). "
            "Qwen2.5-1.5B-Instruct backbone with $H{=}128$ rank-$4$ LoRA "
            "atoms and $k{=}16$ active per forward pass, five seeds per "
            "method, matched per-step active-parameter budget, $100$ "
            "gradient steps per task. ACC: final mean accuracy across "
            "tasks. FM: mean forgetting. BWT: mean backward transfer. "
            "HS NLL: mean head-start answer-token NLL across tasks~2--6 "
            "(task~1 has no prior context). FTN-LoRA improves retention "
            "(higher ACC, lower FM) and is the only configuration with "
            "positive BWT in this suite; head-start NLL is uninformative "
            "on this suite (every method falls in a $\\sim\\!0.2$ NLL "
            "band, see \\cref{app:lora_ablations}). The decisive "
            "random-per-batch-mask ablation isolates the configurer's "
            "contribution; full ablation table and the LoRA $B$ "
            "initialisation diagnostic are in \\cref{app:lora_ablations}."
        ),
    )


def emit_full_table(summary, out_path: Path):
    _emit_table(
        summary, out_path, METHOD_ORDER,
        label="tab:lora_full",
        caption=(
            "Full FTN-LoRA ablation table. Headline rows (SeqLoRA, SeqLoRA + ER, "
            "Indep.\\ LoRA, Indep.\\ LoRA + Router, FTN-LoRA random per-batch masks, "
            "FTN-LoRA) are computed over five seeds; the three diagnostic FTN "
            "rows that use standard LoRA initialisation ($B_0{=}0$) are computed "
            "over two seeds and are reported only as the diagnostic that "
            "motivated the $B_0{\\sim}\\mathcal{N}(0,0.01)$ choice. "
            "RandMask + Router is two seeds. The diagnostic narrative is in "
            "\\cref{app:lora_ablations}."
        ),
    )


def emit_headstart_per_task_table(summary, out_path: Path):
    """Method × task head-start NLL table (excluding task 0)."""
    if not summary:
        out_path.write_text("\\textit{(no results)}\n")
        return
    any_run = next(iter(summary.values()))
    order = any_run["task_order"]
    tasks = order[1:]   # exclude task 0 (no prior tasks)

    n_cols = len(tasks)
    col_spec = "l" + "c" * n_cols
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Per-task head-start NLL on the six-task multilingual classification stream described in \cref{sec:lora}. Each entry is the answer-token NLL on the held-out batch of that task, evaluated under the method's mask BEFORE any gradient update on the task. Lower is better. Tasks are ordered as canonically presented during training; task~1 (C-STANCE-A) is omitted because no prior context exists.}",
        r"\label{tab:lora_hs_per_task}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Method & " + " & ".join(t.replace("_", " ") for t in tasks) + r" \\",
        r"\midrule",
    ]
    for m in METHOD_ORDER:
        if m not in summary:
            continue
        per = summary[m]["per_task_hs"]
        cells = []
        for t in tasks:
            if t in per:
                mn, _sd = per[t]
                cells.append(f"{mn:.2f}")
            else:
                cells.append("--")
        lines.append(f"{METHOD_DISPLAY.get(m, m)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/lora")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--out-main", default="paper/neurips/tables/lora_main.tex")
    p.add_argument("--out-full", default="paper/neurips/tables/lora_full.tex")
    p.add_argument("--out-hs",   default="paper/neurips/tables/lora_headstart_per_task.tex")
    p.add_argument("--out-arxiv-main", default="paper/arxiv/tables/lora_main.tex")
    p.add_argument("--out-arxiv-full", default="paper/arxiv/tables/lora_full.tex")
    p.add_argument("--out-arxiv-hs",   default="paper/arxiv/tables/lora_headstart_per_task.tex")
    args = p.parse_args()

    summary = aggregate(Path(args.results_dir), args.methods or [])
    if not summary:
        print("No results found.")
        return

    print("Summary:")
    for m, info in summary.items():
        a = info["agg"]
        print(f"  {m:20s} (n={info['n_seeds']}) "
              f"ACC={a['ACC'][0]:.3f}±{a['ACC'][1]:.3f}  "
              f"FM={a['FM'][0]:.3f}±{a['FM'][1]:.3f}  "
              f"BWT={a['BWT'][0]:.3f}±{a['BWT'][1]:.3f}  "
              f"HS_NLL={a['HS_NLL'][0]:.2f}±{a['HS_NLL'][1]:.2f}")

    for path in (args.out_main, args.out_arxiv_main):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        emit_main_table(summary, Path(path))
    for path in (args.out_full, args.out_arxiv_full):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        emit_full_table(summary, Path(path))
    for path in (args.out_hs, args.out_arxiv_hs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        emit_headstart_per_task_table(summary, Path(path))


if __name__ == "__main__":
    main()
