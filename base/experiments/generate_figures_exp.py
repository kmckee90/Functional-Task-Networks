"""
generate_figures_exp.py — Build figures and tables for the post-exp1
(exp2-5) sweeps from results/exp{2,3,4,5}_synthetic.json.

Run after the parallel sweeps in run_kt_exp.py finish.

    uv run python experiments/generate_figures_exp.py

Outputs (per exp):
    figures/synthetic/exp{N}_<axis>_summary.pdf      bar chart of
        head-start, FM, BWT side-by-side across axis values
    figures/synthetic/exp{N}_<axis>_headstart_curve.pdf
        per-task head-start curves, one line per axis value
    paper/neurips/tables/exp{N}_<axis>.tex
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plotting

RES_DIR = "results"
FIG_DIR = "figures/synthetic"
TABLES_DIR = "paper/neurips/tables"

EXPS = {
    "exp2": {"axis": "orthogonal", "axis_label": "Task structure",
             "values_order": ["correlated", "orthogonal"]},
    "exp3": {"axis": "reconfig_freq", "axis_label": "Reconfig frequency",
             "values_order": ["batch", "task"]},
    "exp4": {"axis": "k", "axis_label": "KWTA winners $k$",
             "values_order": ["k512", "k128", "k32", "k8"]},
    "exp5": {"axis": "reconfig_n", "axis_label": "Reconfig batch size",
             "values_order": ["rn256", "rn64", "rn16", "rn4"]},
}

VALUE_DISPLAY = {
    "correlated": "correlated",
    "orthogonal": "orthogonal",
    "batch": "per-batch",
    "task": "per-task",
    "k512": "$k{=}512$",
    "k128": "$k{=}128$",
    "k32": "$k{=}32$",
    "k8": "$k{=}8$",
    "rn256": "$n{=}256$",
    "rn64": "$n{=}64$",
    "rn16": "$n{=}16$",
    "rn4": "$n{=}4$",
}

PALETTE = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"]


def load(exp: str) -> dict | None:
    p = os.path.join(RES_DIR, f"{exp}_synthetic.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _agg(runs: list[dict], path: list[str]) -> tuple[float, float]:
    vals = []
    for r in runs:
        cur = r
        for k in path:
            cur = cur[k]
        if cur is None:
            continue
        vals.append(float(cur))
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _runs_by_value(data: dict) -> dict[str, list[dict]]:
    """Group runs by the human-readable label that aggregate() used."""
    out: dict[str, list[dict]] = defaultdict(list)
    # the raw runs don't carry label, but the aggregator's label() is just
    # str(axis_value) for ints / strings, and (correlated|orthogonal) for bool.
    for r in data["runs"].values():
        v = r["axis_value"]
        axis = r["axis"]
        if axis == "orthogonal":
            label = "orthogonal" if v else "correlated"
        elif axis == "reconfig_freq":
            label = str(v)
        elif axis == "k":
            label = f"k{int(v)}"
        elif axis == "reconfig_n":
            label = f"rn{int(v)}"
        else:
            label = str(v)
        out[label].append(r)
    return out


def _head_start_per_run(r: dict) -> float:
    vals = [
        h["task_loss_with_mask"]
        for h in r["head_start"][1:]
        if h["task_loss_with_mask"] is not None
    ]
    return float(np.mean(vals)) if vals else float("nan")


def render_summary_panel(exp: str, data: dict) -> None:
    cfg = EXPS[exp]
    by_val = _runs_by_value(data)
    order = [v for v in cfg["values_order"] if v in by_val]
    if not order:
        return
    os.makedirs(FIG_DIR, exist_ok=True)

    metrics = [
        ("Head-start $\\downarrow$", lambda runs: (
            np.nanmean([_head_start_per_run(r) for r in runs]),
            np.nanstd([_head_start_per_run(r) for r in runs]),
        )),
        ("FM $\\downarrow$", lambda runs: _agg(runs, ["summary_stored", "FM"])),
        ("BWT $\\uparrow$", lambda runs: _agg(runs, ["summary_stored", "BWT"])),
        ("$r_{\\mathrm{share}}\\uparrow$",
         lambda runs: _agg(runs, ["sharing_corr"])),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 2.8))
    for ax, (name, fn) in zip(axes, metrics):
        means, stds = [], []
        for v in order:
            m, s = fn(by_val[v])
            means.append(m)
            stds.append(s)
        xs = np.arange(len(order))
        bars = ax.bar(
            xs, means, yerr=stds, capsize=3,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(order))],
            edgecolor="black", linewidth=0.6,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [VALUE_DISPLAY.get(v, v) for v in order], fontsize=7,
        )
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"{cfg['axis_label']}", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"{exp}_summary.pdf")
    plotting.save(fig, out)


def render_headstart_curve(exp: str, data: dict) -> None:
    cfg = EXPS[exp]
    by_val = _runs_by_value(data)
    order = [v for v in cfg["values_order"] if v in by_val]
    if not order:
        return
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for i, v in enumerate(order):
        runs = by_val[v]
        n_tasks = max(len(r["head_start"]) for r in runs)
        ys = np.full((len(runs), n_tasks), np.nan)
        for ri, r in enumerate(runs):
            for t, h in enumerate(r["head_start"]):
                lv = h.get("task_loss_with_mask")
                if lv is not None:
                    ys[ri, t] = float(lv)
        m = np.nanmean(ys, axis=0)
        s = np.nanstd(ys, axis=0)
        xs = np.arange(1, n_tasks)
        ax.errorbar(
            xs, m[1:], yerr=s[1:],
            color=PALETTE[i % len(PALETTE)],
            marker="o", capsize=3,
            label=VALUE_DISPLAY.get(v, v),
            linewidth=1.4,
        )
    ax.set_xlabel("Task index")
    ax.set_ylabel("Head-start loss")
    ax.set_title(f"Head-start per task — {cfg['axis_label']}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"{exp}_headstart_curve.pdf")
    plotting.save(fig, out)


def render_table(exp: str, data: dict) -> None:
    cfg = EXPS[exp]
    by_val = _runs_by_value(data)
    order = [v for v in cfg["values_order"] if v in by_val]
    if not order:
        return
    os.makedirs(TABLES_DIR, exist_ok=True)
    out = os.path.join(TABLES_DIR, f"{exp}_{cfg['axis']}.tex")
    with open(out, "w") as f:
        f.write("% Auto-generated by experiments/generate_figures_exp.py\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write(
            f"\\caption{{Effect of {cfg['axis_label']} on head-start loss "
            f"(mean over tasks $2..N$), forgetting (FM), backward transfer "
            f"(BWT), and neuron-sharing fidelity ($r_{{\\mathrm{{share}}}}$). "
            f"AnnealSTE configurer, $S{{=}}30$, $k{{=}}512$, "
            f"reconfig batch $256$, per-batch reconfig (defaults except "
            f"the swept axis). Mean$\\pm$std over 8 seeds.}}\n"
        )
        f.write(f"\\label{{tab:{exp}_{cfg['axis']}}}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write(
            f"{cfg['axis_label']} & Head-start $\\downarrow$ "
            f"& FM $\\downarrow$ & BWT $\\uparrow$ "
            f"& $r_{{\\mathrm{{share}}}}\\uparrow$ \\\\\n\\midrule\n"
        )
        for v in order:
            runs = by_val[v]
            hs_per = [_head_start_per_run(r) for r in runs]
            hs_m = float(np.nanmean(hs_per))
            hs_s = float(np.nanstd(hs_per))
            fm_m, fm_s = _agg(runs, ["summary_stored", "FM"])
            bwt_m, bwt_s = _agg(runs, ["summary_stored", "BWT"])
            sc_m, sc_s = _agg(runs, ["sharing_corr"])
            disp = (
                VALUE_DISPLAY.get(v, v)
                .replace("\\,", "")  # safe inside tabular
            )
            f.write(
                f"{disp} & {hs_m:.3f}\\,$\\pm$\\,{hs_s:.3f} "
                f"& {fm_m:.3f}\\,$\\pm$\\,{fm_s:.3f} "
                f"& {bwt_m:+.3f}\\,$\\pm$\\,{bwt_s:.3f} "
                f"& {sc_m:.3f}\\,$\\pm$\\,{sc_s:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  Wrote: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", nargs="+", default=list(EXPS.keys()))
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}")

    for exp in args.exps:
        data = load(exp)
        if data is None:
            print(f"  [skip] no results for {exp}")
            continue
        print(f"  [exp] {exp} ({len(data['runs'])} runs)")
        render_summary_panel(exp, data)
        render_headstart_curve(exp, data)
        render_table(exp, data)
    print("Done.")


if __name__ == "__main__":
    main()
