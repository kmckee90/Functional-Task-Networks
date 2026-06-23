"""
generate_figures_exp1_rerun.py — Regenerate exp1 figures from the
corrected-metric rerun (results/exp1_rerun_t1000.json), with the
"_rerun_t1000" suffix so existing figures from the original exp1
are not overwritten.

Outputs (written under figures/synthetic/):
    headstart_curves_rerun_t1000.pdf
    forgetting_vs_compute_rerun_t1000.pdf
    bwt_vs_compute_rerun_t1000.pdf
    sharing_corr_vs_compute_rerun_t1000.pdf
    headstart_per_task_b{B}_rerun_t1000.pdf  (one per budget)
    sharing_compare_{Method}_b{B}_rerun_t1000.pdf  (one per method, max budget)
    sharing_gt_rerun_t1000.pdf
"""

from __future__ import annotations

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

RES_PATH = "results/exp1_rerun_t1000.json"
FIG_DIR = "figures/synthetic"
SUFFIX = "_rerun_t1000"

METHOD_COLORS = {
    "Sigmoid":   "#377eb8",
    "AnnealSTE": "#e41a1c",
    "Gumbel":    "#4daf4a",
}
METHOD_MARKERS = {
    "Sigmoid":   "o",
    "AnnealSTE": "s",
    "Gumbel":    "^",
}


def _load():
    with open(RES_PATH) as f:
        return json.load(f)


def _group(data):
    out = defaultdict(lambda: defaultdict(list))
    for r in data["runs"].values():
        out[r["method"]][int(r["budget"])].append(r)
    return out


def _style(ax):
    ax.tick_params(labelsize=6)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    ax.title.set_size(8)


def _arrow(ax, dirn):
    dy = 0.16 if dirn == "up" else -0.16
    y0 = 0.5 - dy / 2
    ax.annotate(
        "", xy=(-0.22, y0 + dy), xytext=(-0.22, y0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.2,
                        mutation_scale=12),
        annotation_clip=False,
    )


def _agg(runs, path):
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


def plot_compute_curves(data):
    grouped = _group(data)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Head-start (mean over tasks 1..N) vs budget
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for method, by_b in sorted(grouped.items()):
        xs, ys, errs = [], [], []
        for b in sorted(by_b):
            rs = by_b[b]
            per_run = []
            for r in rs:
                vals = [
                    h["task_loss_with_mask"]
                    for h in r["head_start"][1:]
                    if h["task_loss_with_mask"] is not None
                ]
                per_run.append(np.mean(vals) if vals else np.nan)
            xs.append(b)
            ys.append(float(np.nanmean(per_run)))
            errs.append(float(np.nanstd(per_run)))
        ax.errorbar(
            xs, ys, yerr=errs,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            capsize=3, label=method, linewidth=1.4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Reconfig steps")
    ax.set_ylabel("Head-start loss (held-out batch)")
    ax.set_title("Head-start vs compute (rerun, t=1000, fresh-batch)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    _style(ax)
    _arrow(ax, "down")
    fig.tight_layout()
    plotting.save(fig, os.path.join(FIG_DIR, f"headstart_curves{SUFFIX}.pdf"))

    # FM vs budget
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for method, by_b in sorted(grouped.items()):
        xs, ys, errs = [], [], []
        for b in sorted(by_b):
            mean, std = _agg(by_b[b], ["summary_stored", "FM"])
            xs.append(b)
            ys.append(mean)
            errs.append(std)
        ax.errorbar(
            xs, ys, yerr=errs,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            capsize=3, label=method, linewidth=1.4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Reconfig steps")
    ax.set_ylabel("Forgetting (FM)")
    ax.set_title("Forgetting vs compute (rerun, t=1000)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    _style(ax)
    _arrow(ax, "down")
    fig.tight_layout()
    plotting.save(fig, os.path.join(FIG_DIR, f"forgetting_vs_compute{SUFFIX}.pdf"))

    # BWT vs budget
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for method, by_b in sorted(grouped.items()):
        xs, ys, errs = [], [], []
        for b in sorted(by_b):
            mean, std = _agg(by_b[b], ["summary_stored", "BWT"])
            xs.append(b)
            ys.append(mean)
            errs.append(std)
        ax.errorbar(
            xs, ys, yerr=errs,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            capsize=3, label=method, linewidth=1.4,
        )
    ax.axhline(0, color="k", linestyle=":", alpha=0.5, linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Reconfig steps")
    ax.set_ylabel("Backward transfer")
    ax.set_title("BWT vs compute (rerun, t=1000)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    _style(ax)
    _arrow(ax, "up")
    fig.tight_layout()
    plotting.save(fig, os.path.join(FIG_DIR, f"bwt_vs_compute{SUFFIX}.pdf"))

    # Sharing fidelity vs budget
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for method, by_b in sorted(grouped.items()):
        xs, ys, errs = [], [], []
        for b in sorted(by_b):
            mean, std = _agg(by_b[b], ["sharing_corr"])
            xs.append(b)
            ys.append(mean)
            errs.append(std)
        ax.errorbar(
            xs, ys, yerr=errs,
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            capsize=3, label=method, linewidth=1.4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Reconfig steps")
    ax.set_ylabel("Sharing fidelity (r)")
    ax.set_title("Sharing fidelity vs compute (rerun, t=1000)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    _style(ax)
    _arrow(ax, "up")
    fig.tight_layout()
    plotting.save(fig, os.path.join(FIG_DIR, f"sharing_corr_vs_compute{SUFFIX}.pdf"))


def plot_headstart_per_budget(data):
    grouped = _group(data)
    methods = list(sorted(grouped.keys()))
    budgets = sorted({b for m in methods for b in grouped[m]})

    # Global y-axis range
    global_max = 0.0
    for b in budgets:
        for m in methods:
            for r in grouped[m].get(b, []):
                for h in r["head_start"]:
                    v = h.get("task_loss_with_mask")
                    if v is not None:
                        global_max = max(global_max, float(v))
    y_top = global_max * 1.05 if global_max > 0 else 1.0

    for b in budgets:
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        any_drawn = False
        for method in methods:
            rs = grouped[method].get(b, [])
            if not rs:
                continue
            n_tasks = max(len(r["head_start"]) for r in rs)
            ys = np.full((len(rs), n_tasks), np.nan)
            for i, r in enumerate(rs):
                for t, h in enumerate(r["head_start"]):
                    v = h.get("task_loss_with_mask")
                    if v is not None:
                        ys[i, t] = float(v)
            mean = np.nanmean(ys, axis=0)
            std = np.nanstd(ys, axis=0)
            xs = np.arange(1, n_tasks)
            ax.errorbar(
                xs, mean[1:], yerr=std[1:],
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                capsize=3, label=method, linewidth=1.4,
            )
            any_drawn = True
        if not any_drawn:
            plt.close(fig)
            continue
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel("Task index")
        ax.set_ylabel("Head-start loss (held-out batch)")
        ax.set_title(f"Head-start per task — S={b} (rerun, t=1000)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, frameon=False)
        _style(ax)
        _arrow(ax, "down")
        fig.tight_layout()
        plotting.save(fig, os.path.join(FIG_DIR, f"headstart_per_task_b{b}{SUFFIX}.pdf"))


def plot_sharing(data):
    grouped = _group(data)
    if not grouped:
        return
    any_run = next(iter(grouped[next(iter(grouped))].values()))[0]
    gt = np.array(any_run["gt_sharing"])
    plotting.plot_sharing_matrix(
        gt,
        title=f"Ground-truth concept sharing (rerun, t=1000)",
        out_path=os.path.join(FIG_DIR, f"sharing_gt{SUFFIX}.pdf"),
    )

    # For each method, plot the largest-budget seed-0 sharing
    for method, by_b in grouped.items():
        b_max = max(by_b)
        rs = by_b[b_max]
        r = next((r for r in rs if r["seed"] == 0), rs[0])
        ms = np.array(r["model_sharing"])
        plotting.plot_sharing_comparison(
            gt, ms,
            title=f"Concept vs neuron sharing — {method} S={b_max} (rerun, t=1000)",
            out_path=os.path.join(
                FIG_DIR, f"sharing_compare_{method}_b{b_max}{SUFFIX}.pdf"
            ),
        )


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}")
    if not os.path.exists(RES_PATH):
        print(f"  [skip] no {RES_PATH}")
        return
    data = _load()
    print(f"  [exp1_rerun] {len(data['runs'])} runs")
    plot_compute_curves(data)
    plot_headstart_per_budget(data)
    plot_sharing(data)
    print("Done.")


if __name__ == "__main__":
    main()
