"""
generate_figures.py — Build figures and tables from results JSONs.

Run after the sweeps in ``run_kt.py`` finish (or while they're still
running -- it tolerates partial results).

    uv run python experiments/generate_figures.py [--task synthetic mnist_shared]

Outputs:
    figures/{task}/sharing_gt.pdf
    figures/{task}/sharing_model_{Method}_b{budget}.pdf
    figures/{task}/sharing_compare_{Method}_b{budget}.pdf  (gt vs model side-by-side)
    figures/{task}/headstart_curves.pdf               (loss vs reconfig budget)
    figures/{task}/forgetting_vs_compute.pdf          (FM vs reconfig budget)
    figures/{task}/sharing_corr_vs_compute.pdf
    figures/{task}/perf_matrix_{Method}_b{budget}.pdf
    paper/icml/tables/main_{task}.tex                 (ACC/FM/BWT table)
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

METHOD_COLORS = {
    "Sigmoid": "#377eb8",
    "AnnealSTE": "#e41a1c",
    "Gumbel": "#4daf4a",
    "FixedMask": "#984ea3",
    "NoMask": "#aaaaaa",
}
METHOD_MARKERS = {
    "Sigmoid": "o",
    "AnnealSTE": "s",
    "Gumbel": "^",
    "FixedMask": "D",
    "NoMask": "x",
}

RES_DIR = "results"
FIG_DIR = "figures"
TABLES_DIR = "paper/icml/tables"


def load(task: str) -> dict | None:
    p = os.path.join(RES_DIR, f"{task}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sharing matrices
# ---------------------------------------------------------------------------


def render_sharing(data: dict, task: str) -> None:
    out = os.path.join(FIG_DIR, task)
    os.makedirs(out, exist_ok=True)

    runs = data.get("runs", {})
    if not runs:
        return

    # Group seed=0 runs by (method, budget) so we emit a map per compute budget.
    by_method_budget: dict[tuple[str, int], dict] = {}
    for r in runs.values():
        if r.get("seed", -1) != 0:
            continue
        key = (r["method"], int(r["budget"]))
        # If multiple seed-0 runs exist for the same (method, budget), keep first.
        by_method_budget.setdefault(key, r)

    if not by_method_budget:
        return

    any_run = next(iter(by_method_budget.values()))
    gt = np.array(any_run["gt_sharing"])
    plotting.plot_sharing_matrix(
        gt, title=f"Ground-truth concept sharing — {task}",
        out_path=os.path.join(out, "sharing_gt.pdf"),
    )

    for (method, budget), r in sorted(by_method_budget.items()):
        ms = np.array(r["model_sharing"])
        plotting.plot_sharing_matrix(
            ms,
            title=f"Model neuron sharing — {method} (budget={budget})",
            out_path=os.path.join(out, f"sharing_model_{method}_b{budget}.pdf"),
        )
        plotting.plot_sharing_comparison(
            gt, ms,
            title=f"Concept vs neuron sharing — {method} budget={budget} ({task})",
            out_path=os.path.join(out, f"sharing_compare_{method}_b{budget}.pdf"),
        )


# ---------------------------------------------------------------------------
# Compute axis: head-start, forgetting, sharing-corr
# ---------------------------------------------------------------------------


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


def _group_by_method_budget(runs: dict) -> dict:
    out: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in runs.values():
        out[r["method"]][int(r["budget"])].append(r)
    return out


def _style_axes(ax) -> None:
    """Compact font sizes for the compute-curve panels."""
    ax.tick_params(labelsize=6)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    ax.title.set_size(8)


def _add_pref_arrow(ax, direction: str) -> None:
    """Draw a small arrow to the left of the y-axis label indicating
    preferred direction. Placed beyond the rotated label so it does not
    overlap the text.

    direction: "up" (higher is better) or "down" (lower is better).
    """
    dy = 0.16 if direction == "up" else -0.16
    y0 = 0.5 - dy / 2
    ax.annotate(
        "",
        xy=(-0.22, y0 + dy),
        xytext=(-0.22, y0),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color="#444444", lw=1.2,
            mutation_scale=12,
        ),
        annotation_clip=False,
    )


def plot_headstart_per_budget(data: dict, task: str) -> None:
    """One figure per compute budget: head-start loss vs task index, with
    one line per method. Task 0 is skipped (no prior model to transfer)."""
    out = os.path.join(FIG_DIR, task)
    os.makedirs(out, exist_ok=True)

    grouped = _group_by_method_budget(data["runs"])
    methods = [m for m in sorted(grouped) if m not in ("FixedMask", "NoMask")]
    if not methods:
        return
    budgets = sorted({b for m in methods for b in grouped[m]})

    # Pre-compute global y-max over (mean + std) so all per-budget plots
    # share the same y-axis range and are visually comparable.
    global_max = 0.0
    for b in budgets:
        for method in methods:
            for r in grouped[method].get(b, []):
                for h in r["head_start"]:
                    v = h.get("task_loss_with_mask")
                    if v is not None:
                        global_max = max(global_max, float(v))
    y_top = global_max * 1.05 if global_max > 0 else 1.0

    for b in budgets:
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        any_drawn = False
        for method in methods:
            runs = grouped[method].get(b, [])
            if not runs:
                continue
            n_tasks = max(len(r["head_start"]) for r in runs)
            ys = np.full((len(runs), n_tasks), np.nan)
            for i, r in enumerate(runs):
                for t, h in enumerate(r["head_start"]):
                    v = h.get("task_loss_with_mask")
                    if v is not None:
                        ys[i, t] = float(v)
            mean = np.nanmean(ys, axis=0)
            std = np.nanstd(ys, axis=0)
            xs = np.arange(1, n_tasks)  # skip task 0
            ax.errorbar(
                xs, mean[1:], yerr=std[1:],
                color=METHOD_COLORS.get(method, "k"),
                marker=METHOD_MARKERS.get(method, "o"),
                capsize=3, label=method, linewidth=1.4,
            )
            any_drawn = True
        if not any_drawn:
            plt.close(fig)
            continue
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel("Task index")
        ax.set_ylabel("Head-start loss")
        ax.set_title(f"Head-start per task — budget={b} ({task})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, frameon=False)
        _style_axes(ax)
        _add_pref_arrow(ax, "down")
        fig.tight_layout()
        plotting.save(fig, os.path.join(out, f"headstart_per_task_b{b}.pdf"))


def plot_compute_curves(data: dict, task: str) -> None:
    out = os.path.join(FIG_DIR, task)
    os.makedirs(out, exist_ok=True)

    grouped = _group_by_method_budget(data["runs"])

    # ---- Head start: average task_loss_with_mask (over tasks 1..N-1, since
    # task 0 has no prior model to transfer from)
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for method, budgets in sorted(grouped.items()):
        if method in ("FixedMask", "NoMask"):
            continue
        xs, ys, errs = [], [], []
        for b in sorted(budgets):
            runs = budgets[b]
            per_run = []
            for r in runs:
                hs = r["head_start"]
                # Skip task 0 -- no transfer is possible from no prior tasks
                vals = [h["task_loss_with_mask"] for h in hs[1:]]
                per_run.append(np.mean(vals) if vals else np.nan)
            mean = float(np.nanmean(per_run))
            std = float(np.nanstd(per_run))
            xs.append(b)
            ys.append(mean)
            errs.append(std)
        if xs:
            ax.errorbar(
                xs, ys, yerr=errs,
                color=METHOD_COLORS.get(method, "k"),
                marker=METHOD_MARKERS.get(method, "o"),
                capsize=3, label=method, linewidth=1.4,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Reconfig steps")
    ax.set_ylabel("Head-start loss")
    ax.set_title(f"Head-start vs compute — {task}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    _style_axes(ax)
    _add_pref_arrow(ax, "down")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "headstart_curves.pdf"))

    # ---- Forgetting (stored protocol) vs budget
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for method, budgets in sorted(grouped.items()):
        if method in ("FixedMask", "NoMask"):
            continue
        xs, ys, errs = [], [], []
        for b in sorted(budgets):
            runs = budgets[b]
            mean, std = _agg(runs, ["summary_stored", "FM"])
            xs.append(b)
            ys.append(mean)
            errs.append(std)
        if xs:
            ax.errorbar(
                xs, ys, yerr=errs,
                color=METHOD_COLORS.get(method, "k"),
                marker=METHOD_MARKERS.get(method, "o"),
                capsize=3, label=method, linewidth=1.4,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Reconfig steps")
    ax.set_ylabel("Forgetting (FM)")
    ax.set_title(f"Forgetting vs compute — {task}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    _style_axes(ax)
    _add_pref_arrow(ax, "down")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "forgetting_vs_compute.pdf"))

    # ---- Sharing correlation vs budget
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for method, budgets in sorted(grouped.items()):
        if method in ("FixedMask", "NoMask"):
            continue
        xs, ys, errs = [], [], []
        for b in sorted(budgets):
            runs = budgets[b]
            mean, std = _agg(runs, ["sharing_corr"])
            xs.append(b)
            ys.append(mean)
            errs.append(std)
        if xs:
            ax.errorbar(
                xs, ys, yerr=errs,
                color=METHOD_COLORS.get(method, "k"),
                marker=METHOD_MARKERS.get(method, "o"),
                capsize=3, label=method, linewidth=1.4,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Reconfig steps")
    ax.set_ylabel("Sharing fidelity (r)")
    ax.set_title(f"Sharing fidelity vs compute — {task}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    _style_axes(ax)
    _add_pref_arrow(ax, "up")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "sharing_corr_vs_compute.pdf"))

    # ---- BWT vs budget
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for method, budgets in sorted(grouped.items()):
        if method in ("FixedMask", "NoMask"):
            continue
        xs, ys, errs = [], [], []
        for b in sorted(budgets):
            runs = budgets[b]
            mean, std = _agg(runs, ["summary_stored", "BWT"])
            xs.append(b)
            ys.append(mean)
            errs.append(std)
        if xs:
            ax.errorbar(
                xs, ys, yerr=errs,
                color=METHOD_COLORS.get(method, "k"),
                marker=METHOD_MARKERS.get(method, "o"),
                capsize=3, label=method, linewidth=1.4,
            )
    ax.axhline(0, color="k", linestyle=":", alpha=0.5, linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Reconfig steps")
    ax.set_ylabel("Backward transfer")
    ax.set_title(f"Backward transfer vs compute — {task}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    _style_axes(ax)
    _add_pref_arrow(ax, "up")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "bwt_vs_compute.pdf"))


# ---------------------------------------------------------------------------
# Main table -- 3 methods × 4 budgets × {ACC, FM, BWT, sharing_corr}
# ---------------------------------------------------------------------------


def write_main_table(data: dict, task: str) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    grouped = _group_by_method_budget(data["runs"])
    methods = [m for m in ["Sigmoid", "AnnealSTE", "Gumbel"] if m in grouped]
    if not methods:
        return
    # collect budgets across all methods
    budgets = sorted({b for m in methods for b in grouped[m]})

    rows = []
    for method in methods:
        for b in budgets:
            runs = grouped[method].get(b, [])
            if not runs:
                continue
            acc_m, acc_s = _agg(runs, ["summary_stored", "ACC"])
            fm_m, fm_s = _agg(runs, ["summary_stored", "FM"])
            bwt_m, bwt_s = _agg(runs, ["summary_stored", "BWT"])
            sc_m, sc_s = _agg(runs, ["sharing_corr"])
            head_runs = []
            for r in runs:
                hs_vals = [
                    h["task_loss_with_mask"] for h in r["head_start"][1:]
                ]
                head_runs.append(np.mean(hs_vals) if hs_vals else np.nan)
            head_m = float(np.nanmean(head_runs))
            head_s = float(np.nanstd(head_runs))
            rows.append(
                {
                    "method": method,
                    "budget": b,
                    "acc": (acc_m, acc_s),
                    "fm": (fm_m, fm_s),
                    "bwt": (bwt_m, bwt_s),
                    "head": (head_m, head_s),
                    "sharing": (sc_m, sc_s),
                    "n": len(runs),
                }
            )

    # baselines
    baseline_rows = []
    for bm in ("FixedMask", "NoMask"):
        if bm not in grouped:
            continue
        runs = sum(grouped[bm].values(), [])
        if not runs:
            continue
        acc_m, acc_s = _agg(runs, ["summary_stored", "ACC"])
        fm_m, fm_s = _agg(runs, ["summary_stored", "FM"])
        bwt_m, bwt_s = _agg(runs, ["summary_stored", "BWT"])
        sc_m, sc_s = _agg(runs, ["sharing_corr"])
        baseline_rows.append(
            {
                "method": bm,
                "acc": (acc_m, acc_s),
                "fm": (fm_m, fm_s),
                "bwt": (bwt_m, bwt_s),
                "sharing": (sc_m, sc_s),
                "n": len(runs),
            }
        )

    out_path = os.path.join(TABLES_DIR, f"main_{task}.tex")
    with open(out_path, "w") as f:
        f.write("% Auto-generated by experiments/generate_figures.py\n")
        task_label = task.replace("_", "\\_")
        f.write(
            f"\\begin{{table}}[t]\n\\caption{{Knowledge-transfer results on "
            f"\\texttt{{{task_label}}}: ACC, forgetting (FM), backward transfer "
            f"(BWT), neuron-sharing fidelity ($r_{{\\mathrm{{share}}}}$), "
            f"and pre-training mask-only loss (head-start) averaged over "
            f"tasks 2 to N. Mean$\\pm$std over seeds.}}\n"
        )
        f.write(f"\\label{{tab:main_{task}}}\n\\centering\n\\small\n")
        f.write(
            "\\begin{tabular}{llccccc}\n\\toprule\n"
            "Method & Budget & ACC$\\uparrow$ & FM$\\downarrow$ & BWT$\\uparrow$ "
            "& Head-start$\\downarrow$ & $r_{\\mathrm{share}}\\uparrow$\\\\\n"
            "\\midrule\n"
        )
        for row in rows:
            f.write(
                f"{row['method']} & {row['budget']} "
                f"& {row['acc'][0]:.3f}\\,$\\pm$\\,{row['acc'][1]:.3f} "
                f"& {row['fm'][0]:.3f}\\,$\\pm$\\,{row['fm'][1]:.3f} "
                f"& {row['bwt'][0]:+.3f}\\,$\\pm$\\,{row['bwt'][1]:.3f} "
                f"& {row['head'][0]:.3f}\\,$\\pm$\\,{row['head'][1]:.3f} "
                f"& {row['sharing'][0]:.3f}\\,$\\pm$\\,{row['sharing'][1]:.3f} \\\\\n"
            )
        if baseline_rows:
            f.write("\\midrule\n")
            for row in baseline_rows:
                f.write(
                    f"{row['method']} & -- "
                    f"& {row['acc'][0]:.3f}\\,$\\pm$\\,{row['acc'][1]:.3f} "
                    f"& {row['fm'][0]:.3f}\\,$\\pm$\\,{row['fm'][1]:.3f} "
                    f"& {row['bwt'][0]:+.3f}\\,$\\pm$\\,{row['bwt'][1]:.3f} "
                    f"& -- "
                    f"& {row['sharing'][0]:.3f}\\,$\\pm$\\,{row['sharing'][1]:.3f} \\\\\n"
                )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Per-task perf matrix (largest budget, seed=0) for each method
# ---------------------------------------------------------------------------


def render_perf_matrices(data: dict, task: str) -> None:
    out = os.path.join(FIG_DIR, task)
    os.makedirs(out, exist_ok=True)
    grouped = _group_by_method_budget(data["runs"])

    for method in ["Sigmoid", "AnnealSTE", "Gumbel"]:
        if method not in grouped:
            continue
        budgets = grouped[method]
        b_max = max(budgets)
        runs = budgets[b_max]
        # Seed 0 if available, else first
        run = next((r for r in runs if r["seed"] == 0), runs[0])
        R = run["perf_matrix_stored"]
        plotting.plot_perf_matrix(
            R,
            title=f"{method} budget={b_max} (stored, seed={run['seed']}) — {task}",
            out_path=os.path.join(
                out, f"perf_stored_{method}_b{b_max}.pdf"
            ),
        )
        R = run["perf_matrix_reconfig"]
        plotting.plot_perf_matrix(
            R,
            title=f"{method} budget={b_max} (reconfig, seed={run['seed']}) — {task}",
            out_path=os.path.join(
                out, f"perf_reconfig_{method}_b{b_max}.pdf"
            ),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", nargs="+",
        default=["synthetic", "mnist_shared"],
    )
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}")

    for task in args.tasks:
        data = load(task)
        if data is None:
            print(f"  [skip] no results for {task}")
            continue
        print(f"  [task] {task} ({len(data['runs'])} runs)")
        render_sharing(data, task)
        plot_compute_curves(data, task)
        plot_headstart_per_budget(data, task)
        write_main_table(data, task)
        render_perf_matrices(data, task)
    print("Done.")


if __name__ == "__main__":
    main()
