"""
generate_figures.py — Regenerate all figures from saved JSON results.

Run from project_c_3/:
    uv run python experiments/generate_figures.py [options]

Options:
    --experiments 1 2 3 4   Which experiment figures to generate (default: all)
    --results-dir results   Path to results directory
    --figures-dir figures   Path to figures output directory

Figures generated per experiment
---------------------------------
For each experiment and each method that has results:
  perf_matrix_{method}.pdf       — stored vs recovered side-by-side heatmap
  forgetting_curve_t{n}.pdf      — stored (solid) + recovered (dashed) per task
  metrics_bar.pdf                — stored vs recovered bars for all methods
  mask_tiling_{method}.pdf       — RGB overlay of training masks across seeds

Additionally for Exp1:
  task_vis_clf.pdf / task_vis_reg.pdf   — 2D scatter of task structure

No re-training required — all data is loaded from JSON.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import plotting
from tasks import SyntheticTasks

# ---------------------------------------------------------------------------
# Shared constants (must match multi_seed_experiments.py)
# ---------------------------------------------------------------------------

GRID_DIM = 32

TASK_COLORS = np.array(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.72, 0.10],
        [0.15, 0.35, 0.95],
        [0.95, 0.75, 0.00],
        [0.85, 0.10, 0.85],
        [0.00, 0.82, 0.82],
        [1.00, 0.50, 0.00],
        [0.55, 0.00, 0.90],
        [0.00, 0.50, 0.25],
        [0.55, 0.30, 0.00],
    ]
)

EXP1_INPUT_SIZE = 2
EXP1_N_TASKS    = 3
EXP1_TASK_SIZE  = 4
EXP1_FREQUENCY  = 8

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> dict | None:
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def _methods(data: dict) -> list[str]:
    """Return sorted list of methods present in ALL seeds."""
    if "aggregated" in data:
        return sorted(data["aggregated"].keys())
    # Fall back to intersection across seeds
    all_methods = None
    for sd in data["seeds"].values():
        m = set(sd.keys())
        all_methods = m if all_methods is None else all_methods & m
    return sorted(all_methods or [])


def _seed_pm(data: dict, method: str, proto: str) -> list[list[float]] | None:
    """Return perf_matrix for the first seed that has this method."""
    key = f"perf_matrix_{proto}"
    for sk in sorted(data["seeds"].keys(), key=int):
        sd = data["seeds"][sk]
        if method in sd and key in sd[method]:
            return sd[method][key]
    return None


def _all_mask_stores(data: dict, method: str) -> list[dict]:
    """Return mask_store for every seed that has this method."""
    stores = []
    for sk in sorted(data["seeds"].keys(), key=int):
        sd = data["seeds"][sk]
        if method in sd and sd[method].get("mask_store"):
            stores.append(sd[method]["mask_store"])
    return stores


def _pm_mean_std(data: dict, method: str, proto: str):
    """Return (mean_pm, std_pm) for the given method and protocol."""
    pm_mean = data.get("pm_mean", {})
    pm_std  = data.get("pm_std",  {})
    if method not in pm_mean:
        return None, None
    return pm_mean[method].get(proto), pm_std[method].get(proto)


def _summaries(data: dict, proto: str) -> list[dict]:
    """Return list of aggregated summaries (one per method) for a protocol."""
    agg = data.get("aggregated", {})
    return [agg[m][proto] for m in sorted(agg.keys()) if proto in agg[m]]


# ---------------------------------------------------------------------------
# Mask tiling
# ---------------------------------------------------------------------------

def _render_mask_rgb(mask_store: dict, grid_dim: int) -> np.ndarray:
    n = grid_dim * grid_dim
    rgb = np.zeros((n, 3), dtype=np.float32)
    for task_id, mask in mask_store.items():
        m = np.array(mask, dtype=np.float32)[:n]
        color = TASK_COLORS[int(task_id) % len(TASK_COLORS)]
        rgb += np.outer(m, color)
    return rgb.clip(0.0, 1.0).reshape(grid_dim, grid_dim, 3)


def save_mask_tiling(
    mask_stores: list[dict],
    grid_dim: int,
    out_path: str,
    n_cols: int = 4,
    title: str = "",
) -> None:
    tiles = [_render_mask_rgb(ms, grid_dim) for ms in mask_stores if ms]
    if not tiles:
        return
    n = len(tiles)
    n_rows = (n + n_cols - 1) // n_cols
    h = w = grid_dim
    delim = 1
    canvas = np.ones((n_rows * h + (n_rows - 1) * delim,
                      n_cols * w + (n_cols - 1) * delim, 3), dtype=np.float32)
    for i, tile in enumerate(tiles):
        r, c = i // n_cols, i % n_cols
        canvas[r*(h+delim):r*(h+delim)+h, c*(w+delim):c*(w+delim)+w] = tile
    scale = 2
    fig, ax = plt.subplots(figsize=(n_cols*w*scale/72, n_rows*h*scale/72), dpi=72)
    ax.imshow(canvas, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Task visualization (Exp1 only)
# ---------------------------------------------------------------------------

def save_task_visualization(is_clf: bool, fig_dir: str, seed: int = 0) -> None:
    n_pts = 2000
    out_size = 2 if is_clf else 1
    task_vis = SyntheticTasks(
        input_size=EXP1_INPUT_SIZE, output_size=out_size, n_tasks=EXP1_N_TASKS,
        task_size=EXP1_TASK_SIZE, frequency=EXP1_FREQUENCY,
        classification=is_clf, seed=seed,
    )
    fig, axes = plt.subplots(1, EXP1_N_TASKS, figsize=(3 * EXP1_N_TASKS, 3))
    for t, ax in enumerate(axes):
        x, y = task_vis.sample_batch(t, n_pts)
        x0, x1 = x[:, 0].numpy(), x[:, 1].numpy()
        if is_clf:
            c = y.argmax(-1).numpy().astype(float)
            ax.scatter(x0, x1, c=c, cmap="bwr", s=3, alpha=0.4, vmin=0, vmax=1)
        else:
            c = y[:, 0].numpy()
            vmax = float(np.abs(c).max()) + 1e-8
            ax.scatter(x0, x1, c=c, cmap="RdBu", s=3, alpha=0.4, vmin=-vmax, vmax=vmax)
        ax.set_title(f"Task {t}", fontsize=9)
        ax.set_xlabel("$x_0$", fontsize=8)
        ax.set_ylabel("$x_1$", fontsize=8)
        ax.tick_params(labelsize=7)
    variant = "clf" if is_clf else "reg"
    fig.suptitle(f"Synthetic ({'classification' if is_clf else 'regression'})", fontsize=10)
    fig.tight_layout()
    out_path = f"{fig_dir}/task_vis_{variant}.pdf"
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Core figure routine — works for any experiment data dict
# ---------------------------------------------------------------------------

def generate_figures_for(
    data: dict,
    fig_dir: str,
    task_labels: list[str] | None = None,
    forgetting_task_indices: list[int] | None = None,
    cmap: str = "YlGn",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    suffix: str = "",          # appended to filenames (e.g. "_clf")
    regression: bool = False,  # if True: clamped MSE colour range, MSE↓ labels
) -> None:
    """
    Generate the full figure suite for one experiment result dict.

    Produces per-method and cross-method figures covering both eval protocols.
    """
    os.makedirs(fig_dir, exist_ok=True)
    methods = _methods(data)
    if not methods:
        print(f"  [skip] no complete methods in {fig_dir}")
        return

    n_tasks = len(list(data.get("pm_mean", {}).values())[0]["stored"]) if data.get("pm_mean") else None

    # Regression display options: ACC entry in summaries is mean MSE;
    # FM and BWT are the (peak-MSE − final-MSE) and (final − first) variants.
    # Lower is better for all three under MSE semantics.
    metric_labels_reg = {"ACC": "MSE↓", "FM": "FM↓", "BWT": "BWT↓"}
    metric_labels = metric_labels_reg if regression else None

    # ----------------------------------------------------------------
    # 1. Per-method: mask tiling + side-by-side perf matrices
    # ----------------------------------------------------------------
    for method in methods:
        # Mask tiling
        stores = _all_mask_stores(data, method)
        if stores:
            save_mask_tiling(
                stores, GRID_DIM,
                out_path=f"{fig_dir}/mask_tiling_{method}{suffix}.pdf",
                title=f"{method} mask allocations ({len(stores)} seeds)",
            )

        # Side-by-side perf matrix (stored | reconfig)
        pm_s = _seed_pm(data, method, "stored")
        pm_r = _seed_pm(data, method, "reconfig")
        if pm_s and pm_r:
            if regression:
                plotting.plot_perf_matrix_both_regression(
                    pm_s, pm_r,
                    method=method,
                    out_path=f"{fig_dir}/perf_matrix_{method}{suffix}.pdf",
                )
            else:
                plotting.plot_perf_matrix_both(
                    pm_s, pm_r,
                    method=method,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    out_path=f"{fig_dir}/perf_matrix_{method}{suffix}.pdf",
                )

        # Also individual stored/reconfig heatmaps for clarity
        if pm_s:
            if regression:
                plotting.plot_perf_matrix_regression(
                    pm_s,
                    title=f"{method} — stored mask (oracle)",
                    out_path=f"{fig_dir}/perf_matrix_{method}_stored{suffix}.pdf",
                )
            else:
                plotting.plot_perf_matrix(
                    pm_s,
                    title=f"{method} — stored mask (oracle)",
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    out_path=f"{fig_dir}/perf_matrix_{method}_stored{suffix}.pdf",
                )
        if pm_r:
            if regression:
                plotting.plot_perf_matrix_regression(
                    pm_r,
                    title=f"{method} — recovered mask",
                    out_path=f"{fig_dir}/perf_matrix_{method}_recovered{suffix}.pdf",
                )
            else:
                plotting.plot_perf_matrix(
                    pm_r,
                    title=f"{method} — recovered mask",
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    out_path=f"{fig_dir}/perf_matrix_{method}_recovered{suffix}.pdf",
                )

        # Mask grid (useful for small task counts, e.g. Exp4)
        if stores and n_tasks is not None and n_tasks <= 6:
            mask_tensors = {int(k): torch.tensor(v) for k, v in stores[0].items()}
            if mask_tensors:
                plotting.plot_mask_grid(
                    mask_tensors, GRID_DIM,
                    title=f"{method} subnetworks",
                    out_path=f"{fig_dir}/mask_grid_{method}{suffix}.pdf",
                )

    # ----------------------------------------------------------------
    # 2. Forgetting curves — one figure per task, all methods, both protos
    # ----------------------------------------------------------------
    if n_tasks is not None:
        if forgetting_task_indices is None:
            forgetting_task_indices = list(range(n_tasks))

        stored_means  = {m: _pm_mean_std(data, m, "stored")[0]   for m in methods}
        stored_stds   = {m: _pm_mean_std(data, m, "stored")[1]   for m in methods}
        reconfig_means = {m: _pm_mean_std(data, m, "reconfig")[0] for m in methods}
        reconfig_stds  = {m: _pm_mean_std(data, m, "reconfig")[1] for m in methods}

        # Filter out Nones
        stored_means   = {m: v for m, v in stored_means.items()   if v}
        stored_stds    = {m: v for m, v in stored_stds.items()    if v}
        reconfig_means = {m: v for m, v in reconfig_means.items() if v}
        reconfig_stds  = {m: v for m, v in reconfig_stds.items()  if v}

        for t in forgetting_task_indices:
            task_label = task_labels[t] if task_labels and t < len(task_labels) else str(t)
            title = f"Task-{t} ({task_label}) Retention — stored vs recovered"

            # Combined dual plot (all methods, both protos)
            plotting.plot_forgetting_curves_both(
                stored_means, reconfig_means,
                task_idx=t,
                title=title,
                out_path=f"{fig_dir}/forgetting_curve_t{t}{suffix}.pdf",
                std_stored=stored_stds,
                std_reconfig=reconfig_stds,
                ylabel=(f"MSE↓ on Task {t}" if regression else None),
            )

    # ----------------------------------------------------------------
    # 3. Metrics bar — all methods, stored vs recovered
    # ----------------------------------------------------------------
    summaries_s = _summaries(data, "stored")
    summaries_r = _summaries(data, "reconfig")
    if summaries_s and summaries_r:
        plotting.plot_multi_metric_both(
            summaries_s, summaries_r,
            metrics=["ACC", "FM", "BWT"],
            title="CL Metrics — stored (oracle) vs recovered",
            out_path=f"{fig_dir}/metrics_bar{suffix}.pdf",
            metric_labels=metric_labels,
        )
        # Also individual protocol bars for cleaner single-figure use
        plotting.plot_multi_metric(
            summaries_s,
            title="CL Metrics — stored mask (oracle)",
            out_path=f"{fig_dir}/metrics_bar_stored{suffix}.pdf",
            metric_labels=metric_labels,
        )
        plotting.plot_multi_metric(
            summaries_r,
            title="CL Metrics — recovered mask",
            out_path=f"{fig_dir}/metrics_bar_recovered{suffix}.pdf",
            metric_labels=metric_labels,
        )
        plotting.plot_metrics_table(
            summaries_s,
            title="CL Metrics (stored mask)",
            out_path=f"{fig_dir}/metrics_table_stored{suffix}.pdf",
            metric_labels=metric_labels,
        )
        plotting.plot_metrics_table(
            summaries_r,
            title="CL Metrics (recovered mask)",
            out_path=f"{fig_dir}/metrics_table_recovered{suffix}.pdf",
            metric_labels=metric_labels,
        )


# ---------------------------------------------------------------------------
# Per-experiment wrappers
# ---------------------------------------------------------------------------

def figures_exp1(data_clf: dict | None, data_reg: dict | None, fig_base: str) -> None:
    for variant, is_clf, data in [("clf", True, data_clf), ("reg", False, data_reg)]:
        if data is None:
            continue
        fig_dir = f"{fig_base}/exp1_synthetic_{variant}"
        save_task_visualization(is_clf, fig_dir, seed=0)

        generate_figures_for(
            data, fig_dir,
            forgetting_task_indices=[0],
            cmap="YlGn",
            vmin=0.0,
            vmax=1.0,
            suffix=f"_{variant}",
            regression=not is_clf,
        )


def figures_exp2(data: dict | None, fig_base: str) -> None:
    if data is None:
        return
    fig_dir = f"{fig_base}/exp2_mnist_shuffled"
    n_tasks = 5
    # Infer n_tasks from data
    methods = _methods(data)
    if methods and data.get("pm_mean"):
        n_tasks = len(data["pm_mean"][methods[0]]["stored"])
    generate_figures_for(
        data, fig_dir,
        forgetting_task_indices=list(range(n_tasks)),
    )


def figures_exp3(data: dict | None, fig_base: str) -> None:
    if data is None:
        return
    fig_dir = f"{fig_base}/exp3_permuted_mnist"
    methods = _methods(data)
    n_tasks = 10
    if methods and data.get("pm_mean"):
        n_tasks = len(data["pm_mean"][methods[0]]["stored"])
    highlight = sorted({0, min(4, n_tasks - 1)})
    generate_figures_for(
        data, fig_dir,
        forgetting_task_indices=highlight,
    )


def figures_exp4(data: dict | None, fig_base: str) -> None:
    if data is None:
        return
    fig_dir = f"{fig_base}/exp4_elec2"
    # Four contiguous time-regime chunks of the 1996-05..1998-12 ELEC2 record
    regime_names = ["Regime 1", "Regime 2", "Regime 3", "Regime 4"]
    generate_figures_for(
        data, fig_dir,
        task_labels=regime_names,
        forgetting_task_indices=list(range(len(regime_names))),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate figures from project_c_3 results JSON files."
    )
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        choices=["1", "2", "3", "4"],
        default=["1", "2", "3", "4"],
        help="Which experiment figures to generate (default: all)",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}")
    print(f"Results from: {args.results_dir}/")
    print(f"Figures to:   {args.figures_dir}/")

    if "1" in args.experiments:
        d_clf = _load(f"{args.results_dir}/exp1_synthetic_clf.json")
        d_reg = _load(f"{args.results_dir}/exp1_synthetic_reg.json")
        figures_exp1(d_clf, d_reg, args.figures_dir)

    if "2" in args.experiments:
        figures_exp2(_load(f"{args.results_dir}/exp2_mnist_shuffled.json"), args.figures_dir)

    if "3" in args.experiments:
        figures_exp3(_load(f"{args.results_dir}/exp3_permuted_mnist.json"), args.figures_dir)

    if "4" in args.experiments:
        figures_exp4(_load(f"{args.results_dir}/exp4_elec2.json"), args.figures_dir)

    print("\nDone.")
