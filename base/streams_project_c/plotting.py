"""
plotting.py — Visualisation utilities for project_c experiments.

All functions save figures to a specified output directory and optionally
return the figure object for further manipulation.
"""

from __future__ import annotations
import os
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
          "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62"]

METHOD_COLORS = {
    "NoMask":    "#aaaaaa",
    "FixedMask": "#984ea3",
    "KWTA":      "#ff7f00",
    "FTN-Fast":  "#377eb8",
    "FTN-Slow":  "#4daf4a",
    "EWC":       "#66c2a5",
}

DPI = 300


def _color(method: str) -> str:
    """Return a stable colour for a method, falling back to a hash-based pick."""
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    # Hash the name so the colour is the same regardless of which other methods
    # are present in the same plot.
    return COLORS[hash(method) % len(COLORS)]


def save(fig, path: str | Path) -> None:
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Performance matrix heatmap
# ---------------------------------------------------------------------------

def plot_perf_matrix(
    R: list[list[float]],
    title: str = "",
    figsize: tuple = (5, 4),
    fmt: str = ".3f",
    cmap: str = "YlGn",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    out_path: str | None = None,
    cbar_label: str | None = None,
    cbar_extend: str = "neither",
) -> plt.Figure:
    """Heatmap of lower-triangular performance matrix R[i][j]."""
    n = len(R)
    mat = np.full((n, n), np.nan)
    for i, row in enumerate(R):
        for j, v in enumerate(row):
            mat[i, j] = v

    finite = mat[np.isfinite(mat)]
    _vmin = float(finite.min()) if vmin is None else vmin
    _vmax = float(finite.max()) if vmax is None else vmax

    fig, ax = plt.subplots(figsize=figsize)
    if cbar_extend in ("max", "both"):
        mat_disp = np.minimum(mat, _vmax)
    else:
        mat_disp = mat
    if cbar_extend in ("min", "both"):
        mat_disp = np.maximum(mat_disp, _vmin)
    im = ax.imshow(mat_disp, vmin=_vmin, vmax=_vmax, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, extend=cbar_extend)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:{fmt}}", ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_xticks(range(n)); ax.set_xticklabels([f"T{j}" for j in range(n)])
    ax.set_yticks(range(n)); ax.set_yticklabels([f"After T{i}" for i in range(n)])
    ax.set_xlabel("Evaluated Task")
    ax.set_ylabel("After Training")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()

    if out_path:
        save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Forgetting curve — per-task performance over sequential training
# ---------------------------------------------------------------------------

def plot_forgetting_curves(
    perf_matrix_dict: dict[str, list[list[float]]],
    task_idx: int = 0,
    title: str = "",
    figsize: tuple = (5, 3.5),
    out_path: str | None = None,
    std_dict: dict[str, list[list[float]]] | None = None,
    ylabel: str | None = None,
) -> plt.Figure:
    """
    Plot how performance on a given task changes as more tasks are trained.
    Shows one line per method. If std_dict is provided, draws a shaded band
    of ±1 std around each line.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for method, R in perf_matrix_dict.items():
        n = len(R)
        x_vals, y_vals = [], []
        for i in range(task_idx, n):
            if i < len(R) and task_idx < len(R[i]):
                x_vals.append(i)
                y_vals.append(R[i][task_idx])
        color = _color(method)
        ax.plot(x_vals, y_vals, "o-", label=method, color=color, linewidth=1.3, markersize=4)

        if std_dict is not None and method in std_dict:
            S = std_dict[method]
            s_vals = [S[i][task_idx] for i in range(task_idx, n)
                      if i < len(S) and task_idx < len(S[i])]
            y = np.array(y_vals)
            s = np.array(s_vals)
            ax.fill_between(x_vals, y - s, y + s, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Tasks trained so far", fontsize=7)
    ax.set_ylabel(ylabel or f"Perf on Task {task_idx}", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xticks(range(len(list(perf_matrix_dict.values())[0])))
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout()

    if out_path:
        save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Final accuracy bar chart
# ---------------------------------------------------------------------------

def plot_final_accuracy(
    summaries: list[dict],
    metric: str = "ACC",
    title: str = "",
    figsize: tuple = (5, 3.5),
    out_path: str | None = None,
) -> plt.Figure:
    """Bar chart comparing final performance across methods."""
    labels = [s.get("label", f"M{i}") for i, s in enumerate(summaries)]
    values = [s.get(metric, 0.0) for s in summaries]
    colors = [_color(l) for l in labels]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()

    if out_path:
        save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    loss_history: list[tuple],
    n_tasks: int,
    smoothing: float = 0.9,
    title: str = "",
    figsize: tuple = (6, 3),
    out_path: str | None = None,
) -> plt.Figure:
    """Plot EMA-smoothed training loss per task."""
    # Separate by task
    task_losses: dict[int, list[float]] = {t: [] for t in range(n_tasks)}
    for (t, epoch, step, loss) in loss_history:
        task_losses[t].append(loss)

    fig, ax = plt.subplots(figsize=figsize)
    for t, losses in task_losses.items():
        if not losses:
            continue
        ema = []
        v = losses[0]
        for l in losses:
            v = smoothing * v + (1 - smoothing) * l
            ema.append(v)
        ax.plot(ema, label=f"Task {t}", color=COLORS[t % len(COLORS)], linewidth=1.2)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (EMA)")
    ax.legend(fontsize=8, ncol=min(n_tasks, 5))
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()

    if out_path:
        save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Mask visualisation (2-D cortical grid)
# ---------------------------------------------------------------------------

def plot_mask_grid(
    masks: dict[int, torch.Tensor],
    grid_dim: int,
    title: str = "",
    figsize: tuple = (5, 2),
    out_path: str | None = None,
) -> plt.Figure:
    """
    RGB visualisation of task masks on the cortical grid.
    Each task is assigned a colour channel; overlap shows as mixed colour.
    """
    n_tasks = len(masks)
    # Use first 3 channels for RGB; stack additional tasks as separate panels
    all_masks = list(masks.values())
    hidden_size = all_masks[0].numel()

    n_panels = math.ceil(n_tasks / 3)
    fig, axes = plt.subplots(1, n_panels, figsize=(figsize[0] * n_panels, figsize[1]))
    if n_panels == 1:
        axes = [axes]

    for panel in range(n_panels):
        task_slice = all_masks[panel*3 : (panel+1)*3]
        rgb = torch.zeros(3, hidden_size)
        for ch, m in enumerate(task_slice):
            rgb[ch] = m.float().cpu()

        img = rgb.reshape(3, grid_dim, grid_dim).permute(1, 2, 0).numpy()
        img = img.clip(0, 1)
        axes[panel].imshow(img, interpolation="nearest")
        t_range = list(range(panel*3, min((panel+1)*3, n_tasks)))
        axes[panel].set_title(f"Tasks {t_range}", fontsize=8)
        axes[panel].axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    if out_path:
        save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Metrics comparison table (rendered as figure for paper)
# ---------------------------------------------------------------------------

def plot_metrics_table(
    summaries: list[dict],
    metrics: list[str] = ("ACC", "FM", "BWT"),
    title: str = "",
    figsize: tuple = (6, 2),
    out_path: str | None = None,
    metric_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Render a metrics comparison table as a matplotlib figure."""
    methods = [s.get("label", f"M{i}") for i, s in enumerate(summaries)]
    cols    = list(metrics)
    data    = [[f"{s.get(m, float('nan')):.4f}" for m in cols] for s in summaries]
    metric_labels = metric_labels or {}
    col_display = [metric_labels.get(m, m) for m in cols]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    col_labels = ["Method"] + col_display
    row_data   = [[m] + row for m, row in zip(methods, data)]

    tbl = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)

    # Highlight header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c2c2c")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    if title:
        ax.set_title(title, fontsize=10, pad=10)
    fig.tight_layout()

    if out_path:
        save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Multi-metric comparison: side-by-side bar charts
# ---------------------------------------------------------------------------

def plot_multi_metric(
    summaries: list[dict],
    metrics: list[str] = ("ACC", "FM", "BWT"),
    title: str = "",
    figsize: tuple = (8, 3),
    out_path: str | None = None,
    metric_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Side-by-side bar chart for multiple CL metrics."""
    n_methods = len(summaries)
    n_metrics = len(metrics)
    labels    = [s.get("label", f"M{i}") for i, s in enumerate(summaries)]
    colors    = [_color(l) for l in labels]
    metric_labels = metric_labels or {}

    x = np.arange(n_methods)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals = [s.get(metric, 0.0) for s in summaries]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(metric_labels.get(metric, metric), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01 * (1 if val >= 0 else -1),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    if title:
        fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()

    if out_path:
        save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Dual forgetting curves: stored mask vs recovered mask
# ---------------------------------------------------------------------------

def plot_forgetting_curves_both(
    stored_dict: dict[str, list[list[float]]],
    reconfig_dict: dict[str, list[list[float]]],
    task_idx: int = 0,
    title: str = "",
    figsize: tuple = (5, 3.5),
    out_path: str | None = None,
    std_stored: dict[str, list[list[float]]] | None = None,
    std_reconfig: dict[str, list[list[float]]] | None = None,
    ylabel: str | None = None,
) -> plt.Figure:
    """
    Plot forgetting curves for one task, showing both evaluation protocols.

    Solid lines = stored-mask eval (oracle: correct subnetwork used).
    Dashed lines = reconfig eval (realistic: mask recovered from data).
    Same colour per method.
    """
    fig, ax = plt.subplots(figsize=figsize)
    methods = list(stored_dict.keys())

    for method in methods:
        R_s = stored_dict[method]
        R_r = reconfig_dict.get(method, R_s)
        n = len(R_s)
        color = _color(method)

        x_vals = [i for i in range(task_idx, n) if task_idx < len(R_s[i])]
        y_s = [R_s[i][task_idx] for i in x_vals]
        y_r = [R_r[i][task_idx] for i in x_vals]

        ax.plot(x_vals, y_s, "o-",  label=f"{method} (oracle)",    color=color, linewidth=1.3, markersize=4)
        ax.plot(x_vals, y_r, "o--", label=f"{method} (recovered)", color=color, linewidth=1.3, markersize=4, alpha=0.7)

        if std_stored and method in std_stored:
            s = np.array([std_stored[method][i][task_idx] for i in x_vals])
            ax.fill_between(x_vals, np.array(y_s) - s, np.array(y_s) + s,
                            color=color, alpha=0.10, linewidth=0)
        if std_reconfig and method in std_reconfig:
            s = np.array([std_reconfig[method][i][task_idx] for i in x_vals])
            ax.fill_between(x_vals, np.array(y_r) - s, np.array(y_r) + s,
                            color=color, alpha=0.08, linewidth=0)

    ax.set_xlabel("Tasks trained so far", fontsize=7)
    ax.set_ylabel(ylabel or f"Perf on Task {task_idx}", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xticks(range(len(list(stored_dict.values())[0])))
    ax.legend(fontsize=6, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.42), framealpha=0.85)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout()

    if out_path:
        save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Dual metrics bar: stored vs recovered, grouped by method
# ---------------------------------------------------------------------------

def plot_multi_metric_both(
    summaries_stored: list[dict],
    summaries_reconfig: list[dict],
    metrics: list[str] = ("ACC", "FM", "BWT"),
    title: str = "",
    figsize: tuple = (10, 3),
    out_path: str | None = None,
    metric_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """
    Side-by-side bar chart comparing stored and reconfig evals across methods.

    For each metric, each method gets two adjacent bars:
      - filled = stored mask (oracle)
      - hatched = recovered mask (realistic)
    """
    n_methods = len(summaries_stored)
    n_metrics = len(metrics)
    labels = [s.get("label", f"M{i}") for i, s in enumerate(summaries_stored)]
    colors = [_color(l) for l in labels]
    metric_labels = metric_labels or {}

    width = 0.38
    x = np.arange(n_methods)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vs = [s.get(metric, 0.0) for s in summaries_stored]
        vr = [s.get(metric, 0.0) for s in summaries_reconfig]

        bars_s = ax.bar(x - width / 2, vs, width, color=colors,
                        edgecolor="white", linewidth=0.5)
        bars_r = ax.bar(x + width / 2, vr, width, color=colors,
                        edgecolor="white", linewidth=0.5, alpha=0.55, hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(metric_labels.get(metric, metric), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in list(zip(bars_s, vs)) + list(zip(bars_r, vr)):
            if abs(val) > 1e-4:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005 * (1 if val >= 0 else -1),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6)

    # Legend below the subplots, centred
    fig.legend(["stored (oracle)", "recovered"], fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, framealpha=0.85)

    if title:
        fig.suptitle(title, fontsize=9, y=1.02)
    fig.tight_layout()

    if out_path:
        save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Side-by-side perf matrices: stored vs recovered
# ---------------------------------------------------------------------------

def plot_perf_matrix_both(
    R_stored: list[list[float]],
    R_reconfig: list[list[float]],
    method: str = "",
    figsize: tuple = (9, 4),
    fmt: str = ".3f",
    cmap: str = "YlGn",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    out_path: str | None = None,
    cbar_label: str | None = None,
    cbar_extend: str = "neither",
) -> plt.Figure:
    """Two heatmaps side by side: stored mask (left) vs recovered mask (right)."""
    n = max(len(R_stored), len(R_reconfig))

    def _to_mat(R):
        mat = np.full((n, n), np.nan)
        for i, row in enumerate(R):
            for j, v in enumerate(row):
                mat[i, j] = v
        return mat

    mat_s = _to_mat(R_stored)
    mat_r = _to_mat(R_reconfig)

    finite_all = np.concatenate([mat_s[np.isfinite(mat_s)], mat_r[np.isfinite(mat_r)]])
    _vmin = float(finite_all.min()) if vmin is None else vmin
    _vmax = float(finite_all.max()) if vmax is None else vmax

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ticklabels = [f"T{j}" for j in range(n)]
    rowlabels  = [f"After T{i}" for i in range(n)]

    def _clip(mat):
        out = mat
        if cbar_extend in ("max", "both"):
            out = np.minimum(out, _vmax)
        if cbar_extend in ("min", "both"):
            out = np.maximum(out, _vmin)
        return out

    for ax, mat, subtitle in zip(axes, [mat_s, mat_r], ["stored (oracle)", "recovered"]):
        im = ax.imshow(_clip(mat), vmin=_vmin, vmax=_vmax, cmap=cmap, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, extend=cbar_extend)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=8)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center", va="center",
                            fontsize=7, color="black")
        ax.set_xticks(range(n)); ax.set_xticklabels(ticklabels)
        ax.set_yticks(range(n)); ax.set_yticklabels(rowlabels)
        ax.set_xlabel("Evaluated Task")
        ax.set_ylabel("After Training")
        title_str = f"{method} — {subtitle}" if method else subtitle
        ax.set_title(title_str, fontsize=9)

    fig.tight_layout()
    if out_path:
        save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Regression-specific perf-matrix plots
# ---------------------------------------------------------------------------

# All regression heatmaps clamp MSE to [0, 0.5] so figures from different
# methods are directly comparable; values above 0.5 share the saturated
# top-of-range colour (colorbar marked with "extend=max" arrow).
REG_VMIN, REG_VMAX = 0.0, 0.5
REG_CMAP = "YlOrRd"  # low MSE → light, high MSE → dark red


def plot_perf_matrix_regression(
    R: list[list[float]],
    title: str = "",
    figsize: tuple = (5, 4),
    fmt: str = ".3f",
    out_path: str | None = None,
) -> plt.Figure:
    """Regression-flavoured heatmap: MSE clamped to [0, 0.5]; ↓ = better."""
    return plot_perf_matrix(
        R,
        title=title,
        figsize=figsize,
        fmt=fmt,
        cmap=REG_CMAP,
        vmin=REG_VMIN,
        vmax=REG_VMAX,
        out_path=out_path,
        cbar_label="MSE↓ (clamped at 0.5)",
        cbar_extend="max",
    )


def plot_perf_matrix_both_regression(
    R_stored: list[list[float]],
    R_reconfig: list[list[float]],
    method: str = "",
    figsize: tuple = (9, 4),
    fmt: str = ".3f",
    out_path: str | None = None,
) -> plt.Figure:
    """Side-by-side stored/recovered heatmaps, MSE clamped to [0, 0.5]."""
    return plot_perf_matrix_both(
        R_stored,
        R_reconfig,
        method=method,
        figsize=figsize,
        fmt=fmt,
        cmap=REG_CMAP,
        vmin=REG_VMIN,
        vmax=REG_VMAX,
        out_path=out_path,
        cbar_label="MSE↓ (clamped at 0.5)",
        cbar_extend="max",
    )
