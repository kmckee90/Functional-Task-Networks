#!/usr/bin/env python
"""
analyze_sweep.py - Aggregate + compare the conditions in one run_seeds.py sweep.

Reads every metrics_*.txt under a sweep (the per-seed files run_seeds.py drops
at <sweep>/<condition>/seed<k>/metrics_seed<k>.txt), groups them by condition,
and for each numeric metric:

  - a bar chart comparing all conditions side by side (one colored bar per
    condition, legend = condition names) with error bars over seeds, and
  - Welch's two-sided t-tests between every pair of conditions (always written
    to results.txt; optionally annotated on the charts as significance brackets
    with --sig-brackets, off by default since the bracket count is O(C^2)).

Everything is written under <sweep>/figs/:
  - bar_<metric>.png            one chart per metric
  - grid.pdf                    all metrics as one grid (4 columns) with a
                                single shared condition legend
  - rankings.pdf                leaderboard: the best 3 conditions per metric
                                (best-first, direction-aware) with mean ± margin
  - results.txt                 per-condition n / mean / std / sem / 95% CI and
                                every pairwise t-test (t, p, significance)

Usage:
  uv run python analyze_sweep.py artifacts/sweeps/replay_vs_noreplay
  uv run python analyze_sweep.py <sweep> --errorbar std   # std instead of 95% CI

Error bars default to the 95% CI of the mean; --errorbar {ci,std,sem} switches.
The t-tests are computed over the raw seed values regardless of the error-bar
choice. scipy is used when available for exact p-values / t critical values and
degrades to a normal approximation otherwise (noted in results.txt).
"""

import argparse
import glob
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

try:
    from scipy import stats as _stats
except Exception:  # scipy optional; fall back to a normal approximation
    _stats = None

# Identifier fields that are not metrics to compare.
_EXCLUDE = {"seed", "run_name", "config", "n_tasks"}

# Ranking direction per metric. Cost metrics: smaller is better. Accuracies:
# larger is better (the default for anything unlisted). Mask overlap has no
# obvious "good" direction, so it is ranked high->low without asserting better.
_LOWER_IS_BETTER = {
    "n_params_trainable", "n_params_total", "flops_forward",
    "wall_time_forward_ms", "wall_time_reconfig_ms", "avg_task_block_wall_s",
}
_NEUTRAL = {"avg_mask_overlap_pct"}

# Accuracy metrics: their charts start the y-axis at the chance/null value 0.5
# (bars show accuracy above chance). NOT the forward-transfer *change* metric
# (a delta that can be negative) or any non-accuracy metric.
_ACCURACY_METRICS = {
    "avg_recall_eval_acc",
    "avg_forward_transfer_eval_acc",
    "avg_train_acc_current_block",
    "avg_eval_acc_current_task",
}

# Preferred metric order (matches metrics.py); extras are appended, sorted.
_METRIC_ORDER = [
    "n_params_trainable",
    "n_params_total",
    "flops_forward",
    "wall_time_forward_ms",
    "wall_time_reconfig_ms",
    "avg_task_block_wall_s",
    "avg_recall_eval_acc",
    "avg_forward_transfer_eval_acc",
    "avg_forward_transfer_change_per_block",
    "avg_train_acc_current_block",
    "avg_eval_acc_current_task",
    "avg_mask_overlap_pct",
]


# =============================================================================
# Parsing
# =============================================================================


def parse_metrics_file(path):
    """Parse a metrics_*.txt into {key: value}; numeric where possible, None for
    'NA', str for non-numeric (run_name/config)."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = (s.strip() for s in line.split("=", 1))
            if val == "NA":
                out[key] = None
                continue
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def load_sweep(sweep_dir):
    """Return (conditions_in_order, {condition: [metrics_dict, ...]}).

    Conditions are ordered by first appearance in manifest.txt when present,
    otherwise alphabetically. Each condition maps to the list of its per-seed
    parsed metrics dicts.
    """
    paths = sorted(glob.glob(os.path.join(sweep_dir, "*", "*", "metrics_*.txt")))
    if not paths:
        raise SystemExit(
            f"No metrics_*.txt found under {sweep_dir}/*/*/. "
            "Point me at a sweep directory (the one containing the condition "
            "folders / manifest.txt)."
        )

    by_cond = {}
    for p in paths:
        cond = os.path.basename(os.path.dirname(os.path.dirname(p)))
        by_cond.setdefault(cond, []).append(parse_metrics_file(p))

    order = _manifest_order(sweep_dir)
    conditions = [c for c in order if c in by_cond]
    conditions += sorted(c for c in by_cond if c not in conditions)
    return conditions, by_cond


def _manifest_order(sweep_dir):
    """First-appearance order of conditions from manifest.txt, or [] if absent."""
    path = os.path.join(sweep_dir, "manifest.txt")
    if not os.path.isfile(path):
        return []
    order = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cond = line.split("\t", 1)[0]
            if cond and cond not in order:
                order.append(cond)
    return order


def collect_metric_keys(by_cond):
    """Numeric metric keys present in any condition, in preferred then sorted
    order, excluding identifier fields and all-missing metrics."""
    keys = set()
    for rows in by_cond.values():
        for r in rows:
            for k, v in r.items():
                if k not in _EXCLUDE and isinstance(v, (int, float)):
                    keys.add(k)
    ordered = [k for k in _METRIC_ORDER if k in keys]
    ordered += sorted(k for k in keys if k not in _METRIC_ORDER)
    return ordered


def values_for(by_cond, cond, metric):
    """Sorted-by-seed list of finite values for one condition/metric."""
    vals = []
    for r in by_cond[cond]:
        v = r.get(metric)
        if isinstance(v, (int, float)) and math.isfinite(v):
            vals.append(float(v))
    return vals


# =============================================================================
# Statistics
# =============================================================================


def describe(vals):
    """n, mean, sample std (ddof=1), sem for a value list."""
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "sem": float("nan")}
    arr = np.asarray(vals, float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else float("nan")
    return {"n": n, "mean": mean, "std": std, "sem": sem}


def ci_halfwidth(std, sem, n):
    """95% CI half-width of the mean (t-based with scipy, else normal approx)."""
    if n < 2:
        return 0.0
    if _stats is not None:
        return float(_stats.t.ppf(0.975, n - 1) * sem)
    return 1.96 * sem


def errorbar_value(stats, kind):
    """Error-bar half-length for the requested kind."""
    if stats["n"] < 2:
        return 0.0
    if kind == "std":
        return stats["std"]
    if kind == "sem":
        return stats["sem"]
    return ci_halfwidth(stats["std"], stats["sem"], stats["n"])  # ci


def welch_ttest(a, b):
    """Welch's two-sided t-test. Returns (t, p); p is None if not computable
    (needs >=2 per group, and non-degenerate variance for a p-value)."""
    if len(a) < 2 or len(b) < 2:
        return None, None
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    # Both groups constant (e.g. param counts, FLOPs): the t-test is degenerate
    # (0/0 or inf); report no test rather than a meaningless / warning-y value.
    if va == 0 and vb == 0:
        return None, None
    if _stats is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t, p = _stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    # Manual Welch t; p left None without scipy (no t CDF available).
    denom = math.sqrt(va / len(a) + vb / len(b))
    return float((a.mean() - b.mean()) / denom), None


def stars(p):
    """Significance marker for a p-value."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# =============================================================================
# Plotting
# =============================================================================


def _pretty(metric):
    return metric.replace("_", " ")


def _draw_bar(ax, metric, conditions, stats_by_cond, pairs, colors,
              errorbar_kind, legend=True, compact=False, annotate_sig=False):
    """Draw one metric's comparison onto `ax`: side-by-side colored bars with
    error bars and value labels. Returns the bar container so a caller can build
    a shared legend. legend=False skips the per-axes legend (used by the grid,
    which adds one figure-level legend); compact shrinks fonts/labels for grid
    cells; annotate_sig draws a bracket + stars for each significant pair (off by
    default - it is O(C^2) brackets with many conditions; the pairwise t-tests
    are always in results.txt regardless)."""
    means = [stats_by_cond[c]["mean"] for c in conditions]
    errs = [errorbar_value(stats_by_cond[c], errorbar_kind) for c in conditions]
    x = np.arange(len(conditions))

    bars = ax.bar(
        x, means, yerr=errs, capsize=5 if compact else 6, color=colors,
        edgecolor="black", linewidth=0.7, width=0.68,
        error_kw={"elinewidth": 1.1, "ecolor": "black"},
    )
    for c, b, m in zip(conditions, bars, means):
        n = stats_by_cond[c]["n"]
        label = f"{m:.3g}" if compact else f"{m:.3g}\n(n={n})"
        ax.annotate(
            label, (b.get_x() + b.get_width() / 2, m),
            textcoords="offset points", xytext=(0, 3),
            ha="center", va="bottom", fontsize=7 if compact else 8, color="#333",
        )

    err_label = {"ci": "95% CI", "std": "std dev", "sem": "SEM"}[errorbar_kind]
    ax.set_title(_pretty(metric), fontsize=11 if compact else 13,
                 fontweight="bold")
    if not compact:
        ax.set_ylabel(f"{_pretty(metric)}  (error bars = {err_label})",
                      fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([])  # identity comes from the legend
    if legend:
        ax.legend(bars, conditions, title="condition", fontsize=9,
                  title_fontsize=9, loc="best", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(x=0.05)
    ax.tick_params(labelsize=8 if compact else 10)

    if annotate_sig:
        _annotate_significance(ax, x, means, errs, conditions, pairs)
    # Accuracy charts: origin at the chance baseline 0.5 (preserves the top,
    # which autoscale / significance brackets may have set).
    if metric in _ACCURACY_METRICS:
        ax.set_ylim(bottom=0.5)
    return bars


def bar_chart(metric, conditions, stats_by_cond, pairs, colors, errorbar_kind,
              path, annotate_sig=False):
    """Standalone one-metric bar chart saved to `path`."""
    fig, ax = plt.subplots(figsize=(max(5.0, 1.7 * len(conditions) + 2.0), 5.2))
    _draw_bar(ax, metric, conditions, stats_by_cond, pairs, colors,
              errorbar_kind, legend=True, compact=False, annotate_sig=annotate_sig)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def grid_pdf(metrics, per_metric, conditions, colors, errorbar_kind, path,
             ncols=4, annotate_sig=False):
    """All metrics as one grid of bar charts (default 4 columns -> 3 rows for
    the 12 metrics) written to a single PDF, with one shared condition legend.

    per_metric maps metric -> (stats_by_cond, pairs).
    """
    n = len(metrics)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows),
                             squeeze=False)
    flat = axes.flatten()
    bars = None
    for ax, metric in zip(flat, metrics):
        stats_by_cond, pairs = per_metric[metric]
        bars = _draw_bar(ax, metric, conditions, stats_by_cond, pairs, colors,
                         errorbar_kind, legend=False, compact=True,
                         annotate_sig=annotate_sig)
    for ax in flat[n:]:  # hide any unused cells
        ax.set_visible(False)

    err_label = {"ci": "95% CI", "std": "std dev", "sem": "SEM"}[errorbar_kind]
    # Shared legend: use as many columns as fit across the figure width so the
    # legend stays SHORT (fewer rows = less vertical space taken from the
    # charts), and never let a column hold more than MAX_PER_COL entries.
    MAX_PER_COL = 10
    legend_rows = 0
    if bars is not None:
        maxlen = max(len(c) for c in conditions)
        fig_w = 4.6 * ncols  # figure width (inches)
        entry_w = 0.45 + 0.062 * maxlen  # marker + label + padding, inches
        fit_cols = max(1, int(fig_w / entry_w))
        # >= ceil(n / MAX_PER_COL) columns caps each column at MAX_PER_COL rows;
        # more columns than that (up to what fits) only makes it shorter.
        min_cols = math.ceil(len(conditions) / MAX_PER_COL)
        legend_ncol = min(len(conditions), max(min_cols, fit_cols))
        legend_rows = math.ceil(len(conditions) / legend_ncol)
        fig.legend(bars, conditions, title="condition", ncol=legend_ncol,
                   loc="lower center", fontsize=9, title_fontsize=10,
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.005))
    sig_note = ";  * p<0.05  ** p<0.01  *** p<0.001" if annotate_sig else ""
    fig.suptitle(
        f"Sweep metrics by condition  (error bars = {err_label}{sig_note})",
        fontsize=14, fontweight="bold",
    )
    # Leave room for the suptitle (top) and the wrapped legend (bottom): a title
    # line plus one line per legend row, as a fraction of the figure height.
    fig_h = 4.0 * nrows
    bottom = min(0.4, (0.35 + 0.26 * legend_rows) / fig_h)
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def _annotate_significance(ax, x, means, errs, conditions, pairs):
    """Draw a bracket + stars for each SIGNIFICANT pair (ns pairs stay in the
    plaintext only, to keep the chart readable)."""
    sig = [(i, j, s) for (i, j, t, p, s) in pairs if s != "ns"]
    if not sig:
        return
    finite = [m + e for m, e in zip(means, errs) if math.isfinite(m)]
    top = max(finite) if finite else 1.0
    bottom = min([m for m in means if math.isfinite(m)] + [0.0])
    span = (top - bottom) or (abs(top) or 1.0)
    step = 0.09 * span
    y = top + 0.13 * span  # clear the value labels sitting just above the bars
    for level, (i, j, s) in enumerate(sig):
        yy = y + level * step
        ax.plot([x[i], x[i], x[j], x[j]],
                [yy, yy + 0.02 * span, yy + 0.02 * span, yy],
                lw=1.2, color="black")
        ax.text((x[i] + x[j]) / 2, yy + 0.02 * span, s,
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(top=y + (len(sig) - 1) * step + 0.12 * span)


# =============================================================================
# Rankings (top-3-per-metric leaderboard)
# =============================================================================


def _direction(metric):
    """('lower'|'neutral'|'higher', human label, sort_descending)."""
    if metric in _LOWER_IS_BETTER:
        return "lower", "lower is better", False
    if metric in _NEUTRAL:
        return "neutral", "ranked high→low", True
    return "higher", "higher is better", True


def _num(v):
    """Compact number formatting: sci notation for very large/small, else 4 sig
    figs."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "NA"
    a = abs(v)
    if a != 0 and (a >= 1e5 or a < 1e-3):
        return f"{v:.3e}"
    return f"{v:.4g}"


def rank_metric(metric, stats_by_cond, conditions, errorbar_kind, top=3):
    """Best `top` conditions for a metric as [(rank, cond, mean, margin)],
    ordered best-first per the metric's direction."""
    rows = []
    for c in conditions:
        s = stats_by_cond[c]
        if s["n"] > 0 and math.isfinite(s["mean"]):
            rows.append((c, s["mean"], errorbar_value(s, errorbar_kind)))
    _, _, desc = _direction(metric)
    rows.sort(key=lambda r: r[1], reverse=desc)
    return [(i + 1, c, m, e) for i, (c, m, e) in enumerate(rows[:top])]


_MEDAL = {1: "#D4AF37", 2: "#9AA0A6", 3: "#CD7F32"}  # gold / silver / bronze


def _draw_ranking_card(ax, metric, ranking, errorbar_kind, name_w=22):
    """One metric card: title, direction, and up to three medal-ranked rows."""
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch(
        (0.015, 0.015), 0.97, 0.97,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        transform=ax.transAxes, fc="#f7f7f7", ec="#cccccc", lw=1.0,
    ))
    _, dir_label, _ = _direction(metric)
    ax.text(0.06, 0.86, _pretty(metric), transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="center")
    ax.text(0.06, 0.70, dir_label, transform=ax.transAxes, fontsize=8,
            style="italic", color="#666", va="center")

    if not ranking:
        ax.text(0.06, 0.4, "no data", transform=ax.transAxes, fontsize=9,
                color="#999", va="center")
        return
    y = 0.52
    for rank, cond, mean, margin in ranking:
        ax.text(0.07, y, f"{rank}", transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=_MEDAL.get(rank, "#333"),
                va="center", ha="center")
        name = cond if len(cond) <= name_w else cond[: name_w - 1] + "…"
        ax.text(0.14, y, f"{name:<{name_w}s}", transform=ax.transAxes,
                fontsize=9, family="monospace", va="center")
        ax.text(0.98, y, f"{_num(mean)} ± {_num(margin)}",
                transform=ax.transAxes, fontsize=9, family="monospace",
                va="center", ha="right")
        y -= 0.17


def rankings_pdf(metrics, per_metric, conditions, errorbar_kind, path, ncols=3):
    """Leaderboard PDF: a card per metric listing the best 3 conditions
    (best-first) with mean +/- error margin."""
    n = len(metrics)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 2.3 * nrows),
                             squeeze=False)
    flat = axes.flatten()
    for ax, metric in zip(flat, metrics):
        stats_by_cond, _ = per_metric[metric]
        ranking = rank_metric(metric, stats_by_cond, conditions, errorbar_kind)
        _draw_ranking_card(ax, metric, ranking, errorbar_kind)
    for ax in flat[n:]:
        ax.set_visible(False)

    err_label = {"ci": "95% CI", "std": "std dev", "sem": "SEM"}[errorbar_kind]
    fig.suptitle(
        f"Top-3 conditions per metric  (value ± {err_label}, "
        f"over {len(conditions)} conditions)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Plaintext report
# =============================================================================


def write_report(path, sweep_dir, conditions, by_cond, metrics, errorbar_kind):
    lines = []
    lines.append(f"# Sweep analysis: {os.path.abspath(sweep_dir)}")
    lines.append(f"# conditions: {len(conditions)}   metrics: {len(metrics)}")
    lines.append(f"# error bars in figures: {errorbar_kind}")
    lines.append(
        "# t-tests: Welch's two-sided (equal_var=False)"
        + ("" if _stats is not None else "  [scipy missing: p-values NA]")
    )
    lines.append("# significance: * p<0.05  ** p<0.01  *** p<0.001  ns = not significant")
    lines.append("")
    lines.append("Seeds per condition:")
    for c in conditions:
        lines.append(f"  {c}: n={len(by_cond[c])}")
    lines.append("")

    for metric in metrics:
        lines.append("=" * 72)
        lines.append(metric)
        lines.append("-" * 72)
        vals_by_cond = {c: values_for(by_cond, c, metric) for c in conditions}
        stats_by_cond = {c: describe(vals_by_cond[c]) for c in conditions}
        for c in conditions:
            s = stats_by_cond[c]
            hw = ci_halfwidth(s["std"], s["sem"], s["n"])
            lo, hi = s["mean"] - hw, s["mean"] + hw
            lines.append(
                f"  {c:<40s} n={s['n']:<2d} mean={s['mean']:.6g}  "
                f"std={s['std']:.4g}  sem={s['sem']:.4g}  "
                f"95%CI=[{lo:.6g}, {hi:.6g}]"
            )
        lines.append("  pairwise Welch t-tests:")
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                a = vals_by_cond[conditions[i]]
                b = vals_by_cond[conditions[j]]
                t, p = welch_ttest(a, b)
                t_s = "NA" if t is None else f"{t:.3g}"
                p_s = "NA" if p is None else f"{p:.4g}"
                lines.append(
                    f"    {conditions[i]} vs {conditions[j]}: "
                    f"t={t_s}  p={p_s}  {stars(p)}"
                )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sweep", help="Sweep directory (contains the condition folders / "
        "manifest.txt), e.g. artifacts/sweeps/replay_vs_noreplay.",
    )
    parser.add_argument(
        "--errorbar", choices=["ci", "std", "sem"], default="ci",
        help="Error-bar type in the bar charts (default ci = 95%% CI of mean). "
        "t-tests are always over the raw seed values.",
    )
    parser.add_argument(
        "--sig-brackets", action="store_true",
        help="Draw significance brackets (horizontal bracket + * / ** / ***) "
        "over each significantly different pair on the bar charts. Off by "
        "default - the count grows combinatorially with conditions. The pairwise "
        "t-tests are always written to results.txt regardless.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default <sweep>/figs).",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep.rstrip("/")
    out_dir = args.out or os.path.join(sweep_dir, "figs")
    os.makedirs(out_dir, exist_ok=True)

    conditions, by_cond = load_sweep(sweep_dir)
    metrics = collect_metric_keys(by_cond)
    if not metrics:
        raise SystemExit("No numeric metrics found to compare.")

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(conditions))]

    print(f"Sweep      : {sweep_dir}")
    print(f"Conditions : {conditions}")
    print(f"Metrics    : {len(metrics)}")
    print(f"Output     : {out_dir}")

    per_metric = {}
    for metric in metrics:
        vals_by_cond = {c: values_for(by_cond, c, metric) for c in conditions}
        stats_by_cond = {c: describe(vals_by_cond[c]) for c in conditions}
        pairs = []
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                t, p = welch_ttest(vals_by_cond[conditions[i]],
                                   vals_by_cond[conditions[j]])
                pairs.append((i, j, t, p, stars(p)))
        per_metric[metric] = (stats_by_cond, pairs)
        png = os.path.join(out_dir, f"bar_{metric}.png")
        bar_chart(metric, conditions, stats_by_cond, pairs, colors,
                  args.errorbar, png, annotate_sig=args.sig_brackets)
        print(f"  wrote {png}")

    grid = os.path.join(out_dir, "grid.pdf")
    grid_pdf(metrics, per_metric, conditions, colors, args.errorbar, grid,
             annotate_sig=args.sig_brackets)
    print(f"  wrote {grid}")

    rankings = os.path.join(out_dir, "rankings.pdf")
    rankings_pdf(metrics, per_metric, conditions, args.errorbar, rankings)
    print(f"  wrote {rankings}")

    report = os.path.join(out_dir, "results.txt")
    write_report(report, sweep_dir, conditions, by_cond, metrics, args.errorbar)
    print(f"  wrote {report}")


if __name__ == "__main__":
    main()
