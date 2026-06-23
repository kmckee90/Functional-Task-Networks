"""
plot.py — Visualise LoRA-Mux experiment results.

Reads `results/lora_styles.json` and produces:
    figures/perf_matrix_<method>_<protocol>.pdf  -- per-task forgetting heatmap
    figures/forgetting_curves.pdf                -- per-task loss-over-tasks lines
    figures/summary_bar.pdf                      -- avg_loss / FM bars across methods

Run from project_c_4/:
    uv run python experiments/lora/plot.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "lora_styles.json")
FIGS = os.path.join(HERE, "figures")


def _pad(matrix):
    n = len(matrix)
    out = np.full((n, n), np.nan)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            out[i, j] = v
    return out


def perf_matrix(matrix, title, path, vmin=None, vmax=None):
    M = _pad(matrix)
    fig, ax = plt.subplots(figsize=(0.55 * len(M) + 1.2, 0.55 * len(M) + 1.0))
    im = ax.imshow(M, cmap="viridis_r", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Eval task j")
    ax.set_ylabel("After training task i")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(M)))
    ax.set_yticks(range(len(M)))
    for i in range(len(M)):
        for j in range(len(M)):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > (vmax or 3) * 0.55 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="CE")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def forgetting_curves(results, path, protocol="reconfig"):
    methods = list(results["methods"].keys())
    n_tasks = len(results["methods"][methods[0]]["perf_matrix_" + protocol])
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.0),
                             sharey=True)
    if len(methods) == 1:
        axes = [axes]
    style_names = results["methods"][methods[0]].get("style_names", [str(i) for i in range(n_tasks)])
    cmap = plt.cm.tab10
    for ax, method in zip(axes, methods):
        M = _pad(results["methods"][method]["perf_matrix_" + protocol])
        for j in range(n_tasks):
            col = M[:, j]
            xs = np.arange(j, n_tasks)
            ys = col[j:]
            ax.plot(xs, ys, marker="o", color=cmap(j % 10), label=style_names[j])
        ax.set_xlabel("After training task")
        ax.set_title(f"{method} — {protocol}", fontsize=10)
        ax.set_xticks(range(n_tasks))
    axes[0].set_ylabel("Per-token CE")
    axes[0].legend(loc="upper left", fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summary_bars(results, path):
    methods = list(results["methods"].keys())
    metrics = ("avg_loss", "FM")
    protos = ("stored", "reconfig")
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0, 3.0))
    width = 0.4
    x = np.arange(len(methods))
    for ax, key in zip(axes, metrics):
        vals_s = [results["methods"][m]["metrics_stored"][key] for m in methods]
        vals_r = [results["methods"][m]["metrics_reconfig"][key] for m in methods]
        ax.bar(x - width / 2, vals_s, width, label="stored", color="#377eb8")
        ax.bar(x + width / 2, vals_r, width, label="reconfig", color="#e41a1c")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_title(key)
        ax.axhline(0.0, color="k", lw=0.5)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.exists(RESULTS):
        print(f"No results at {RESULTS}; run experiments/lora/run.py first.")
        sys.exit(1)
    with open(RESULTS) as f:
        results = json.load(f)
    os.makedirs(FIGS, exist_ok=True)

    # Common color scale across all matrices for fair comparison.
    all_vals = []
    for d in results["methods"].values():
        for m in (d["perf_matrix_stored"], d["perf_matrix_reconfig"]):
            for row in m:
                all_vals.extend(row)
    vmin, vmax = min(all_vals), max(all_vals)

    for method, d in results["methods"].items():
        for proto in ("stored", "reconfig"):
            out = os.path.join(FIGS, f"perf_matrix_{method}_{proto}.pdf")
            perf_matrix(d[f"perf_matrix_{proto}"], f"{method} — {proto}",
                        out, vmin=vmin, vmax=vmax)
            print(f"  wrote {out}")

    fc = os.path.join(FIGS, "forgetting_curves.pdf")
    forgetting_curves(results, fc, protocol="reconfig")
    print(f"  wrote {fc}")

    sb = os.path.join(FIGS, "summary_bar.pdf")
    summary_bars(results, sb)
    print(f"  wrote {sb}")


if __name__ == "__main__":
    main()
