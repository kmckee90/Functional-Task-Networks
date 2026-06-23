"""
generate_revision_figs.py — figures for the revision results.

Produces:
  figures/revision/methods_acc_fm.pdf   # grouped bar chart: ACC, FM per method per benchmark
  figures/revision/headstart_per_bench.pdf  # head-start per benchmark
  figures/revision/supportsize.pdf      # support-set size sweep curves on permuted/rotated

Run after experiments/run_revision_merge.py + experiments/generate_revision_tables.py.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASK_DISPLAY = {
    "synthetic": "Synthetic",
    "mnist_shared": "MNIST (shared)",
    "permuted": "PermutedMNIST",
    "rotated": "RotatedMNIST",
}

METHOD_ORDER = [
    "NoMask", "NoMask+ER",
    "FixedMask", "FixedMask+Router",
    "FTN", "FTN+Router", "FTN_noreplay",
]
METHOD_DISPLAY = {
    "NoMask": "NoMask", "NoMask+ER": "ER",
    "FixedMask": "FixedM", "FixedMask+Router": "FixedM+R",
    "FTN": "FTN", "FTN+Router": "FTN+R", "FTN_noreplay": "FTN-NR",
}

METHOD_COLORS = {
    "NoMask": "#bdbdbd",
    "NoMask+ER": "#9e9e9e",
    "FixedMask": "#80b1d3",
    "FixedMask+Router": "#377eb8",
    "FTN": "#e41a1c",
    "FTN+Router": "#fd8d3c",
    "FTN_noreplay": "#984ea3",
}


def load_aggregated(suffix: str = "") -> dict:
    out: dict = {}
    for task in TASK_DISPLAY.keys():
        path = f"results/revision_{task}{suffix}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        agg = data.get("aggregated", {})
        for method, by_n in agg.items():
            entry = by_n.get("256") or next(iter(by_n.values()))
            metrics = {}
            for fld in ("ACC", "FM", "BWT"):
                key = f"summary_reconfig.{fld}"
                if key in entry:
                    metrics[fld] = (entry[key]["mean"], entry[key]["std"])
            if "head_start_mean" in entry:
                metrics["HS"] = (entry["head_start_mean"]["mean"],
                                 entry["head_start_mean"]["std"])
            out.setdefault(task, {})[method] = metrics
    return out


def methods_grouped_bar(results: dict, metric: str, ylabel: str, out_path: str,
                         lower_is_better: bool = False):
    tasks = [t for t in TASK_DISPLAY.keys() if t in results]
    methods = [m for m in METHOD_ORDER
               if any(m in results.get(t, {}) for t in tasks)]
    if not tasks or not methods:
        return
    x = np.arange(len(tasks))
    width = 0.85 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for i, method in enumerate(methods):
        means = []
        stds = []
        for t in tasks:
            m = results.get(t, {}).get(method, {}).get(metric)
            if m is None:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(m[0])
                stds.append(m[1])
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2,
               label=METHOD_DISPLAY[method], color=METHOD_COLORS[method])
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY[t] for t in tasks], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="upper left" if not lower_is_better else "upper right",
              fontsize=6, ncol=4, frameon=False, bbox_to_anchor=(0, 1.18))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    results = load_aggregated()
    if not results:
        print("No revision results found; skipping figures.")
        return
    methods_grouped_bar(
        results, "ACC", "Final ACC (reconfig protocol)",
        "figures/revision/methods_acc.pdf",
    )
    methods_grouped_bar(
        results, "FM", "Forgetting (FM, lower better)",
        "figures/revision/methods_fm.pdf",
        lower_is_better=True,
    )
    methods_grouped_bar(
        results, "HS", "Head-start loss (lower better)",
        "figures/revision/methods_hs.pdf",
        lower_is_better=True,
    )

    # Support-size sweep: aggregate across tasks per n.
    support_data: dict = {}
    for task in TASK_DISPLAY.keys():
        path = f"results/revision_{task}_support_size.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        agg = data.get("aggregated", {}).get("FTN", {})
        for n_str, entry in agg.items():
            metrics = {}
            for fld in ("ACC", "FM"):
                key = f"summary_reconfig.{fld}"
                if key in entry:
                    metrics[fld] = (entry[key]["mean"], entry[key]["std"])
            if "head_start_mean" in entry:
                metrics["HS"] = (entry["head_start_mean"]["mean"],
                                 entry["head_start_mean"]["std"])
            support_data.setdefault(task, {})[int(n_str)] = metrics
    if support_data:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
        for task, by_n in support_data.items():
            ns = sorted(by_n.keys())
            acc_m = [by_n[n]["ACC"][0] for n in ns if "ACC" in by_n[n]]
            acc_s = [by_n[n]["ACC"][1] for n in ns if "ACC" in by_n[n]]
            fm_m  = [by_n[n]["FM"][0] for n in ns if "FM" in by_n[n]]
            fm_s  = [by_n[n]["FM"][1] for n in ns if "FM" in by_n[n]]
            axes[0].errorbar(ns, acc_m, yerr=acc_s, marker="o", label=TASK_DISPLAY[task])
            axes[1].errorbar(ns, fm_m, yerr=fm_s, marker="o", label=TASK_DISPLAY[task])
        for ax, ylabel in zip(axes, ("ACC", "FM")):
            ax.set_xscale("log")
            ax.set_xlabel("Support size $n$")
            ax.set_ylabel(ylabel)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        out_path = "figures/revision/supportsize.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
