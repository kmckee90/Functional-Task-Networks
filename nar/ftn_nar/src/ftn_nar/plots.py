"""Figures for the final report (design 23.3). Matplotlib, Agg backend."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import algorithms as A
from .aggregate import OOD_CONDS, induction_frame, load_evals, retention_metrics

WIDTH_CONDS = ["id_test", "ood_width_32", "ood_width_64", "ood_width_128"]
DEPTH_CONDS = ["id_test", "ood_depth_8", "ood_depth_16", "ood_depth_32"]


def _mode_curve(df, method, mode, conds):
    xs, ys, es = [], [], []
    for c in conds:
        sub = df[(df.method == method) & (df["mode"] == mode) & (df.condition == c)
                 & df.identifiable]
        if not sub.empty:
            xs.append(c.replace("ood_", "").replace("id_test", "ID"))
            ys.append(sub.exact.mean())
            es.append(sub.exact.std())
    return xs, ys, es


def plot_exact_vs_size(df, out_path, conds, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    curves = [("FTN_FULL", "ftn", "FTN configurer"),
              ("FTN_FULL", "retrieval", "FTN retrieval"),
              ("DENSE_ACTIVE_MATCHED", "dense", "Dense+FiLM"),
              ("SOFT_MIXTURE", "soft", "Soft mixture"),
              ("ORACLE_COMPOSITION", "oracle", "Oracle")]
    for method, mode, label in curves:
        xs, ys, es = _mode_curve(df, method, mode, conds)
        if xs:
            ax.errorbar(range(len(xs)), ys, yerr=es, marker="o", capsize=3, label=label)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs)
    ax.set_ylabel("exact graph value accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_accuracy_matrix(ev, out_path):
    recs = [r for r in ev.get("boundary_records", []) if r.get("event") == "boundary"]
    if not recs:
        return
    tasks = []
    for r in recs:
        for t in r.get("trained", []):
            if t not in tasks:
                tasks.append(t)
    all_tasks = sorted({t for r in recs for t in r["r_row"] if "/" not in t})
    M = np.full((len(recs), len(all_tasks)), np.nan)
    for i, r in enumerate(recs):
        for j, t in enumerate(all_tasks):
            if t in r["r_row"]:
                M[i, j] = r["r_row"][t]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(all_tasks)))
    ax.set_xticklabels(all_tasks, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(recs)))
    ax.set_yticklabels([f"after block {i}" for i in range(len(recs))], fontsize=7)
    ax.set_title(f"Seen-task exact accuracy matrix ({ev['method']}, held-out {ev['held_out']})")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_mask_overlap(runs_dir, out_path, method="FTN_FULL"):
    protos = {}
    for pj in sorted(Path(runs_dir).glob(f"{method}__*/prototypes.json")):
        for name, e in json.loads(pj.read_text()).items():
            if "medoid_ext" in e:
                protos.setdefault(name, e)
    names = [n for n in A.ALGORITHM_NAMES if n in protos]
    if len(names) < 2:
        return
    from .masks import overlap
    for bank, key in (("extender", "medoid_ext"), ("reducer", "medoid_red")):
        M = np.zeros((len(names), len(names)))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                k = int(np.sum(protos[a][key]))
                M[i, j] = overlap(protos[a][key], protos[b][key], k)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"{bank} mask overlap")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(str(out_path).replace(".png", f"_{bank}.png"), dpi=130)
        plt.close(fig)


def build_figures(runs_dir: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    evals = load_evals(runs_dir)
    df = induction_frame(evals)
    if not df.empty:
        plot_exact_vs_size(df, out / "exact_vs_width.png", WIDTH_CONDS,
                           "Held-out exact accuracy vs graph width")
        plot_exact_vs_size(df, out / "exact_vs_depth.png", DEPTH_CONDS,
                           "Held-out exact accuracy vs depth")
    for ev in evals:
        rd = out / f"accmatrix_{ev['method']}_{ev['held_out']}_s{ev['seed']}.png"
        plot_accuracy_matrix(ev, rd)
    plot_mask_overlap(runs_dir, out / "mask_overlap.png")
    print(f"figures -> {out}")
