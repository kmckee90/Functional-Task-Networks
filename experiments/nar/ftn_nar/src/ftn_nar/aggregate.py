"""Aggregate run results: tables, paired bootstrap CIs, retention metrics, mask geometry
(design sections 14, 15, 23.3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import algorithms as A
from .masks import overlap

OOD_CONDS = ["id_val", "id_test", "ood_width_32", "ood_width_64", "ood_width_128",
             "ood_depth_8", "ood_depth_16", "ood_depth_32",
             "ood_both_32", "ood_both_64", "ood_both_128"]


def load_evals(runs_dir: str) -> list[dict]:
    out = []
    for p in sorted(Path(runs_dir).glob("*/eval/eval.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"skip {p}: {e}")
    return out


def induction_frame(evals: list[dict]) -> pd.DataFrame:
    """One row per (method, mode, held_out, seed, identifiable, condition)."""
    rows = []
    for ev in evals:
        for mode_name, agg in ev.get("held_out_eval", {}).items():
            for cond in OOD_CONDS:
                key = f"{cond}/exact_mean"
                if key not in agg:
                    continue
                rows.append({
                    "method": ev["method"], "mode": mode_name,
                    "held_out": ev["held_out"], "seed": ev["seed"],
                    "identifiable": agg.get("identifiable", True),
                    "condition": cond, "exact": agg[key], "exact_std": agg[f"{cond}/exact_std"],
                })
    return pd.DataFrame(rows)


def retention_metrics(ev: dict) -> dict | None:
    """Final average seen accuracy, forgetting, backward transfer from boundary R-rows."""
    recs = [r for r in ev.get("boundary_records", []) if r.get("event") == "boundary"]
    if not recs:
        return None
    final = recs[-1]["r_row"]
    tasks = []
    for r in recs:
        for t in r.get("trained", []):
            if t not in tasks:
                tasks.append(t)
    forget, bwt = {}, {}
    for j, t in enumerate(tasks):
        # rows at blocks where t was already trained
        rows = [r["r_row"][t] for r in recs if t in r.get("trained", []) and t in r["r_row"]]
        if not rows:
            continue
        forget[t] = max(rows) - final.get(t, rows[-1])
        bwt[t] = final.get(t, rows[-1]) - rows[0]
    seen_final = [final[t] for t in tasks if t in final]
    return {
        "final_mean_seen_exact": float(np.mean(seen_final)) if seen_final else None,
        "mean_forgetting": float(np.mean(list(forget.values()))) if forget else None,
        "mean_backward_transfer": float(np.mean(list(bwt.values()))) if bwt else None,
        "forgetting": forget,
        "final_row": final,
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    """95% bootstrap CI for mean(a - b) over paired cells."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return {"mean": None}
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "lo95": float(np.quantile(means, 0.025)),
            "hi95": float(np.quantile(means, 0.975)), "n_cells": len(d)}


def method_comparison(df: pd.DataFrame, mode_a: tuple, mode_b: tuple, cond: str) -> dict:
    """Paired diff between (method, mode) pairs over shared (held_out, seed) cells."""
    da = df[(df.method == mode_a[0]) & (df["mode"] == mode_a[1]) & (df.condition == cond)
            & df.identifiable]
    db = df[(df.method == mode_b[0]) & (df["mode"] == mode_b[1]) & (df.condition == cond)
            & df.identifiable]
    m = pd.merge(da, db, on=["held_out", "seed"], suffixes=("_a", "_b"))
    if m.empty:
        return {"mean": None, "n_cells": 0}
    res = paired_bootstrap(m.exact_a.values, m.exact_b.values)
    res["positive_folds"] = int((m.groupby("held_out")
                                 .apply(lambda g: (g.exact_a - g.exact_b).mean() > 0)).sum())
    res["n_folds"] = int(m.held_out.nunique())
    return res


def mask_geometry(runs_dir: str, method: str = "FTN_FULL") -> dict:
    """H4: prototype-mask overlap grouped by shared/different primitive (design 14.6)."""
    ext_same, ext_diff, red_same, red_diff = [], [], [], []
    for pj in sorted(Path(runs_dir).glob(f"{method}__*/prototypes.json")):
        protos = json.loads(pj.read_text())
        names = list(protos)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if "medoid_ext" not in protos[a] or "medoid_ext" not in protos[b]:
                    continue
                ia, ib = A.alg_id(a), A.alg_id(b)
                oe = overlap(protos[a]["medoid_ext"], protos[b]["medoid_ext"],
                             int(np.sum(protos[a]["medoid_ext"])))
                orr = overlap(protos[a]["medoid_red"], protos[b]["medoid_red"],
                              int(np.sum(protos[a]["medoid_red"])))
                (ext_same if A.EXTENDER_OF[ia] == A.EXTENDER_OF[ib] else ext_diff).append(oe)
                (red_same if A.REDUCER_OF[ia] == A.REDUCER_OF[ib] else red_diff).append(orr)

    def stat(x):
        return {"mean": float(np.mean(x)), "n": len(x)} if x else {"mean": None, "n": 0}

    out = {"ext_overlap_same_extender": stat(ext_same),
           "ext_overlap_diff_extender": stat(ext_diff),
           "red_overlap_same_reducer": stat(red_same),
           "red_overlap_diff_reducer": stat(red_diff)}
    if ext_same and ext_diff:
        out["ext_gap_bootstrap"] = _unpaired_bootstrap(ext_same, ext_diff)
    if red_same and red_diff:
        out["red_gap_bootstrap"] = _unpaired_bootstrap(red_same, red_diff)
    return out


def _unpaired_bootstrap(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    d = da - db
    return {"mean": float(a.mean() - b.mean()), "lo95": float(np.quantile(d, 0.025)),
            "hi95": float(np.quantile(d, 0.975))}


def build_report(runs_dir: str, out_dir: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evals = load_evals(runs_dir)
    df = induction_frame(evals)
    report: dict = {"n_runs": len(evals)}

    if not df.empty:
        df.to_csv(out_dir / "induction_cells.csv", index=False)
        tbl = (df[df.identifiable]
               .groupby(["method", "mode", "condition"])["exact"]
               .agg(["mean", "std", "count"]).reset_index())
        tbl.to_csv(out_dir / "induction_table.csv", index=False)
        report["induction_table"] = tbl.to_dict("records")

        comparisons = {}
        for cond in ("id_test", "ood_width_64", "ood_width_128", "ood_depth_16"):
            comparisons[cond] = {
                "FTN_vs_RETRIEVAL": method_comparison(
                    df, ("FTN_FULL", "ftn"), ("FTN_FULL", "retrieval"), cond),
                "FTN_vs_DENSE": method_comparison(
                    df, ("FTN_FULL", "ftn"), ("DENSE_ACTIVE_MATCHED", "dense"), cond),
                "FTN_vs_SOFT": method_comparison(
                    df, ("FTN_FULL", "ftn"), ("SOFT_MIXTURE", "soft"), cond),
                "FTN_vs_RANDOM_TRAIN": method_comparison(
                    df, ("FTN_FULL", "ftn"), ("FTN_RANDOM_MASK_TRAIN", "ftn"), cond),
            }
        report["comparisons"] = comparisons

    retention = {}
    for ev in evals:
        r = retention_metrics(ev)
        if r:
            retention.setdefault(ev["method"], []).append(
                {"held_out": ev["held_out"], "seed": ev["seed"], **{
                    k: r[k] for k in ("final_mean_seen_exact", "mean_forgetting",
                                      "mean_backward_transfer")}})
    for m, rows in retention.items():
        vals = [r["final_mean_seen_exact"] for r in rows if r["final_mean_seen_exact"] is not None]
        fg = [r["mean_forgetting"] for r in rows if r["mean_forgetting"] is not None]
        report.setdefault("retention", {})[m] = {
            "final_mean_seen_exact": float(np.mean(vals)) if vals else None,
            "mean_forgetting": float(np.mean(fg)) if fg else None,
            "cells": rows}

    report["mask_geometry"] = mask_geometry(runs_dir)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report
