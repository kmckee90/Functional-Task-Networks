"""Figures for the temporal (recurrent FTN) experiment."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ftn_core.viz import (BASELINE, CAT, GRID, INK, INK2, MUTED, SEQ_BLUE,
                          SURFACE, apply_style, save)

apply_style()

CHANCE_POS = 1 / 11


def _rows_by_level(rows, key):
    """{level: [values]} for rows that have `key`."""
    out = {}
    for r in rows:
        if key in r:
            out.setdefault(r["level"], []).append(r[key])
    return dict(sorted(out.items()))


def fig_task(out, results, tasks_obj, cfg):
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)

    # non-commutativity worked example
    ax = fig.add_subplot(gs[0, 0:2])
    ax.axis("off")
    ax.text(0, 1.0, "The task family: words over two generators of Aff(Z₁₁)",
            fontsize=12, fontweight="bold", color=INK, va="top")
    ax.text(0, 0.82,
            "T1:  x → x + 1 (mod 11)          M2:  x → 2x (mod 11)\n\n"
            "Order matters (non-commutative):\n"
            "    x=3:   T1 then M2:  3 → 4 → 8        M2 then T1:  3 → 6 → 7\n\n"
            "A task = one affine map g(x) = c·x + a, applied elementwise to a\n"
            "sequence of L=4 tokens. Every g has a SHORTEST WORD over {T1, M2}\n"
            "— its program length ℓ(g). The model sees only (x, g(x)) pairs:\n"
            "it must infer the operator sequence AND its length from a support\n"
            "set, then execute it step by step.",
            fontsize=10.5, color=INK2, va="top", linespacing=1.5)

    # level size histogram with split shading
    ax = fig.add_subplot(gs[0, 2])
    sizes = results["level_sizes"]
    lv = sorted(int(k) for k in sizes)
    vals = [sizes[l] if l in sizes else sizes[str(l)] for l in lv]
    colors = ["#cde2fb" if l <= cfg.train_max_len else "#2a78d6" for l in lv]
    ax.bar(lv, vals, 0.7, color=colors, edgecolor=BASELINE, linewidth=0.5)
    for l, v in zip(lv, vals):
        ax.text(l, v + 0.5, str(v), ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("shortest program length ℓ")
    ax.set_ylabel("# group elements")
    ax.set_title(f"the 110 affine maps by program length\n"
                 f"light = trained (ℓ≤{cfg.train_max_len}), dark = HELD OUT",
                 fontsize=9.5)
    ax.set_xticks(lv)

    # recurrent FTN schematic (text)
    ax = fig.add_subplot(gs[1, 0:3])
    ax.axis("off")
    ax.text(0, 1.0, "Recurrent FTN: one weight-tied stage, the mask sequence is the program",
            fontsize=12, fontweight="bold", color=INK, va="top")
    ax.text(0, 0.78,
            "s₀ = x   →   s₁ = stage(s₀; m₁)   →   s₂ = stage(s₁; m₂)   →   …   →   s_T = ŷ        "
            "(argmax re-encoded between steps at execution)\n\n"
            "One bank of H=2048 deep neurons; a binary top-k mask per STEP. Time replaces depth: an atom learned once is reusable at every\n"
            "position of every program — loops are literally repeated masks. Program inference = trace search with θ frozen, two routes:\n\n"
            "   A. LibraryBFS (hard states): consolidate one consensus mask per trained operator (+identity); breadth-first search over atom\n"
            "       sequences on a support batch, memoized on produced states; halting = shortest trace that matches the support.\n"
            "   B. UnrolledAnnealSTE (soft states): the abelian configurer unrolled through time — per-step top-k logits optimized jointly by\n"
            "       gradient through a soft rollout; halting = external sweep over T, best support loss wins.",
            fontsize=10.5, color=INK2, va="top", linespacing=1.5)
    save(fig, out, "fig1_task.png")


def fig_length_ood(out, results, cfg):
    """Headline: accuracy vs required program length, beyond the training horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    series = []
    if "v1_noid" in results:
        series.append(("V1 · BFS trace, id-free vocabulary", results["v1_noid"],
                       "library", CAT[0]))
    if "v1" in results:
        series.append(("V1 (library-EM) · BFS trace" +
                       (" (id in vocab)" if "v1_noid" in results else ""),
                       results["v1"], "library",
                       CAT[5] if "v1_noid" in results else CAT[0]))
        series.append(("V1 · gradient trace", results["v1"], "gradient", CAT[4]))
        series.append(("V1 · greedy-grow trace", results["v1"], "greedy", CAT[1]))
    if "v2" in results:
        series.append(("V2 (BPTT) · BFS trace", results["v2"], "library", CAT[6]))
        series.append(("V2 · gradient trace", results["v2"], "gradient", CAT[3]))
        series.append(("V2 · greedy-grow trace", results["v2"], "greedy", CAT[7]))

    for ax, metric, ttl in ((axes[0], "pos", "per-position accuracy"),
                            (axes[1], "exact", "exact-match")):
        for label, res, method, color in series:
            rows = res["trained"] + res["held"]
            by_lvl = {}
            for r in rows:
                if method in r:
                    by_lvl.setdefault(r["level"], []).append(r[method][metric])
            lv = sorted(by_lvl)
            ax.plot(lv, [np.mean(by_lvl[l]) for l in lv], marker="o",
                    markersize=4.5, linewidth=1.9, color=color, label=label)
        # random bank + joint baseline (held levels only)
        if "v1" in results and results["v1"].get("random_bank"):
            rows = results["v1"]["random_bank"]
            by_lvl = {}
            for r in rows:
                if "library" in r:
                    by_lvl.setdefault(r["level"], []).append(r["library"][metric])
            lv = sorted(by_lvl)
            ax.plot(lv, [np.mean(by_lvl[l]) for l in lv], marker="s",
                    markersize=4, linewidth=1.4, linestyle="--", color=CAT[2],
                    label="random bank · BFS trace")
        jr = {}
        for r in results["joint_train"] + results["joint_held"]:
            jr.setdefault(r["level"], []).append(r[metric if metric != "pos" else "pos"])
        lv = sorted(jr)
        ax.plot(lv, [np.mean(jr[l]) for l in lv], marker="^", markersize=4,
                linewidth=1.4, linestyle="--", color=INK2,
                label="joint task-conditioned MLP")

        chance = CHANCE_POS if metric == "pos" else CHANCE_POS ** cfg.L
        ax.axhline(chance, color=BASELINE, linewidth=1, linestyle=":")
        ax.axvspan(-0.5, cfg.train_max_len + 0.5, color="#f0f4fa", zorder=0)
        ax.text(cfg.train_max_len / 2, 1.06, "trained regime", fontsize=8,
                color=MUTED, ha="center")
        ax.text((cfg.train_max_len + 1 + cfg.eval_max_len) / 2, 1.06,
                "held out: longer programs than ever trained", fontsize=8,
                color=INK2, ha="center")
        ax.set_xlabel("required program length ℓ (shortest word)")
        ax.set_ylabel(ttl)
        ax.set_ylim(-0.02, 1.12)
        ax.set_xlim(-0.4, cfg.eval_max_len + 0.4)
        ax.set_xticks(range(cfg.eval_max_len + 1))
        if metric == "pos":
            ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, out, "fig2_length_ood.png")


def fig_halting(out, results, cfg):
    n_panels = len([v for v in ("v1", "v2") if v in results]) * 3
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.9),
                             squeeze=False)
    axes = axes[0]
    i = 0
    for v in ("v1", "v2"):
        if v not in results:
            continue
        for method in ("library", "gradient", "greedy"):
            ax = axes[i]
            i += 1
            rows = results[v]["trained"] + results[v]["held"]
            H = np.zeros((cfg.T_max + 1, cfg.eval_max_len + 1))
            for r in rows:
                if method in r:
                    H[min(r[method]["T"], cfg.T_max), r["level"]] += 1
            ax.imshow(H, cmap=SEQ_BLUE, origin="lower", aspect="auto",
                      vmin=0, vmax=max(H.max(), 1))
            ax.plot([0, cfg.eval_max_len], [0, cfg.eval_max_len], color=MUTED,
                    linewidth=1, linestyle=":")
            for t in range(H.shape[0]):
                for l in range(H.shape[1]):
                    if H[t, l] > 0:
                        ax.text(l, t, int(H[t, l]), ha="center", va="center",
                                fontsize=7,
                                color="white" if H[t, l] > H.max() * 0.5 else INK2)
            ax.set_xlabel("true shortest length ℓ")
            ax.set_ylabel("inferred trace length T")
            ax.set_title(f"{v} · {method}\n(halting: dotted = T = ℓ)", fontsize=9)
    save(fig, out, "fig3_halting.png")


def fig_programs(out, results, tasks_obj, cfg, n_show=16):
    """Rendered program table for held-out tasks + step faithfulness."""
    if "v1" not in results:
        return
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.axis("off")
    rows = [r for r in results["v1"]["held"] if "library" in r]
    rows = sorted(rows, key=lambda r: (r["level"], str(r["task"])))
    step = max(1, len(rows) // n_show)
    shown = rows[::step][:n_show]
    header = f"{'task (c,a)':>12} {'ℓ':>2}  {'BFS shortest word':<24} {'inferred program':<26} {'pos':>5} {'exact':>6}"
    lines = [header, "─" * len(header)]
    for r in shown:
        g = r["task"]
        sw = "·".join(r["shortest_word"]) or "(id)"
        tr = "·".join(r["library"]["trace"]) or "(id)"
        lines.append(f"{str(tuple(g) if not isinstance(g, str) else g):>12} "
                     f"{r['level']:>2}  {sw:<24} {tr:<26} "
                     f"{r['library']['pos']:>5.2f} {r['library']['exact']:>6.2f}")
    ax.text(0, 1.0, "\n".join(lines), fontsize=8.6, family="monospace",
            color=INK2, va="top")
    ax.set_title("held-out tasks: inferred programs (V1, BFS trace)",
                 fontsize=10, color=INK)

    ax = fig.add_subplot(gs[0, 2])
    faith = results["v1"].get("faithfulness", [])
    if faith:
        vals = [f["step_faithfulness"] for f in faith]
        max_T = max(len(v) for v in vals)
        means = [np.mean([v[t] for v in vals if len(v) > t])
                 for t in range(max_T)]
        ax.bar(range(1, max_T + 1), means, 0.65, color=CAT[0])
        for t, m in enumerate(means):
            ax.text(t + 1, m + 0.015, f"{m:.2f}", ha="center", fontsize=8,
                    color=INK2)
        ax.axhline(CHANCE_POS, color=BASELINE, linewidth=1, linestyle=":")
        ax.set_xlabel("step index within inferred program")
        ax.set_ylabel("step faithfulness (matches atom ground truth)")
        ax.set_ylim(0, 1.08)
        ax.set_title("is each step really running its\nnamed operator?", fontsize=9.5)
    save(fig, out, "fig4_programs.png")


def fig_training(out, results, cfg):
    variants = [v for v in ("v1", "v2") if v in results]
    fig, axes = plt.subplots(1, len(variants), figsize=(6.5 * len(variants), 4),
                             squeeze=False)
    for ax, v in zip(axes[0], variants):
        hist = results[v]["history"]
        blocks = sorted({h["block"] for h in hist})
        for b in blocks:
            hs = [h for h in hist if h["block"] == b]
            xs = [b * cfg.steps_per_task + h["step"] for h in hs]
            ax.plot(xs, [h["pos"] for h in hs],
                    color=SEQ_BLUE(0.25 + 0.7 * b / max(len(blocks) - 1, 1)),
                    linewidth=1.2)
        ax.axhline(CHANCE_POS, color=BASELINE, linewidth=1, linestyle=":")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("sequential training step")
        ax.set_ylabel("per-position accuracy (test inputs)")
        ax.set_title(f"{v}: block-sequential stream "
                     f"({len(blocks)} visits, {cfg.cycles} cycles)", fontsize=10)
    save(fig, out, "fig5_training.png")


def fig_methods(out, results, cfg):
    """Mean over held-out (OOD) tasks, all method/model combinations."""
    entries = []
    colors = {("v1_noid", "library"): CAT[0],
              ("v1", "library"): CAT[5], ("v1", "gradient"): CAT[4],
              ("v1", "greedy"): CAT[1], ("v2", "library"): CAT[6],
              ("v2", "gradient"): CAT[3], ("v2", "greedy"): CAT[7]}
    for v in ("v1_noid", "v1", "v2"):
        if v not in results:
            continue
        for method in ("library", "gradient", "greedy"):
            vals_p = [r[method]["pos"] for r in results[v]["held"] if method in r]
            vals_e = [r[method]["exact"] for r in results[v]["held"] if method in r]
            if vals_p:
                disp = {"v1_noid": "v1 id-free"}.get(v, v)
                entries.append((f"{disp}\n{method}", np.mean(vals_p),
                                np.mean(vals_e), colors[(v, method)]))
    rnd = [r["library"]["pos"] for r in results.get("v1", {}).get("random_bank", [])
           if "library" in r]
    if rnd:
        entries.append(("random\nbank", np.mean(rnd), 0.0, CAT[2]))
    jp = [r["pos"] for r in results["joint_held"]]
    je = [r["exact"] for r in results["joint_held"]]
    entries.append(("joint\ncond. MLP", np.mean(jp), np.mean(je), CAT[5]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, idx, ttl in ((axes[0], 1, "per-position accuracy"),
                         (axes[1], 2, "exact-match")):
        xs = np.arange(len(entries))
        ax.bar(xs, [e[idx] for e in entries], 0.6, color=[e[3] for e in entries])
        for x, e in zip(xs, entries):
            ax.text(x, e[idx] + 0.015, f"{e[idx]:.2f}", ha="center", fontsize=8.5,
                    color=INK2)
        chance = CHANCE_POS if idx == 1 else CHANCE_POS ** cfg.L
        ax.axhline(chance, color=BASELINE, linewidth=1, linestyle=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([e[0] for e in entries], fontsize=8.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"held-out tasks (ℓ>{cfg.train_max_len}): {ttl}", fontsize=10)
    save(fig, out, "fig6_methods.png")


def make_all_figures(out, results, tasks_obj, cfg):
    fig_task(out, results, tasks_obj, cfg)
    fig_length_ood(out, results, cfg)
    fig_halting(out, results, cfg)
    fig_programs(out, results, tasks_obj, cfg)
    fig_training(out, results, cfg)
    fig_methods(out, results, cfg)
