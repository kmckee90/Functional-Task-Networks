"""All figures for the FTN abelian-grid experiment.

Static matplotlib PNGs (light mode). Color jobs follow the dataviz method:
sequential single-hue blue ramp for magnitudes (accuracy, overlap), a fixed
categorical order for method identity, reserved ink/grid tokens for chrome.
"""

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle

# ---- palette (reference instance, light mode) ----
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BASELINE = "#c3c2b7"
CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834",
       "#4a3aa7", "#e34948"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.size": 10, "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, out, name):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _seq_matrix(ax, seq, p, title, highlight="#2a78d6"):
    L = len(seq)
    M = np.zeros((L, p))
    for i, v in enumerate(seq):
        M[i, v] = 1
    ax.imshow(M, cmap=LinearSegmentedColormap.from_list("b", [SURFACE, highlight]),
              aspect="equal", vmin=0, vmax=1)
    for i, v in enumerate(seq):
        ax.text(v, i, str(v), ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
    ax.set_xticks(range(p))
    ax.set_yticks(range(L))
    ax.set_xticklabels(range(p), fontsize=7)
    ax.set_yticklabels([f"pos {i}" for i in range(L)], fontsize=7)
    ax.set_title(title, fontsize=9, color=INK)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRID)


def fig_task_explainer(out, grid, train_cells, held_cells, cfg):
    """What the task is, why it's hard, and the grid split."""
    p, L = cfg.p, cfg.L
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.45, wspace=0.35)

    # Row 1: worked example x -> T_a -> R_r -> y for (a, r) = (2, 1)
    seq = [3, 7, 0, 9]
    a, r = 2, 1
    shifted = [(v + a) % p for v in seq]
    rotated = shifted[-r:] + shifted[:-r]
    ax0 = fig.add_subplot(gs[0, 0])
    _seq_matrix(ax0, seq, p, "input  x = [3,7,0,9]", CAT[0])
    ax1 = fig.add_subplot(gs[0, 1])
    _seq_matrix(ax1, shifted, p, f"value shift $T_{{{a}}}$: cols rotate by {a}", CAT[4])
    ax2 = fig.add_subplot(gs[0, 2])
    _seq_matrix(ax2, rotated, p, f"rotation $R_{{{r}}}$: rows rotate by {r}", CAT[1])
    for x0 in (0.245, 0.475):
        fig.patches.append(FancyArrowPatch((x0, 0.76), (x0 + 0.045, 0.76),
                           transform=fig.transFigure, arrowstyle="-|>",
                           mutation_scale=16, color=INK2))

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    ax3.text(0, 0.95, "One task cell (a, r):", fontsize=10, color=INK,
             fontweight="bold", va="top")
    ax3.text(0, 0.80,
             "$y_i = x_{(i-r)\\ \\mathrm{mod}\\ L} + a \\ (\\mathrm{mod}\\ p)$\n\n"
             "• a fixed permutation of the\n   $L{\\times}p$ input bits\n"
             "• operators form the abelian\n   group $Z_p \\times Z_L$ — all cells\n"
             "   commute\n"
             "• task identity is NOT in the\n   input; it lives only in the mask",
             fontsize=9, color=INK2, va="top")

    # Row 2: the task grid split + why it's hard
    axg = fig.add_subplot(gs[1, 0:2])
    for aa in range(p):
        for rr in range(L):
            cell = (aa, rr)
            if cell in train_cells:
                fc, ec, lbl = "#cde2fb", "#2a78d6", "T"
            elif cell == (0, 0):
                fc, ec, lbl = "#fdf1d7", "#eda100", "id"
            else:
                fc, ec, lbl = SURFACE, BASELINE, ""
            axg.add_patch(Rectangle((rr, aa), 0.92, 0.92, facecolor=fc,
                                    edgecolor=ec, linewidth=1.2))
            if lbl:
                axg.text(rr + 0.46, aa + 0.46, lbl, ha="center", va="center",
                         fontsize=8, color=INK2)
    axg.set_xlim(-0.3, L + 1.5)
    axg.set_ylim(p, -0.8)
    axg.set_xticks([x + 0.46 for x in range(L)])
    axg.set_xticklabels([f"r={x}" for x in range(L)])
    axg.set_yticks([y + 0.46 for y in range(p)])
    axg.set_yticklabels([f"a={y}" for y in range(p)])
    axg.set_title(f"task grid: {len(train_cells)} trained (blue), "
                  f"{len(held_cells)} held out (empty; identity marked)",
                  fontsize=10, color=INK)
    axg.spines[:].set_visible(False)
    axg.tick_params(length=0)

    axh = fig.add_subplot(gs[1, 2:4])
    axh.axis("off")
    axh.text(0, 0.98, "Why this is hard for FTN", fontsize=10, color=INK,
             fontweight="bold", va="top")
    axh.text(0, 0.86,
             "Same input, different outputs. The network sees only x; the mask\n"
             "is the entire program. Cells are trained block-sequentially —\n"
             "weights shift under every later task (interference), and no joint\n"
             "gradient ever ties the grid together.\n\n"
             "Held-out cells get NO weight updates, ever. A solution must be\n"
             "composed by spatial selection alone — either data-free mask\n"
             "algebra over trained cells' masks (μ + α(a) + β(r) → top-k), or\n"
             "support-conditioned mask search over the frozen bank.\n\n"
             "A composite cell is NOT the sum of two solutions — additive\n"
             "readout provably fails — so the two banks must factor the\n"
             "algorithm through depth: one stage's assembly shifts values,\n"
             "the next stage's assembly rotates positions. Commutativity of\n"
             "the group is what makes order-free spatial selection sufficient.",
             fontsize=9, color=INK2, va="top", linespacing=1.45)

    _save(fig, out, "fig1_task.png")


def fig_training(out, results, cfg):
    hist = results["history"]
    blocks = sorted({h["block"] for h in hist})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    n = len(blocks)
    for b in blocks:
        hs = [h for h in hist if h["block"] == b]
        xs = [b * cfg.steps_per_task + h["step"] for h in hs]
        ax.plot(xs, [h["pos_acc"] for h in hs], color=SEQ_BLUE(0.25 + 0.7 * b / max(n - 1, 1)),
                linewidth=1.6)
    for b in blocks[1:]:
        ax.axvline(b * cfg.steps_per_task, color=GRID, linewidth=0.7, zorder=0)
    ax.axhline(1 / cfg.p, color=BASELINE, linewidth=1, linestyle=":")
    ax.text(0, 1 / cfg.p + 0.02, "chance", fontsize=8, color=MUTED)
    ax.set_xlabel("sequential training step (block boundaries in gray)")
    ax.set_ylabel("per-position accuracy (held-out inputs)")
    ax.set_title("block-sequential training: each block = one grid cell", fontsize=10)
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    rows = results["trained_rows"]
    stored = [r["stored"][0] for r in rows]
    reconf = [r["reconfig"][0] for r in rows]
    idx = np.arange(len(rows))
    if "final" in rows[0]:
        ax.bar(idx - 0.27, stored, 0.25, color=CAT[3], label="block-end mask (stale)")
        ax.bar(idx, [r["final"][0] for r in rows], 0.25, color=CAT[0],
               label="post-stream consensus mask")
        ax.bar(idx + 0.27, reconf, 0.25, color=CAT[4], label="reconfig protocol")
    else:
        ax.bar(idx - 0.2, stored, 0.38, color=CAT[0], label="stored mask")
        ax.bar(idx + 0.2, reconf, 0.38, color=CAT[4], label="reconfig protocol")
    ax.axhline(1 / cfg.p, color=BASELINE, linewidth=1, linestyle=":")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(tuple(r["cell"])) for r in rows], rotation=90, fontsize=7)
    ax.set_ylabel("per-position accuracy")
    ax.set_title("final accuracy per trained cell (after ALL blocks)", fontsize=10)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    _save(fig, out, "fig2_training.png")


def _grid_heat(ax, values, p, L, title, na_cells=()):
    M = np.full((p, L), np.nan)
    for (a, r), v in values.items():
        M[a, r] = v
    im = ax.imshow(M, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    for a in range(p):
        for r in range(L):
            if (a, r) in na_cells:
                ax.add_patch(Rectangle((r - 0.5, a - 0.5), 1, 1, fill=False,
                                       hatch="///", edgecolor=GRID, linewidth=0))
            elif not np.isnan(M[a, r]):
                ax.text(r, a, f"{M[a, r]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if M[a, r] > 0.55 else INK2)
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"r={r}" for r in range(L)], fontsize=8)
    ax.set_yticks(range(p))
    ax.set_yticklabels([f"a={a}" for a in range(p)], fontsize=8)
    ax.set_title(title, fontsize=9.5)
    return im


def fig_grid_accuracy(out, results, cfg):
    p, L = cfg.p, cfg.L
    trained = {tuple(r["cell"]): r["stored"][0] for r in results["trained_rows"]}
    synth = {tuple(r["cell"]): r["additive"][0] for r in results["held_rows"]}
    reconf = {tuple(r["cell"]): r["reconfig"][0] for r in results["held_rows"]}
    reconf.update({tuple(r["cell"]): r["reconfig"][0] for r in results["trained_rows"]})
    joint = {tuple(r["cell"]): r["pos"] for r in results["joint_held"]}
    joint.update({tuple(r["cell"]): r["pos"] for r in results["joint_train"]})

    fig, axes = plt.subplots(1, 4, figsize=(14, 6.5))
    _grid_heat(axes[0], trained, p, L,
               "trained cells\n(stored mask)", na_cells=set(synth))
    _grid_heat(axes[1], synth, p, L,
               "HELD-OUT cells\nzero-shot synthesized mask", na_cells=set(trained))
    _grid_heat(axes[2], reconf, p, L, "all cells\nreconfig protocol (support set)")
    im = _grid_heat(axes[3], joint, p, L, "all cells\njoint task-conditioned MLP")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("per-position accuracy (held-out input sequences)", fontsize=8)
    cbar.outline.set_visible(False)
    _save(fig, out, "fig3_grid_accuracy.png")


def fig_methods(out, results, cfg):
    held = results["held_rows"]
    methods = [
        ("additive\nsynthesis", "additive", CAT[0]),
        ("assigned\n(b1:a, b2:r)", "assigned", CAT[6]),
        ("assigned\nswapped", "assigned_swap", CAT[7]),
        ("majority\nunion", "majority_union", CAT[4]),
        ("reconfig\nprotocol", "reconfig", CAT[1]),
        ("$\\mu$-only\n(no factors)", "mu_only", CAT[3]),
        ("nearest\nneighbor", "nearest_neighbor", CAT[5]),
        ("random\nk-mask", "random", CAT[2]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    for ax, j, ttl in ((axes[0], 0, "per-position accuracy"),
                       (axes[1], 1, "exact-match (whole sequence)")):
        means, cols = [], []
        for _, key, c in methods:
            vals = [r[key][j] for r in held]
            means.append(np.mean(vals))
            cols.append(c)
        xs = np.arange(len(methods))
        ax.bar(xs, means, 0.62, color=cols)
        for x, mval in zip(xs, means):
            ax.text(x, mval + 0.015, f"{mval:.2f}", ha="center", fontsize=8,
                    color=INK2)
        jb = np.mean([r["pos" if j == 0 else "exact"] for r in results["joint_held"]])
        ax.axhline(jb, color=INK2, linewidth=1.2, linestyle="--")
        ax.text(len(methods) - 0.55, jb + 0.06, f"joint task-cond. MLP: {jb:.2f}",
                fontsize=8, color=INK2, ha="right")
        chance = (1 / cfg.p) if j == 0 else (1 / cfg.p) ** cfg.L
        ax.axhline(chance, color=BASELINE, linewidth=1, linestyle=":")
        ax.text(-0.45, chance + 0.03, "chance", fontsize=8, color=MUTED,
                ha="left")
        ax.set_xticks(xs)
        ax.set_xticklabels([m[0].replace("\n", " ") for m in methods],
                           fontsize=7.5, rotation=24, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"held-out cells ({len(held)}): {ttl}", fontsize=10)
    _save(fig, out, "fig4_methods.png")


def fig_mask_structure(out, results, final_masks, final_scores, train_cells, cfg):
    stored_masks, mask_emas = final_masks, final_scores  # post-stream basis
    from .synthesis import overlap_matrix
    cells = sorted(train_cells)  # a-major order -> same-a blocks
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    for col, (name, sl, k) in enumerate((
            ("bank 1", slice(0, cfg.H1), cfg.k1),
            ("bank 2", slice(cfg.H1, None), cfg.k2))):
        ax = fig.add_subplot(gs[0, col])
        O = overlap_matrix(stored_masks, cells, sl, k).numpy()
        np.fill_diagonal(O, np.nan)
        ax.imshow(O, cmap=SEQ_BLUE, vmin=0, vmax=max(np.nanmax(O), 1e-6))
        labels = [str(c) for c in cells]
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"{name}: consensus-mask overlap |m∩m'|/k\n"
                     "(cells sorted by a, then r)", fontsize=9)

    # same-a / same-r / neither overlap summary
    ax = fig.add_subplot(gs[0, 2])
    groups = {"same a": [], "same r": [], "neither": []}
    per_bank = {"bank 1": (slice(0, cfg.H1), cfg.k1),
                "bank 2": (slice(cfg.H1, None), cfg.k2)}
    width = 0.38
    for bi, (bname, (sl, k)) in enumerate(per_bank.items()):
        O = overlap_matrix(stored_masks, cells, sl, k).numpy()
        vals = {g: [] for g in groups}
        for i, ci in enumerate(cells):
            for j, cj in enumerate(cells):
                if i >= j:
                    continue
                if ci[0] == cj[0]:
                    vals["same a"].append(O[i, j])
                elif ci[1] == cj[1]:
                    vals["same r"].append(O[i, j])
                else:
                    vals["neither"].append(O[i, j])
        xs = np.arange(3) + (bi - 0.5) * width
        means = [np.mean(vals[g]) for g in groups]
        ax.bar(xs, means, width, color=[CAT[0], CAT[4]][bi], label=bname)
        for x, mval in zip(xs, means):
            ax.text(x, mval + 0.005, f"{mval:.2f}", ha="center", fontsize=7.5,
                    color=INK2)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(list(groups))
    ax.set_ylabel("mean pairwise overlap")
    ax.set_title("overlap by shared factor\n(factorization ⇒ same-a > neither in one bank,\n"
                 "same-r > neither in the other)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    # ANOVA decomposition
    for col, (name, an) in enumerate((("consensus scores", results["anova_stored"]),
                                      ("training mask EMAs", results["anova_ema"]))):
        ax = fig.add_subplot(gs[1, col])
        banks = ["bank1", "bank2"]
        comps = ["a", "r", "residual"]
        colors = [CAT[0], CAT[4], BASELINE]
        xs = np.arange(len(banks))
        bottom = np.zeros(len(banks))
        for comp, c in zip(comps, colors):
            vals = np.array([an[b][comp] for b in banks])
            ax.bar(xs, vals, 0.5, bottom=bottom, color=c,
                   label={"a": "main effect a (value shift)",
                          "r": "main effect r (rotation)",
                          "residual": "interaction + residual"}[comp])
            for x, v, b0 in zip(xs, vals, bottom):
                if v > 0.04:
                    ax.text(x, b0 + v / 2, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if c != BASELINE else INK2)
            bottom += vals
        ax.set_xticks(xs)
        ax.set_xticklabels(["bank 1", "bank 2"])
        ax.set_ylabel("fraction of cross-cell mask variance")
        ax.set_title(f"ANOVA of {name} across grid cells", fontsize=9.5)
        if col == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    # factor-sorted mask raster (bank with clearest structure: show both)
    ax = fig.add_subplot(gs[1, 2])
    E = torch.stack([mask_emas[c].float() for c in cells]).numpy()
    n_show = 130
    var_a, var_r = [], []
    for bank_sl in (slice(0, cfg.H1), slice(cfg.H1, cfg.H1 + cfg.H2)):
        Eb = E[:, bank_sl]
        a_lv = sorted({c[0] for c in cells})
        r_lv = sorted({c[1] for c in cells})
        ga = np.stack([Eb[[i for i, c in enumerate(cells) if c[0] == a]].mean(0)
                       for a in a_lv])
        gr = np.stack([Eb[[i for i, c in enumerate(cells) if c[1] == r]].mean(0)
                       for r in r_lv])
        var_a.append(ga.var(0))
        var_r.append(gr.var(0))
    sel = np.concatenate([
        np.argsort(-var_a[0])[:n_show // 2],                    # bank1 a-selective
        cfg.H1 + np.argsort(-var_r[1])[:n_show // 2],           # bank2 r-selective
    ])
    ax.imshow(E[:, sel], cmap=SEQ_BLUE, aspect="auto", vmin=0,
              vmax=max(E[:, sel].max(), 1e-6))
    ax.axvline(n_show // 2 - 0.5, color=INK2, linewidth=1)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([str(c) for c in cells], fontsize=6)
    ax.set_xlabel("units: bank-1 a-selective | bank-2 r-selective", fontsize=8)
    ax.set_title("consensus-score raster (factor-sorted units)", fontsize=9)

    _save(fig, out, "fig5_mask_structure.png")


def fig_consistency(out, results, cfg):
    rows = results["consistency"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ks = sorted({r["k"] for r in rows})
    xs = np.arange(len(ks))
    for off, key, c, lbl in (
            (-0.27, None, CAT[0], "temporal route (iterate trained mask, zero data)"),
            (0.0, "spatial_pos", CAT[4], "spatial route (synthesized mask, zero data)"),
            (0.27, "route_agreement", BASELINE, "route agreement")):
        if key is None:
            vals = [np.mean([r["temporal_step_pos"][-1] for r in rows
                             if r["k"] == k]) for k in ks]
        else:
            vals = [np.mean([r[key] for r in rows if r["k"] == k]) for k in ks]
        ax.bar(xs + off, vals, 0.25, color=c, label=lbl)
        for x, v in zip(xs, vals):
            ax.text(x + off, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5,
                    color=INK2)
    ax.axhline(1 / cfg.p, color=BASELINE, linewidth=1, linestyle=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"k={k}\n(n={sum(1 for r in rows if r['k'] == k)})"
                        for k in ks], fontsize=8)
    ax.set_ylabel("per-position accuracy")
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("k-fold iterates of trained cells, scored on the\n"
                 "HELD-OUT composite (ka, kr), mean over composites", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    ax = axes[1]
    for r in rows:
        steps = r["temporal_step_pos"]
        ax.plot(range(1, len(steps) + 1), steps, linewidth=1.1,
                color=CAT[0], alpha=0.25)
    max_k = max(len(r["temporal_step_pos"]) for r in rows)
    means = [np.mean([r["temporal_step_pos"][j] for r in rows
                      if len(r["temporal_step_pos"]) > j]) for j in range(max_k)]
    ax.plot(range(1, max_k + 1), means, marker="o", markersize=5, linewidth=2.4,
            color=SEQ_BLUE(0.85), label="mean over composites")
    ax.axhline(1 / cfg.p, color=BASELINE, linewidth=1, linestyle=":")
    ax.set_xlabel("autoregressive step j (each verified vs the generator)")
    ax.set_ylabel("per-position accuracy at step j")
    ax.set_title("error compounding along the temporal route\n"
                 f"(one line per composite, n={len(rows)})", fontsize=9.5)
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(range(1, max_k + 1))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _save(fig, out, "fig6_consistency.png")


def make_all_figures(out, model, grid, results, final_masks, final_scores,
                     synthesizer, cfg):
    train_cells = [tuple(c) for c in results["train_cells"]]
    held_cells = [tuple(c) for c in results["held_cells"]]
    fig_task_explainer(out, grid, train_cells, held_cells, cfg)
    fig_training(out, results, cfg)
    fig_grid_accuracy(out, results, cfg)
    fig_methods(out, results, cfg)
    fig_mask_structure(out, results, final_masks, final_scores, train_cells, cfg)
    fig_consistency(out, results, cfg)
