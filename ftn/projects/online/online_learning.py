"""
online_learning.py - Point-wise online learning with Functional Task Networks.

Idea
====
Can an FTN learn a function by first overfitting mask-routed subspaces to
INDIVIDUAL data points, then composing the function via reconfiguration across
the set of them? There are no tasks here: a single fixed set of N_POINTS
(x, y) pairs is drawn once from one function, and

  Phase 1 (point-wise overfitting): the points are presented sequentially.
     For each point, the mask is reconfigured "per batch" (each batch being
     that one point) and the model weights are trained to convergence on it,
     so each point claims its own mask-routed subspace.

  Phase 2 (batched reconfiguration): the SAME points are batched together
     (all at once, or EVAL_BATCH_SIZE at a time) and the mask is reconfigured
     on the batch -- under each entry of EVAL_SETTINGS -- with the model
     weights frozen. If reconfiguration can compose the point-wise subspaces,
     the batch loss should approach the per-point overfit losses.

Reference points reported alongside the phase-2 settings:
  - retention: each point scored under its OWN stored phase-1 mask (did later
    points overwrite earlier subspaces?).
  - union_mask: all points scored under the union of the stored point masks
    (composition by brute-force superposition, no reconfiguration).
  - reference_ff (phase 3, REFERENCE_FF): a FRESH copy of the same
    architecture with the mask FIXED to the first EVAL_K units, trained
    JOINTLY on the full N_POINTS batch for up to REF_TRAIN_STEPS steps. A
    plain feed-forward fit of the same points (it will overfit them --
    that's the intended reference).

Reuse / isolation
=================
The FTN model (models.NeuroModel), mask configurers (masks.*), and data
generator (tasks.DataGenSine, single function via n_tasks=1) are imported
UNMODIFIED from ../base. All parameters live in this directory's config.py,
with train-time reconfig params (TRAIN_MASK_PARS) separated from eval-time
reconfig params (EVAL_MASK_PARS / EVAL_SETTINGS).
"""

import argparse
import os
from collections import defaultdict

import ftn.projects.online.config as config  # online config, this directory
import ftn.projects.online.masks as masks  # snapshot of the pre-v2 ftn/masks.py, this directory
import ftn.projects.online.models as models  # snapshot of the pre-v2 ftn/models.py, this directory
import numpy as np
import ftn.projects.online.tasks as tasks  # snapshot of the pre-v2 ftn/tasks.py, this directory
import torch
import torch.nn.functional as F

Model = models.NeuroModel
DataGen = tasks.DataGenSine

# ParallelMLP squeezes ALL size-1 dims, so a true single-point batch (1, D)
# collapses to (D,) and breaks its einsum. Single points are replicated to
# REPEAT identical rows instead -- losses and gradients are unchanged.
REPEAT = 2

Config = config.Default()  # replaced by --config in __main__


# =============================================================================
# Data: one function, one fixed set of points
# =============================================================================


def make_rng(seed):
    """Create a local torch.Generator on the current default device."""
    device = torch.empty(0).device
    return torch.Generator(device=device).manual_seed(seed)


def make_data_gen(seed):
    """Single-function generator (n_tasks=1, disjoint dims => valid task mask)."""
    return DataGen(
        input_dim=Config.X_DIM,
        output_dim=Config.Y_DIM,
        n_tasks=1,
        task_size=Config.TASK_SIZE,
        frequency=Config.FREQUENCY,
        classify=Config.CLASSIFICATION,
        n_samples=Config.N_POINTS,
        data_temp=Config.DATA_TEMP,
        correlated_tasks=False,
        concept_shift=True,
        domain_shift_factor=0.0,
        rng=make_rng(seed),
    )


def as_point_batch(x_all, y_all, i):
    """Point i as a (REPEAT, dim) batch of identical rows (see REPEAT note)."""
    xi = x_all[i : i + 1].expand(REPEAT, -1).contiguous()
    yi = y_all[i : i + 1].expand(REPEAT, -1).contiguous()
    return xi, yi


def ensure_min_batch(xb, yb):
    """Replicate a 1-row eval batch so it survives ParallelMLP's squeeze."""
    if xb.shape[0] == 1:
        xb = xb.expand(REPEAT, -1).contiguous()
        yb = yb.expand(REPEAT, -1).contiguous()
    return xb, yb


# =============================================================================
# Loss / scoring
# =============================================================================


def compute_loss(pred, y):
    p = pred.squeeze()
    if Config.CLASSIFICATION:
        target = torch.where(y == 1)[1]
        return F.cross_entropy(p, target)
    return F.mse_loss(p, y.squeeze())


def score(pred, y):
    """Loss and accuracy-like score for one batch (mirrors piecewise _score)."""
    p = pred.squeeze()
    if Config.CLASSIFICATION:
        target = torch.where(y == 1)[1]
        loss = F.cross_entropy(p, target).item()
        acc = (p.argmax(-1) == target).float().mean().item()
    else:
        mse = F.mse_loss(p, y.squeeze()).item()
        loss = mse
        var = float(y.float().var().item()) if y.numel() > 1 else 1.0
        acc = max(0.0, 1.0 - mse / max(var, 1e-8))
    return loss, acc


# =============================================================================
# Phase 1: sequential point-wise overfitting
# =============================================================================


def fit_point(model, optimizer, mask_cfg, xi, yi):
    """Overfit the model to one (x, y) pair under train-time reconfiguration.

    Reconfigures the mask every gradient step (TRAIN_CONFIGURE_PER_STEP) or
    once up front, then trains until CONVERGENCE_LOSS or MAX_STEPS_PER_POINT.
    Returns (final_mask, loss_history).
    """
    m = torch.zeros(model.hidden_size)
    if Config.RESET_OPTIMIZER_PER_POINT:
        optimizer.state = defaultdict(dict)
    if not Config.TRAIN_CONFIGURE_PER_STEP:
        m = mask_cfg(model, (xi, yi), m)

    history = []
    for _ in range(Config.MAX_STEPS_PER_POINT):
        if Config.TRAIN_CONFIGURE_PER_STEP:
            m = mask_cfg(model, (xi, yi), m)
        optimizer.zero_grad()
        pred = model(xi.unsqueeze(0), m)
        loss = compute_loss(pred, yi)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if history[-1] < Config.CONVERGENCE_LOSS:
            break
    return m.detach(), history


def make_fixed_mask(hidden_size, n_active):
    """Static binary mask with the first n_active units on (as in piecewise)."""
    n_active = min(n_active, hidden_size)
    m = torch.zeros(hidden_size)
    m[:n_active] = 1.0
    return m


def fit_reference_ff(model, optimizer, fixed_mask, x_all, y_all):
    """Train the reference feed-forward model on the FULL point batch jointly.

    One fixed mask, all N_POINTS at once, up to REF_TRAIN_STEPS gradient steps
    with early stop at CONVERGENCE_LOSS on the batch loss. Returns the loss
    history.
    """
    history = []
    for step in range(Config.REF_TRAIN_STEPS):
        optimizer.zero_grad()
        pred = model(x_all.unsqueeze(0), fixed_mask)
        loss = compute_loss(pred, y_all)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if step % max(1, Config.REF_TRAIN_STEPS // 10) == 0:
            print(f"  step {step}/{Config.REF_TRAIN_STEPS}: loss={history[-1]:.6f}")
        if history[-1] < Config.CONVERGENCE_LOSS:
            break
    return history


# =============================================================================
# Retro/forward transfer matrix over individual points (optional)
# =============================================================================


def point_score(pred, y, y_var):
    """[0, 1] score for a single (repeated) point, for the transfer matrix.

    Classification: probability assigned to the correct class (continuous
    accuracy). Regression: 1 - mse/var(y over the WHOLE point set), floored at
    0 -- a single point has no variance of its own, so the point set's global
    variance is the reference scale.
    """
    p = pred.squeeze()
    if Config.CLASSIFICATION:
        target = torch.where(y == 1)[1]
        probs = torch.softmax(p, -1)
        return probs[torch.arange(len(target)), target].mean().item()
    mse = F.mse_loss(p, y.squeeze()).item()
    return max(0.0, 1.0 - mse / max(y_var, 1e-8))


def evaluate_point_row(model, mask_cfg, x_all, y_all, y_var):
    """Score EVERY point under a fresh single-point reconfiguration.

    The reconfiguration support for point j is point j itself (the support set
    IS the evaluation problem -- deliberate: this asks whether the network
    stored the overfit solution such that reconfiguration can retrieve it).
    Points not yet trained fill the forward-transfer (upper) triangle.
    """
    row = []
    for j in range(x_all.shape[0]):
        xj, yj = as_point_batch(x_all, y_all, j)
        m = mask_cfg(model, (xj, yj), torch.zeros(model.hidden_size))
        with torch.no_grad():
            pred = model(xj.unsqueeze(0), m)
        row.append(point_score(pred, yj, y_var))
    return row


def save_xfer_matrix(rows, row_labels, n_cols, suffix):
    """Score matrix over points, adapted from base's save_forgetting_matrix.

    rows[i] holds per-point scores captured after point row_labels[i] was
    overfit. White=1, red=0; the upper triangle is forward transfer.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n_rows = len(rows)
    if n_rows == 0 or n_cols == 0:
        return
    mat = np.array(rows, dtype=float)

    cmap = LinearSegmentedColormap.from_list("red_white", ["red", "white"])

    cell = 0.45 if n_cols <= 24 else 0.18
    _, ax = plt.subplots(
        figsize=(max(4, n_cols * cell), max(4, n_rows * cell)),
    )
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(0, n_cols, max(1, n_cols // 32)))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(
        [str(t) for t in range(0, n_cols, max(1, n_cols // 32))], fontsize=6
    )
    ax.set_yticklabels([str(t) for t in row_labels], fontsize=6)
    ax.set_xlabel("Evaluated point")
    ax.set_ylabel("After overfitting point")
    ax.set_title("Point score (white=1, red=0); upper triangle = forward transfer")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.tick_params(which="minor", length=0)

    # Per-cell score text only while it stays legible.
    if n_cols <= 24:
        fontsize = max(4, min(9, int(70 / max(max(n_rows, n_cols), 1))))
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=fontsize, color="black",
                )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Score")
    plt.tight_layout()
    plt.savefig(f"images/xfer_{suffix}.png", dpi=120)
    plt.close()


# =============================================================================
# Phase 2: batched reconfiguration over the stored points
# =============================================================================


def index_batches(n, batch_size):
    """Sequential chunks of the point set; None/oversized => one full batch."""
    bs = n if batch_size is None else min(batch_size, n)
    return [list(range(s, min(s + bs, n))) for s in range(0, n, bs)]


def evaluate_setting(model, mask_cfg, x_all, y_all, batch_size):
    """Reconfigure on batches of stored points and score on the same batch.

    Each batch is reconfigured N_EVAL_RECONFIGS times (reconfig is stochastic);
    weights stay frozen. Returns (mean_loss, mean_acc, example_masks) with one
    example mask kept per batch.
    """
    losses, accs, example_masks = [], [], []
    for idx in index_batches(x_all.shape[0], batch_size):
        xb, yb = ensure_min_batch(x_all[idx], y_all[idx])
        for rep in range(Config.N_EVAL_RECONFIGS):
            m = mask_cfg(model, (xb, yb), torch.zeros(model.hidden_size))
            with torch.no_grad():
                pred = model(xb.unsqueeze(0), m)
            l, a = score(pred, yb)
            losses.append(l)
            accs.append(a)
            if rep == 0:
                example_masks.append(m.detach())
    return float(np.mean(losses)), float(np.mean(accs)), example_masks


def evaluate_fixed_mask(model, m, x_all, y_all):
    """Score all points at once under one fixed mask (no reconfiguration)."""
    with torch.no_grad():
        pred = model(x_all.unsqueeze(0), m)
    return score(pred, y_all)


def evaluate_retention(model, point_masks, x_all, y_all):
    """Per-point loss under each point's OWN stored phase-1 mask."""
    losses = []
    for i, m in enumerate(point_masks):
        xi, yi = as_point_batch(x_all, y_all, i)
        with torch.no_grad():
            pred = model(xi.unsqueeze(0), m)
        losses.append(compute_loss(pred, yi).item())
    return losses


# =============================================================================
# Plotting
# =============================================================================


def save_point_loss_curves(histories, suffix, labels=None):
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [f"pt {i}" for i in range(len(histories))]
    for h, lbl in zip(histories, labels):
        plt.plot(h, linewidth=1, label=lbl)
    plt.axhline(Config.CONVERGENCE_LOSS, color="k", ls="--", lw=0.8)
    plt.xlabel("Gradient step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(f"images/point_losses_{suffix}.png", dpi=120)
    plt.close()


def save_mask_overlap(point_masks, suffix):
    """IoU between the per-point masks: do points share subspaces?"""
    import matplotlib.pyplot as plt

    n = len(point_masks)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = (point_masks[i] * point_masks[j]).sum().item()
            union = (point_masks[i] + point_masks[j]).clamp(max=1).sum().item()
            mat[i, j] = inter / max(union, 1e-8)
    _, ax = plt.subplots(figsize=(max(4, n * 0.3), max(4, n * 0.3)))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Point")
    ax.set_ylabel("Point")
    ax.set_title("Point-mask overlap (IoU)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"images/mask_overlap_{suffix}.png", dpi=120)
    plt.close()


def save_mask_montage(mask_list, labels, suffix, ncols=8):
    import matplotlib.pyplot as plt

    n = len(mask_list)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, m, lbl in zip(axes, mask_list, labels):
        grid = masks.mask_to_grid(m.detach(), Config.SQUARE_DIM).cpu().numpy()
        ax.imshow(grid, cmap="hot", vmin=0, vmax=1)
        ax.set_title(lbl, fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"images/masks_{suffix}.png", dpi=120)
    plt.close()


def _truth_field(data_gen, X):
    """The data-generating function on query points X (mirrors DataGenSine)."""
    with torch.no_grad():
        m = data_gen.m[0]
        z = data_gen.enc(X) * m
        z = torch.sin(data_gen.frequency * z) * m
        if Config.CLASSIFICATION:
            # class 1 iff z.sum() <= 0, matching the decoder in __call__
            return (z.sum(-1) <= 0).float()
        return z.sum(-1) / data_gen.task_size


def _model_field(model, m, X):
    """The model's prediction on query points X under mask m.

    Rendered in eval mode (dropout off), as vis_ftn's decision_surface does;
    the numeric scores elsewhere keep the base pipeline's train-mode forward.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(X.unsqueeze(0), m)
    if was_training:
        model.train()
    if Config.CLASSIFICATION:
        return torch.softmax(out, -1)[..., 1]
    return out.squeeze(-1)


def save_fit_figure(model, data_gen, m, xb, yb, suffix):
    """Three-panel eval-fit figure (adapted from vis_ftn.render_panel):

    Left  : the true data-generating function over the input domain.
    Mid   : the model's prediction surface under the eval mask `m`, with the
            eval batch points (the ones the mask was reconfigured on) on top.
    Right : the eval mask itself, as a SQUARE_DIM x SQUARE_DIM raster.
    """
    if Config.X_DIM != 2:
        return
    import matplotlib.pyplot as plt

    res = Config.GRID_RES
    r = Config.PLOT_RANGE
    lin = torch.linspace(-r, r, res)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    truth = _truth_field(data_gen, X).reshape(res, res).cpu().numpy()
    pred = _model_field(model, m, X).reshape(res, res).cpu().numpy()

    if Config.CLASSIFICATION:
        cmap, vmin, vmax = "coolwarm", 0.0, 1.0
        pt_vals = yb.argmax(-1).cpu().numpy()
    else:
        # Shared color scale (from the truth field) so the panels compare.
        cmap, vmin, vmax = "viridis", truth.min(), truth.max()
        pt_vals = yb.squeeze(-1).cpu().numpy()

    fig, (axT, axM, axK) = plt.subplots(1, 3, figsize=(13, 4.4), dpi=100)

    axT.imshow(
        truth, extent=[-r, r, -r, r], origin="lower", aspect="equal",
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    axT.set_xlabel("x0")
    axT.set_ylabel("x1")
    axT.set_title("true data-generating function")

    axM.imshow(
        pred, extent=[-r, r, -r, r], origin="lower", aspect="equal",
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    axM.scatter(
        xb[:, 0].cpu().numpy(), xb[:, 1].cpu().numpy(),
        c=pt_vals, cmap=cmap, vmin=vmin, vmax=vmax,
        s=28, edgecolors="white", linewidths=0.6,
    )
    axM.set_xlim(-r, r)
    axM.set_ylim(-r, r)
    axM.set_xlabel("x0")
    axM.set_ylabel("x1")
    axM.set_title("eval batch + model under eval mask")

    axK.imshow(
        masks.mask_to_grid(m.detach(), Config.SQUARE_DIM).cpu().numpy(),
        cmap="hot", vmin=0, vmax=1, interpolation="nearest", aspect="equal",
    )
    axK.set_xticks([])
    axK.set_yticks([])
    axK.set_title("eval mask m")

    fig.suptitle(suffix, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"images/fit_{suffix}.png", dpi=120)
    plt.close(fig)


def save_eval_summary(results, suffix):
    import matplotlib.pyplot as plt

    names = list(results.keys())
    vals = [results[k]["loss"] for k in names]
    plt.bar(range(len(names)), vals, tick_label=names)
    plt.ylabel("Loss")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"images/eval_summary_{suffix}.png", dpi=120)
    plt.close()


# =============================================================================
# Main experiment
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        default="",
        help="Tag appended to output filenames (e.g. 'seed1' -> 'point_losses_seed1.png').",
    )
    parser.add_argument(
        "--config",
        default="Default",
        help="Config class from config.py (e.g. Default, Smoke).",
    )
    args = parser.parse_args()
    Config = getattr(config, args.config)()
    MaskConfigurer = Config.MASK_CONFIGURER
    RUN_TAG = args.name if args.name else "run"

    if Config.CLASSIFICATION and Config.Y_DIM != 2:
        raise ValueError(
            "CLASSIFICATION=True requires Y_DIM=2: DataGenSine's classify "
            "decoder always emits 2-column one-hot targets."
        )
    if not Config.CLASSIFICATION and Config.Y_DIM != 1:
        raise ValueError(
            "CLASSIFICATION=False requires Y_DIM=1: DataGenSine's regression "
            "decoder always emits 1-column targets, so a wider model output "
            "silently broadcasts against them (wrong losses)."
        )

    print("Running online (point-wise) learning:")
    print(Model.__name__)
    print(DataGen.__name__)
    print(MaskConfigurer.__name__)
    print(f"Config: {args.config} | Run tag: {RUN_TAG}")

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("images", exist_ok=True)

    if Config.SEED is not None:
        torch.manual_seed(Config.SEED)

    # --- The fixed point set: drawn ONCE, reused for both phases ---
    data_gen = make_data_gen(Config.SEED + 100)
    x_all, y_all = data_gen(0)  # (N_POINTS, X_DIM), (N_POINTS, Y_DIM)

    model = Model(
        input_size=Config.X_DIM,
        output_size=Config.Y_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        **Config.MODEL_PARS,
    )
    train_mask_cfg = MaskConfigurer(
        hidden_size=Config.HIDDEN_SIZE,
        square_dim=Config.SQUARE_DIM,
        classification=Config.CLASSIFICATION,
        **Config.TRAIN_MASK_PARS,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.TRAIN_LR, weight_decay=Config.WEIGHT_DECAY
    )

    print(
        f"Model: hidden={Config.HIDDEN_SIZE} ({Config.SQUARE_DIM}x{Config.SQUARE_DIM}); "
        f"{Config.N_POINTS} points; train_k={Config.TRAIN_K} eval_k={Config.EVAL_K}"
    )

    # ----- Phase 1: sequential point-wise overfitting -----
    print(f"\n{'=' * 60}\nPHASE 1: Point-wise overfitting "
          f"({Config.N_POINTS} points)\n{'=' * 60}")
    xfer_on = getattr(Config, "XFER_MATRIX", False)
    xfer_every = getattr(Config, "XFER_EVERY", 1)
    if xfer_on:
        n_rows = len(range(xfer_every - 1, Config.N_POINTS, xfer_every))
        print(
            f"Transfer matrix on: {n_rows} rows x {Config.N_POINTS} points "
            f"= {n_rows * Config.N_POINTS} extra reconfigs"
        )
    y_var = float(y_all.float().var().item()) if y_all.numel() > 1 else 1.0
    xfer_rows, xfer_row_labels = [], []

    point_masks, histories = [], []
    for i in range(Config.N_POINTS):
        xi, yi = as_point_batch(x_all, y_all, i)
        m_i, hist = fit_point(model, optimizer, train_mask_cfg, xi, yi)
        point_masks.append(m_i)
        histories.append(hist)
        converged = hist[-1] < Config.CONVERGENCE_LOSS
        print(
            f"  point {i:3d}: steps={len(hist):4d} "
            f"final_loss={hist[-1]:.6f} converged={converged}"
        )

        # Retro + forward transfer row: fresh single-point reconfig on every
        # point (seen and unseen) after this point's overfit.
        if xfer_on and ((i + 1) % xfer_every == 0 or i == Config.N_POINTS - 1):
            if not xfer_row_labels or xfer_row_labels[-1] != i:
                row = evaluate_point_row(model, train_mask_cfg, x_all, y_all, y_var)
                xfer_rows.append(row)
                xfer_row_labels.append(i)
                retro = row[: i + 1]
                fwd = row[i + 1 :]
                print(
                    f"    xfer row: retro_mean={np.mean(retro):.3f}"
                    + (f" fwd_mean={np.mean(fwd):.3f}" if fwd else "")
                )
                save_xfer_matrix(
                    xfer_rows, xfer_row_labels, Config.N_POINTS, RUN_TAG
                )

    final_losses = [h[-1] for h in histories]
    n_converged = sum(h[-1] < Config.CONVERGENCE_LOSS for h in histories)
    print(
        f"Phase 1 summary: {n_converged}/{Config.N_POINTS} converged, "
        f"mean_final_loss={np.mean(final_losses):.6f}, "
        f"mean_steps={np.mean([len(h) for h in histories]):.1f}"
    )

    save_point_loss_curves(histories, RUN_TAG)
    save_mask_overlap(point_masks, RUN_TAG)
    save_mask_montage(
        point_masks, [f"pt {i}" for i in range(len(point_masks))], f"points_{RUN_TAG}"
    )

    results = {}

    # Retention: each point under its own phase-1 mask (interference check).
    retention = evaluate_retention(model, point_masks, x_all, y_all)
    results["retention"] = {"loss": float(np.mean(retention)), "acc": float("nan")}
    print(
        f"\nRetention (own stored masks): mean_loss={np.mean(retention):.6f} "
        f"worst={np.max(retention):.6f}"
    )

    # Union of point masks: composition by superposition, no reconfiguration.
    m_union = torch.stack(point_masks).max(0).values
    u_loss, u_acc = evaluate_fixed_mask(model, m_union, x_all, y_all)
    results["union_mask"] = {"loss": u_loss, "acc": u_acc}
    print(
        f"Union mask ({int(m_union.sum().item())} active units): "
        f"loss={u_loss:.6f} acc={u_acc:.3f}"
    )

    # ----- Phase 2: batched reconfiguration under each eval setting -----
    print(f"\n{'=' * 60}\nPHASE 2: Batched reconfiguration eval\n{'=' * 60}")
    for name, overrides in Config.EVAL_SETTINGS.items():
        overrides = dict(overrides)
        batch_size = overrides.pop("eval_batch_size", Config.EVAL_BATCH_SIZE)
        pars = {**Config.EVAL_MASK_PARS, **overrides}
        eval_mask_cfg = MaskConfigurer(
            hidden_size=Config.HIDDEN_SIZE,
            square_dim=Config.SQUARE_DIM,
            classification=Config.CLASSIFICATION,
            **pars,
        )
        n_batches = len(index_batches(Config.N_POINTS, batch_size))
        loss, acc, eval_masks = evaluate_setting(
            model, eval_mask_cfg, x_all, y_all, batch_size
        )
        results[name] = {"loss": loss, "acc": acc}
        print(
            f"  [{name}] batch_size={batch_size or Config.N_POINTS} "
            f"({n_batches} batch(es)) x {Config.N_EVAL_RECONFIGS} reconfigs: "
            f"loss={loss:.6f} acc={acc:.3f}"
        )
        save_mask_montage(
            eval_masks,
            [f"batch {b}" for b in range(len(eval_masks))],
            f"eval_{name}_{RUN_TAG}",
        )
        # Eval-fit solution figures: one per batch, rendered with that batch's
        # first reconfigured mask (the same one kept in the montage above).
        for b, idx in enumerate(index_batches(Config.N_POINTS, batch_size)):
            save_fit_figure(
                model,
                data_gen,
                eval_masks[b],
                x_all[idx],
                y_all[idx],
                f"{name}_b{b}_{RUN_TAG}",
            )

    # ----- Phase 3: reference feed-forward baseline (fixed mask) -----
    if getattr(Config, "REFERENCE_FF", True):
        print(f"\n{'=' * 60}\nPHASE 3: Reference feed-forward baseline "
              f"(fixed mask, first {Config.EVAL_K} units, full batch)\n{'=' * 60}")
        ref_model = Model(
            input_size=Config.X_DIM,
            output_size=Config.Y_DIM,
            hidden_size=Config.HIDDEN_SIZE,
            **Config.MODEL_PARS,
        )
        ref_optimizer = torch.optim.AdamW(
            ref_model.parameters(), lr=Config.TRAIN_LR,
            weight_decay=Config.WEIGHT_DECAY,
        )
        fixed_mask = make_fixed_mask(Config.HIDDEN_SIZE, Config.EVAL_K)

        # Joint fit of all N_POINTS at once (it will overfit them -- that is
        # the intended reference), unlike phase 1's sequential stream.
        ref_hist = fit_reference_ff(ref_model, ref_optimizer, fixed_mask, x_all, y_all)
        print(
            f"  finished: steps={len(ref_hist)} final_loss={ref_hist[-1]:.6f} "
            f"converged={ref_hist[-1] < Config.CONVERGENCE_LOSS}"
        )
        save_point_loss_curves([ref_hist], f"reference_{RUN_TAG}", labels=["full batch"])

        ref_loss, ref_acc = evaluate_fixed_mask(ref_model, fixed_mask, x_all, y_all)
        results["reference_ff"] = {"loss": ref_loss, "acc": ref_acc}
        print(f"Reference feed-forward (all points): loss={ref_loss:.6f} "
              f"acc={ref_acc:.3f}")
        save_fit_figure(
            ref_model, data_gen, fixed_mask, x_all, y_all,
            f"reference_{RUN_TAG}",
        )

    # ----- Summary -----
    print(f"\n{'=' * 60}\nEXPERIMENT COMPLETE\n{'=' * 60}")
    print("\nComposition summary (stored points):")
    for label, r in results.items():
        acc_str = "" if np.isnan(r["acc"]) else f"  acc={r['acc']:.3f}"
        print(f"  {label:14s}: loss={r['loss']:.6f}{acc_str}")
    save_eval_summary(results, RUN_TAG)
    print("\nOutputs saved to images/")
