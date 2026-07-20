"""
piecewise_learning.py - Piecewise continual learning with Functional Task Networks.

Idea
====
Instead of presenting N *different* functions (as base/run_experiments.py does),
we present N contiguous SUB-DOMAINS of ONE shared, highly nonlinear latent-sinusoid
function. Each sub-domain is an "extreme domain shift" from the others (a disjoint
slab of the input space). The FTN must:

  1. Cover the whole function by learning its sub-domains block-sequentially
     (one task = one sub-domain, trained to convergence before the next), and
  2. Perform inference by reconfiguring its mask to the correct subspace given a
     small sample of whichever sub-domain it is shown.

This probes two things at once: (a) FTN as a defense against extreme domain shift
in continual learning, and (b) whether learning a complicated nonlinear function
piece-wise (a different mask-selected subnetwork per region) has advantages.

Reuse / isolation
=================
The FTN model (models.NeuroModel) and the mask configurers (masks.*) are imported
UNMODIFIED from ../base. Only the *task* changes, via DataGenPiecewise in this
directory's tasks.py (a copy of base/tasks.py with the piecewise generator added). The
training/eval driver and all visualizations are the same as base/run_experiments.py
(copied here so this experiment is self-contained and ../base stays untouched).

Config: see config.py in this directory (number of sub-domains, function hardness,
number of components, domain bounds).
"""

import argparse
import os
from collections import defaultdict, deque

import base.projects.piecewise.config as config  # piecewise config, this directory
import base.projects.piecewise.masks as masks  # snapshot of the pre-v2 base/masks.py, this directory
import base.projects.piecewise.models as models  # snapshot of the pre-v2 base/models.py, this directory
import numpy as np
import base.projects.piecewise.tasks as tasks  # this directory (base tasks + DataGenPiecewise)
import torch
import torch.nn.functional as F
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

Model = models.NeuroModel
Config = config.Default()
MaskConfigurer = Config.MASK_CONFIGURER


# =============================================================================
# Task: one nonlinear function, split into N input sub-domains
# =============================================================================
# The piecewise data generator lives in this directory's tasks.py (a copy of
# base/tasks.py with DataGenPiecewise added). `import tasks` above resolves to
# it because this script's directory is first on sys.path.

DataGen = tasks.DataGenPiecewise


def make_rng(seed):
    """Create a local torch.Generator on the current default device."""
    device = torch.empty(0).device
    return torch.Generator(device=device).manual_seed(seed)


def make_data_gen(seed):
    """Create the piecewise data generator with its own local RNG."""
    return DataGen(
        input_dim=Config.X_DIM,
        output_dim=Config.Y_DIM,
        n_domains=Config.N_TASK,
        n_components=Config.N_COMPONENTS,
        frequency=Config.FREQUENCY,
        classify=Config.CLASSIFICATION,
        n_samples=Config.T,
        domain_half=Config.DOMAIN_HALF,
        split_dim=Config.SPLIT_DIM,
        data_temp=Config.DATA_TEMP,
        rng=make_rng(seed),
    )


def sample_full_domain(data_gen):
    """Draw one batch uniformly over the WHOLE bounded box (not segmented).

    The piecewise generator's __call__ restricts the split axis to one
    sub-domain's slab; the simultaneous baseline instead sees the entire
    function at once, so we sample every axis uniformly across the full
    [-DOMAIN_HALF, DOMAIN_HALF] box and apply the shared latent function.
    """
    h = Config.DOMAIN_HALF
    x = torch.rand(Config.T, Config.X_DIM, generator=data_gen.rng) * (2 * h) - h
    with torch.no_grad():
        y = data_gen.transform(x)
    return x, y


def make_fixed_mask(hidden_size, n_active):
    """A static binary mask with the first `n_active` units switched on.

    This is the simultaneous baseline's analog of an FTN mask: instead of
    reconfiguring a K-unit subnetwork per sub-domain, it hands the model one
    fixed subnetwork sized to the FTN's *total* budget (K per sub-domain x
    N_TASK sub-domains). Placement is arbitrary for this unstructured MLP with
    lateral dynamics disabled (LATERAL_STEPS=0), so the first n_active units are
    used for determinism.
    """
    n_active = min(n_active, hidden_size)
    m = torch.zeros(hidden_size)
    m[:n_active] = 1.0
    return m


# =============================================================================
# Replay buffer (persists across phases) -- copied from base/run_experiments.py
# =============================================================================


class ReplayBuffer:
    """Stores (data, mask) pairs for offline replay across all experiment phases."""

    def __init__(self, maxlen, batch_size=None, shuffle_on_replay=False):
        self.data = deque(maxlen=maxlen)
        self.masks = deque(maxlen=maxlen)
        # Rows kept per replay minibatch; None keeps the full observed batch.
        self.batch_size = batch_size
        # If True, keep the full observed batch and draw a fresh random
        # `batch_size` rows on each replay; if False, subsample once at store
        # time and reuse that fixed subset.
        self.shuffle_on_replay = shuffle_on_replay

    def add(self, dat, m):
        x, y = dat
        n = x.shape[0]
        if (
            not self.shuffle_on_replay
            and self.batch_size is not None
            and self.batch_size < n
        ):
            idx = torch.randperm(n, device=x.device)[: self.batch_size]
            dat = (x[idx], y[idx])
        self.data.append(dat)
        self.masks.append(m.clone())

    def get(self, i):
        """Return stored (x, y) at index i.

        When SHUFFLE_ON_REPLAY is set, draw a fresh random `batch_size` rows
        from the full stored batch; otherwise return the stored batch as-is
        (already subsampled at store time).
        """
        x, y = self.data[i]
        if (
            self.shuffle_on_replay
            and self.batch_size is not None
            and self.batch_size < x.shape[0]
        ):
            idx = torch.randperm(x.shape[0], device=x.device)[: self.batch_size]
            return x[idx], y[idx]
        return x, y

    def __len__(self):
        return len(self.data)


# =============================================================================
# Evaluation helpers -- copied from base/run_experiments.py
# =============================================================================


def evaluate_tasks(model, data_gen, mask_configurer, task_indices):
    """Quick loss evaluation over multiple batches. Returns {task_idx: mean_loss}."""
    results = {}
    m = torch.zeros(model.hidden_size)
    for task_idx in task_indices:
        losses = []
        for _ in range(Config.N_EVAL_EPOCHS):
            dat = data_gen(task_idx)
            m = mask_configurer(model, dat, m)
            x, y = dat
            with torch.no_grad():
                pred = model(x.unsqueeze(0), m)
                if Config.CLASSIFICATION:
                    target = torch.where(y == 1)[1]
                    losses.append(F.cross_entropy(pred.squeeze(), target).item())
                else:
                    losses.append(F.mse_loss(pred.squeeze(), y.squeeze()).item())
        results[task_idx] = np.mean(losses)
    return results


def evaluate_backwards(model, data_gen, mask_configurer, task_indices):
    """Reconfigure & eval on each prior sub-domain for N_RETRO_EVAL_BATCHES batches.

    Respects CONFIGURE_PER_BATCH (reconfig every batch vs. once per task).
    Returns {task_idx: {"loss": mean_loss, "acc": mean_score}}.
    """
    results = {}
    for task_idx in task_indices:
        m = torch.zeros(mask_configurer.hidden_size)
        losses = []
        accs = []
        if not Config.CONFIGURE_PER_BATCH:
            m = mask_configurer(model, data_gen(task_idx), m)
        for _ in range(Config.N_RETRO_EVAL_BATCHES):
            if Config.CONFIGURE_PER_BATCH:
                m = mask_configurer(model, data_gen(task_idx), m)
            # Eval on a fresh holdout batch -- not the one used to reconfigure.
            x, y = data_gen(task_idx)
            with torch.no_grad():
                pred = model(x.unsqueeze(0), m)
                p = pred.squeeze()
                if Config.CLASSIFICATION:
                    target = torch.where(y == 1)[1]
                    losses.append(F.cross_entropy(p, target).item())
                    accs.append((p.argmax(-1) == target).float().mean().item())
                else:
                    mse = F.mse_loss(p, y.squeeze()).item()
                    losses.append(mse)
                    var = float(y.float().var().item()) if y.numel() > 1 else 1.0
                    accs.append(max(0.0, 1.0 - mse / max(var, 1e-8)))
        results[task_idx] = {
            "loss": float(np.mean(losses)),
            "acc": float(np.mean(accs)),
        }
    return results


def _score(pred, y):
    """Loss and accuracy for one (pred, y) batch (mirrors evaluate_backwards)."""
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


def evaluate_full_domain(predict, data_gen, n_batches):
    """Mean loss/acc over `n_batches` of uniformly-sampled full-domain points.

    `predict(x) -> logits/values` lets the same metric score either the FTN
    (route each point to its sub-domain mask) or the simultaneous model (one
    fixed mask for all points), so the two are directly comparable.
    """
    losses, accs = [], []
    for _ in range(n_batches):
        x, y = sample_full_domain(data_gen)
        with torch.no_grad():
            pred = predict(x)
        loss, acc = _score(pred, y)
        losses.append(loss)
        accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))


def make_piecewise_predict(model, data_gen, mask_configurer):
    """Predictor that routes each point to its sub-domain's reconfigured mask.

    Masks are reconfigured once per sub-domain (as the func_* reconstruction
    does), then reused, so this measures the trained FTN's whole-function
    accuracy under mask reconfiguration at inference time.
    """
    masks_by_t = [
        mask_configurer(model, data_gen(t), torch.zeros(model.hidden_size))
        for t in range(Config.N_TASK)
    ]

    def predict(x):
        sub = torch.bucketize(
            x[:, Config.SPLIT_DIM].contiguous(), data_gen.edges[1:-1].contiguous()
        )
        out = torch.zeros(x.shape[0], Config.Y_DIM)
        for t in range(Config.N_TASK):
            idx = torch.where(sub == t)[0]
            if len(idx) == 0:
                continue
            out[idx] = model(x[idx].unsqueeze(0), masks_by_t[t]).squeeze(0)
        return out

    return predict


def run_simultaneous(model, data_gen, fixed_mask, n_steps, optimizer, suffix):
    """Train the model on the whole function at once with one fixed mask.

    Baseline for the piecewise FTN: same architecture, same total active-unit
    budget, but no per-sub-domain masking or block-sequential schedule. Returns
    the EMA loss history.
    """
    optimizer.state = defaultdict(dict)
    loss_history = []
    loss_smooth = None
    for i in range(n_steps):
        x, y = sample_full_domain(data_gen)
        optimizer.zero_grad()
        pred = model(x.unsqueeze(0), fixed_mask)
        if Config.CLASSIFICATION:
            target = torch.where(y == 1)[1]
            loss = F.cross_entropy(pred.squeeze(), target)
        else:
            loss = F.mse_loss(pred.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()

        li = loss.item()
        if i == 0:
            print(f"SIMUL INITIAL_LOSS: {li}")
            loss_smooth = li
        else:
            loss_smooth = (
                Config.LOSS_EMA_SMOOTHING * loss_smooth
                + (1 - Config.LOSS_EMA_SMOOTHING) * li
            )
        loss_history.append(loss_smooth)
        if i % max(1, n_steps // 10) == 0:
            print(f"  [{suffix}] step {i}/{n_steps}: loss={loss_smooth:.4f}")
    return loss_history


# =============================================================================
# Plotting -- copied from base/run_experiments.py (visualizations identical)
# =============================================================================


def save_loss_plot(per_task_losses, suffix):
    import matplotlib.pyplot as plt

    for task_idx in sorted(per_task_losses.keys()):
        if per_task_losses[task_idx]:
            plt.plot(per_task_losses[task_idx], linewidth=1, label=f"Task {task_idx}")
    plt.xlabel("Training step")
    plt.ylim(bottom=-0.05, top=1)
    plt.ylabel("Loss")
    plt.legend(fontsize=7)
    plt.savefig(f"images/loss_{suffix}.png")
    plt.close()


def save_eval_bar(eval_dict, labels, suffix):
    import matplotlib.pyplot as plt

    ent = -Config.Y_DIM * np.log(1.0 / Config.Y_DIM) * (1.0 / Config.Y_DIM)
    keys = list(eval_dict.keys())
    vals = [eval_dict[k] for k in keys]
    lbls = [labels.get(k, str(k)) for k in keys]
    plt.bar(range(len(keys)), vals, tick_label=lbls)
    plt.ylabel("Loss")
    plt.ylim(bottom=0, top=ent + 0.25)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"images/eval_{suffix}.png")
    plt.close()


def save_forgetting_matrix(rows, row_order, col_order, suffix):
    """Save accuracy matrix to images/xfer_{suffix}.png.

    rows[i] is a dict {task_idx: acc} captured after row_order[i] was first
    trained. Columns span `col_order`, so prospective evals on not-yet-trained
    sub-domains fill the upper triangle. Entries with no data render white.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n_rows = len(row_order)
    n_cols = len(col_order)
    if n_rows == 0 or n_cols == 0:
        return
    mat = np.full((n_rows, n_cols), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, t in enumerate(col_order):
            if t in row:
                mat[i, j] = row[t]

    cmap = LinearSegmentedColormap.from_list("red_white", ["red", "white"])
    cmap = cmap.copy()
    cmap.set_bad(color="white")

    _, ax = plt.subplots(
        figsize=(max(4, n_cols * 0.45), max(4, n_rows * 0.45)),
    )
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([str(t) for t in col_order], fontsize=7)
    ax.set_yticklabels([str(t) for t in row_order], fontsize=7)
    ax.set_xlabel("Evaluated sub-domain")
    ax.set_ylabel("After training sub-domain")
    ax.set_title("Accuracy (white=1, red=0)")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    fontsize = max(4, min(9, int(70 / max(max(n_rows, n_cols), 1))))
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=fontsize, color="black",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    plt.tight_layout()
    plt.savefig(f"images/xfer_{suffix}.png", dpi=120)
    plt.close()


def save_eval_snapshots(snapshots, suffix):
    import matplotlib.pyplot as plt

    if not snapshots:
        return
    all_tasks = sorted(snapshots[0].keys())
    for t in all_tasks:
        plt.plot([s[t] for s in snapshots], label=f"Task {t}", linewidth=1)
    plt.xlabel("Eval checkpoint")
    plt.ylabel("Loss")
    plt.legend(fontsize=7)
    plt.savefig(f"images/eval_over_time_{suffix}.png")
    plt.close()


# =============================================================================
# Piecewise-specific visualization: the function and its reconstruction
# =============================================================================


def _field(y):
    """Map model/generator output to a scalar field for imshow."""
    if Config.CLASSIFICATION:
        return y.argmax(-1).float()
    return y.squeeze(-1)


def save_function_figure(
    data_gen, suffix, model=None, mask_configurer=None, fixed_mask=None, res=200
):
    """Visualize the shared function over the 2D domain, with sub-domain edges.

    When `model` and `mask_configurer` are given, render the FTN's piecewise
    reconstruction: for each sub-domain we reconfigure the mask from a small
    sample of that sub-domain (inference), then predict on the grid points that
    fall in its slab. This is the piecewise analog of base's task_vis.png.

    When `model` and `fixed_mask` are given instead, render the simultaneous
    baseline's reconstruction: the single fixed mask predicts the whole grid at
    once (no per-sub-domain reconfiguration).
    """
    if Config.X_DIM != 2:
        return
    import matplotlib.pyplot as plt

    h = Config.DOMAIN_HALF
    axis = torch.linspace(-h, h, res)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    with torch.no_grad():
        true_field = _field(data_gen.transform(X)).reshape(res, res)

    panels = [("True function", true_field)]

    if model is not None and fixed_mask is not None:
        # Simultaneous baseline: one fixed mask predicts the whole domain.
        with torch.no_grad():
            pred = model(X.unsqueeze(0), fixed_mask).squeeze(0)
        recon = _field(pred)
        panels.append(("Simultaneous reconstruction", recon.reshape(res, res)))
    elif model is not None and mask_configurer is not None:
        # Assign each grid point to a sub-domain by its split-axis coordinate.
        sub = torch.bucketize(
            X[:, Config.SPLIT_DIM].contiguous(),
            data_gen.edges[1:-1].contiguous(),
        )
        recon = torch.zeros(X.shape[0])
        for t in range(Config.N_TASK):
            # Reconfigure to sub-domain t from a small sample (inference).
            m_t = mask_configurer(model, data_gen(t), torch.zeros(model.hidden_size))
            idx = torch.where(sub == t)[0]
            if len(idx) == 0:
                continue
            with torch.no_grad():
                pred = model(X[idx].unsqueeze(0), m_t).squeeze(0)
            recon[idx] = _field(pred)
        panels.append(("Piecewise reconstruction", recon.reshape(res, res)))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    edges = data_gen.edges.detach().cpu().numpy()
    for ax, (title, fld) in zip(axes, panels):
        ax.imshow(
            fld.detach().cpu().numpy(),
            origin="lower",
            extent=[-h, h, -h, h],
            aspect="equal",
            cmap="viridis",
        )
        for e in edges[1:-1]:
            if Config.SPLIT_DIM == 0:
                ax.axvline(e, color="w", lw=0.8, alpha=0.7)
            else:
                ax.axhline(e, color="w", lw=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
    plt.tight_layout()
    plt.savefig(f"images/func_{suffix}.png", dpi=130)
    plt.close()


# =============================================================================
# Core: unified train/eval phase -- copied from base/run_experiments.py
# =============================================================================


def run_phase(
    model,
    data_gen,
    mask_configurer,
    task_indices,
    n_epochs,
    suffix,
    optimizer,
    replay_buffer,
    train=True,
    eval_gen=None,
    eval_tasks=None,
    eval_mask=None,
    block_dim=None,
):
    """Run a training or evaluation phase with visualization.

    When train=True: optimizes model weights with offline replay.
    When train=False: forward-only passes (for visualization / loss tracking).

    Returns (per_task_losses, eval_snapshots).
    """
    if block_dim is None:
        block_dim = Config.SQUARE_DIM

    # Reset Adam (m, v) at each training phase so optimizer state from a prior
    # phase can't drift parameters whose gradient is zero under the new mask.
    if train:
        optimizer.state = defaultdict(dict)

    hidden = mask_configurer.hidden_size
    n_blocks = hidden // (block_dim**2)
    gh, gw = block_dim, n_blocks * block_dim

    per_task_losses = {t: [] for t in task_indices}
    eval_snapshots = []

    # Gate the bw_*/rgb_*/loss_* training visualizations (and their per-step
    # frame building) for speed. func_* and xfer_* are produced elsewhere.
    save_vis = getattr(Config, "SAVE_TRAINING_VIS", True)

    frames, rgb_frames = [], []
    m = torch.zeros(hidden)
    m_ema = torch.zeros(gh, gw, 3)
    red_net = torch.zeros(hidden)
    green_net = torch.zeros(hidden)
    blue_net = torch.zeros(hidden)

    blank = Image.fromarray(np.zeros((gh, gw, 3), dtype=np.uint8), mode="RGB")
    frames.extend([blank.copy() for _ in range(2)])
    rgb_frames.extend([blank.copy() for _ in range(10)])

    gif_save_epoch = min(Config.GIF_SAVE_EPOCH, n_epochs) or 1
    seen_tasks = []
    forgetting_rows = []
    for epoch in range(n_epochs):
        for task in task_indices:
            loss_history = []
            if not Config.CONFIGURE_PER_BATCH:
                m = mask_configurer(model, data_gen(task), m)
            optimizer.state = defaultdict(dict)
            for i in range(Config.TRAIN_DURATION):
                dat = data_gen(task)
                if Config.CONFIGURE_PER_BATCH:
                    m = mask_configurer(model, dat, m)

                if train and i < Config.N_SAMPLE_PER_TASK:
                    replay_buffer.add(dat, m)

                optimizer.zero_grad()

                with torch.set_grad_enabled(train):
                    x, y = dat
                    pred = model(x.unsqueeze(0), m)
                    if Config.CLASSIFICATION:
                        target = torch.where(y == 1)[1]
                        loss = F.cross_entropy(pred.squeeze(), target)
                    else:
                        loss = F.mse_loss(pred.squeeze(), y.squeeze())

                if epoch == 0 and i == 0:
                    print(f"TASK {task} INITIAL_LOSS: {loss.item()}")
                if train:
                    loss.backward()

                    # Offline replay: train on stored samples with overlapping masks
                    if len(replay_buffer) > 0:
                        stored = torch.stack(list(replay_buffer.masks))
                        corrs = (stored * m).sum(-1) / m.sum()
                        keep = (corrs > 0.2) & (corrs <= 0.8)
                        overlap = torch.where(keep)[0]
                        if len(overlap) > 0:
                            idxes = torch.randint(
                                0, len(overlap), (Config.N_REPLAY_SAMPLES,)
                            )
                            for idx in idxes:
                                x_r, y_r = replay_buffer.get(idx.item())
                                m_r = replay_buffer.masks[idx.item()]
                                r_pred = model(x_r.unsqueeze(0), m_r)
                                if Config.CLASSIFICATION:
                                    r_tgt = torch.where(y_r == 1)[1]
                                    F.cross_entropy(r_pred.squeeze(), r_tgt).backward()
                                else:
                                    F.mse_loss(
                                        r_pred.squeeze(), y_r.squeeze()
                                    ).backward()

                    optimizer.step()

                    # --- End-of-task visualization ---
                    if save_vis:
                        m_2d = masks.mask_to_grid(m.detach(), block_dim)
                        m_ema[:, :, 0] = torch.max(0.95 * m_ema[:, :, 0], m_2d)
                        m_ema[:, :, 1] = torch.max(0.30 * m_ema[:, :, 1], m_2d)
                        m_ema[:, :, 2] = torch.max(0.975 * m_ema[:, :, 2], m_2d)

                        if epoch % Config.PLOT_EVERY == 0:
                            frame = Image.fromarray(
                                (m_ema.cpu().numpy() * 255).astype(np.uint8), mode="RGB"
                            )
                            frames.append(frame)

                li = loss.item()
                if i == 0:
                    loss_smooth = li
                else:
                    loss_smooth = (
                        Config.LOSS_EMA_SMOOTHING * loss_smooth
                        + (1 - Config.LOSS_EMA_SMOOTHING) * li
                    )
                loss_history.append(loss_smooth)

            # RGB channel assignment: task%3 -> 0:blue, 1:red, 2:green
            ch = task % 3
            if ch == 0:
                blue_net = m.clone()
            elif ch == 1:
                red_net = m.clone()
            else:
                green_net = m.clone()

            if save_vis and epoch % Config.PLOT_EVERY == 0:
                rgb = torch.stack(
                    [
                        masks.mask_to_grid(c, block_dim)
                        for c in [red_net, green_net, blue_net]
                    ]
                )
                rgb = masks.normalize_mask(rgb) * 255
                rgb = rgb.permute(1, 2, 0)
                rgb_frames.append(
                    Image.fromarray(rgb.detach().cpu().numpy().astype(np.uint8))
                )
                rgb_frames[-1].save(f"images/rgb_{suffix}.png")

            per_task_losses[task].extend(loss_history)

            # Backwards eval: report loss on every sub-domain seen so far.
            newly_seen = task not in seen_tasks
            if newly_seen:
                seen_tasks.append(task)
            if train and getattr(Config, "EVAL_BACKWARDS", False):
                prospective = getattr(Config, "EVAL_PROSPECTIVE", False)
                eval_set = list(task_indices) if prospective else list(seen_tasks)
                bw = evaluate_backwards(model, data_gen, mask_configurer, eval_set)
                losses = {t: bw[t]["loss"] for t in eval_set}
                accs = {t: bw[t]["acc"] for t in eval_set}
                loss_str = " ".join(
                    f"t{t}={losses[t]:.4f}" + ("*" if t not in seen_tasks else "")
                    for t in eval_set
                )
                acc_str = " ".join(
                    f"t{t}={accs[t]:.3f}" + ("*" if t not in seen_tasks else "")
                    for t in eval_set
                )
                print(
                    f"  [{suffix}] backwards eval (epoch {epoch}, "
                    f"after sub-domain {task}): loss[{loss_str}] | "
                    f"acc[{acc_str}] | "
                    f"mean_loss={np.mean(list(losses.values())):.4f} "
                    f"mean_acc={np.mean(list(accs.values())):.3f}"
                )
                if newly_seen:
                    forgetting_rows.append(accs)
                    col_order = list(task_indices) if prospective else seen_tasks
                    save_forgetting_matrix(
                        forgetting_rows, seen_tasks, col_order, suffix
                    )

        if save_vis and epoch % Config.PLOT_EVERY == 0:
            save_loss_plot(per_task_losses, suffix)

        if eval_gen is not None and eval_tasks is not None:
            if epoch % Config.PLOT_EVERY == 0:
                mc = eval_mask or mask_configurer
                eval_snapshots.append(evaluate_tasks(model, eval_gen, mc, eval_tasks))

        if save_vis and (epoch + 1) % gif_save_epoch == 0 and len(frames) > 1:
            print(f"  [{suffix}] Saving GIFs at epoch {epoch + 1}")
            frames[0].save(
                f"images/bw_{suffix}.gif",
                save_all=True,
                append_images=frames[1:],
                duration=Config.GIF_DURATION,
                loop=0,
            )
            if len(rgb_frames) > 1:
                rgb_frames[0].save(
                    f"images/rgb_{suffix}.gif",
                    save_all=True,
                    append_images=rgb_frames[1:],
                    duration=Config.GIF_DURATION,
                    loop=0,
                )

    return per_task_losses, eval_snapshots


# =============================================================================
# Main experiment
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        default="",
        help="Tag appended to output filenames (e.g. 'seed1' -> 'p1_piecewise_seed1.png').",
    )
    args = parser.parse_args()
    RUN_TAG = f"_{args.name}" if args.name else ""

    print("Running piecewise learning:")
    print(Model.__name__)
    print(DataGen.__name__)
    print(MaskConfigurer.__name__)
    if RUN_TAG:
        print(f"Run tag: {RUN_TAG}")

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("images", exist_ok=True)

    if Config.SEED is not None:
        torch.manual_seed(Config.SEED)

    # --- Shared setup ---
    data_gen = make_data_gen(Config.SEED + 100)

    model = Model(
        input_size=Config.X_DIM,
        output_size=Config.Y_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        **Config.MODEL_PARS,
    )
    mask_cfg = MaskConfigurer(
        hidden_size=Config.HIDDEN_SIZE,
        square_dim=Config.SQUARE_DIM,
        classification=Config.CLASSIFICATION,
        **Config.MASK_PARS,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.TRAIN_LR, weight_decay=Config.WEIGHT_DECAY
    )
    replay = ReplayBuffer(
        maxlen=Config.REPLAY_MAXLEN,
        batch_size=getattr(Config, "REPLAY_BATCH_SIZE", None),
        shuffle_on_replay=getattr(Config, "SHUFFLE_ON_REPLAY", False),
    )

    ent = -Config.Y_DIM * np.log(1.0 / Config.Y_DIM) * (1.0 / Config.Y_DIM)
    print(
        f"Model: hidden={Config.HIDDEN_SIZE} ({Config.SQUARE_DIM}x{Config.SQUARE_DIM}); "
        f"function: {Config.N_COMPONENTS} components, freq={Config.FREQUENCY}; "
        f"{Config.N_TASK} sub-domains along axis {Config.SPLIT_DIM}"
    )
    print(f"Chance-level loss: {ent:.4f}")
    print(f"Active phases: {Config.PHASES}")

    # Preview: the true function and its sub-domain partition (analog of task_vis).
    save_function_figure(data_gen, f"preview{RUN_TAG}")

    task0_trajectory = {}
    comparison = {}  # {"piecewise"/"simultaneous": {"loss", "acc"}} for the summary

    # ----- Phase 0: Pre-train sub-domain 0 -----
    if 0 in Config.PHASES:
        print(f"\n{'=' * 60}\nPHASE 0: Pre-train sub-domain 0\n{'=' * 60}")
        run_phase(
            model,
            data_gen,
            mask_cfg,
            [0],
            Config.N_EPOCHS,
            f"p0_pretrain{RUN_TAG}",
            optimizer,
            replay,
            train=True,
        )
        task0_trajectory["baseline"] = evaluate_tasks(model, data_gen, mask_cfg, [0])[0]
        print(f"Sub-domain 0 baseline loss: {task0_trajectory['baseline']:.4f}")

    # ----- Phase 1: Block-sequential learning over all sub-domains -----
    if 1 in Config.PHASES:
        print(
            f"\n{'=' * 60}\nPHASE 1: Piecewise continual learning "
            f"({Config.N_TASK} sub-domains)\n{'=' * 60}"
        )
        run_phase(
            model,
            data_gen,
            mask_cfg,
            list(range(Config.N_TASK)),
            Config.N_EPOCHS,
            f"p1_piecewise{RUN_TAG}",
            optimizer,
            replay,
            train=True,
        )
        pw_eval = evaluate_tasks(model, data_gen, mask_cfg, list(range(Config.N_TASK)))
        print("Per-sub-domain eval:", {k: f"{v:.4f}" for k, v in pw_eval.items()})
        if 0 in Config.PHASES:
            task0_trajectory["after_piecewise"] = evaluate_tasks(
                model, data_gen, mask_cfg, [0]
            )[0]

        # Reconstruct the whole function piece-wise (inference by reconfiguration).
        save_function_figure(
            data_gen, f"p1_piecewise{RUN_TAG}", model=model, mask_configurer=mask_cfg
        )

        # Whole-function accuracy under mask reconfiguration (comparable to the
        # simultaneous baseline in Phase 2).
        pw_predict = make_piecewise_predict(model, data_gen, mask_cfg)
        pw_loss, pw_acc = evaluate_full_domain(
            pw_predict, data_gen, Config.N_RETRO_EVAL_BATCHES
        )
        comparison["piecewise"] = {"loss": pw_loss, "acc": pw_acc}
        print(f"Piecewise full-domain: loss={pw_loss:.4f} acc={pw_acc:.3f}")

    # ----- Phase 2: Simultaneous baseline (one fixed mask, whole domain) -----
    if 2 in Config.PHASES:
        n_active = Config.K * Config.N_TASK
        n_steps = Config.TRAIN_DURATION * Config.N_TASK
        print(
            f"\n{'=' * 60}\nPHASE 2: Simultaneous fit "
            f"({n_active} fixed active units, {n_steps} steps)\n{'=' * 60}"
        )

        # Fresh copy of the same architecture -- an independent comparison, not a
        # continuation of the FTN-trained weights.
        simul_model = Model(
            input_size=Config.X_DIM,
            output_size=Config.Y_DIM,
            hidden_size=Config.HIDDEN_SIZE,
            **Config.MODEL_PARS,
        )
        simul_opt = torch.optim.AdamW(
            simul_model.parameters(),
            lr=Config.TRAIN_LR,
            weight_decay=Config.WEIGHT_DECAY,
        )
        fixed_mask = make_fixed_mask(Config.HIDDEN_SIZE, n_active)

        loss_history = run_simultaneous(
            simul_model, data_gen, fixed_mask, n_steps, simul_opt, f"p1_simul{RUN_TAG}"
        )
        if getattr(Config, "SAVE_TRAINING_VIS", True):
            save_loss_plot({0: loss_history}, f"p1_simul{RUN_TAG}")

        # Same figure style as func_p1_piecewise, but reconstructed with the
        # single fixed mask over the whole domain.
        save_function_figure(
            data_gen, f"p1_simul{RUN_TAG}", model=simul_model, fixed_mask=fixed_mask
        )

        def _simul_predict(x):
            return simul_model(x.unsqueeze(0), fixed_mask).squeeze(0)

        simul_loss, simul_acc = evaluate_full_domain(
            _simul_predict, data_gen, Config.N_RETRO_EVAL_BATCHES
        )
        comparison["simultaneous"] = {"loss": simul_loss, "acc": simul_acc}
        print(f"Simultaneous full-domain: loss={simul_loss:.4f} acc={simul_acc:.3f}")

    # ----- Summary -----
    print(f"\n{'=' * 60}\nEXPERIMENT COMPLETE\n{'=' * 60}")
    if task0_trajectory:
        print("\nSub-domain 0 trajectory:")
        for label, val in task0_trajectory.items():
            print(f"  {label:20s}: {val:.4f}")
        print(f"  {'(chance level)':20s}: {ent:.4f}")
    if comparison:
        print("\nWhole-function performance (uniform full-domain samples):")
        for label, m in comparison.items():
            print(f"  {label:14s}: loss={m['loss']:.4f}  acc={m['acc']:.3f}")
    print("\nOutputs saved to images/")
