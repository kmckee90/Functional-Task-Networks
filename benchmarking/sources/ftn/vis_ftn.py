"""
vis_ftn.py - Live-demo visualization of a functional task network (FTN).

This is a trimmed copy of run_experiments.py (phase-1 multi-task learning only)
with the analysis machinery removed and two multi-pane animations added.

It produces two GIFs:

  1. images/demo_live{TAG}.gif  — the "live demo". One frame per FRAME_EVERY
     training batches, with three panes:
        - LEFT : a static, full red/blue image of the *true* data-generating
                 function for the current task, in the same grid geometry as the
                 model's decision surface — so the learned function can be
                 compared at a glance to the target.
        - MID  : scatter of the current batch's 2D inputs (x0, x1) coloured
                 red/blue by label, drawn over a light-red / light-blue decision
                 surface produced by the model under the *current* mask. This
                 shows the data, the model's current predictive function, and how
                 they relate.
        - RIGHT: the per-batch mask EMA visualization from run_experiments.py
                 (the coloured `m_ema` raster).

  2. images/demo_reconfig{TAG}.gif — a frame-by-frame animation of mask
     reconfiguration on RECONFIG_VIZ_TASK at epoch RECONFIG_VIZ_EPOCH, captured
     *before* that task's first training session. This visualizes the forward-
     transfer solution: how the mask reorganizes (using the already-trained
     weights) to solve a new task from scratch. Same two-pane layout, with the
     fixed reconfiguration sample on the left and the evolving `m` mask (coloured
     EMA) on the right.

All design choices live in VisConfig. This script does not modify config.py,
run_experiments.py, masks.py, or models.py — the recording mask configurer is a
local subclass of masks.PolyakKWTAFast.
"""

import argparse
import os
from collections import defaultdict, deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import models
import numpy as np
import tasks
import torch
from reconfig_optimizer import make_reconfig_optimizer
import torch.nn as nn
import torch.nn.functional as F
from masks import PolyakKWTAFast, kwta, mask_to_grid
from PIL import Image

# =============================================================================
# Configuration (separate from config.py — all knobs for the demo live here)
# =============================================================================

Model = models.NeuroModel
DataGen = tasks.DataGenSine


class VisConfig:
    """Configuration for the FTN live-demo visualization."""

    # --- Experiment scope (kept small; this is a demo, not a benchmark) ---
    SEED = 123
    RECONFIG_OPTIMIZER = "Adam"
    N_TASK = 3
    N_EPOCHS = 3

    # --- Data generator: 3 orthogonal binary-classification tasks ---
    CLASSIFICATION = True
    X_DIM = 2
    Y_DIM = 2
    TASK_SIZE = 4
    FREQUENCY = 12
    T = 256
    DATA_TEMP = 1e-8
    CORRELATED_TASKS = True  # orthogonal (disjoint) task subspaces
    CONCEPT_SHIFT = True
    DOMAIN_SHIFT_FACTOR = 0

    # --- Model / mask: 32x32 grid, k=256 active units, PolyakKWTAFast ---
    SQUARE_DIM = 48
    K = 256
    LAYERS = 8
    DROPOUT = 0.2
    INNER_DIM = 8

    LATERAL_B = 0.0
    LATERAL_STEPS = 0
    KERNEL_SIZE = 3
    TORUS = True

    CONFIGURE_PER_BATCH = True  # reconfigure the mask every batch (live config)

    # --- Training sizes (short, for a watchable demo) ---
    TRAIN_DURATION = 1000
    N_SAMPLE_PER_TASK = 50
    N_REPLAY_SAMPLES = 1

    TRAIN_LR = 3e-4
    WEIGHT_DECAY = 0.0
    LOSS_EMA_SMOOTHING = 0.8

    # --- Reconfiguration animation ---
    RECONFIG_VIZ_TASK = 2  # which task to animate the reconfig on (0-indexed)
    RECONFIG_VIZ_EPOCH = 0  # animate the reconfig at the start of this epoch

    # --- Decision-surface rendering ---
    GRID_RES = 120  # decision-surface grid resolution (GRID_RES x GRID_RES)
    PLOT_RANGE = 2.0  # input domain shown: [-PLOT_RANGE, PLOT_RANGE]^2
    POINT_SIZE = 14  # scatter marker size

    # --- GIF output ---
    FRAME_EVERY = 10  # render a live-demo frame every N training batches
    GIF_DURATION = 50  # ms per frame, live-demo gif
    RECONFIG_GIF_DURATION = 100  # ms per frame, reconfig gif

    @property
    def HIDDEN_SIZE(self):
        return self.SQUARE_DIM**2

    @property
    def REPLAY_MAXLEN(self):
        return self.N_SAMPLE_PER_TASK * self.N_TASK

    @property
    def MODEL_PARS(self):
        return {
            "b": self.LATERAL_B,
            "steps": self.LATERAL_STEPS,
            "layers": self.LAYERS,
            "kernel_size": self.KERNEL_SIZE,
            "dropout_p": self.DROPOUT,
            "inner_size": self.INNER_DIM,
        }

    @property
    def MASK_PARS(self):
        return {
            "k": self.K,
            "torus": self.TORUS,
            "reconfig_steps": 60,
            "reconfig_lr": 0.3,
            "reconfig_optimizer": self.RECONFIG_OPTIMIZER,
            "reconfig_sample_size": 256,
        }


Config = VisConfig()


# =============================================================================
# Recording mask configurer (local subclass; does not touch masks.py)
# =============================================================================


class RecordingPolyakKWTAFast(PolyakKWTAFast):
    """PolyakKWTAFast that, when `recording` is True, stores the discrete mask at
    every reconfiguration step into `self.recorded`.

    The reconfiguration logic mirrors masks.SmoothKWTA.__call__ exactly; the only
    additions are the recording hooks, so the produced mask is identical to the
    parent's for a given input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recording = False
        self.recorded = []

    def __call__(self, model, data, m):
        x, y = data
        n = min(self.reconfig_sample_size, x.shape[0])

        if self.recording:
            self.recorded = []

        m = nn.Parameter(torch.randn_like(m) * 0.0001)
        optimizer = make_reconfig_optimizer([m], lr=self.reconfig_lr, name=self.reconfig_optimizer)

        z = model.parallel(x[:n]).detach()
        lowest_loss = torch.inf
        best = None
        for i in range(self.reconfig_steps):
            sm = torch.sigmoid(m)
            tau = m.detach().std() * torch.exp(
                -torch.ones(1) * i / (self.reconfig_steps / 4)
            )
            noise = torch.randn_like(m) * tau
            mask = kwta(m + noise, k=self.k) + sm - sm.detach()

            y_pred = model.linear(z * mask)
            if self.classification:
                target = torch.where(y == 1)[1]
                loss = F.cross_entropy(input=y_pred[:n], target=target[:n])
            else:
                loss = F.mse_loss(y_pred.squeeze()[:n], y[:n].squeeze())

            if self.entropy_coef != 0:
                entropy = (
                    torch.distributions.Categorical(logits=y_pred).entropy().mean()
                )
                loss = loss + self.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < lowest_loss:
                best = mask.clone().detach()
                lowest_loss = loss

            if self.recording:
                self.recorded.append(mask.detach().clone())

        # Lateral dynamics (Polyak-averaged smoothing) + final kwta.
        if self.lateral_polyak < 1.0:
            for _ in range(self.lateral_steps):
                m_lateral = self.step_lateral(best)
                best = (
                    self.lateral_polyak * best + (1 - self.lateral_polyak) * m_lateral
                )
                if self.recording:
                    self.recorded.append(kwta(best, k=self.k).clone())
            best = kwta(best, k=self.k)

        if self.recording:
            self.recorded.append(best.clone())

        return best


# =============================================================================
# Replay buffer (mirrors run_experiments.py)
# =============================================================================


class ReplayBuffer:
    """Stores (data, mask) pairs for offline replay during training."""

    def __init__(self, maxlen):
        self.data = deque(maxlen=maxlen)
        self.masks = deque(maxlen=maxlen)

    def add(self, dat, m):
        self.data.append(dat)
        self.masks.append(m.clone())

    def __len__(self):
        return len(self.data)


# =============================================================================
# Helpers
# =============================================================================


def make_rng(seed):
    device = torch.empty(0).device
    return torch.Generator(device=device).manual_seed(seed)


def make_data_gen(seed):
    return DataGen(
        input_dim=Config.X_DIM,
        output_dim=Config.Y_DIM,
        n_tasks=Config.N_TASK,
        task_size=Config.TASK_SIZE,
        frequency=Config.FREQUENCY,
        classify=Config.CLASSIFICATION,
        n_samples=Config.T,
        data_temp=Config.DATA_TEMP,
        correlated_tasks=Config.CORRELATED_TASKS,
        concept_shift=Config.CONCEPT_SHIFT,
        domain_shift_factor=Config.DOMAIN_SHIFT_FACTOR,
        rng=make_rng(seed),
    )


# --- EMA mask raster (same recipe as run_experiments.py) ---

LIGHT_RED = np.array([1.0, 0.5, 0.5])
LIGHT_BLUE = np.array([0.5, 0.55, 1.0])

# Full-saturation colours for the ground-truth (data-generating) surface.
FULL_RED = np.array([1.0, 0.0, 0.0])
FULL_BLUE = np.array([0.0, 0.0, 1.0])


def update_mask_ema(m_ema, m):
    """Update the 3-channel EMA raster in place from a flat mask `m`."""
    m_2d = mask_to_grid(m.detach(), Config.SQUARE_DIM)
    m_ema[:, :, 0] = torch.max(0.95 * m_ema[:, :, 0], m_2d)
    m_ema[:, :, 1] = torch.max(0.30 * m_ema[:, :, 1], m_2d)
    m_ema[:, :, 2] = torch.max(0.975 * m_ema[:, :, 2], m_2d)
    return m_ema


def make_grid(cfg):
    """Build the (R*R, 2) decision-surface query grid over [-R, R]^2."""
    lin = torch.linspace(-cfg.PLOT_RANGE, cfg.PLOT_RANGE, cfg.GRID_RES)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")  # gx: cols=x0, gy: rows=x1
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    return grid


def decision_surface(model, m, grid, cfg):
    """Return an (R, R, 3) light-coloured RGB image of P(class=1) under mask m."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(grid.unsqueeze(0), m)  # (R*R, Y_DIM)
        p1 = torch.softmax(logits, dim=-1)[..., 1]
    if was_training:
        model.train()
    p1 = p1.reshape(cfg.GRID_RES, cfg.GRID_RES).cpu().numpy()
    bg = (1 - p1)[..., None] * LIGHT_RED + p1[..., None] * LIGHT_BLUE
    return np.clip(bg, 0, 1)


def ground_truth_surface(data_gen, task, grid, cfg):
    """Return an (R, R, 3) full red/blue RGB image of the *true* data-generating
    label over the query grid for `task`.

    Mirrors DataGenSine.__call__ exactly (minus the random sampling of x): a
    point is class 0 (red) iff sin(freq * enc(x)·m)·m sums positive, else class
    1 (blue). Same grid geometry as `decision_surface`, but full saturation so
    it can be compared at a glance against what the model has learned.
    """
    with torch.no_grad():
        m = data_gen.m[task]
        x = grid + data_gen.dshift[task]
        z = data_gen.enc(x) * m
        z = torch.sin(data_gen.frequency * z) * m
        # class 1 (blue) iff z.sum() <= 0, matching the decoder in __call__
        p1 = (z.sum(-1) <= 0).float()
    p1 = p1.reshape(cfg.GRID_RES, cfg.GRID_RES).cpu().numpy()
    bg = (1 - p1)[..., None] * FULL_RED + p1[..., None] * FULL_BLUE
    return np.clip(bg, 0, 1)


def render_panel(model, m, batch, m_ema, title, cfg, grid, gt_bg):
    """Render one three-pane frame -> PIL Image.

    Left  : static ground-truth data-generating function (full red/blue).
    Mid   : decision surface under `m` + scatter of `batch` coloured by label.
    Right : the coloured mask EMA raster `m_ema`.
    """
    x, y = batch
    bg = decision_surface(model, m, grid, cfg)
    labels = y.argmax(dim=-1).cpu().numpy()
    xs = x[:, 0].cpu().numpy()
    ys = x[:, 1].cpu().numpy()
    pt_colors = np.where(labels == 0, "#c81e1e", "#1e46c8")
    r = cfg.PLOT_RANGE

    fig, (axGT, axL, axR) = plt.subplots(1, 3, figsize=(13, 4.4), dpi=100)

    axGT.imshow(gt_bg, extent=[-r, r, -r, r], origin="lower", aspect="equal")
    axGT.set_xlim(-r, r)
    axGT.set_ylim(-r, r)
    axGT.set_xlabel("x₀")
    axGT.set_ylabel("x₁")
    axGT.set_title("true data-generating function")

    axL.imshow(bg, extent=[-r, r, -r, r], origin="lower", aspect="equal")
    axL.scatter(
        xs, ys, c=pt_colors, s=cfg.POINT_SIZE, edgecolors="white", linewidths=0.3
    )
    axL.set_xlim(-r, r)
    axL.set_ylim(-r, r)
    axL.set_xlabel("x₀")
    axL.set_ylabel("x₁")
    axL.set_title("data + current model (decision surface)")

    axR.imshow(m_ema.cpu().numpy(), interpolation="nearest", aspect="equal")
    axR.set_title("mask  m  (per-batch EMA)")
    axR.set_xticks([])
    axR.set_yticks([])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(img)


def save_gif(frames, path, duration):
    if not frames:
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"  saved {path} ({len(frames)} frames)")


# =============================================================================
# Reconfiguration animation
# =============================================================================


def render_reconfig_gif(model, rec_mask, data_gen, task, cfg, grid, tag):
    """Capture and render the frame-by-frame mask reconfiguration for `task`.

    Uses a single fixed sample (the forward-transfer reconfiguration batch) and
    the already-trained model weights; the mask starts from scratch.
    """
    print(
        f"\nRendering reconfiguration animation for task {task} "
        f"(epoch {cfg.RECONFIG_VIZ_EPOCH}, forward-transfer solution)..."
    )
    batch = data_gen(task)  # fixed reconfiguration sample
    gt_bg = ground_truth_surface(data_gen, task, grid, cfg)

    rec_mask.recording = True
    rec_mask(model, batch, torch.zeros(cfg.HIDDEN_SIZE))
    rec_mask.recording = False
    recorded = rec_mask.recorded

    gh = gw = cfg.SQUARE_DIM
    m_ema = torch.zeros(gh, gw, 3)
    frames = []
    n = len(recorded)
    for step, mask in enumerate(recorded):
        update_mask_ema(m_ema, mask)
        frames.append(
            render_panel(
                model,
                mask,
                batch,
                m_ema,
                f"reconfiguration · task {task} · step {step + 1}/{n}",
                cfg,
                grid,
                gt_bg,
            )
        )
    save_gif(
        frames,
        f"images/demo_reconfig{tag}.gif",
        cfg.RECONFIG_GIF_DURATION,
    )


# =============================================================================
# Main: simplified multi-task training with live visualization
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="", help="Tag appended to output filenames.")
    args = parser.parse_args()
    tag = f"_{args.name}" if args.name else ""

    print("FTN live-demo visualization")
    print(f"  model={Model.__name__}  datagen={DataGen.__name__}  mask=PolyakKWTAFast")
    print(
        f"  N_TASK={Config.N_TASK}  {Config.SQUARE_DIM}x{Config.SQUARE_DIM}"
        f"  K={Config.K}  N_EPOCHS={Config.N_EPOCHS}"
    )

    torch.set_default_device("cuda")
    os.makedirs("images", exist_ok=True)
    torch.manual_seed(Config.SEED)

    data_gen = make_data_gen(Config.SEED + 100)
    model = Model(
        input_size=Config.X_DIM,
        output_size=Config.Y_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        **Config.MODEL_PARS,
    )
    mask_cfg = PolyakKWTAFast(
        hidden_size=Config.HIDDEN_SIZE,
        square_dim=Config.SQUARE_DIM,
        classification=Config.CLASSIFICATION,
        **Config.MASK_PARS,
    )
    rec_mask = RecordingPolyakKWTAFast(
        hidden_size=Config.HIDDEN_SIZE,
        square_dim=Config.SQUARE_DIM,
        classification=Config.CLASSIFICATION,
        **Config.MASK_PARS,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.TRAIN_LR, weight_decay=Config.WEIGHT_DECAY
    )
    replay = ReplayBuffer(maxlen=Config.REPLAY_MAXLEN)

    grid = make_grid(Config)
    gt_surfaces = [
        ground_truth_surface(data_gen, t, grid, Config) for t in range(Config.N_TASK)
    ]

    m = torch.zeros(Config.HIDDEN_SIZE)
    m_ema = torch.zeros(Config.SQUARE_DIM, Config.SQUARE_DIM, 3)
    frames = []
    global_step = 0

    for epoch in range(Config.N_EPOCHS):
        for task in range(Config.N_TASK):
            # Reconfiguration animation: capture the forward-transfer solution on
            # the configured task, *before* its first training session.
            if epoch == Config.RECONFIG_VIZ_EPOCH and task == Config.RECONFIG_VIZ_TASK:
                render_reconfig_gif(model, rec_mask, data_gen, task, Config, grid, tag)

            if not Config.CONFIGURE_PER_BATCH:
                m = mask_cfg(model, data_gen(task), m)
            optimizer.state = defaultdict(dict)

            for i in range(Config.TRAIN_DURATION):
                dat = data_gen(task)
                if Config.CONFIGURE_PER_BATCH:
                    m = mask_cfg(model, dat, m)

                if i < Config.N_SAMPLE_PER_TASK:
                    replay.add(dat, m)

                optimizer.zero_grad()
                x, y = dat
                pred = model(x.unsqueeze(0), m)
                target = torch.where(y == 1)[1]
                loss = F.cross_entropy(pred.squeeze(), target)
                loss.backward()

                # Offline replay on stored samples with overlapping masks.
                if len(replay) > 0:
                    stored = torch.stack(list(replay.masks))
                    corrs = (stored * m).sum(-1) / m.sum()
                    keep = (corrs > 0.2) & (corrs <= 0.8)
                    overlap = torch.where(keep)[0]
                    if len(overlap) > 0:
                        idxes = torch.randint(
                            0, len(overlap), (Config.N_REPLAY_SAMPLES,)
                        )
                        for idx in idxes:
                            x_r, y_r = replay.data[idx.item()]
                            m_r = replay.masks[idx.item()]
                            r_pred = model(x_r.unsqueeze(0), m_r)
                            r_tgt = torch.where(y_r == 1)[1]
                            F.cross_entropy(r_pred.squeeze(), r_tgt).backward()

                optimizer.step()

                update_mask_ema(m_ema, m)

                if global_step % Config.FRAME_EVERY == 0:
                    frames.append(
                        render_panel(
                            model,
                            m,
                            dat,
                            m_ema,
                            f"live demo · epoch {epoch} · task {task} · step {i}",
                            Config,
                            grid,
                            gt_surfaces[task],
                        )
                    )
                if i == 0:
                    print(
                        f"  epoch {epoch} task {task} initial loss: {loss.item():.4f}"
                    )
                global_step += 1

        save_gif(frames, f"images/demo_live{tag}.gif", Config.GIF_DURATION)

    save_gif(frames, f"images/demo_live{tag}.gif", Config.GIF_DURATION)
    print(f"\nDone. Outputs in images/: demo_live{tag}.gif, demo_reconfig{tag}.gif")


if __name__ == "__main__":
    main()
