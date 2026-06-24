"""
viz_latents.py - Visualize MaskedMLPSpatial latent activations on DataGenSine.

Trains the model on CUDA, then captures pre-output latents (h + h_lin)
for a few eval observations and plots them as 32x32 rasters.
Images saved to images/ with prefix "latents".
"""

import os

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from models import MaskedMLPSpatial
from tasks import DataGenSine, make_rng

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Config:
    SEED = 42

    # Task
    X_DIM = 2
    Y_DIM = 2
    N_TASK = 1
    TASK_SIZE = 2
    FREQUENCY = 6
    CLASSIFICATION = True
    CORRELATED_TASKS = False
    CONCEPT_SHIFT = False
    DOMAIN_SHIFT_FACTOR = 0.0
    T = 2048  # samples per training batch

    # Model
    SQUARE_DIM = 32
    LAYERS = 3
    DROPOUT = 0.0
    LATERAL_B = 0.0
    LATERAL_STEPS = 1
    KERNEL_SIZE = 3

    # Training
    N_EPOCHS = 1000
    LR = 3e-4

    # Viz
    N_OBS_TO_PLOT = 32
    GRID_COLS = 8  # obs per row in the raster grid
    IMAGES_DIR = "images"
    IMAGE_PREFIX = "latents"

    @property
    def HIDDEN_SIZE(self):
        return self.SQUARE_DIM**2

    @property
    def MODEL_PARS(self):
        return {
            "b": self.LATERAL_B,
            "steps": self.LATERAL_STEPS,
            "layers": self.LAYERS,
            "kernel_size": self.KERNEL_SIZE,
            "dropout": self.DROPOUT,
        }


# ---------------------------------------------------------------------------
# Training  (no hooks — clean forward pass)
# ---------------------------------------------------------------------------


def train(cfg: Config, device: torch.device):
    rng = make_rng(cfg.SEED)
    dg = DataGenSine(
        input_dim=cfg.X_DIM,
        output_dim=cfg.Y_DIM,
        n_tasks=cfg.N_TASK,
        task_size=cfg.TASK_SIZE,
        frequency=cfg.FREQUENCY,
        classify=cfg.CLASSIFICATION,
        n_samples=cfg.T,
        correlated_tasks=cfg.CORRELATED_TASKS,
        concept_shift=cfg.CONCEPT_SHIFT,
        domain_shift_factor=cfg.DOMAIN_SHIFT_FACTOR,
        rng=rng,
    )

    model = MaskedMLPSpatial(
        cfg.X_DIM, cfg.Y_DIM, cfg.HIDDEN_SIZE, **cfg.MODEL_PARS
    ).to(device)
    mask = torch.ones(cfg.HIDDEN_SIZE, device=device)

    loss_fn = nn.MSELoss() if not cfg.CLASSIFICATION else nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    model.train()
    for epoch in range(cfg.N_EPOCHS):
        total_loss = 0.0
        for task_idx in range(cfg.N_TASK):
            x, y = dg(task_idx)
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, mask), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(
                f"  epoch {epoch + 1}/{cfg.N_EPOCHS}  loss={total_loss / cfg.N_TASK:.4f}"
            )

    return model, dg, mask


# ---------------------------------------------------------------------------
# Latent capture  (hook registered only during eval)
# ---------------------------------------------------------------------------


def capture_latents(
    model: MaskedMLPSpatial, x: torch.Tensor, mask: torch.Tensor, sq: int
):
    """
    Return latents as (B, sq, sq) by hooking the input to model.out.
    The input to model.out is h + h_lin, the full pre-output hidden state.
    """
    captured = []

    def pre_hook(_module, inp):
        # inp is a tuple; inp[0] is the (B, hidden_size) tensor
        captured.append(inp[0].detach().cpu())

    handle = model.out.register_forward_pre_hook(pre_hook)
    model.eval()
    with torch.no_grad():
        model(x, mask)
    handle.remove()

    # captured has one entry per forward call (should be exactly 1)
    latent = captured[0]  # (B, hidden_size)
    return latent.view(latent.shape[0], sq, sq)  # (B, sq, sq)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_task_grid(latents, task_idx, cfg: Config):
    """
    Plot latents as a grid (GRID_ROWS x GRID_COLS) of sq x sq rasters.
    latents: (N, sq, sq) numpy array
    """
    n = latents.shape[0]
    ncols = cfg.GRID_COLS
    nrows = (n + ncols - 1) // ncols

    vmin, vmax = latents.min(), latents.max()
    cell = 2.2  # inches per cell
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell, nrows * cell))
    axes = axes.flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            im = ax.imshow(
                latents[i],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_title(f"obs {i}", fontsize=7, pad=2)
        ax.axis("off")

    # single shared colorbar on the right
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.suptitle(f"Latents — Task {task_idx}", fontsize=10, y=1.01)
    return fig


def plot_latent_rasters(model, dg, mask, cfg: Config, device: torch.device):
    os.makedirs(cfg.IMAGES_DIR, exist_ok=True)
    sq = cfg.SQUARE_DIM
    mean_paths = []

    for task_idx in range(cfg.N_TASK):
        x, _ = dg(task_idx)
        x_sub = x[: cfg.N_OBS_TO_PLOT].to(device)

        latents = capture_latents(model, x_sub, mask, sq).numpy()  # (N, sq, sq)

        # --- grid of all obs ---
        fig = plot_task_grid(latents, task_idx, cfg)
        fpath = os.path.join(cfg.IMAGES_DIR, f"{cfg.IMAGE_PREFIX}_task{task_idx}.png")
        fig.savefig(fpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        mean_paths.append(fpath)
        print(f"  task {task_idx}: {latents.shape[0]} obs → {fpath}")

    # --- GIF cycling through tasks ---
    frames = [imageio.imread(p) for p in mean_paths]
    gif_path = os.path.join(cfg.IMAGES_DIR, f"{cfg.IMAGE_PREFIX}_tasks.gif")
    imageio.mimwrite(gif_path, frames, duration=0.8)
    print(f"  GIF → {gif_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()

    print("Training...")
    model, dg, mask = train(cfg, device)

    print("Capturing latents and plotting...")
    plot_latent_rasters(model, dg, mask, cfg, device)

    print("Done.")
