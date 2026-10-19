"""
viz_latents_2d.py - 2D coordinate-mapped latent activity mosaic for MaskedMLPSpatial.

For X_DIM=2 inputs, builds a J×J grid of coordinate points spanning the input
domain and places the corresponding 32×32 latent raster at each grid position,
producing a (J*32) × (J*32) mosaic where spatial position = input coordinate.
"""

import os

import imageio.v2 as imageio
import matplotlib
import matplotlib.cm as cm
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from models import MaskedMLPSpatial, Spatial0
from tasks import DataGenSine, make_rng

assert True  # X_DIM must be 2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

torch.manual_seed(123)


class Config:
    SEED = 42

    # Task  (X_DIM fixed at 2)
    Y_DIM = 2
    N_TASK = 2
    TASK_SIZE = 2
    FREQUENCY = 6
    CLASSIFICATION = True
    CORRELATED_TASKS = False
    CONCEPT_SHIFT = False
    DOMAIN_SHIFT_FACTOR = 0.0
    T = 2048  # samples per training batch

    # Input domain for the 2D coordinate grid
    X_MIN = -1.6  # ~= -pi/2
    X_MAX = 1.6  # ~=  pi/2

    # Model
    SQUARE_DIM = 32
    LAYERS = 3
    DROPOUT = 0.0
    LATERAL_B = 0.0
    LATERAL_STEPS = 1
    KERNEL_SIZE = 3
    TORUS = False

    # Training
    N_EPOCHS = 5000
    LR = 3e-4
    WEIGHT_DECAY = 0
    # Viz
    J = 8  # grid resolution: J×J rasters in the mosaic
    EVAL_EVERY = 500  # save a mosaic every N epochs (0 = only at end)
    PRINT_EVERY = 100
    IMAGES_DIR = "images"
    IMAGE_PREFIX = "latents_2d"

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
            "torus": self.TORUS,
        }


# ---------------------------------------------------------------------------
# Training  (no hooks)
# ---------------------------------------------------------------------------


def train(cfg: Config, device: torch.device, x_grid: torch.Tensor, mask: torch.Tensor):
    rng = make_rng(cfg.SEED)
    dg = DataGenSine(
        input_dim=2,
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

    model = Spatial0(2, cfg.Y_DIM, cfg.HIDDEN_SIZE, **cfg.MODEL_PARS).to(device)

    loss_fn = nn.MSELoss() if not cfg.CLASSIFICATION else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )

    os.makedirs(cfg.IMAGES_DIR, exist_ok=True)

    for epoch in range(cfg.N_EPOCHS):
        model.train()
        total_loss = 0.0
        for task_idx in range(cfg.N_TASK):
            x, y = dg(task_idx)
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, mask), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if cfg.PRINT_EVERY > 0 and (epoch + 1) % cfg.PRINT_EVERY == 0:
            print(
                f"  epoch {epoch + 1}/{cfg.N_EPOCHS}  loss={total_loss / cfg.N_TASK:.4f}"
            )
        if cfg.EVAL_EVERY > 0 and (epoch + 1) % cfg.EVAL_EVERY == 0:
            save_mosaic(model, x_grid, mask, cfg, epoch=epoch + 1)

    return model


# ---------------------------------------------------------------------------
# Latent capture
# ---------------------------------------------------------------------------


def capture_latents(
    model: MaskedMLPSpatial, x: torch.Tensor, mask: torch.Tensor, sq: int
):
    """Hook model.out input to get final (h + h_lin) latent, shape (B, sq, sq)."""
    captured = []

    def pre_hook(_, inp):
        captured.append(inp[0].detach().cpu())

    handle = model.out.register_forward_pre_hook(pre_hook)
    model.eval()
    with torch.no_grad():
        model(x, mask)
    handle.remove()

    latent = captured[0]  # (B, hidden_size)
    return latent.view(latent.shape[0], sq, sq)  # (B, sq, sq)


# ---------------------------------------------------------------------------
# Build coordinate grid and mosaic
# ---------------------------------------------------------------------------


def make_coord_grid(cfg: Config, device: torch.device):
    """
    Returns x_batch (J*J, 2) — a regular grid over [X_MIN, X_MAX]^2.
    Row-major: outer dim = x2 (vertical, top=max), inner dim = x1 (horizontal).
    """
    coords = torch.linspace(cfg.X_MIN, cfg.X_MAX, cfg.J)
    # meshgrid: rows = x2 descending (so top of image = high x2), cols = x1
    x2, x1 = torch.meshgrid(coords.flip(0), coords, indexing="ij")  # both (J, J)
    x_batch = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1)  # (J*J, 2)
    return x_batch.to(device)


def build_mosaic(latents: np.ndarray, J: int, sq: int) -> np.ndarray:
    """
    latents: (J*J, sq, sq)
    Returns: (J*sq + J-1, J*sq + J-1) mosaic with 1-px NaN delimiters between panes.
    """
    H = W = J * sq + (J - 1)
    mosaic = np.full((H, W), np.nan)
    grid = latents.reshape(J, J, sq, sq)
    for row in range(J):
        for col in range(J):
            r0 = row * (sq + 1)
            c0 = col * (sq + 1)
            mosaic[r0 : r0 + sq, c0 : c0 + sq] = grid[row, col]
    return mosaic


def apply_viridis(arr: np.ndarray, delim_rgba=(20, 20, 20, 255)) -> np.ndarray:
    """Normalize and apply viridis colormap → uint8 RGBA (H, W, 4).
    NaN pixels (delimiters) are filled with delim_rgba."""
    delim_mask = np.isnan(arr)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    normed = np.nan_to_num((arr - lo) / (hi - lo + 1e-8), nan=0.0)
    rgba = (cm.viridis(normed) * 255).astype(np.uint8)
    rgba[delim_mask] = delim_rgba
    return rgba


# ---------------------------------------------------------------------------
# Mosaic save  (called during training and at the end)
# ---------------------------------------------------------------------------


def save_mosaic(
    model, x_grid: torch.Tensor, mask: torch.Tensor, cfg: Config, epoch: int
):
    """Capture latents, build mosaic, write annotated PNG tagged with epoch."""
    sq, J = cfg.SQUARE_DIM, cfg.J

    latents_np = capture_latents(model, x_grid, mask, sq).numpy()  # (J*J, sq, sq)
    mosaic = build_mosaic(latents_np, J, sq)
    rgba = apply_viridis(mosaic)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(
        rgba,
        extent=[cfg.X_MIN, cfg.X_MAX, cfg.X_MIN, cfg.X_MAX],
        origin="upper",
        aspect="equal",
    )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"Latent mosaic  epoch {epoch} — {J}×{J} coords, {sq}×{sq} rasters")

    lo, hi = np.nanmin(latents_np), np.nanmax(latents_np)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(lo, hi))
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    path = os.path.join(cfg.IMAGES_DIR, f"{cfg.IMAGE_PREFIX}_epoch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()
    mask = torch.ones(cfg.HIDDEN_SIZE, device=device)
    x_grid = make_coord_grid(cfg, device)

    print("Training...")
    model = train(cfg, device, x_grid, mask)

    print("Done.")
