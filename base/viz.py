"""
viz.py - Visualization for FTN experiments.

All plotting/animation lives here, deliberately outside the FTN class:
  - MaskAnimator: mask-activity GIFs (EMA grayscale + per-task RGB rasters)
  - save_loss_plot: per-task training-loss curves
  - save_forgetting_matrix: retro/prospective accuracy matrix
"""

import numpy as np
import torch
from PIL import Image

from ftn import mask_to_grid, normalize_mask


def tag(suffix):
    """Filename tag: '_suffix', or '' when suffix is empty (no dangling '_')."""
    return f"_{suffix}" if suffix else ""


class MaskAnimator:
    """
    Tracks mask activity during a training phase and writes GIF/PNG rasters.

    - record_step(m): update the EMA raster each training step; optionally
      snapshot it as a GIF frame.
    - record_task(task, m): update the per-task RGB raster at end of task
      (task % 3 -> blue/red/green channel); optionally snapshot + save PNG.
    - maybe_save_gifs(epoch, suffix): flush GIFs every gif_save_epoch epochs.
    """

    def __init__(self, hidden_size, block_dim, gif_duration, gif_save_epoch,
                 out_dir="artifacts", ema_scale=1.0):
        self.block_dim = block_dim
        self.gif_duration = gif_duration
        self.gif_save_epoch = gif_save_epoch
        self.out_dir = out_dir

        # Trail decay of the bw_*.gif EMA channels. Green decays fast (bright
        # head of the trail); red + blue persist, giving the purple mid-trail
        # and the blue tail. ema_scale stretches the red/blue time constants:
        # decay**(1/scale), so scale=2 doubles the time constant (longer
        # purple/blue trails), scale=0.5 halves it.
        ema_scale = max(float(ema_scale), 1e-6)
        self.decay = (0.95 ** (1 / ema_scale), 0.30, 0.975 ** (1 / ema_scale))

        n_blocks = hidden_size // (block_dim**2)
        gh, gw = block_dim, n_blocks * block_dim
        self.m_ema = torch.zeros(gh, gw, 3)
        self.red_net = torch.zeros(hidden_size)
        self.green_net = torch.zeros(hidden_size)
        self.blue_net = torch.zeros(hidden_size)

        blank = Image.fromarray(np.zeros((gh, gw, 3), dtype=np.uint8), mode="RGB")
        self.frames = [blank.copy() for _ in range(2)]
        self.rgb_frames = [blank.copy() for _ in range(10)]

    def record_step(self, m, snapshot=True):
        """Per-step EMA update of mask activity; snapshot appends a GIF frame."""
        m_2d = mask_to_grid(m.detach(), self.block_dim)
        for c, decay in enumerate(self.decay):
            self.m_ema[:, :, c] = torch.max(decay * self.m_ema[:, :, c], m_2d)

        if snapshot:
            frame = Image.fromarray(
                (self.m_ema.cpu().numpy() * 255).astype(np.uint8), mode="RGB"
            )
            self.frames.append(frame)

    def record_task(self, task, m, suffix, snapshot=True):
        """End-of-task RGB channel assignment: task%3 -> 0:blue, 1:red, 2:green."""
        ch = task % 3
        if ch == 0:
            self.blue_net = m.clone()
        elif ch == 1:
            self.red_net = m.clone()
        else:
            self.green_net = m.clone()

        if snapshot:
            rgb = torch.stack(
                [
                    mask_to_grid(c, self.block_dim)
                    for c in [self.red_net, self.green_net, self.blue_net]
                ]
            )
            rgb = normalize_mask(rgb) * 255
            rgb = rgb.permute(1, 2, 0)
            self.rgb_frames.append(
                Image.fromarray(rgb.detach().cpu().numpy().astype(np.uint8))
            )
            self.rgb_frames[-1].save(f"{self.out_dir}/rgb{tag(suffix)}.png")

    def maybe_save_gifs(self, epoch, suffix):
        if (epoch + 1) % self.gif_save_epoch != 0 or len(self.frames) <= 1:
            return
        print(f"  [{suffix}] Saving GIFs at epoch {epoch + 1}")
        self.frames[0].save(
            f"{self.out_dir}/bw{tag(suffix)}.gif",
            save_all=True,
            append_images=self.frames[1:],
            duration=self.gif_duration,
            loop=0,
        )
        if len(self.rgb_frames) > 1:
            self.rgb_frames[0].save(
                f"{self.out_dir}/rgb{tag(suffix)}.gif",
                save_all=True,
                append_images=self.rgb_frames[1:],
                duration=self.gif_duration,
                loop=0,
            )


def save_loss_plot(per_task_losses, suffix, out_dir="artifacts"):
    import matplotlib.pyplot as plt

    for task_idx in sorted(per_task_losses.keys()):
        if per_task_losses[task_idx]:
            plt.plot(per_task_losses[task_idx], linewidth=1, label=f"Task {task_idx}")
    plt.xlabel("Training step")
    plt.ylim(bottom=-0.05, top=1)
    plt.ylabel("Loss")
    plt.legend(fontsize=7)
    plt.savefig(f"{out_dir}/loss{tag(suffix)}.png")
    plt.close()


def save_value_matrix(
    mat,
    row_order,
    col_order,
    prefix,
    suffix,
    out_dir="artifacts",
    title="",
    xlabel="",
    ylabel="",
    cmap_colors=("red", "white"),
    vmin=0.0,
    vmax=1.0,
    fmt="{:.2f}",
    cbar_label="",
):
    """Save an (n_rows, n_cols) value matrix as a heatmap to
    {out_dir}/{prefix}{tag(suffix)}.png.

    Generalizes save_forgetting_matrix for arbitrary value matrices (accuracy,
    variance, neuron sharing, ...). Non-finite cells (NaN/inf) render white,
    every finite cell is annotated with fmt, and thin black cell borders are
    drawn. cmap_colors is a low->high color list; vmax=None autoscales to the
    finite maximum of the matrix.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    mat = np.asarray(mat, dtype=float)
    n_rows, n_cols = mat.shape
    if n_rows == 0 or n_cols == 0:
        return
    if vmax is None:
        finite = mat[np.isfinite(mat)]
        vmax = float(finite.max()) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-8

    cmap = LinearSegmentedColormap.from_list("vm", list(cmap_colors)).copy()
    cmap.set_bad(color="white")

    _, ax = plt.subplots(figsize=(max(4, n_cols * 0.45), max(4, n_rows * 0.45)))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([str(t) for t in col_order], fontsize=7)
    ax.set_yticklabels([str(t) for t in row_order], fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    fontsize = max(4, min(9, int(70 / max(max(n_rows, n_cols), 1))))
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            ax.text(
                j, i, fmt.format(v), ha="center", va="center",
                fontsize=fontsize, color="black",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}{tag(suffix)}.png", dpi=120)
    plt.close()


def save_forgetting_matrix(rows, row_order, col_order, suffix, out_dir="artifacts"):
    """Save accuracy matrix to {out_dir}/xfer{tag(suffix)}.png.

    rows[i] is a dict {task_idx: acc} captured after row_order[i] was first
    trained. Columns span col_order (typically the full task sequence), so
    prospective evals on not-yet-trained tasks fill the upper triangle.
    Entries with no data are rendered white.
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
    ax.set_xlabel("Evaluated task")
    ax.set_ylabel("After training task")
    ax.set_title("Accuracy (white=1, red=0)")

    # Thin black cell borders.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    # Per-cell accuracy text (only for filled cells).
    fontsize = max(4, min(9, int(70 / max(max(n_rows, n_cols), 1))))
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color="black",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/xfer{tag(suffix)}.png", dpi=120)
    plt.close()
