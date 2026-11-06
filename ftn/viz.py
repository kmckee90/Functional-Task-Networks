"""
viz.py - Visualization for FTN experiments.

All plotting/animation lives here, deliberately outside the FTN class:
  - MaskAnimator: mask-activity GIFs (EMA grayscale + per-task RGB rasters)
  - save_loss_plot: per-task training-loss curves
  - save_forgetting_matrix: retro/prospective accuracy matrix
"""

import colorsys

import numpy as np
import torch
from PIL import Image

from ftn import mask_to_grid
from spatial_permute import spatial_permutation


def tag(suffix):
    """Filename tag: '_suffix', or '' when suffix is empty (no dangling '_')."""
    return f"_{suffix}" if suffix else ""


def task_color(task):
    """Stable categorical color; golden-angle hues separate neighboring tasks."""
    hue = (0.60 + task * 0.6180339887498949) % 1.0
    return tuple(round(255 * c) for c in colorsys.hsv_to_rgb(hue, 0.8, 1.0))


class MaskAnimator:
    """
    Tracks mask activity during a training phase and writes GIF/PNG rasters.

    - record_step(m): update the EMA raster each training step; optionally
      snapshot it as a GIF frame every bw_frame_skip steps.
    - record_task(task, m): retain each task's latest mask, color it with a
      distinct categorical hue, and blend colors where task masks overlap.
    - maybe_save_gifs(epoch, suffix): flush GIFs every gif_save_epoch epochs.
    """

    def __init__(self, hidden_size, block_dim, gif_duration, gif_save_epoch,
                 out_dir="artifacts", ema_scale=1.0, bw_frame_skip=1,
                 spatial_permute=False, spatial_options=None):
        if bw_frame_skip < 1:
            raise ValueError("bw_frame_skip must be at least 1")
        self.block_dim = block_dim
        self.gif_duration = gif_duration
        self.gif_save_epoch = gif_save_epoch
        self.out_dir = out_dir
        self.bw_frame_skip = bw_frame_skip
        self._step_count = 0
        self.task_masks = {}
        self.spatial_permute = spatial_permute
        self.spatial_options = dict(spatial_options or {})
        self.permutation = np.arange(hidden_size)
        self._spatial_initialized = False

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

        blank = Image.fromarray(np.zeros((gh, gw, 3), dtype=np.uint8), mode="RGB")
        self.frames = [blank.copy() for _ in range(2)]
        self.rgb_frames = [blank.copy() for _ in range(10)]

    def _display(self, raster):
        """Gather spatial axes only; EMA and archived masks keep neuron IDs."""
        if not self.spatial_permute:
            return raster
        shape = raster.shape
        indices = torch.as_tensor(self.permutation, device=raster.device)
        return raster.reshape(-1, *shape[2:])[indices].reshape(shape)

    def _bw_frame(self):
        return Image.fromarray(
            (self._display(self.m_ema).cpu().numpy() * 255).astype(np.uint8),
            mode="RGB",
        )

    def record_step(self, m, snapshot=True):
        """Per-step EMA update of mask activity; snapshot appends a GIF frame."""
        m_2d = mask_to_grid(m.detach(), self.block_dim)
        for c, decay in enumerate(self.decay):
            self.m_ema[:, :, c] = torch.max(decay * self.m_ema[:, :, c], m_2d)

        self._step_count += 1
        if snapshot and self._step_count % self.bw_frame_skip == 0:
            self.frames.append(self._bw_frame())

    def record_task(self, task, m, suffix, snapshot=True):
        """Save the latest mask of each task, including tasks beyond the first three."""
        self.task_masks[task] = m.detach().clone()

        if self.spatial_permute:
            grids = np.stack([
                mask_to_grid(self.task_masks[t], self.block_dim).cpu().numpy().ravel()
                for t in sorted(self.task_masks)
            ])
            self.permutation = spatial_permutation(
                grids, tuple(self.m_ema.shape[:2]),
                self.permutation if self._spatial_initialized else None,
                **self.spatial_options,
            )
            self._spatial_initialized = True
            # Show the updated layout even at the final block of a run. Older
            # frames retain the layout in effect when they were recorded.
            if snapshot:
                self.frames.append(self._bw_frame())

        if snapshot:
            tasks = sorted(self.task_masks)
            masks = torch.stack([mask_to_grid(self.task_masks[t], self.block_dim)
                                 for t in tasks]).float()
            colors = torch.tensor([task_color(t) for t in tasks],
                                  device=masks.device, dtype=masks.dtype)
            # A neuron shared by tasks displays their mean color. Inactive
            # neurons remain black; no task disappears when another is added.
            weights = masks.unsqueeze(-1)
            rgb = (weights * colors[:, None, None, :]).sum(0)
            rgb = rgb / weights.sum(0).clamp_min(1)
            raster = self._display(rgb).clamp(0, 255).byte().cpu().numpy()
            self.rgb_frames.append(Image.fromarray(raster, mode="RGB"))
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

    n_rows = len(row_order)
    n_cols = len(col_order)
    if n_rows == 0 or n_cols == 0:
        return
    mat = np.full((n_rows, n_cols), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, t in enumerate(col_order):
            if t in row:
                mat[i, j] = row[t]

    cmap = plt.get_cmap("viridis").copy()
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
    ax.set_title("Accuracy (viridis, 0--1)")

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
