"""
run_experiments_2.py - Experiment pipeline for functional task networks.

Phases:
  0: Pre-train task 0 (baseline for long-term recovery)
  1: Multi-task learning on N tasks
  2: Continual learning (2a: OOD neuron combos, 2b: new data generator)
  3: Task 0 long-term recovery report
  4: Neurogenesis (expand model, train on new generator)

Design:
  - All phases share a single model, optimizer, and replay buffer.
  - run_phase() handles both training and evaluation via the `train` flag.
  - The replay buffer persists across phases for continual learning.
  - Set PHASES to any subset of [0,1,2,3,4] to run selected phases.
"""

import argparse
import os
import random
from collections import defaultdict, deque

import config
import masks
import models
import numpy as np
import tasks
import torch
import torch.nn.functional as F
import utils
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

Model = models.NeuroModel
DataGen = tasks.DataGenSine
MaskConfigurer = masks.KWTA
Config = config.Default()


# Generate the task images
import subprocess
import sys

subprocess.run([sys.executable, "tasks.py"], check=True)

# =============================================================================
# Replay buffer (persists across phases)
# =============================================================================


class ReplayBuffer:
    """Stores (data, mask) pairs for offline replay across all experiment phases."""

    def __init__(self, maxlen):
        self.data = deque(maxlen=maxlen)
        self.masks = deque(maxlen=maxlen)

    def add(self, dat, m):
        self.data.append(dat)
        self.masks.append(m.clone())

    def resize_masks(self, new_size):
        """Zero-pad stored masks after model expansion (neurogenesis)."""
        for i in range(len(self.masks)):
            old = self.masks[i]
            if old.shape[0] < new_size:
                padded = torch.zeros(new_size, device=old.device)
                padded[: old.shape[0]] = old
                self.masks[i] = padded

    def __len__(self):
        return len(self.data)


# =============================================================================
# Helpers
# =============================================================================
def wrap_datagen_ood(dg: DataGen):
    dg.m = torch.bernoulli(torch.ones_like(dg.m) / Config.N_TASK)
    return dg


def make_rng(seed):
    """Create a local torch.Generator on the current default device."""
    device = torch.empty(0).device
    return torch.Generator(device=device).manual_seed(seed)


def make_data_gen(seed):
    """Create a DataGen with its own local RNG (no global state mutation)."""
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
    """Reconfigure & eval on each prior task for N_RETRO_EVAL_BATCHES batches.

    Respects CONFIGURE_PER_BATCH (reconfig every batch vs. once per task).
    Returns {task_idx: {"loss": mean_loss, "acc": mean_score}}.
    `acc` is classification accuracy for CLASSIFICATION, otherwise
    1 - MSE/var(y), clipped to [0, 1].
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
            # Eval on a fresh holdout batch — not the one used to reconfigure.
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


def expand_model(model, new_hidden_size):
    """Create a larger model, copying old weights into the corresponding submatrix."""
    new_model = Model(
        input_size=Config.X_DIM,
        output_size=Config.Y_DIM,
        hidden_size=new_hidden_size,
        **Config.MODEL_PARS,
    )
    with torch.no_grad():
        for old_fc, new_fc in zip(model.fc, new_model.fc):
            oh, ow = old_fc.weight.shape
            new_fc.weight[:oh, :ow] = old_fc.weight
            new_fc.bias[:oh] = old_fc.bias
        ow = model.out.weight.shape[1]
        new_model.out.weight[:, :ow] = model.out.weight
        new_model.out.bias[:] = model.out.bias
    return new_model


# =============================================================================
# Plotting
# =============================================================================


def save_loss_plot(per_task_losses, suffix):
    import matplotlib.pyplot as plt

    # ent = -Config.Y_DIM * np.log(1.0 / Config.Y_DIM) * (1.0 / Config.Y_DIM)
    for task_idx in sorted(per_task_losses.keys()):
        if per_task_losses[task_idx]:
            plt.plot(per_task_losses[task_idx], linewidth=1, label=f"Task {task_idx}")
    plt.xlabel("Training step")
    # plt.ylim(bottom=-0.05, top=ent + 0.25)
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
    """Save accuracy matrix to images/forgetting_{suffix}.png.

    rows[i] is a dict {task_idx: acc} captured after row_order[i] was first
    trained. Columns span `col_order` (typically the full task sequence), so
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
# Core: unified train/eval phase
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
    """
    Run a training or evaluation phase with visualization.

    When train=True: optimizes model weights with offline replay.
    When train=False: forward-only passes (for visualization / loss tracking).

    Returns (per_task_losses, eval_snapshots).
    """
    if block_dim is None:
        block_dim = Config.SQUARE_DIM

    # Reset Adam (m, v) at each training phase so optimizer state from a prior
    # phase can't drift parameters whose gradient is zero under the new mask
    # (matches the per-task reset in project_c_4/trainer.py).
    if train:
        optimizer.state = defaultdict(dict)

    hidden = mask_configurer.hidden_size
    n_blocks = hidden // (block_dim**2)
    gh, gw = block_dim, n_blocks * block_dim

    # Loss tracking
    per_task_losses = {t: [] for t in task_indices}
    eval_snapshots = []

    # Visualization state
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
    # Cumulative lower-triangular forgetting matrix: one row per first-time task.
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

                # Clear stale gradients (from mask reconfiguration or prior step)
                optimizer.zero_grad()

                # Forward pass (no graph when not training)
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
                # if epoch == 5:
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
                                x_r, y_r = replay_buffer.data[idx.item()]
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
                    m_2d = masks.mask_to_grid(m.detach(), block_dim)
                    m_ema[:, :, 0] = torch.max(0.95 * m_ema[:, :, 0], m_2d)
                    m_ema[:, :, 1] = torch.max(0.30 * m_ema[:, :, 1], m_2d)
                    m_ema[:, :, 2] = torch.max(0.975 * m_ema[:, :, 2], m_2d)

                    if epoch % Config.PLOT_EVERY == 0:
                        frame = Image.fromarray(
                            (m_ema.cpu().numpy() * 255).astype(np.uint8), mode="RGB"
                        )
                        frames.append(frame)

                # EMA loss tracking
                li = loss.item()
                if i == 0:
                    loss_smooth = li
                else:
                    loss_smooth = (
                        Config.LOSS_EMA_SMOOTHING * loss_smooth
                        + (1 - Config.LOSS_EMA_SMOOTHING) * li
                    )
                loss_history.append(loss_smooth)

            # RGB channel assignment: task%3 → 0:blue, 1:red, 2:green
            ch = task % 3
            if ch == 0:
                blue_net = m.clone()
            elif ch == 1:
                red_net = m.clone()
            else:
                green_net = m.clone()

            if epoch % Config.PLOT_EVERY == 0:
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

            # Backwards eval: report loss on every task seen so far. If
            # EVAL_PROSPECTIVE is set, also eval on tasks not yet trained so
            # the upper triangle of the matrix gets filled.
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
                    f"after task {task}): loss[{loss_str}] | "
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

        # Periodic plot saves
        if epoch % Config.PLOT_EVERY == 0:
            save_loss_plot(per_task_losses, suffix)

        # Periodic held-out evaluation
        if eval_gen is not None and eval_tasks is not None:
            if epoch % Config.PLOT_EVERY == 0:
                mc = eval_mask or mask_configurer
                eval_snapshots.append(evaluate_tasks(model, eval_gen, mc, eval_tasks))

        # GIF saves
        if (epoch + 1) % gif_save_epoch == 0 and len(frames) > 1:
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
        help="Tag appended to output filenames (e.g. 'kwta_seed1' → 'p1_mtl_kwta_seed1.png').",
    )
    args = parser.parse_args()
    RUN_TAG = f"_{args.name}" if args.name else ""

    print("Running:")
    print(Model.__name__)
    print(DataGen.__name__)
    print(MaskConfigurer.__name__)
    if RUN_TAG:
        print(f"Run tag: {RUN_TAG}")

    torch.set_default_device("cuda")
    os.makedirs("images", exist_ok=True)

    if Config.SEED is not None:
        torch.manual_seed(Config.SEED)

    # --- Shared setup (always created, regardless of PHASES) ---
    task0_gen = make_data_gen(Config.SEED)
    data_gen = make_data_gen(Config.SEED + 100)
    ood_gen_a = wrap_datagen_ood(make_data_gen(Config.SEED + 100))
    ood_gen_b = make_data_gen(Config.SEED + 1000)
    ng_gen = make_data_gen(Config.SEED + 2000)

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
    replay = ReplayBuffer(maxlen=Config.REPLAY_MAXLEN)

    ent = -Config.Y_DIM * np.log(1.0 / Config.Y_DIM) * (1.0 / Config.Y_DIM)
    print(
        f"Model: {utils.count_parameters(model)} params, "
        f"hidden={Config.HIDDEN_SIZE} ({Config.SQUARE_DIM}x{Config.SQUARE_DIM})"
    )
    print(f"Chance-level loss: {ent:.4f}")
    print(f"Active phases: {Config.PHASES}")

    task0_trajectory = {}

    # ----- Phase 0: Pre-train task 0 -----
    if 0 in Config.PHASES:
        print(f"\n{'=' * 60}\nPHASE 0: Pre-train task 0\n{'=' * 60}")
        run_phase(
            model,
            task0_gen,
            mask_cfg,
            [0],
            Config.N_EPOCHS,
            f"p0_pretrain{RUN_TAG}",
            optimizer,
            replay,
            train=True,
        )
        task0_trajectory["baseline"] = evaluate_tasks(model, task0_gen, mask_cfg, [0])[
            0
        ]
        print(f"Task 0 baseline loss: {task0_trajectory['baseline']:.4f}")

    # ----- Phase 1: Multi-task learning -----
    if 1 in Config.PHASES:
        print(
            f"\n{'=' * 60}\nPHASE 1: Multi-task learning "
            f"({Config.N_TASK} tasks)\n{'=' * 60}"
        )
        run_phase(
            model,
            data_gen,
            mask_cfg,
            list(range(Config.N_TASK)),
            Config.N_EPOCHS,
            f"p1_mtl{RUN_TAG}",
            optimizer,
            replay,
            train=True,
        )
        mtl_eval = evaluate_tasks(model, data_gen, mask_cfg, list(range(Config.N_TASK)))
        print("MTL eval:", {k: f"{v:.4f}" for k, v in mtl_eval.items()})
        task0_trajectory["after_mtl"] = evaluate_tasks(model, task0_gen, mask_cfg, [0])[
            0
        ]

    # ----- Phase 2: Continual learning with OOD tasks -----
    if 2 in Config.PHASES:
        # 2a: OOD neuron combinations (same feature space)
        print(f"\n{'=' * 60}\nPHASE 2a: CL - OOD neuron combinations\n{'=' * 60}")
        mask_kwta = masks.KWTA(Config.HIDDEN_SIZE, Config.SQUARE_DIM)
        _, snaps_a = run_phase(
            model,
            ood_gen_a,
            mask_kwta,
            list(range(Config.N_OOD_TASKS)),
            Config.N_EVAL_EPOCHS,
            f"p2a_ood_neurons{RUN_TAG}",
            optimizer,
            replay,
            train=False,
            eval_gen=data_gen,
            eval_tasks=list(range(Config.N_TASK)),
        )
        save_eval_snapshots(snaps_a, f"p2a_holdout{RUN_TAG}")
        cl_a_eval = evaluate_tasks(
            model, data_gen, mask_cfg, list(range(Config.N_TASK))
        )
        print("MTL tasks after CL-a:", {k: f"{v:.4f}" for k, v in cl_a_eval.items()})
        task0_trajectory["after_cl_a"] = evaluate_tasks(
            model, task0_gen, mask_cfg, [0]
        )[0]
        print(f"Task 0 after CL-a: {task0_trajectory['after_cl_a']:.4f}")

        # 2b: New data generator (different seed)
        print(f"\n{'=' * 60}\nPHASE 2b: CL - new data generator\n{'=' * 60}")
        _, snaps_b = run_phase(
            model,
            ood_gen_b,
            mask_kwta,
            list(range(Config.N_OOD_TASKS)),
            Config.N_EVAL_EPOCHS,
            f"p2b_ood_data{RUN_TAG}",
            optimizer,
            replay,
            train=False,
            eval_gen=data_gen,
            eval_tasks=list(range(Config.N_TASK)),
        )
        save_eval_snapshots(snaps_b, f"p2b_holdout{RUN_TAG}")
        cl_b_eval = evaluate_tasks(
            model, data_gen, mask_cfg, list(range(Config.N_TASK))
        )
        print("MTL tasks after CL-b:", {k: f"{v:.4f}" for k, v in cl_b_eval.items()})
        task0_trajectory["after_cl_b"] = evaluate_tasks(
            model, task0_gen, mask_cfg, [0]
        )[0]
        print(f"Task 0 after CL-b: {task0_trajectory['after_cl_b']:.4f}")

    # ----- Phase 3: Task 0 long-term recovery report -----
    if 3 in Config.PHASES:
        print(f"\n{'=' * 60}\nPHASE 3: Task 0 recovery\n{'=' * 60}")
        if task0_trajectory:
            for label, val in task0_trajectory.items():
                print(f"  {label:20s}: {val:.4f}")
            save_eval_bar(
                {i: v for i, v in enumerate(task0_trajectory.values())},
                {i: k for i, k in enumerate(task0_trajectory.keys())},
                f"task0_recovery{RUN_TAG}",
            )
        else:
            print("  (no trajectory data - earlier phases were skipped)")

    # ----- Phase 4: Neurogenesis -----
    if 4 in Config.PHASES:
        print(f"\n{'=' * 60}\nPHASE 4: Neurogenesis\n{'=' * 60}")

        NEW_HIDDEN = Config.HIDDEN_SIZE * 2
        print(f"Expanding: {Config.HIDDEN_SIZE} -> {NEW_HIDDEN} neurons")

        model = expand_model(model, NEW_HIDDEN)
        replay.resize_masks(NEW_HIDDEN)
        mask_cfg = MaskConfigurer(
            hidden_size=NEW_HIDDEN,
            square_dim=Config.SQUARE_DIM,
            classification=Config.CLASSIFICATION,
            **Config.MASK_PARS,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.TRAIN_LR,
            weight_decay=Config.WEIGHT_DECAY,
        )
        print(f"Expanded model: {utils.count_parameters(model)} params")

        # Train on new generator
        run_phase(
            model,
            ng_gen,
            mask_cfg,
            list(range(Config.N_TASK)),
            Config.N_EPOCHS,
            f"p4_ng_new{RUN_TAG}",
            optimizer,
            replay,
            train=True,
        )

        # Re-train original tasks on expanded model
        run_phase(
            model,
            data_gen,
            mask_cfg,
            list(range(Config.N_TASK)),
            Config.N_EPOCHS,
            f"p4_ng_orig{RUN_TAG}",
            optimizer,
            replay,
            train=True,
        )

        # Mask allocation analysis
        print("\nMask allocation (new-gen tasks):")
        m_probe = torch.zeros(NEW_HIDDEN)
        for t in range(Config.N_TASK):
            m_probe = mask_cfg(model, ng_gen(t), m_probe)
            old = m_probe[: Config.HIDDEN_SIZE].sum().item()
            new = m_probe[Config.HIDDEN_SIZE :].sum().item()
            total = m_probe.sum().item()
            print(
                f"  Task {t}: old={old:.0f} new={new:.0f} "
                f"new_frac={new / max(total, 1):.1%}"
            )

        print("\nMask allocation (orig tasks on expanded model):")
        ng_eval = evaluate_tasks(model, data_gen, mask_cfg, list(range(Config.N_TASK)))
        m_probe = torch.zeros(NEW_HIDDEN)
        for t in range(Config.N_TASK):
            m_probe = mask_cfg(model, data_gen(t), m_probe)
            old = m_probe[: Config.HIDDEN_SIZE].sum().item()
            new = m_probe[Config.HIDDEN_SIZE :].sum().item()
            total = m_probe.sum().item()
            print(
                f"  Task {t}: old={old:.0f} new={new:.0f} "
                f"old_frac={old / max(total, 1):.1%} loss={ng_eval[t]:.4f}"
            )

        # Neurogenesis mask visualization
        import matplotlib.pyplot as plt

        m_viz = torch.zeros(NEW_HIDDEN)
        fig, axes = plt.subplots(Config.N_TASK, 3, figsize=(12, 3 * Config.N_TASK))
        if Config.N_TASK == 1:
            axes = axes[np.newaxis, :]
        for t in range(Config.N_TASK):
            m_viz = mask_cfg(model, ng_gen(t), m_viz)
            full = masks.mask_to_grid(m_viz.detach(), Config.SQUARE_DIM).cpu().numpy()
            axes[t, 0].imshow(full[:, : Config.SQUARE_DIM], cmap="hot", vmin=0, vmax=1)
            axes[t, 0].set_title(f"Task {t} - Old neurons")
            axes[t, 1].imshow(full[:, Config.SQUARE_DIM :], cmap="hot", vmin=0, vmax=1)
            axes[t, 1].set_title(f"Task {t} - New neurons")
            axes[t, 2].imshow(full, cmap="hot", vmin=0, vmax=1)
            axes[t, 2].set_title(f"Task {t} - Full mask")
        plt.tight_layout()
        plt.savefig(f"images/ng_mask_allocation{RUN_TAG}.png", dpi=150)
        plt.close()

    # ----- Summary -----
    print(f"\n{'=' * 60}\nEXPERIMENT COMPLETE\n{'=' * 60}")
    if task0_trajectory:
        print("\nTask 0 trajectory:")
        for label, val in task0_trajectory.items():
            print(f"  {label:20s}: {val:.4f}")
        print(f"  {'(chance level)':20s}: {ent:.4f}")
    print("\nOutputs saved to images/")
