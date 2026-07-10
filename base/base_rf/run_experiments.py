"""
run_experiments.py - Multi-task learning experiment for functional task networks.

Trains a single FTN on a stream of N tasks (the former "phase 1"), with
offline replay and a backwards/prospective forgetting monitor.

Design:
  - The FTN owns its mask and reconfiguration: `ftn(x, y)` reconfigures the
    mask on that batch and predicts; `ftn(x)` predicts with the current mask;
    `ftn(x, m=...)` predicts under a stored mask (replay). This file only
    optimizes the weights.
"""

import argparse
import os
from collections import defaultdict, deque

import numpy as np
import torch

import config
import tasks
import utils
import viz
from ftn import FTN

# =============================================================================
# Replay buffer
# =============================================================================


class ReplayBuffer:
    """Stores (data, mask) pairs for offline replay.

    If batch_size is set, each stored batch is subsampled to batch_size rows
    once at store time.
    """

    def __init__(self, maxlen, batch_size=None):
        self.data = deque(maxlen=maxlen)
        self.masks = deque(maxlen=maxlen)
        self.batch_size = batch_size

    def add(self, dat, m):
        x, y = dat
        n = x.shape[0]
        if self.batch_size is not None and self.batch_size < n:
            idx = torch.randperm(n, device=x.device)[: self.batch_size]
            dat = (x[idx], y[idx])
        self.data.append(dat)
        self.masks.append(m.clone())

    def get(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


# =============================================================================
# Helpers
# =============================================================================


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
        rng=tasks.make_rng(seed),
    )


def accuracy(pred, y):
    """Classification accuracy, or 1 - MSE/var(y) clipped to [0, 1]."""
    if Config.CLASSIFICATION:
        target = torch.where(y == 1)[1]
        return (pred.argmax(-1) == target).float().mean().item()
    mse = torch.nn.functional.mse_loss(pred.squeeze(), y.squeeze()).item()
    var = float(y.float().var().item()) if y.numel() > 1 else 1.0
    return max(0.0, 1.0 - mse / max(var, 1e-8))


def evaluate_tasks(ftn, data_gen, task_indices):
    """Quick loss evaluation over multiple batches. Returns {task_idx: mean_loss}.

    Reconfigures per batch and evaluates on the same batch.
    """
    results = {}
    for task_idx in task_indices:
        losses = []
        for _ in range(Config.N_EVAL_EPOCHS):
            x, y = data_gen(task_idx)
            ftn.reconfigure(x, y)
            with torch.no_grad():
                losses.append(ftn.loss(ftn(x), y).item())
        results[task_idx] = np.mean(losses)
    return results


def evaluate_backwards(ftn, data_gen, task_indices):
    """Reconfigure & eval on each task for N_RETRO_EVAL_BATCHES batches.

    Reconfigures per batch, then evaluates on a fresh holdout batch.
    Returns {task_idx: {"loss": mean_loss, "acc": mean_score}}.
    """
    results = {}
    for task_idx in task_indices:
        losses, accs = [], []
        for _ in range(Config.N_RETRO_EVAL_BATCHES):
            ftn.reconfigure(*data_gen(task_idx))
            # Eval on a fresh holdout batch — not the one used to reconfigure.
            x, y = data_gen(task_idx)
            with torch.no_grad():
                pred = ftn(x)
                losses.append(ftn.loss(pred, y).item())
                accs.append(accuracy(pred, y))
        results[task_idx] = {
            "loss": float(np.mean(losses)),
            "acc": float(np.mean(accs)),
        }
    return results


def replay_step(ftn, replay_buffer):
    """Accumulate gradients on stored batches whose masks overlap the current one."""
    if len(replay_buffer) == 0:
        return
    stored = torch.stack(list(replay_buffer.masks))
    corrs = (stored * ftn.m).sum(-1) / ftn.m.sum()
    keep = (corrs > 0.2) & (corrs <= 0.8)
    overlap = torch.where(keep)[0]
    if len(overlap) == 0:
        return
    choice = torch.randint(0, len(overlap), (Config.N_REPLAY_SAMPLES,))
    for idx in overlap[choice]:
        x_r, y_r = replay_buffer.get(idx.item())
        m_r = replay_buffer.masks[idx.item()]
        r_pred = ftn(x_r, m=m_r)
        ftn.loss(r_pred, y_r).backward()


# =============================================================================
# Training
# =============================================================================


def train(ftn, data_gen, task_indices, n_epochs, suffix, optimizer, replay_buffer):
    """
    Multi-task training over the task stream, with visualization and the
    backwards/prospective forgetting monitor.

    Returns per_task_losses.
    """
    per_task_losses = {t: [] for t in task_indices}
    animator = viz.MaskAnimator(
        hidden_size=ftn.hidden_size,
        block_dim=Config.SQUARE_DIM,
        gif_duration=Config.GIF_DURATION,
        gif_save_epoch=min(Config.GIF_SAVE_EPOCH, n_epochs) or 1,
    )

    seen_tasks = []
    # Cumulative lower-triangular forgetting matrix: one row per first-time task.
    forgetting_rows = []
    for epoch in range(n_epochs):
        snapshot = epoch % Config.PLOT_EVERY == 0
        for task in task_indices:
            loss_history = []
            # Reset Adam (m, v) per task so optimizer state from a prior task
            # can't drift parameters whose gradient is zero under the new mask.
            optimizer.state = defaultdict(dict)
            for i in range(Config.TRAIN_DURATION):
                x, y = data_gen(task)

                # Forward pass; reconfigures the mask on this batch internally.
                pred = ftn(x, y)
                loss = ftn.loss(pred, y)

                if i < Config.N_SAMPLE_PER_TASK:
                    replay_buffer.add((x, y), ftn.m)

                if epoch == 0 and i == 0:
                    print(f"TASK {task} INITIAL_LOSS: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                replay_step(ftn, replay_buffer)
                optimizer.step()
                animator.record_step(ftn.m, snapshot=snapshot)

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

            animator.record_task(task, ftn.m, suffix, snapshot=snapshot)
            per_task_losses[task].extend(loss_history)

            # Backwards eval: report loss on every task seen so far. If
            # EVAL_PROSPECTIVE is set, also eval on tasks not yet trained so
            # the upper triangle of the matrix gets filled.
            newly_seen = task not in seen_tasks
            if newly_seen:
                seen_tasks.append(task)
            if Config.EVAL_BACKWARDS:
                prospective = Config.EVAL_PROSPECTIVE
                eval_set = list(task_indices) if prospective else list(seen_tasks)
                bw = evaluate_backwards(ftn, data_gen, eval_set)
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
                    viz.save_forgetting_matrix(
                        forgetting_rows, seen_tasks, col_order, suffix
                    )

        # Periodic plot saves
        if snapshot:
            viz.save_loss_plot(per_task_losses, suffix)

        animator.maybe_save_gifs(epoch, suffix)

    return per_task_losses


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
    parser.add_argument(
        "--config",
        default="Default",
        help="Config class name from config.py (e.g. Default, Smoke).",
    )
    args = parser.parse_args()
    RUN_TAG = f"_{args.name}" if args.name else ""

    Config = getattr(config, args.config)()
    DataGen = tasks.DataGenSine

    print("Running:")
    print(f"Config: {type(Config).__name__}")
    print(f"DataGen: {DataGen.__name__}")
    if RUN_TAG:
        print(f"Run tag: {RUN_TAG}")

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("images", exist_ok=True)

    if Config.SEED is not None:
        torch.manual_seed(Config.SEED)

    tasks.save_task_vis(Config)

    data_gen = make_data_gen(Config.SEED + 100)
    ftn = FTN(
        input_size=Config.X_DIM,
        output_size=Config.Y_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        **Config.FTN_PARS,
    )
    optimizer = torch.optim.AdamW(
        ftn.parameters(), lr=Config.TRAIN_LR, weight_decay=Config.WEIGHT_DECAY
    )
    replay = ReplayBuffer(
        maxlen=Config.REPLAY_MAXLEN,
        batch_size=Config.REPLAY_BATCH_SIZE,
    )

    ent = -Config.Y_DIM * np.log(1.0 / Config.Y_DIM) * (1.0 / Config.Y_DIM)
    print(
        f"Model: {utils.count_parameters(ftn)} params, "
        f"hidden={Config.HIDDEN_SIZE} ({Config.SQUARE_DIM}x{Config.SQUARE_DIM})"
    )
    print(f"Chance-level loss: {ent:.4f}")

    print(
        f"\n{'=' * 60}\nMulti-task learning ({Config.N_TASK} tasks)\n{'=' * 60}"
    )
    train(
        ftn,
        data_gen,
        list(range(Config.N_TASK)),
        Config.N_EPOCHS,
        f"p1_mtl{RUN_TAG}",
        optimizer,
        replay,
    )
    mtl_eval = evaluate_tasks(ftn, data_gen, list(range(Config.N_TASK)))

    print(f"\n{'=' * 60}\nEXPERIMENT COMPLETE\n{'=' * 60}")
    print("MTL eval:", {k: f"{v:.4f}" for k, v in mtl_eval.items()})
    print(f"Chance-level loss: {ent:.4f}")
    print("\nOutputs saved to images/")
