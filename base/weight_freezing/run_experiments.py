"""
run_experiments.py - Multi-task learning experiment for functional task networks.

Trains a single FTN on a stream of N tasks (the former "phase 1"), with
USAGE-DENSITY WEIGHT FREEZING IN PLACE OF REPLAY, and a backwards/prospective
forgetting monitor.

Design:
  - The FTN owns its mask and reconfiguration: `ftn(x, y)` reconfigures the
    mask on that batch and predicts; `ftn(x)` predicts with the current mask.
    This file only optimizes the weights. The core model and training loop
    are identical to base_rf with replay disabled.
  - No replay buffer, no task ids, no boundaries: freezing.UsageFreezer keeps
    one (hidden_size,) vector of batch-involvement counts and, each step,
    freezes the discretely-senior cohort of the current mask ("only train the
    least-globally-used neurons in the mask"). See freezing.py.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch

import config
import tasks
import utils
import viz
from freezing import UsageFreezer
from ftn import FTN

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
        domain_shift=Config.DOMAIN_SHIFT,
        domain_shift_stride=Config.DOMAIN_SHIFT_STRIDE,
        domain_dense_projection=Config.DOMAIN_DENSE_PROJECTION,
        domain_per_task_projection=Config.DOMAIN_PER_TASK_PROJECTION,
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
    Restores the training mask on exit so eval never leaks state into
    training (reconfigure() mutates ftn.m in place).
    """
    m_train = ftn.m.clone()
    results = {}
    for task_idx in task_indices:
        losses = []
        for _ in range(Config.N_EVAL_EPOCHS):
            x, y = data_gen(task_idx)
            ftn.reconfigure(x, y)
            with torch.no_grad():
                losses.append(ftn.loss(ftn(x), y).item())
        results[task_idx] = np.mean(losses)
    ftn.m.copy_(m_train)
    return results


def evaluate_backwards(ftn, data_gen, task_indices):
    """Reconfigure & eval on each task for N_RETRO_EVAL_BATCHES batches.

    Reconfigures per batch (or once per task when RECONFIG_EVERY == 0,
    matching base/'s CONFIGURE_PER_BATCH=False), then evaluates on a fresh
    holdout batch.
    Restores the training mask on exit so eval never leaks state into
    training (reconfigure() mutates ftn.m in place; without this, reactive
    mode would train under the last-evaluated task's support mask).
    Returns {task_idx: {"loss": mean_loss, "acc": mean_score}}.
    """
    m_train = ftn.m.clone()
    results = {}
    for task_idx in task_indices:
        losses, accs = [], []
        if Config.RECONFIG_EVERY == 0:
            ftn.reconfigure(*data_gen(task_idx))
        for _ in range(Config.N_RETRO_EVAL_BATCHES):
            if Config.RECONFIG_EVERY != 0:
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
    ftn.m.copy_(m_train)
    return results


# =============================================================================
# Training
# =============================================================================


def train(ftn, data_gen, task_indices, n_epochs, suffix, optimizer, freezer):
    """
    Multi-task training over the task stream, with visualization and the
    backwards/prospective forgetting monitor.

    Returns per_task_losses.
    """
    per_task_losses = {t: [] for t in task_indices}
    # RENDER_STEPS=False disables per-step mask rendering (GIFs); loss plots
    # and the forgetting/transfer matrices still save.
    animator = None
    if Config.RENDER_STEPS:
        animator = viz.MaskAnimator(
            hidden_size=ftn.hidden_size,
            block_dim=Config.SQUARE_DIM,
            gif_duration=Config.GIF_DURATION,
            gif_save_epoch=min(Config.GIF_SAVE_EPOCH, n_epochs) or 1,
        )

    # Reactive reconfiguration (RECONFIG_REACTIVE): reconfigure once on the
    # first training batch, then only when the batch loss under the current
    # mask exceeds the EMA mean by RECONFIG_REACTIVE_SD standard deviations.
    # State persists across task blocks — the mechanism has no boundary
    # knowledge. Triggers are suppressed for ~1/(1-EMA) batches after each
    # reconfig while the statistics re-form.
    reactive = Config.RECONFIG_REACTIVE
    r_sd = Config.RECONFIG_REACTIVE_SD
    r_alpha = Config.RECONFIG_REACTIVE_EMA
    r_warmup = int(np.ceil(1.0 / max(1.0 - r_alpha, 1e-8)))
    r_mean, r_var, r_since = None, 0.0, 0

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
            n_frozen_hist = []
            n_block_reconfigs = 0
            for i in range(Config.TRAIN_DURATION):
                x, y = data_gen(task)
                if i == 0:
                    print(x[0],y[0])

                if reactive:
                    # Loss-triggered reconfiguration. r_mean is None only on
                    # the very first training batch: do the one-time initial
                    # reconfig there.
                    if r_mean is None:
                        ftn.reconfigure(x, y)
                        n_block_reconfigs += 1
                        r_since = 0
                    pred = ftn(x)
                    loss = ftn.loss(pred, y)
                    li_r = loss.item()
                    sd = float(np.sqrt(max(r_var, 0.0)))
                    if (
                        r_mean is not None
                        and r_since >= r_warmup
                        and li_r > r_mean + r_sd * sd
                    ):
                        ftn.reconfigure(x, y)
                        n_block_reconfigs += 1
                        r_since = 0
                        pred = ftn(x)
                        loss = ftn.loss(pred, y)
                        li_r = loss.item()
                    # EMA of loss mean/variance under the trained mask (the
                    # pre-reconfig spike is never recorded).
                    if r_mean is None:
                        r_mean = li_r
                    else:
                        diff = li_r - r_mean
                        r_mean = r_alpha * r_mean + (1 - r_alpha) * li_r
                        r_var = r_alpha * r_var + (1 - r_alpha) * diff * diff
                    r_since += 1
                else:
                    # Reconfigure the mask every RECONFIG_EVERY batches; 0 =
                    # once at the start of each task block (CONFIGURE_PER_BATCH
                    # =False in base/). Otherwise predict with the current mask.
                    every = Config.RECONFIG_EVERY
                    reconfig = (i == 0) if every == 0 else (i % every == 0)
                    pred = ftn(x, y) if reconfig else ftn(x)
                    loss = ftn.loss(pred, y)

                if epoch == 0 and i == 0:
                    print(f"TASK {task} INITIAL_LOSS: {loss.item()}")

                # Freeze decision from counts BEFORE this batch is recorded.
                frz = freezer.frozen(ftn.m)
                freezer.record(ftn.m)
                n_frozen_hist.append(int(frz.sum().item()))

                optimizer.zero_grad()
                loss.backward()
                freezer.zero_frozen_grads(ftn, frz)
                optimizer.step()
                if animator:
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

            if animator:
                animator.record_task(task, ftn.m, suffix, snapshot=snapshot)
            per_task_losses[task].extend(loss_history)
            reactive_str = (
                f" | reactive_reconfigs={n_block_reconfigs}"
                f" loss_ema={r_mean:.4f} sd={np.sqrt(max(r_var, 0.0)):.4f}"
                if reactive
                else ""
            )
            print(
                f"  [{suffix}] task {task} freeze summary: "
                f"mean_frozen={np.mean(n_frozen_hist):.1f} "
                f"final_frozen={n_frozen_hist[-1]} | {freezer.stats(frz)}"
                f"{reactive_str}"
            )

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

        # Periodic plot saves (kept even with RENDER_STEPS=False; only the
        # per-step mask rendering is disabled)
        if snapshot:
            viz.save_loss_plot(per_task_losses, suffix)

        if animator:
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
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Override Config.FREEZE_MARGIN (<= 0 disables freezing).",
    )
    parser.add_argument(
        "--reconfig-every",
        type=int,
        default=None,
        help="Override Config.RECONFIG_EVERY (batches between mask "
        "reconfigurations; 0 = once per task block).",
    )
    parser.add_argument(
        "--render-steps",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override Config.RENDER_STEPS (0 = skip per-step mask GIFs and "
        "task vis; loss plots and forgetting/transfer matrices still save).",
    )
    args = parser.parse_args()
    RUN_TAG = f"_{args.name}" if args.name else ""

    Config = getattr(config, args.config)()
    if args.margin is not None:
        Config.FREEZE_MARGIN = args.margin if args.margin > 0 else None
    if args.reconfig_every is not None:
        Config.RECONFIG_EVERY = args.reconfig_every
    if args.render_steps is not None:
        Config.RENDER_STEPS = bool(args.render_steps)
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

    if Config.RENDER_STEPS:
        tasks.save_task_vis(Config)

    data_gen = make_data_gen(Config.SEED + 100)
    # With DOMAIN_SHIFT the input vector is wider than X_DIM; the generator
    # is the single source of truth for the model's input width.
    ftn = FTN(
        input_size=data_gen.total_input_dim,
        output_size=Config.Y_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        **Config.FTN_PARS,
    )
    optimizer = torch.optim.AdamW(
        ftn.parameters(), lr=Config.TRAIN_LR, weight_decay=Config.WEIGHT_DECAY
    )
    freezer = UsageFreezer(
        hidden_size=Config.HIDDEN_SIZE,
        margin=Config.FREEZE_MARGIN,
        lag=Config.FREEZE_LAG,
        min_plastic=Config.FREEZE_MIN_PLASTIC,
        low_q=Config.FREEZE_LOW_Q,
    )
    print(
        f"Freezing: margin={Config.FREEZE_MARGIN} lag={Config.FREEZE_LAG} "
        f"min_plastic={Config.FREEZE_MIN_PLASTIC} low_q={Config.FREEZE_LOW_Q}"
    )
    if Config.RECONFIG_REACTIVE:
        print(
            f"Reconfig: REACTIVE sd={Config.RECONFIG_REACTIVE_SD} "
            f"ema={Config.RECONFIG_REACTIVE_EMA} (RECONFIG_EVERY ignored)"
        )
    else:
        print(f"Reconfig: every={Config.RECONFIG_EVERY} (0 = per task block)")

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
        freezer,
    )
    mtl_eval = evaluate_tasks(ftn, data_gen, list(range(Config.N_TASK)))

    print(f"\n{'=' * 60}\nEXPERIMENT COMPLETE\n{'=' * 60}")
    print("MTL eval:", {k: f"{v:.4f}" for k, v in mtl_eval.items()})
    print(f"Chance-level loss: {ent:.4f}")
    print("\nOutputs saved to images/")
