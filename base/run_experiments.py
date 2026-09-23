"""
run_experiments.py - Multi-task learning experiment for functional task networks.

Trains a single FTN on a stream of N tasks (the former "phase 1"), with
offline replay and a backwards/prospective forgetting monitor.

v2 = base_rf + domain-shift options (see tasks.py), reconfiguration
scheduling (RECONFIG_EVERY: per-batch / every-N / once per task block) and
reactive loss-triggered reconfiguration (RECONFIG_REACTIVE), and the
SmoothKWTA mask parameters exposed through FTN_PARS. No weight freezing.

Design:
  - The FTN owns its mask and reconfiguration: `ftn(x, y)` reconfigures the
    mask on that batch and predicts; `ftn(x)` predicts with the current mask;
    `ftn(x, m=...)` predicts under a stored mask (replay). This file only
    optimizes the weights and decides *when* to reconfigure.
"""

import argparse
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F

import config
import metrics as metrics_mod
import task_comp
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


class NoiseProbeBuffer:
    """Stores frozen (noise, generated_output) probes + their masks.

    Structurally this is the replay buffer with two substitutions: the input is
    random noise instead of task data, and the target is the network's OWN
    response to that noise, captured (detached) at store time — a fixed
    fingerprint of the function while the mask's task still lives in the weights.
    Both are frozen so that when a later, overlapping task overwrites the shared
    neurons, the stored target still represents the OLD function instead of
    drifting with the overwrite.
    """

    def __init__(self, maxlen):
        self.data = deque(maxlen=maxlen)
        self.masks = deque(maxlen=maxlen)

    def add(self, dat, m):
        # dat = (x_noise, y_noise), both already detached at generation time.
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
    """Create the task's DataGen with its own local RNG (no global state
    mutation). TASK selects the sine or composition generator; both expose the
    same data_gen(task_idx) -> (x, y) / .total_input_dim / .output_dim API."""
    if getattr(Config, "TASK", "sine").lower() == "compose":
        return task_comp.DataGenComposition(
            n_tasks=Config.N_TASK,
            n_samples=Config.T,
            rng=tasks.make_rng(seed),
            seed=Config.SEED,
            **Config.COMP_PARS,
        )
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
    """Per-cell binary accuracy (BINARY), single-label accuracy
    (CLASSIFICATION), or 1 - MSE/var(y) clipped to [0, 1] (regression)."""
    if getattr(Config, "BINARY", False):
        # pred are logits; threshold at 0 (sigmoid > 0.5). Fraction of cells
        # matching the {0,1} target, averaged over cells and batch.
        return ((pred > 0) == (y > 0.5)).float().mean().item()
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
            ftn._reconfig(x, y)
            with torch.no_grad():
                losses.append(ftn.loss(ftn(x), y).item())
        results[task_idx] = np.mean(losses)
    ftn.m.copy_(m_train)
    return results


def evaluate_backwards(ftn, data_gen, task_indices):
    """Reconfigure & eval on each task for N_RETRO_EVAL_BATCHES batches.

    Reconfigures per batch (or once per task when RECONFIG_EVERY == 0,
    matching base/'s CONFIGURE_PER_BATCH=False), then evaluates on a fresh
    holdout batch. EVAL_RECONFIG_STEPS overrides the training search length
    for both backwards and prospective tasks when set.
    Restores the training mask on exit so eval never leaks state into
    training (reconfigure() mutates ftn.m in place; without this, reactive
    mode would train under the last-evaluated task's support mask).
    Returns {task_idx: {"loss": mean_loss, "acc": mean_score}}.
    """
    m_train = ftn.m.clone()
    eval_steps = Config.EVAL_RECONFIG_STEPS
    train_steps = ftn.reconfig_steps
    primary_spec = getattr(ftn, "spec1", None)
    train_spec_steps = primary_spec.steps if primary_spec is not None else None
    if eval_steps is not None:
        ftn.reconfig_steps = eval_steps
        if primary_spec is not None:
            primary_spec.steps = eval_steps
    results = {}
    try:
        for task_idx in task_indices:
            losses, accs = [], []
            if Config.RECONFIG_EVERY == 0:
                ftn._reconfig(*data_gen(task_idx))
            for _ in range(Config.N_RETRO_EVAL_BATCHES):
                if Config.RECONFIG_EVERY != 0:
                    ftn._reconfig(*data_gen(task_idx))
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
    finally:
        ftn.m.copy_(m_train)
        ftn.reconfig_steps = train_steps
        if primary_spec is not None:
            primary_spec.steps = train_spec_steps
    return results


def entropy_tests(ftn, data_gen, task_indices, suffix, out_dir="artifacts"):
    """Post-hoc out-of-distribution subspace probe (ENTROPY_TESTS).

    For every ordered task pair (i, j): reconfigure on a fresh task-i batch to
    recover task i's mask m_i, then run task j's inputs through the network UNDER
    m_i and score the output against task j's target. This asks how the subspace
    trained on task i behaves when driven by task j's input distribution (out of
    its trained domain) — the diagonal (i == j) is the in-distribution
    reference.

    Repeated for ENTROPY_TEST_TRIALS trials. Each trial reconfigures a fresh
    task-i mask on a new task-i batch and draws fresh task-j data, so the
    variance matrix reflects both mask and data resampling. Emits three
    matrices (rows = mask task i, cols = input task j):
      - entropy_acc:   mean accuracy over trials,
      - entropy_std:   across-trial standard deviation of that accuracy,
      - entropy_share: mean fraction of task i's active neurons also active in
                       task j's mask (|m_i & m_j| / |m_i|).

    Dropout stays ENABLED during output generation (the model is kept in
    train() mode) so dropout can be explored as a lever for OOD entropy. The
    reconfiguration algorithms detach the backbone and don't use dropout, so
    train() mode only affects the scored forward pass. Restores the training
    mask and the model's train/eval state on exit.

    Returns (acc_mean, acc_std, share_mean) as (n, n) numpy arrays.
    """
    tasks_l = list(task_indices)
    n = len(tasks_l)
    n_trials = max(1, int(Config.ENTROPY_TEST_TRIALS))

    was_training = ftn.training
    ftn.train()  # keep dropout active during the scored forward pass
    m_train = ftn.m.clone()

    accs = np.full((n_trials, n, n), np.nan, dtype=float)
    shares = np.full((n_trials, n, n), np.nan, dtype=float)

    for trial in range(n_trials):
        # Recover a fresh mask per task from a new reconfiguration batch.
        masks = []
        for ti in tasks_l:
            ftn._reconfig(*data_gen(ti))
            masks.append(ftn.m.clone())
        # Fresh task-j data drawn once per task this trial (reused across the
        # mask tasks i so a column shares one input batch within the trial).
        data_j = [data_gen(tj) for tj in tasks_l]
        for a in range(n):
            m_i = masks[a]
            k_i = float(m_i.sum().item())
            for b in range(n):
                x_j, y_j = data_j[b]
                with torch.no_grad():
                    pred = ftn(x_j, m=m_i)
                accs[trial, a, b] = accuracy(pred, y_j)
                inter = float((m_i * masks[b]).sum().item())
                shares[trial, a, b] = inter / k_i if k_i > 0 else np.nan

    ftn.m.copy_(m_train)
    if not was_training:
        ftn.eval()

    acc_mean = np.nanmean(accs, axis=0)
    acc_std = np.nanstd(accs, axis=0)
    share_mean = np.nanmean(shares, axis=0)

    print("ENTROPY_TESTS mean accuracy (row=mask task i, col=input task j):")
    for a, ti in enumerate(tasks_l):
        cells = " ".join(f"{acc_mean[a, b]:.2f}" for b in range(n))
        print(f"  i={ti}: {cells}")

    viz.save_value_matrix(
        acc_mean, tasks_l, tasks_l, "entropy_acc", suffix, out_dir=out_dir,
        title="OOD accuracy (white=1, red=0)",
        xlabel="Input domain (task j)", ylabel="Mask task (task i)",
        cmap_colors=("red", "white"), vmin=0.0, vmax=1.0, cbar_label="Accuracy",
    )
    viz.save_value_matrix(
        acc_std, tasks_l, tasks_l, "entropy_std", suffix, out_dir=out_dir,
        title="OOD accuracy std across trials",
        xlabel="Input domain (task j)", ylabel="Mask task (task i)",
        cmap_colors=("white", "purple"), vmin=0.0, vmax=None, fmt="{:.3f}",
        cbar_label="Std. dev.",
    )
    viz.save_value_matrix(
        share_mean, tasks_l, tasks_l, "entropy_share", suffix, out_dir=out_dir,
        title="Shared active neurons (|m_i & m_j| / K)",
        xlabel="Task j", ylabel="Task i",
        cmap_colors=("white", "blue"), vmin=0.0, vmax=1.0, cbar_label="Fraction",
    )
    return acc_mean, acc_std, share_mean


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


def _sample_noise(n, d, distribution, scale):
    """A (n, d) batch of random-noise inputs for the noise probe.

    "gaussian" -> unit normal scaled by `scale`; "uniform" -> centered uniform
    drawn from [-scale, scale].
    """
    if distribution == "uniform":
        return (torch.rand(n, d) * 2.0 - 1.0) * scale
    return torch.randn(n, d) * scale


def snapshot_noise_probes(ftn, probe_buffer, input_dim):
    """Freeze noise probes representing the function under the current mask.

    Called once a task block has finished training (before any later task can
    overwrite the shared neurons). For each of NOISE_PROBE_SAMPLE_PER_TASK
    batches, we draw a fresh noise input and run it through the network AS IT IS
    NOW, under the current mask, to generate the target output. Dropout is
    disabled during generation so the target is a clean, deterministic
    fingerprint of the learned function (teacher in eval, student trains with
    dropout at replay time). Input and output are stored detached, so both stay
    fixed while the weights are later overwritten.
    """
    if probe_buffer is None:
        return
    was_training = ftn.training
    ftn.eval()
    m = ftn.m.clone()
    with torch.no_grad():
        for _ in range(Config.NOISE_PROBE_SAMPLE_PER_TASK):
            x_noise = _sample_noise(
                Config.NOISE_PROBE_BATCH,
                input_dim,
                Config.NOISE_PROBE_DISTRIBUTION,
                Config.NOISE_PROBE_SCALE,
            )
            y_noise = ftn(x_noise, m=m)
            probe_buffer.add((x_noise, y_noise), m)
    if was_training:
        ftn.train()


def noise_probe_step(ftn, probe_buffer):
    """Replay-free alternative to replay_step(): preserve learned structure by
    replaying frozen noise probes whose masks overlap the current one.

    Mirrors replay_step exactly — the same mask-overlap selection over the same
    (LOW, HIGH] band — but the replayed batches are noise probes: predict on the
    stored noise under the stored (full) mask and regress toward the stored
    generated output. Because that output was frozen at snapshot time, the MSE
    gradient pulls the neurons shared with the current task back toward the old
    function instead of chasing a target that drifts with the overwrite.
    """
    if probe_buffer is None or len(probe_buffer) == 0:
        return
    stored = torch.stack(list(probe_buffer.masks))
    corrs = (stored * ftn.m).sum(-1) / ftn.m.sum()
    keep = (corrs > Config.NOISE_PROBE_OVERLAP_LOW) & (
        corrs <= Config.NOISE_PROBE_OVERLAP_HIGH
    )
    overlap = torch.where(keep)[0]
    if len(overlap) == 0:
        return
    choice = torch.randint(0, len(overlap), (Config.NOISE_PROBE_SAMPLES,))
    for idx in overlap[choice]:
        x_n, y_n = probe_buffer.get(idx.item())
        m_r = probe_buffer.masks[idx.item()]
        pred = ftn(x_n, m=m_r)  # full stored mask, exactly like replay
        # MSE (not ftn.loss): y_n is a generated continuous output, not a label.
        F.mse_loss(pred, y_n).backward()


# =============================================================================
# Training
# =============================================================================


def train(
    ftn,
    data_gen,
    task_indices,
    n_epochs,
    suffix,
    optimizer,
    replay_buffer,
    probe_buffer=None,
    input_dim=None,
    out_dir="artifacts",
    collector=None,
    save_viz=True,
):
    """
    Multi-task training over the task stream, with visualization and the
    backwards/prospective forgetting monitor.

    probe_buffer / input_dim are used only by the noise-probe strategy
    (Config.NOISE_PROBE) and ignored when it is disabled.

    collector (metrics.MetricsCollector): when given, records one row per task
    block (tail training accuracy + backwards-eval accuracies) for the scalar
    metrics file. save_viz=False suppresses every visual artifact (mask GIFs,
    loss plots, forgetting matrices) while still running the eval that feeds the
    collector — used by --metrics-only runs.

    Returns per_task_losses.
    """
    per_task_losses = {t: [] for t in task_indices}
    # Tail window of each block used for its "training accuracy on the current
    # task" metric (the converged accuracy, not the early ramp).
    tail_n = max(1, Config.TRAIN_DURATION // 5)
    block_order = 0
    # RENDER_STEPS=False disables per-step mask rendering (GIFs); loss plots
    # and the forgetting/transfer matrices still save. save_viz=False (metrics-
    # only) disables all of them.
    animator = None
    if Config.RENDER_STEPS and save_viz:
        animator = viz.MaskAnimator(
            hidden_size=ftn.hidden_size,
            block_dim=Config.SQUARE_DIM,
            gif_duration=Config.GIF_DURATION,
            gif_save_epoch=min(Config.GIF_SAVE_EPOCH, n_epochs) or 1,
            out_dir=out_dir,
            ema_scale=getattr(Config, "GIF_EMA_SCALE", 1.0),
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
    for epoch in range(n_epochs):
        snapshot = epoch % Config.PLOT_EVERY == 0
        # Transfer/accuracy matrix for THIS epoch: one row per task as it is
        # (re)trained during the epoch, evaluated across tasks. Reset each epoch
        # so a multi-epoch run emits one xfer png per epoch (xfer_<name>_ep<e>);
        # a single-epoch run keeps the original xfer_<name> filename.
        forgetting_rows = []
        epoch_row_tasks = []
        for task in task_indices:
            loss_history = []
            # Training accuracy over the block's tail (converged-solution acc);
            # feeds the "current task train accuracy" metric.
            block_accs = []
            # Reset Adam (m, v) per task so optimizer state from a prior task
            # can't drift parameters whose gradient is zero under the new mask.
            optimizer.state = defaultdict(dict)
            n_block_reconfigs = 0
            # Wall time of the whole block's training loop (reconfig + forward +
            # backward + replay + optimizer, all together) for the metrics log.
            time_block = collector is not None
            if time_block and torch.cuda.is_available():
                torch.cuda.synchronize()
            block_t0 = time.perf_counter()
            for i in range(Config.TRAIN_DURATION):
                x, y = data_gen(task)
                if reactive:
                    # Loss-triggered reconfiguration. r_mean is None only on
                    # the very first training batch: do the one-time initial
                    # reconfig there.
                    if r_mean is None:
                        ftn._reconfig(x, y)
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
                        ftn._reconfig(x, y)
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

                if i < Config.N_SAMPLE_PER_TASK:
                    replay_buffer.add((x, y), ftn.m)

                if epoch == 0 and i == 0:
                    print(f"TASK {task} INITIAL_LOSS: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                replay_step(ftn, replay_buffer)
                if Config.NOISE_PROBE:
                    noise_probe_step(ftn, probe_buffer)
                optimizer.step()
                if animator:
                    animator.record_step(ftn.m, snapshot=snapshot)

                # Training accuracy over the block tail (for the metrics log).
                if collector is not None and i >= Config.TRAIN_DURATION - tail_n:
                    with torch.no_grad():
                        block_accs.append(accuracy(pred, y))

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

            if time_block and torch.cuda.is_available():
                torch.cuda.synchronize()
            block_wall_s = time.perf_counter() - block_t0

            if animator:
                animator.record_task(task, ftn.m, suffix, snapshot=snapshot)
            per_task_losses[task].extend(loss_history)
            # Freeze noise probes for this task's now-learned function, under its
            # final mask, before any later task can overwrite the shared neurons.
            if Config.NOISE_PROBE:
                snapshot_noise_probes(ftn, probe_buffer, input_dim)
            if reactive:
                print(
                    f"  [{suffix}] task {task}: "
                    f"reactive_reconfigs={n_block_reconfigs} "
                    f"loss_ema={r_mean:.4f} sd={np.sqrt(max(r_var, 0.0)):.4f}"
                )

            # Backwards eval: report loss on every task seen so far. If
            # EVAL_PROSPECTIVE is set, also eval on tasks not yet trained so
            # the upper triangle of the matrix gets filled.
            newly_seen = task not in seen_tasks
            if newly_seen:
                seen_tasks.append(task)
            accs = {}
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
                # One row per task per epoch. In epoch 0 every task is newly
                # seen, so this reproduces the previous single-epoch matrix;
                # in later epochs it captures the full re-evaluated matrix.
                forgetting_rows.append(accs)
                epoch_row_tasks.append(task)
                col_order = list(task_indices) if prospective else list(seen_tasks)
                if save_viz:
                    if n_epochs > 1:
                        xfer_tag = f"{suffix}_ep{epoch}" if suffix else f"ep{epoch}"
                    else:
                        xfer_tag = suffix
                    viz.save_forgetting_matrix(
                        forgetting_rows, epoch_row_tasks, col_order, xfer_tag,
                        out_dir=out_dir,
                    )

            # Record this block for the scalar metrics file. seen_tasks includes
            # the just-trained task, so summary() splits accs into recall (prior
            # tasks), current-task, and forward transfer (not-yet-seen) itself.
            if collector is not None:
                train_acc_block = float(np.mean(block_accs)) if block_accs else None
                collector.record_block(
                    block_order, task, train_acc_block, accs, seen_tasks,
                    block_wall_s=block_wall_s,
                )
                block_order += 1

        # Periodic plot saves (kept even with RENDER_STEPS=False; only the
        # per-step mask rendering is disabled). Suppressed entirely by save_viz.
        if snapshot and save_viz:
            viz.save_loss_plot(per_task_losses, suffix, out_dir=out_dir)

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
        default="auto",
        help="Run name: outputs go to artifacts/<name>/ with <name> suffixed "
        "to filenames (e.g. 'run1' → artifacts/run1/bw_run1.gif). Pass 'auto' "
        "to derive the name from the --config subconfigs joined with '_' plus "
        "an incrementing integer (e.g. Reactive_Low_Large_0).",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["Default"],
        help="One or more config class names from config.py, combined left "
        "to right — later ones override earlier ones on conflicts "
        "(e.g. --config Default DomainShift NoReplay).",
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
    parser.add_argument(
        "--noidx",
        action="store_true",
        help="With --name auto, don't scan for the lowest free integer; "
        "always use <base>_0 and overwrite any existing artifacts/ directory "
        "of that name instead of incrementing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override Config.SEED (varies the run's RNG; used by the "
        "multi-seed sweep in run_seeds.py).",
    )
    parser.add_argument(
        "--out-root",
        default="artifacts",
        help="Root directory for run outputs: files go to <out-root>/<name>/. "
        "Default 'artifacts'. Lets a sweep route runs into its own subtree "
        "without polluting the filename suffix.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Emit only the scalar metrics log (metrics_<name>.txt) and the "
        "effective config; skip every visualization (mask GIFs, loss plots, "
        "forgetting/transfer matrices, task vis, entropy-test matrices). The "
        "backwards/prospective eval still runs so the metrics are complete.",
    )
    args = parser.parse_args()

    # --name auto: derive the run name from the subconfigs (the set of --config
    # class names, which is exactly what lands after "# config:" in config.txt),
    # joined with '_', plus the lowest free incrementing integer. The integer
    # lets the same subconfig set be run repeatedly (e.g. with other params
    # changed) without clobbering earlier artifacts/ directories.
    if args.name == "auto":
        base = "_".join(args.config)
        i = 0
        if not args.noidx:
            existing = set()
            if os.path.isdir("artifacts"):
                existing = {
                    d for d in os.listdir("artifacts")
                    if os.path.isdir(os.path.join("artifacts", d))
                }
            while f"{base}_{i}" in existing:
                i += 1
        args.name = f"{base}_{i}"
        print(f"--name auto resolved to: {args.name}")

    RUN_TAG = f"_{args.name}" if args.name else ""
    # Per-run output directory: <out-root>/<name>/ (files still carry the
    # <name> suffix so runs can later be pooled into one folder).
    OUT_DIR = os.path.join(args.out_root, args.name) if args.name else args.out_root

    # Combine configs via MRO: bases are reversed so the LAST class listed
    # wins conflicting attributes (properties included).
    _bases = tuple(getattr(config, c) for c in reversed(args.config))
    Config = type("+".join(args.config), _bases, {})()
    if args.reconfig_every is not None:
        Config.RECONFIG_EVERY = args.reconfig_every
    if args.render_steps is not None:
        Config.RENDER_STEPS = bool(args.render_steps)
    if args.seed is not None:
        Config.SEED = args.seed
    # --metrics-only: force off every visual artifact; the metrics log is always
    # written regardless.
    SAVE_VIZ = not args.metrics_only
    if args.metrics_only:
        Config.RENDER_STEPS = False
    DataGen = tasks.DataGenSine
    _task = getattr(Config, "TASK", "sine").lower()

    print("Running:")
    print(f"Config: {type(Config).__name__}")
    print(f"Task: {_task}")
    print(
        f"DataGen: "
        f"{'DataGenComposition' if _task == 'compose' else DataGen.__name__}"
    )
    if RUN_TAG:
        print(f"Run tag: {RUN_TAG}")

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Record the effective config (post-composition, post-CLI-overrides) so
    # every run directory is self-describing.
    with open(os.path.join(OUT_DIR, f"config{RUN_TAG}.txt"), "w") as f:
        f.write(f"# config: {type(Config).__name__}\n")
        f.write(f"# argv: {' '.join(sys.argv)}\n")
        for k in sorted(k for k in dir(Config) if k.isupper()):
            f.write(f"{k} = {getattr(Config, k)!r}\n")

    if Config.SEED is not None:
        torch.manual_seed(Config.SEED)

    # Task visualization is sine-specific (plots per-task target surfaces).
    if Config.RENDER_STEPS and getattr(Config, "TASK", "sine").lower() == "sine":
        tasks.save_task_vis(
            Config, path=os.path.join(OUT_DIR, f"task_vis{RUN_TAG}.png")
        )

    data_gen = make_data_gen(Config.SEED + 100)
    # For the composition task, report the fixed per-task operator sequences.
    if getattr(Config, "TASK", "sine").lower() == "compose":
        print("Composition tasks:")
        for t in range(Config.N_TASK):
            print(f"  task {t}: {data_gen.task_description(t)}")
    # With DOMAIN_SHIFT the input vector is wider than X_DIM; the generator
    # is the single source of truth for the model's input width.
    arch = getattr(Config, "ARCHITECTURE", "base").lower()
    if arch == "lora":
        from lora_ftn import LoRAFTN
        ftn = LoRAFTN(
            input_size=data_gen.total_input_dim,
            output_size=data_gen.output_dim,
            hidden_size=Config.HIDDEN_SIZE,
            **Config.LORA_PARS,
        )
    elif arch in ("stack", "stack_residual", "rnn", "rnn_residual"):
        from stack_ftn import StackFTN
        ftn = StackFTN(
            input_size=data_gen.total_input_dim,
            output_size=data_gen.output_dim,
            hidden_size=Config.HIDDEN_SIZE,
            recurrent=arch in ("rnn", "rnn_residual"),
            residual=arch in ("stack_residual", "rnn_residual"),
            **Config.STACK_PARS,
        )
    elif arch == "wave":
        from wave_ftn import WaveFTN
        ftn = WaveFTN(
            input_size=data_gen.total_input_dim,
            output_size=data_gen.output_dim,
            hidden_size=Config.HIDDEN_SIZE,
            **Config.WAVE_PARS,
        )
    elif arch == "lorarnn":
        from lora_rnn_ftn import LoRARNNFTN
        ftn = LoRARNNFTN(
            input_size=data_gen.total_input_dim,
            output_size=data_gen.output_dim,
            hidden_size=Config.HIDDEN_SIZE,
            **Config.LORARNN_PARS,
        )
    elif arch == "hml":
        from hml_ftn_stable import HMLFTN
        ftn = HMLFTN(
            input_size=data_gen.total_input_dim,
            output_size=data_gen.output_dim,
            hidden_size=Config.HIDDEN_SIZE,
            **Config.HML_PARS,
        )
    elif arch == "quad":
        # Same FTN (mask / reconfig / readout) as base, per-unit backbone swapped
        # for the einsum-vectorized ParallelQuadMLP (backbone="quad" in QUAD_PARS).
        ftn = FTN(
            input_size=data_gen.total_input_dim,
            output_size=data_gen.output_dim,
            hidden_size=Config.HIDDEN_SIZE,
            **Config.QUAD_PARS,
        )
    else:
        ftn = FTN(
            input_size=data_gen.total_input_dim,
            output_size=data_gen.output_dim,
            hidden_size=Config.HIDDEN_SIZE,
            **Config.FTN_PARS,
        )
    # Only trainable params: the base FTN trains everything; FTN-LoRA freezes the
    # backbone and trains only the atom bank.
    optimizer = torch.optim.Adam(
        [p for p in ftn.parameters() if p.requires_grad], lr=Config.TRAIN_LR
    )
    replay = ReplayBuffer(
        maxlen=Config.REPLAY_MAXLEN,
        batch_size=Config.REPLAY_BATCH_SIZE,
    )
    # Frozen (noise, generated-output) buffer for the noise-probe strategy
    # (alternative to replay).
    probe_buffer = NoiseProbeBuffer(maxlen=Config.NOISE_PROBE_MAXLEN)

    # Match-stored reconfiguration option: give the FTN a live reference to the
    # active mask buffer (noise probe if enabled, else replay) so its annealste
    # search can snap onto a previous task's mask. No-op when the buffer is empty
    # or the FTN class doesn't support it.
    if getattr(Config, "RECONFIG_MATCH_STORED", False) and hasattr(
        ftn, "set_stored_masks_ref"
    ):
        src = probe_buffer if Config.NOISE_PROBE else replay
        ftn.set_stored_masks_ref(src.masks)

    if Config.RECONFIG_REACTIVE:
        print(
            f"Reconfig: REACTIVE sd={Config.RECONFIG_REACTIVE_SD} "
            f"ema={Config.RECONFIG_REACTIVE_EMA} (RECONFIG_EVERY ignored)"
        )
    else:
        print(f"Reconfig: every={Config.RECONFIG_EVERY} (0 = per task block)")
    print(
        f"Replay: {Config.N_REPLAY_SAMPLES}/step "
        f"(buffer {Config.REPLAY_MAXLEN} x {Config.REPLAY_BATCH_SIZE} rows)"
    )
    if Config.NOISE_PROBE:
        print(
            f"Noise probe: {Config.NOISE_PROBE_SAMPLES}/step "
            f"({Config.NOISE_PROBE_DISTRIBUTION}, scale={Config.NOISE_PROBE_SCALE}, "
            f"batch={Config.NOISE_PROBE_BATCH}, "
            f"overlap=({Config.NOISE_PROBE_OVERLAP_LOW}, "
            f"{Config.NOISE_PROBE_OVERLAP_HIGH}], "
            f"mask buffer {Config.NOISE_PROBE_MAXLEN})"
        )

    # Chance-level cross-entropy baseline (per loss mode; None for regression).
    if getattr(Config, "BINARY", False):
        # Per-cell BCE of a constant base-rate predictor: binary entropy H(p).
        p = float(getattr(Config, "COMP_P", 0.5))
        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p)) if 0.0 < p < 1.0 else 0.0
    elif Config.CLASSIFICATION:
        c = data_gen.output_dim
        ent = -c * np.log(1.0 / c) * (1.0 / c)
    else:
        ent = None
    print(
        f"Model: {utils.count_parameters(ftn)} params, "
        f"hidden={Config.HIDDEN_SIZE} ({Config.SQUARE_DIM}x{Config.SQUARE_DIM})"
    )
    if ent is not None:
        print(f"Chance-level loss: {ent:.4f}")

    print(
        f"\n{'=' * 60}\nMulti-task learning ({Config.N_TASK} tasks)\n{'=' * 60}"
    )
    collector = metrics_mod.MetricsCollector()
    train(
        ftn,
        data_gen,
        list(range(Config.N_TASK)),
        Config.N_EPOCHS,
        args.name,
        optimizer,
        replay,
        probe_buffer=probe_buffer,
        input_dim=data_gen.total_input_dim,
        out_dir=OUT_DIR,
        collector=collector,
        save_viz=SAVE_VIZ,
    )
    mtl_eval = evaluate_tasks(ftn, data_gen, list(range(Config.N_TASK)))

    # Scalar metrics log (always written): continual-learning accuracies from
    # the collector plus post-hoc cost/structure metrics measured on the trained
    # model (params, forward FLOPs, forward/reconfig wall time, mean task-mask
    # overlap). Parseable KEY = value; pooled across seeds by run_seeds.py.
    all_tasks = list(range(Config.N_TASK))
    n_trainable, n_total = metrics_mod.count_params(ftn)
    fwd_ms, recon_ms = metrics_mod.benchmark_timings(ftn, data_gen, all_tasks[0])
    metric_fields = {
        "run_name": args.name,
        "config": type(Config).__name__,
        "seed": Config.SEED,
        "n_tasks": Config.N_TASK,
        "n_params_trainable": n_trainable,
        "n_params_total": n_total,
        "flops_forward": metrics_mod.measure_flops(ftn, data_gen(all_tasks[0])[0]),
        "wall_time_forward_ms": fwd_ms,
        "wall_time_reconfig_ms": recon_ms,
        "avg_mask_overlap_pct": metrics_mod.mean_pairwise_mask_overlap(
            ftn, data_gen, all_tasks
        ),
    }
    metric_fields.update(collector.summary())
    metrics_path = os.path.join(OUT_DIR, f"metrics{RUN_TAG}.txt")
    metrics_mod.write_metrics(metrics_path, metric_fields)
    print(f"Metrics written to {metrics_path}")

    # Post-hoc ENTROPY_TESTS: how each task's masked subspace behaves out of its
    # trained input distribution (accuracy, across-trial variance, and neuron
    # sharing). Dropout stays enabled during output generation. Skipped in
    # --metrics-only runs (it emits visualization matrices).
    if getattr(Config, "ENTROPY_TESTS", False) and SAVE_VIZ:
        print(f"\n{'=' * 60}\nENTROPY_TESTS (OOD subspace probe)\n{'=' * 60}")
        print(f"Trials: {Config.ENTROPY_TEST_TRIALS} (dropout enabled)")
        entropy_tests(
            ftn, data_gen, list(range(Config.N_TASK)), args.name, out_dir=OUT_DIR
        )

    print(f"\n{'=' * 60}\nEXPERIMENT COMPLETE\n{'=' * 60}")
    print("MTL eval:", {k: f"{v:.4f}" for k, v in mtl_eval.items()})
    if ent is not None:
        print(f"Chance-level loss: {ent:.4f}")
    print(f"\nOutputs saved to {OUT_DIR}/")
