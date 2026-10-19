"""
run_experiments.py - Continual-learning experiment for the bayesian FTN
subproject.

Trains a plain-kWTA FTN on a stream of N sine classification tasks, with
optional offline replay and a backwards/prospective forgetting monitor. Pared
down from ftn/ to exactly this subproject: base and persistent-logit Bayesian
FTNs, one task (tasks.DataGenSine, classification), a single mask, replay
(no noise probe / Hopfield / dual mask / domain-agnostic branches).

The training loop below (train()) is intentionally flat and unabstracted: the
FTN owns the mask search (ftn(x, y) reconfigures then predicts; ftn(x) predicts
with the current mask; ftn(x, m=...) predicts under a stored mask), and this
file only optimizes the weights and decides WHEN to reconfigure. That "when"
lives inline in the task loop - the natural place to write a new reconfig
protocol by hand.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

# Keep the inherited experiment modules in the parent directory while using
# this folder's local config and routed model subclasses.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
import metrics as metrics_mod
import tasks
import utils
import viz
from ftn import BayesFTN, FTN
from lora_ftn import LoRAFTN
from routing import BayesLoRAFTN, BayesLoRARNNFTN, ReconstructionPrior
from autoencoder_prior import AutoencoderLogitPrior
from replay_logit_prior import ReplayLogitPrior
from bayes_lora_ftn import BayesLoRAFTN as PlainBayesLoRAFTN
from bayes_lora_rnn_ftn import BayesLoRARNNFTN as PlainBayesLoRARNNFTN
from bayes_stack_ftn import BayesRNNResidualFTN

# =============================================================================
# Replay buffer
# =============================================================================


class ReplayBuffer:
    """Stores (data, mask) pairs for offline replay.

    If batch_size is set, each stored batch is subsampled to batch_size rows
    once at store time.
    """

    def __init__(self, maxlen, batch_size=None, subspace_rank=None):
        self.data = deque(maxlen=maxlen)
        self.masks = deque(maxlen=maxlen)
        self.logits = deque(maxlen=maxlen)
        self.bases = deque(maxlen=maxlen)
        self.tasks = deque(maxlen=maxlen)
        self.batch_size = batch_size
        self.subspace_rank = subspace_rank

    def add(self, dat, m, logits=None, task=None):
        x, y = dat
        n = x.shape[0]
        if self.batch_size is not None and self.batch_size < n:
            idx = torch.randperm(n, device=x.device)[: self.batch_size]
            dat = (x[idx], y[idx])
        self.data.append(dat)
        self.masks.append(m.clone())
        if logits is not None:
            self.logits.append(logits.detach().clone())
            self.tasks.append(task)
            if self.subspace_rank is not None:
                with torch.no_grad():
                    _, _, right = torch.linalg.svd(dat[0], full_matrices=False)
                    self.bases.append(right[:self.subspace_rank].T.contiguous())

    def get(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


class ChangepointReplayBuffer(ReplayBuffer):
    """Refresh each task's existing slots with its recent outgoing mask/data."""

    def __init__(self, maxlen, batch_size, tail_batches):
        super().__init__(maxlen, batch_size)
        self.entry_tasks = deque(maxlen=maxlen)
        self.late_slots = deque(maxlen=maxlen)
        self.tails = defaultdict(lambda: deque(maxlen=tail_batches))
        self.last_completed_task = None
        self.current_train_task = None
        self.priority_replay = False
        self.priority_every = 1
        self.replay_calls = 0
        self.priority_draws = 0
        self.filtered_draws = 0
        self.events = []

    def add(self, dat, m, logits=None, task=None):
        super().add(dat, m, logits=logits, task=task)
        self.entry_tasks.append(task)
        self.late_slots.append(False)

    def observe_completed_batch(self, dat, task):
        self.tails[task].append(dat)
        self.last_completed_task = task

    def snapshot_before_change(self, old_mask):
        task = self.last_completed_task
        recent = list(self.tails.get(task, ())) if task is not None else []
        slots = [i for i, stored_task in enumerate(self.entry_tasks)
                 if stored_task == task] if task is not None else []
        n = min(len(recent), len(slots)) if old_mask.sum() > 0 else 0
        for slot, (x, y) in (zip(slots[-n:], recent[-n:]) if n else ()):
            if self.batch_size is not None:
                x, y = x[:self.batch_size], y[:self.batch_size]
            self.data[slot] = (x, y)
            self.masks[slot] = old_mask.detach().clone()
            self.late_slots[slot] = True
        event = {"task": task, "stored_slots": len(slots),
                 "recent_batches": len(recent), "replaced_slots": n,
                 "buffer_size": len(self)}
        self.events.append(event)
        if task is not None:
            self.tails[task].clear()
        return event


# =============================================================================
# Helpers
# =============================================================================


def make_data_gen(seed):
    """Create the sine task's DataGen with its own local RNG."""
    return tasks.DataGenSine(
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
    """Single-label classification accuracy: fraction of rows whose argmax
    prediction matches the one-hot target."""
    target = torch.where(y == 1)[1]
    return (pred.argmax(-1) == target).float().mean().item()


def eval_reconfigure(ftn, x, y, auto_prior=None, task=None):
    """Run the architecture-independent, stateless evaluation mask search."""
    if auto_prior is not None:
        return auto_prior.eval_reconfigure(ftn, x, y, task=task)
    return ftn.cold_reconfigure(x, y, **Config.EVAL_FTN_PARS)


def probe_routing(ftn, data_gen, task_indices, trials):
    """Compare known-mask retrieval and search on identical held-out queries.

    Oracle is the best stored hard mask *on the query labels*, so it is only an
    upper bound. Direct and refined use input-only retrieval. This diagnostic
    runs after training and never updates the input memory or model weights.
    """
    saved_state = ftn.get_reconfig_state()
    saved_stats = ftn.ic_stats.copy()
    saved_mode = ftn.ic_mode
    saved_eval_prior = ftn.ic_eval_prior
    saved_strategy = ftn.ic_eval_strategy
    scores = {key: [] for key in ("direct", "refined", "zero", "full_support", "oracle",
                                  "matched", "reconstruction_error")}
    entries = ftn._ic_candidates()
    try:
        for task in task_indices:
            for _ in range(trials):
                xs, ys = data_gen(task)
                xq, yq = data_gen(task)
                entry, error = ftn._ic_match(xs)
                scores["matched"].append(float(entry is not None))
                if entry is not None:
                    scores["reconstruction_error"].append(error)
                    direct_mask = ftn._topk_mask(entry.logits)
                    with torch.no_grad():
                        scores["direct"].append(
                            accuracy(ftn(xq, m=direct_mask), yq))
                with torch.no_grad():
                    scores["oracle"].append(max(
                        accuracy(ftn(xq, m=ftn._topk_mask(candidate.logits)), yq)
                        for candidate in entries))

                # The parent cold search, regardless of this run's IC mode.
                super(ReconstructionPrior, ftn).cold_reconfigure(
                    xs, ys, **Config.EVAL_FTN_PARS)
                with torch.no_grad():
                    scores["zero"].append(accuracy(ftn(xq), yq))

                # Diagnostic upper benchmark: same detached backbone search,
                # but with the complete labeled support batch available.
                full_pars = dict(Config.EVAL_FTN_PARS)
                full_pars["sample_size"] = xs.shape[0]
                super(ReconstructionPrior, ftn).cold_reconfigure(
                    xs, ys, **full_pars)
                with torch.no_grad():
                    scores["full_support"].append(accuracy(ftn(xq), yq))

                ftn.ic_mode = "reconstruction"
                ftn.ic_eval_prior = True
                ftn.ic_eval_strategy = "refine"
                ftn.cold_reconfigure(xs, ys, **Config.EVAL_FTN_PARS)
                with torch.no_grad():
                    scores["refined"].append(accuracy(ftn(xq), yq))
    finally:
        ftn.ic_mode = saved_mode
        ftn.ic_eval_prior = saved_eval_prior
        ftn.ic_eval_strategy = saved_strategy
        ftn.set_reconfig_state(saved_state)
        ftn.ic_stats = saved_stats
    return {f"ic_probe_{key}": float(np.mean(vals)) if vals else None
            for key, vals in scores.items()}


def evaluate_tasks(ftn, data_gen, task_indices, auto_prior=None):
    """Quick loss evaluation over multiple batches. Returns {task_idx: mean_loss}.

    Every trial cold-searches on a support batch and scores a fresh query batch.
    Restores the training inference state so eval never leaks into training.
    """
    train_state = ftn.get_reconfig_state()
    results = {}
    for task_idx in task_indices:
        losses = []
        for _ in range(Config.N_EVAL_EPOCHS):
            eval_reconfigure(ftn, *data_gen(task_idx), auto_prior=auto_prior,
                             task=task_idx)
            x, y = data_gen(task_idx)  # fresh query batch
            with torch.no_grad():
                losses.append(ftn.loss(ftn(x), y).item())
        results[task_idx] = np.mean(losses)
    ftn.set_reconfig_state(train_state)
    return results


def evaluate_backwards(ftn, data_gen, task_indices, auto_prior=None):
    """Reconfigure & eval on each task for N_RETRO_EVAL_BATCHES batches.

    Every trial independently cold-searches on a support batch and evaluates on
    a fresh holdout batch. Training mask state and training reconfiguration
    schedule are ignored; the complete training inference state is restored.
    Returns {task_idx: {"loss": mean_loss, "acc": mean_acc}}.
    """
    train_state = ftn.get_reconfig_state()
    results = {}
    for task_idx in task_indices:
        losses, accs = [], []
        for _ in range(Config.N_RETRO_EVAL_BATCHES):
            eval_reconfigure(ftn, *data_gen(task_idx), auto_prior=auto_prior,
                             task=task_idx)
            x, y = data_gen(task_idx)  # fresh holdout batch
            with torch.no_grad():
                pred = ftn(x)
                losses.append(ftn.loss(pred, y).item())
                accs.append(accuracy(pred, y))
            if auto_prior is not None:
                auto_prior.record_query(ftn, x, y, task_idx)
        results[task_idx] = {
            "loss": float(np.mean(losses)),
            "acc": float(np.mean(accs)),
        }
    ftn.set_reconfig_state(train_state)
    return results


def entropy_tests(ftn, data_gen, task_indices, suffix, out_dir="artifacts",
                  auto_prior=None):
    """Post-hoc out-of-distribution subspace probe.

    For every ordered task pair (i, j): reconfigure on a fresh task-i batch to
    recover task i's mask m_i, then run task j's inputs through the network UNDER
    m_i and score against task j's target - how the subspace trained on task i
    behaves out of its trained input distribution. Averaged over
    ENTROPY_TEST_TRIALS trials (each resamples masks and data). Emits three
    (n, n) matrices (rows = mask task i, cols = input task j): mean accuracy,
    across-trial std, and mean neuron sharing |m_i & m_j| / |m_i|.

    Dropout stays ENABLED during output generation (model kept in train() mode)
    so it can be explored as an OOD-entropy lever. Restores mask + train state.
    """
    tasks_l = list(task_indices)
    n = len(tasks_l)
    n_trials = max(1, int(Config.ENTROPY_TEST_TRIALS))

    was_training = ftn.training
    ftn.train()
    train_state = ftn.get_reconfig_state()

    accs = np.full((n_trials, n, n), np.nan, dtype=float)
    shares = np.full((n_trials, n, n), np.nan, dtype=float)

    for trial in range(n_trials):
        masks = []
        for ti in tasks_l:
            eval_reconfigure(ftn, *data_gen(ti), auto_prior=auto_prior, task=ti)
            masks.append(ftn.m.clone())
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

    ftn.set_reconfig_state(train_state)
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
    """Replay one stored batch, optionally prioritizing old late-mask snapshots."""
    if len(replay_buffer) == 0:
        return
    stored = torch.stack(list(replay_buffer.masks))
    priority = []
    if getattr(replay_buffer, "priority_replay", False):
        replay_buffer.replay_calls += 1
        if replay_buffer.replay_calls % replay_buffer.priority_every == 0:
            priority = [i for i, (late, task) in enumerate(
                zip(replay_buffer.late_slots, replay_buffer.entry_tasks))
                if late and task != replay_buffer.current_train_task]
    if priority:
        overlap = torch.tensor(priority, device=stored.device)
        replay_buffer.priority_draws += 1
    else:
        corrs = (stored * ftn.m).sum(-1) / ftn.m.sum()
        keep = (corrs > Config.REPLAY_OVERLAP_LOW) & (
            corrs <= Config.REPLAY_OVERLAP_HIGH
        )
        overlap = torch.where(keep)[0]
        if hasattr(replay_buffer, "filtered_draws") and len(overlap):
            replay_buffer.filtered_draws += 1
    if len(overlap) == 0:
        return
    choice = torch.randint(0, len(overlap), (Config.N_REPLAY_SAMPLES,))
    replay_scale = getattr(Config, "IC_REPLAY_LOSS_SCALE", 1.0)
    for idx in overlap[choice]:
        x_r, y_r = replay_buffer.get(idx.item())
        m_r = replay_buffer.masks[idx.item()]
        r_pred = ftn(x_r, m=m_r)
        (ftn.loss(r_pred, y_r) * replay_scale).backward()


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
    out_dir="artifacts",
    collector=None,
    save_viz=True,
    auto_prior=None,
):
    """Continual-learning training over the task stream, with visualization,
    the backwards/prospective forgetting monitor, and metrics collection.

    The reconfiguration policy is inline in the task loop below - either
    loss-triggered (RECONFIG_REACTIVE) or scheduled every RECONFIG_EVERY batches
    (0 = once per task block). Replace it there to try a new protocol.

    collector (metrics.MetricsCollector): when given, records one row per task
    block for the scalar metrics file. save_viz=False suppresses every visual
    artifact (used by --metrics-only) while still running the eval that feeds the
    collector.

    Returns per_task_losses.
    """
    per_task_losses = {t: [] for t in task_indices}
    tail_n = max(1, Config.TRAIN_DURATION // 5)  # block-tail window for train acc
    block_order = 0

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

    # Reactive change-point state. The detector observes loss under the current
    # hard mask before any update. Base FTN reconfigures only on triggers;
    # BayesFTN still makes a cheap persistent update on stable batches, but uses
    # its reset/refit parameter set on triggers. Task boundaries are never used.
    reactive = Config.RECONFIG_REACTIVE
    r_sd = Config.RECONFIG_REACTIVE_SD
    r_alpha = Config.RECONFIG_REACTIVE_EMA
    r_warmup = int(np.ceil(1.0 / max(1.0 - r_alpha, 1e-8)))
    r_mean, r_var, r_since = None, 0.0, 0
    rng_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    pending_train_batch = None

    seen_tasks = []
    for epoch in range(n_epochs):
        snapshot = epoch % Config.PLOT_EVERY == 0
        forgetting_rows = []
        epoch_row_tasks = []
        for task in task_indices:
            loss_history = []
            block_accs = []               # training accuracy over the block tail
            # Reset Adam (m, v) per task so optimizer state from a prior task
            # can't drift parameters whose gradient is zero under the new mask.
            optimizer.state = defaultdict(dict)
            n_block_changepoints = 0
            block_cp_recorded = False

            time_block = collector is not None
            if time_block and torch.cuda.is_available():
                torch.cuda.synchronize()
            block_t0 = time.perf_counter()

            # ================= inner training loop (one task block) ============
            for i in range(Config.TRAIN_DURATION):
                x, y = data_gen(task)
                if hasattr(replay_buffer, "observe_completed_batch"):
                    replay_buffer.current_train_task = task
                    if pending_train_batch is not None:
                        replay_buffer.observe_completed_batch(*pending_train_batch)
                    pending_train_batch = ((x, y), task)

                # ---- decide whether to reconfigure the mask, then predict ----
                if reactive:
                    with torch.no_grad():
                        detector_loss = ftn.loss(ftn(x), y).item()
                    sd = float(np.sqrt(max(r_var, 0.0)))
                    changepoint = r_mean is None or (
                        r_mean is not None
                        and r_since >= r_warmup
                        and detector_loss > r_mean + r_sd * sd
                    )
                    if hasattr(ftn, "ic_novelty"):
                        changepoint = changepoint or ftn.ic_novelty(x)
                    if changepoint:
                        snapshot_event = None
                        if hasattr(replay_buffer, "snapshot_before_change"):
                            snapshot_event = replay_buffer.snapshot_before_change(
                                ftn.m.detach().clone())
                            old_mask = ftn.m.detach().clone()
                        ftn.reconfigure_changepoint(x, y)
                        if snapshot_event is not None and old_mask.sum() > 0:
                            overlap = float((old_mask * ftn.m).sum() / ftn.m.sum())
                            snapshot_event["replay_overlap_after_change"] = overlap
                            snapshot_event["replay_eligible_after_change"] = (
                                0.2 < overlap <= 0.8)
                        n_block_changepoints += 1
                        r_since = 0
                    elif ftn.continuous_reactive_updates:
                        ftn._reconfig(x, y)

                    pred = ftn(x)
                    if (changepoint and epoch > 0 and not block_cp_recorded
                            and hasattr(ftn, "ic_cp_holdout_acc")):
                        # Only the first four rows informed the mask search.
                        # The other rows assess immediate recall before any
                        # weight update on this returning domain.
                        n_support = Config.IC_SUPPORT_SIZE
                        ftn.ic_cp_holdout_acc.append(
                            accuracy(pred[n_support:], y[n_support:]))
                        ftn.ic_cp_holdout_loss.append(
                            ftn.loss(pred[n_support:], y[n_support:]).item())
                        block_cp_recorded = True
                    loss = ftn.loss(pred, y)
                    li_r = loss.item()
                    if changepoint:
                        # Restart detector statistics in the new mask regime.
                        r_mean, r_var = li_r, 0.0
                    # EMA of loss mean/variance under the trained mask (the
                    # pre-reconfig spike is never recorded).
                    if not changepoint:
                        diff = li_r - r_mean
                        r_mean = r_alpha * r_mean + (1 - r_alpha) * li_r
                        r_var = r_alpha * r_var + (1 - r_alpha) * diff * diff
                    r_since += 1
                else:
                    # Scheduled: reconfigure every RECONFIG_EVERY batches; 0 =
                    # once at the start of each task block.
                    every = Config.RECONFIG_EVERY
                    reconfig = (i == 0) if every == 0 else (i % every == 0)
                    pred = ftn(x, y) if reconfig else ftn(x)
                    loss = ftn.loss(pred, y)

                # ---- store a few batches for replay ----
                if i < Config.N_SAMPLE_PER_TASK:
                    replay_buffer.add(
                        (x, y), ftn.m,
                        logits=(ftn.mask_logits if getattr(
                            Config, "REPLAY_LOGIT_PRIOR_ENABLED", False) else None),
                        task=task)

                if epoch == 0 and i == 0:
                    print(f"TASK {task} INITIAL_LOSS: {loss.item()}")

                # ---- weight update (+ replay-augmented gradient) ----
                optimizer.zero_grad()
                loss.backward()
                replay_step(ftn, replay_buffer)
                optimizer.step()

                if auto_prior is not None:
                    # Auxiliary FTN search must not change the predictor's
                    # subsequent mask-sampling RNG stream.
                    with torch.random.fork_rng(devices=rng_devices):
                        auto_prior.train_step(x, y)

                if animator:
                    animator.record_step(ftn.m, snapshot=snapshot)

                # ---- training accuracy over the block tail (metrics log) ----
                if collector is not None and i >= Config.TRAIN_DURATION - tail_n:
                    with torch.no_grad():
                        block_accs.append(accuracy(pred, y))

                # ---- EMA loss tracking (for the loss plot) ----
                li = loss.item()
                if i == 0:
                    loss_smooth = li
                else:
                    loss_smooth = (
                        Config.LOSS_EMA_SMOOTHING * loss_smooth
                        + (1 - Config.LOSS_EMA_SMOOTHING) * li
                    )
                loss_history.append(loss_smooth)
            # =============== end inner training loop ============================

            if auto_prior is not None:
                with torch.random.fork_rng(devices=rng_devices):
                    auto_prior.train_block_end(ftn, task)

            if hasattr(ftn, "ic_refresh_memory"):
                ftn.ic_refresh_memory()
            if hasattr(ftn, "ic_repair_from_replay"):
                ftn.ic_repair_from_replay(replay_buffer)

            if time_block and torch.cuda.is_available():
                torch.cuda.synchronize()
            block_wall_s = time.perf_counter() - block_t0

            if animator:
                animator.record_task(task, ftn.m, suffix, snapshot=snapshot)
            per_task_losses[task].extend(loss_history)
            if save_viz:
                viz.save_loss_plot(per_task_losses, suffix, out_dir=out_dir)
            if reactive:
                print(
                    f"  [{suffix}] task {task}: "
                    f"detected_changepoints={n_block_changepoints} "
                    f"loss_ema={r_mean:.4f} sd={np.sqrt(max(r_var, 0.0)):.4f}"
                )

            # ---- backwards / prospective eval (forgetting monitor) ----
            newly_seen = task not in seen_tasks
            if newly_seen:
                seen_tasks.append(task)
            accs = {}
            if Config.EVAL_BACKWARDS:
                prospective = Config.EVAL_PROSPECTIVE
                eval_set = list(task_indices) if prospective else list(seen_tasks)
                # The two routing methods consume different mask-sampling RNG
                # during evaluation. Keep subsequent training streams paired.
                devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices):
                    bw = evaluate_backwards(ftn, data_gen, eval_set,
                                            auto_prior=auto_prior)
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

            # ---- record this block for the scalar metrics file ----
            if collector is not None:
                train_acc_block = float(np.mean(block_accs)) if block_accs else None
                collector.record_block(
                    block_order, task, train_acc_block, accs, seen_tasks,
                    block_wall_s=block_wall_s,
                )
                block_order += 1

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
        help="Run name: outputs go to <out-root>/<name>/ with <name> suffixed "
        "to filenames. 'auto' derives the name from the --config subconfigs "
        "joined with '_' plus an incrementing integer.",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["Default"],
        help="One or more config class names from config.py, combined left to "
        "right - later ones override earlier ones (e.g. --config Default NoReplay "
        "Bayes).",
    )
    parser.add_argument(
        "--architecture",
        choices=["base", "lora", "bayes", "bayes_lora", "bayes_rnn_residual", "bayes_lora_rnn"],
        default=None,
        help="Override Config.ARCHITECTURE.",
    )
    parser.add_argument(
        "--reconfig-every",
        type=int,
        default=None,
        help="Override Config.RECONFIG_EVERY (batches between reconfigurations; "
        "0 = once per task block).",
    )
    parser.add_argument(
        "--render-steps",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override Config.RENDER_STEPS (0 = skip per-step mask GIFs and task "
        "vis; loss plots and forgetting/transfer matrices still save).",
    )
    parser.add_argument(
        "--noidx",
        action="store_true",
        help="With --name auto, don't scan for the lowest free integer; always "
        "use <base>_0 and overwrite any existing directory of that name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override Config.SEED (varies the run's RNG; used by run_seeds.py).",
    )
    parser.add_argument(
        "--out-root",
        default="artifacts",
        help="Root directory for run outputs: files go to <out-root>/<name>/.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Emit only the scalar metrics log (metrics_<name>.txt) and the "
        "effective config; skip every visualization (mask GIFs, loss plots, "
        "forgetting/transfer matrices, task vis, entropy-test matrices). The "
        "backwards/prospective eval still runs so the metrics are complete.",
    )
    parser.add_argument(
        "--checkpoint-out",
        default=None,
        help="Optional pod-local path for trained weights, routing memory, and replay.",
    )
    args = parser.parse_args()

    # --name auto: derive the run name from the subconfigs joined with '_' plus
    # the lowest free incrementing integer (so the same set can be re-run).
    if args.name == "auto":
        base = "_".join(args.config)
        i = 0
        if not args.noidx:
            existing = set()
            root = args.out_root
            if os.path.isdir(root):
                existing = {
                    d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d))
                }
            while f"{base}_{i}" in existing:
                i += 1
        args.name = f"{base}_{i}"
        print(f"--name auto resolved to: {args.name}")

    RUN_TAG = f"_{args.name}" if args.name else ""
    OUT_DIR = os.path.join(args.out_root, args.name) if args.name else args.out_root

    # Combine configs via MRO: bases reversed so the LAST class listed wins.
    _bases = tuple(getattr(config, c) for c in reversed(args.config))
    Config = type("+".join(args.config), _bases, {})()
    if args.reconfig_every is not None:
        Config.RECONFIG_EVERY = args.reconfig_every
    if args.architecture is not None:
        Config.ARCHITECTURE = args.architecture
    if args.render_steps is not None:
        Config.RENDER_STEPS = bool(args.render_steps)
    if args.seed is not None:
        Config.SEED = args.seed
    SAVE_VIZ = not args.metrics_only
    if args.metrics_only:
        Config.RENDER_STEPS = False

    print("Running:")
    print(f"Config: {type(Config).__name__}")
    if RUN_TAG:
        print(f"Run tag: {RUN_TAG}")

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Record the effective config (post-composition, post-CLI-overrides).
    with open(os.path.join(OUT_DIR, f"config{RUN_TAG}.txt"), "w") as f:
        f.write(f"# config: {type(Config).__name__}\n")
        f.write(f"# argv: {' '.join(sys.argv)}\n")
        for k in sorted(k for k in dir(Config) if k.isupper()):
            f.write(f"{k} = {getattr(Config, k)!r}\n")

    if Config.SEED is not None:
        torch.manual_seed(Config.SEED)

    # Task visualization (plots per-task target surfaces).
    if Config.RENDER_STEPS:
        tasks.save_task_vis(
            Config, path=os.path.join(OUT_DIR, f"task_vis{RUN_TAG}.png")
        )

    data_gen = make_data_gen(Config.SEED + 100)
    # With DOMAIN_SHIFT the input vector is wider than X_DIM; the generator is
    # the single source of truth for the model's input width.
    architectures = {
        "base": FTN,
        "lora": LoRAFTN,
        "bayes": BayesFTN,
        "bayes_lora": BayesLoRAFTN,
        "bayes_rnn_residual": BayesRNNResidualFTN,
        "bayes_lora_rnn": BayesLoRARNNFTN,
    }
    if Config.ARCHITECTURE not in architectures:
        raise ValueError(
            f"Unknown ARCHITECTURE={Config.ARCHITECTURE!r}; expected one of "
            f"{sorted(architectures)}"
        )
    auto_prior_enabled = getattr(Config, "AE_PRIOR_ENABLED", False)
    replay_prior_enabled = getattr(Config, "REPLAY_LOGIT_PRIOR_ENABLED", False)
    if auto_prior_enabled and replay_prior_enabled:
        raise ValueError("Choose one evaluation prior at a time")
    if auto_prior_enabled or replay_prior_enabled:
        if Config.ARCHITECTURE == "bayes_lora":
            architectures["bayes_lora"] = PlainBayesLoRAFTN
        elif Config.ARCHITECTURE == "bayes_lora_rnn":
            architectures["bayes_lora_rnn"] = PlainBayesLoRARNNFTN
        elif Config.ARCHITECTURE != "bayes":
            raise ValueError("Evaluation priors require a Bayesian architecture")
    # Per-architecture constructor parameters. The three Bayesian variants share
    # the same persistent-posterior hyperparameters (BAYES_FTN_PARS); only the
    # backbone bundle differs (ParallelMLP / LoRA atom bank / recurrent cell).
    pars_by_arch = {
        "base": dict(Config.FTN_PARS),
        "lora": dict(Config.LORA_PARS),
        "bayes": {**Config.FTN_PARS, **Config.BAYES_FTN_PARS},
        "bayes_lora": {**Config.LORA_FTN_PARS, **Config.BAYES_FTN_PARS},
        "bayes_rnn_residual": {**Config.RNN_FTN_PARS, **Config.BAYES_FTN_PARS},
        "bayes_lora_rnn": {**Config.LORA_RNN_FTN_PARS, **Config.BAYES_FTN_PARS},
    }
    ftn_pars = dict(pars_by_arch[Config.ARCHITECTURE])
    ftn = architectures[Config.ARCHITECTURE](
        input_size=data_gen.total_input_dim,
        output_size=data_gen.output_dim,
        hidden_size=Config.HIDDEN_SIZE,
        **ftn_pars,
    )
    if hasattr(ftn, "configure_ic"):
        ftn.configure_ic(Config)
    optimizer = torch.optim.Adam(
        [p for p in ftn.parameters() if p.requires_grad], lr=Config.TRAIN_LR
    )
    cp_replay = getattr(Config, "IC_CP_REPLAY_REFRESH", False)
    if cp_replay and replay_prior_enabled:
        raise ValueError("Changepoint replay refresh is incompatible with replay-logit retrieval")
    replay = (ChangepointReplayBuffer(
        maxlen=Config.REPLAY_MAXLEN,
        batch_size=Config.REPLAY_BATCH_SIZE,
        tail_batches=Config.IC_CP_REPLAY_TAIL_BATCHES,
    ) if cp_replay else ReplayBuffer(
        maxlen=Config.REPLAY_MAXLEN,
        batch_size=Config.REPLAY_BATCH_SIZE,
        subspace_rank=(Config.RP_SUBSPACE_RANK if replay_prior_enabled
                       and Config.RP_MATCH == "subspace" else None),
    ))
    if cp_replay:
        replay.priority_replay = getattr(Config, "IC_CP_REPLAY_PRIORITY", False)
        replay.priority_every = getattr(Config, "IC_CP_REPLAY_PRIORITY_EVERY", 1)
        if replay.priority_every < 1:
            raise ValueError("IC_CP_REPLAY_PRIORITY_EVERY must be positive")
    auto_prior = (AutoencoderLogitPrior(
        Config, data_gen.total_input_dim, ftn.m.numel(),
        groups=getattr(ftn, "n_stack", 1)) if auto_prior_enabled else
        ReplayLogitPrior(Config, replay) if replay_prior_enabled else None)
    if Config.RECONFIG_MATCH_STORED and hasattr(ftn, "set_stored_masks_ref"):
        ftn.set_stored_masks_ref(replay.masks)

    if Config.RECONFIG_REACTIVE:
        print(
            f"Reconfig: REACTIVE sd={Config.RECONFIG_REACTIVE_SD} "
            f"ema={Config.RECONFIG_REACTIVE_EMA} (RECONFIG_EVERY ignored)"
        )
        if Config.ARCHITECTURE.startswith("bayes"):
            print(
                "Bayes change-point update: "
                f"steps={Config.BAYES_CHANGEPOINT_RECONFIG_STEPS} "
                f"lr={Config.BAYES_CHANGEPOINT_RECONFIG_LR} "
                f"kl={Config.BAYES_CHANGEPOINT_KL_WEIGHT} "
                f"prior_decay={Config.BAYES_CHANGEPOINT_PRIOR_DECAY}"
            )
    else:
        print(f"Reconfig: every={Config.RECONFIG_EVERY} (0 = per task block)")
    if auto_prior_enabled:
        print(f"Autoencoder-logit prior: eval={Config.AE_EVAL_STRATEGY} "
              f"support={Config.IC_SUPPORT_SIZE} "
              f"autoencoder={Config.AE_SQUARE_DIM ** 2} units/{Config.AE_K} active")
    elif replay_prior_enabled:
        print(f"Replay-logit prior: eval={Config.RP_EVAL_STRATEGY} "
              f"match={Config.RP_MATCH} support={Config.IC_SUPPORT_SIZE}")
    else:
        print(
            "Evaluation mask search: stateless zero-logit | "
            f"steps={Config.EVAL_RECONFIG_STEPS} "
            f"lr={Config.EVAL_RECONFIG_LR} "
            f"restarts={Config.EVAL_RECONFIG_RESTARTS}"
        )
    print(
        f"Replay: {Config.N_REPLAY_SAMPLES}/step "
        f"(buffer {Config.REPLAY_MAXLEN} x {Config.REPLAY_BATCH_SIZE} rows, "
        f"overlap=({Config.REPLAY_OVERLAP_LOW}, {Config.REPLAY_OVERLAP_HIGH}])"
    )

    # Chance-level cross-entropy baseline for classification: log(n_classes).
    ent = math.log(data_gen.output_dim)
    print(
        f"Model: {Config.ARCHITECTURE} | {utils.count_parameters(ftn)} params, "
        f"hidden={Config.HIDDEN_SIZE} ({Config.SQUARE_DIM}x{Config.SQUARE_DIM})"
    )
    print(f"Chance-level loss: {ent:.4f}")

    print(
        f"\n{'=' * 60}\nContinual learning ({Config.N_TASK} tasks)\n{'=' * 60}"
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
        out_dir=OUT_DIR,
        collector=collector,
        save_viz=SAVE_VIZ,
        auto_prior=auto_prior,
    )
    if cp_replay:
        with open(os.path.join(OUT_DIR, f"cp_snapshots{RUN_TAG}.json"), "w") as f:
            json.dump(replay.events, f, indent=2)
    if args.checkpoint_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_out)), exist_ok=True)
        checkpoint = {
            "model": ftn.state_dict(),
            "replay": replay,
            "config_names": args.config,
            "seed": Config.SEED,
        }
        if auto_prior_enabled:
            checkpoint["autoencoder"] = auto_prior.autoencoder.state_dict()
            checkpoint["logit_net"] = auto_prior.logit_net.state_dict()
            checkpoint["ae_train_masks"] = auto_prior.train_masks
            checkpoint["ae_train_targets"] = auto_prior.train_targets
        elif not replay_prior_enabled:
            checkpoint.update({
                "ic_entries": ftn.ic_entries,
                "ic_cov": ftn.ic_cov,
                "ic_samples": ftn.ic_samples,
                "ic_support_x": ftn.ic_support_x,
                "ic_support_y": ftn.ic_support_y,
            })
        torch.save(checkpoint, args.checkpoint_out)
        print(f"Checkpoint written to {args.checkpoint_out}")
    probe_metrics = {}
    if hasattr(ftn, "_ic_match") and getattr(Config, "IC_DIAGNOSTIC_TRIALS", 0):
        probe_metrics = probe_routing(
            ftn, data_gen, list(range(Config.N_TASK)), Config.IC_DIAGNOSTIC_TRIALS)
        print("Routing probe:", probe_metrics)
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        mtl_eval = evaluate_tasks(ftn, data_gen, list(range(Config.N_TASK)),
                                  auto_prior=auto_prior)

    # Scalar metrics log (always written): continual-learning accuracies from
    # the collector plus post-hoc cost/structure metrics on the trained model.
    # Timings invoke reconfiguration, so restore the trained posterior before
    # the subsequent structural probe.
    all_tasks = list(range(Config.N_TASK))
    n_trainable, n_total = metrics_mod.count_params(ftn)
    trained_state = ftn.get_reconfig_state()
    fwd_ms, recon_ms = metrics_mod.benchmark_timings(ftn, data_gen, all_tasks[0])
    ftn.set_reconfig_state(trained_state)
    avg_mask_overlap_pct = metrics_mod.mean_pairwise_mask_overlap(
        ftn, data_gen, all_tasks
    )
    ftn.set_reconfig_state(trained_state)
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
        "avg_mask_overlap_pct": avg_mask_overlap_pct,
    }
    if cp_replay:
        metric_fields.update({
            "ic_cp_snapshot_events": len(replay.events),
            "ic_cp_replaced_slots": sum(e["replaced_slots"] for e in replay.events),
            "ic_cp_replay_eligible_events": sum(
                e.get("replay_eligible_after_change", False) for e in replay.events),
            "ic_cp_priority_draws": replay.priority_draws,
            "ic_cp_filtered_draws": replay.filtered_draws,
        })
    metric_fields.update(collector.summary())
    metric_fields.update(probe_metrics)
    if auto_prior is not None:
        metric_fields.update(auto_prior.metrics())
        if auto_prior_enabled:
            metric_fields["ae_autoencoder_params"] = utils.count_parameters(
                auto_prior.autoencoder)
            metric_fields["ae_logit_net_params"] = utils.count_parameters(
                auto_prior.logit_net)
    if hasattr(ftn, "ic_stats"):
        metric_fields.update({f"ic_{k}": v for k, v in ftn.ic_stats.items()})
        metric_fields["ic_cp_holdout_acc"] = (
            float(np.mean(ftn.ic_cp_holdout_acc)) if ftn.ic_cp_holdout_acc else None)
        metric_fields["ic_cp_holdout_loss"] = (
            float(np.mean(ftn.ic_cp_holdout_loss)) if ftn.ic_cp_holdout_loss else None)
        metric_fields["ic_cp_holdout_n"] = len(ftn.ic_cp_holdout_acc)
        metric_fields["ic_refresh_before_loss"] = (
            float(np.mean(ftn.ic_refresh_before)) if ftn.ic_refresh_before else None)
        metric_fields["ic_refresh_after_loss"] = (
            float(np.mean(ftn.ic_refresh_after)) if ftn.ic_refresh_after else None)
        metric_fields["ic_refresh_count"] = len(ftn.ic_refresh_after)
        metric_fields["ic_repair_before_loss"] = (
            float(np.mean(ftn.ic_repair_before)) if ftn.ic_repair_before else None)
        metric_fields["ic_repair_after_loss"] = (
            float(np.mean(ftn.ic_repair_after)) if ftn.ic_repair_after else None)
        metric_fields["ic_repair_accepted"] = ftn.ic_repair_accepted
    metrics_path = os.path.join(OUT_DIR, f"metrics{RUN_TAG}.txt")
    metrics_mod.write_metrics(metrics_path, metric_fields)
    print(f"Metrics written to {metrics_path}")

    # Post-hoc ENTROPY_TESTS (skipped in --metrics-only: it emits vis matrices).
    if getattr(Config, "ENTROPY_TESTS", False) and SAVE_VIZ:
        print(f"\n{'=' * 60}\nENTROPY_TESTS (OOD subspace probe)\n{'=' * 60}")
        print(f"Trials: {Config.ENTROPY_TEST_TRIALS} (dropout enabled)")
        entropy_tests(
            ftn, data_gen, list(range(Config.N_TASK)), args.name,
            out_dir=OUT_DIR, auto_prior=auto_prior
        )

    print(f"\n{'=' * 60}\nEXPERIMENT COMPLETE\n{'=' * 60}")
    print("MTL eval:", {k: f"{v:.4f}" for k, v in mtl_eval.items()})
    print(f"Chance-level loss: {ent:.4f}")
    print(f"\nOutputs saved to {OUT_DIR}/")
