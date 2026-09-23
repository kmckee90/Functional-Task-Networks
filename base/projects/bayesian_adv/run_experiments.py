"""
run_experiments.py - Continual-learning experiment for the bayesian FTN
subproject.

Trains a plain-kWTA FTN on a stream of N sine classification tasks, with
optional offline replay and a backwards/prospective forgetting monitor. Pared
down from base/ to exactly this subproject: base and persistent-logit Bayesian
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
import math
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch

import config
import metrics as metrics_mod
import tasks
import utils
import viz
from ftn import BayesFTN, FTN
from bayes_lora_ftn import BayesLoRAFTN
from bayes_lora_rnn_ftn import BayesLoRARNNFTN
from bayes_stack_ftn import BayesRNNResidualFTN

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


def eval_reconfigure(ftn, x, y):
    """Run the architecture-independent, stateless evaluation mask search."""
    return ftn.cold_reconfigure(x, y, **Config.EVAL_FTN_PARS)


def evaluate_tasks(ftn, data_gen, task_indices):
    """Quick loss evaluation over multiple batches. Returns {task_idx: mean_loss}.

    Every trial cold-searches on a support batch and scores a fresh query batch.
    Restores the training inference state so eval never leaks into training.
    """
    train_state = ftn.get_reconfig_state()
    results = {}
    for task_idx in task_indices:
        losses = []
        for _ in range(Config.N_EVAL_EPOCHS):
            eval_reconfigure(ftn, *data_gen(task_idx))
            x, y = data_gen(task_idx)  # fresh query batch
            with torch.no_grad():
                losses.append(ftn.loss(ftn(x), y).item())
        results[task_idx] = np.mean(losses)
    ftn.set_reconfig_state(train_state)
    return results


def evaluate_backwards(ftn, data_gen, task_indices):
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
            eval_reconfigure(ftn, *data_gen(task_idx))
            x, y = data_gen(task_idx)  # fresh holdout batch
            with torch.no_grad():
                pred = ftn(x)
                losses.append(ftn.loss(pred, y).item())
                accs.append(accuracy(pred, y))
        results[task_idx] = {
            "loss": float(np.mean(losses)),
            "acc": float(np.mean(accs)),
        }
    ftn.set_reconfig_state(train_state)
    return results


def entropy_tests(ftn, data_gen, task_indices, suffix, out_dir="artifacts"):
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
            eval_reconfigure(ftn, *data_gen(ti))
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

            time_block = collector is not None
            if time_block and torch.cuda.is_available():
                torch.cuda.synchronize()
            block_t0 = time.perf_counter()

            # ================= inner training loop (one task block) ============
            for i in range(Config.TRAIN_DURATION):
                x, y = data_gen(task)

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
                    if changepoint:
                        ftn.reconfigure_changepoint(x, y)
                        n_block_changepoints += 1
                        r_since = 0
                    elif ftn.continuous_reactive_updates:
                        ftn._reconfig(x, y)

                    pred = ftn(x)
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
                    replay_buffer.add((x, y), ftn.m)

                if epoch == 0 and i == 0:
                    print(f"TASK {task} INITIAL_LOSS: {loss.item()}")

                # ---- weight update (+ replay-augmented gradient) ----
                optimizer.zero_grad()
                loss.backward()
                replay_step(ftn, replay_buffer)
                optimizer.step()

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

            if time_block and torch.cuda.is_available():
                torch.cuda.synchronize()
            block_wall_s = time.perf_counter() - block_t0

            if animator:
                animator.record_task(task, ftn.m, suffix, snapshot=snapshot)
            per_task_losses[task].extend(loss_history)
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
        choices=["base", "bayes", "bayes_lora", "bayes_rnn_residual", "bayes_lora_rnn"],
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
    # Per-architecture constructor parameters. The three Bayesian variants share
    # the same persistent-posterior hyperparameters (BAYES_FTN_PARS); only the
    # backbone bundle differs (ParallelMLP / LoRA atom bank / recurrent cell).
    pars_by_arch = {
        "base": dict(Config.FTN_PARS),
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
    optimizer = torch.optim.Adam(
        [p for p in ftn.parameters() if p.requires_grad], lr=Config.TRAIN_LR
    )
    replay = ReplayBuffer(
        maxlen=Config.REPLAY_MAXLEN,
        batch_size=Config.REPLAY_BATCH_SIZE,
    )

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
    print(
        "Evaluation mask search: stateless zero-logit | "
        f"steps={Config.EVAL_RECONFIG_STEPS} "
        f"lr={Config.EVAL_RECONFIG_LR} "
        f"restarts={Config.EVAL_RECONFIG_RESTARTS}"
    )
    print(
        f"Replay: {Config.N_REPLAY_SAMPLES}/step "
        f"(buffer {Config.REPLAY_MAXLEN} x {Config.REPLAY_BATCH_SIZE} rows)"
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
    )
    mtl_eval = evaluate_tasks(ftn, data_gen, list(range(Config.N_TASK)))

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
    metric_fields.update(collector.summary())
    metrics_path = os.path.join(OUT_DIR, f"metrics{RUN_TAG}.txt")
    metrics_mod.write_metrics(metrics_path, metric_fields)
    print(f"Metrics written to {metrics_path}")

    # Post-hoc ENTROPY_TESTS (skipped in --metrics-only: it emits vis matrices).
    if getattr(Config, "ENTROPY_TESTS", False) and SAVE_VIZ:
        print(f"\n{'=' * 60}\nENTROPY_TESTS (OOD subspace probe)\n{'=' * 60}")
        print(f"Trials: {Config.ENTROPY_TEST_TRIALS} (dropout enabled)")
        entropy_tests(
            ftn, data_gen, list(range(Config.N_TASK)), args.name, out_dir=OUT_DIR
        )

    print(f"\n{'=' * 60}\nEXPERIMENT COMPLETE\n{'=' * 60}")
    print("MTL eval:", {k: f"{v:.4f}" for k, v in mtl_eval.items()})
    print(f"Chance-level loss: {ent:.4f}")
    print(f"\nOutputs saved to {OUT_DIR}/")
