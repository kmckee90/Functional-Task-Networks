"""
run_compose.py - Shuffled multi-task composition experiment (NON-continual).

Unlike run_experiments.py (sequential per-task blocks, with replay and a
forgetting monitor), this trains one FTN on all training tasks *simultaneously,
in shuffled batches*:

  - J training + K held-out composition tasks (task_comp.DataGenComposition,
    built with N_TASK = SHUFFLE_N_TRAIN + SHUFFLE_N_HOLDOUT). Tasks 0..J-1 are
    trained; tasks J..J+K-1 are held out and never trained.
  - Each gradient step draws SHUFFLE_TASKS_PER_STEP (= N) batches, each from a
    DIFFERENT random training task. The model does its usual COLD-START
    reconfiguration on every batch (`ftn(x, y)`), producing a unique
    task-specific mask per batch, and the weight gradients from all N masked
    forward passes are ACCUMULATED before a single optimizer step.
  - So the shared weights (the weight-tied recurrent cell / the LoRA atom bank,
    plus the readout) are trained across many tasks at once, each reached
    through its own reconfigured mask - shuffled task training instead of
    block-sequential, but still task-specific batches with the model finding
    task-specific masks.

Focused on the recurrent architectures - the RNN-residual FTN ("rnn_residual",
stack_ftn.StackFTN) and the RNN-LoRA FTN ("lorarnn", lora_rnn_ftn.LoRARNNFTN) -
where one weight-tied cell (or one low-rank atom bank) is reused across tasks and
the mask carries the task identity. Held-out tasks are never trained; evaluating
them probes whether reconfiguration can compose the learned primitives into an
unseen function.

Usage:
    uv run python run_compose.py --config ComposeShuffle
    uv run python run_compose.py --config ComposeShuffleLoRA
    uv run python run_compose.py --config ComposeShuffle RNN_Resid_1x8 --seed 1
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import metrics as metrics_mod
import task_comp
import tasks
import utils
from ftn import FTN


# =============================================================================
# Model construction (same dispatch as run_experiments.py)
# =============================================================================


def build_model(Config, input_size, output_size):
    """Construct the FTN backbone selected by Config.ARCHITECTURE. Focused on
    the recurrent architectures but supports the full dispatch for flexibility."""
    arch = getattr(Config, "ARCHITECTURE", "base").lower()
    common = dict(
        input_size=input_size,
        output_size=output_size,
        hidden_size=Config.HIDDEN_SIZE,
    )
    if arch == "lora":
        from lora_ftn import LoRAFTN
        return LoRAFTN(**common, **Config.LORA_PARS)
    if arch in ("stack", "stack_residual", "rnn", "rnn_residual"):
        from stack_ftn import StackFTN
        return StackFTN(
            **common,
            recurrent=arch in ("rnn", "rnn_residual"),
            residual=arch in ("stack_residual", "rnn_residual"),
            **Config.STACK_PARS,
        )
    if arch == "wave":
        from wave_ftn import WaveFTN
        return WaveFTN(**common, **Config.WAVE_PARS)
    if arch == "lorarnn":
        from lora_rnn_ftn import LoRARNNFTN
        return LoRARNNFTN(**common, **Config.LORARNN_PARS)
    if arch == "hml":
        from hml_ftn_stable import HMLFTN
        return HMLFTN(**common, **Config.HML_PARS)
    return FTN(**common, **Config.FTN_PARS)


# =============================================================================
# Helpers
# =============================================================================


def make_compose_data_gen(Config, n_tasks, seed):
    """DataGenComposition with its own local RNG. Built with n_tasks = J + K so
    the same generator serves both the training and held-out task functions."""
    return task_comp.DataGenComposition(
        n_tasks=n_tasks,
        n_samples=Config.T,
        rng=tasks.make_rng(seed),
        seed=Config.SEED,
        **Config.COMP_PARS,
    )


def accuracy(pred, y, Config):
    """Per-cell binary accuracy (compose is BINARY), single-label accuracy
    (CLASSIFICATION), or 1 - MSE/var(y) clipped to [0, 1] (regression)."""
    if getattr(Config, "BINARY", False):
        return ((pred > 0) == (y > 0.5)).float().mean().item()
    if Config.CLASSIFICATION:
        target = torch.where(y == 1)[1]
        return (pred.argmax(-1) == target).float().mean().item()
    mse = torch.nn.functional.mse_loss(pred.squeeze(), y.squeeze()).item()
    var = float(y.float().var().item()) if y.numel() > 1 else 1.0
    return max(0.0, 1.0 - mse / max(var, 1e-8))


@contextmanager
def eval_reconfig_steps(ftn, Config):
    """Temporarily set the model's reconfiguration step count to
    SHUFFLE_EVAL_RECONFIG_STEPS for the duration of eval (None = keep the
    training value). All architectures read self.reconfig_steps in their search,
    and the anneal schedule rescales with it, so this changes only how thorough
    the eval-time cold starts are. Restored on exit."""
    n = getattr(Config, "SHUFFLE_EVAL_RECONFIG_STEPS", None)
    if n is None or not hasattr(ftn, "reconfig_steps"):
        yield
        return
    saved = ftn.reconfig_steps
    ftn.reconfig_steps = int(n)
    try:
        yield
    finally:
        ftn.reconfig_steps = saved


def evaluate(ftn, data_gen, task_indices, Config):
    """Cold-start reconfigure on a fresh batch, then score on a second fresh
    batch, averaged over SHUFFLE_EVAL_BATCHES. Returns {task: {loss, acc}}.
    Saves/restores ftn.m and train/eval state so eval never leaks into training.
    Uses SHUFFLE_EVAL_RECONFIG_STEPS for the cold-start search when set.
    """
    m_save = ftn.m.clone()
    was_training = ftn.training
    ftn.eval()
    results = {}
    with eval_reconfig_steps(ftn, Config):
        for t in task_indices:
            losses, accs = [], []
            for _ in range(Config.SHUFFLE_EVAL_BATCHES):
                ftn._reconfig(*data_gen(t))      # cold start on a fresh batch
                x, y = data_gen(t)               # score on a different batch
                with torch.no_grad():
                    pred = ftn(x)
                    losses.append(ftn.loss(pred, y).item())
                    accs.append(accuracy(pred, y, Config))
            results[t] = {"loss": float(np.mean(losses)), "acc": float(np.mean(accs))}
    ftn.m.copy_(m_save)
    if was_training:
        ftn.train()
    return results


def _mean(results, key):
    return float(np.mean([r[key] for r in results.values()])) if results else float("nan")


# =============================================================================
# Checkpointing (self-overwriting; consumed by eval_compose.py)
# =============================================================================


def effective_config_dict(Config):
    """Flat dict of the config's settable UPPERCASE attributes, excluding the
    computed @property fields. Enough to reconstruct an equivalent Config via
    config_from_dict when reloading a checkpoint."""
    out = {}
    for k in dir(Config):
        if not k.isupper():
            continue
        if isinstance(getattr(type(Config), k, None), property):
            continue
        out[k] = getattr(Config, k)
    return out


def config_from_dict(d):
    """Rebuild a Config from effective_config_dict output: a fresh subclass of
    config.Default carrying the saved attributes, so every derived @property
    (HIDDEN_SIZE, *_PARS, BINARY, ...) recomputes correctly and independently of
    which config classes were originally composed."""
    cls = type("CkptConfig", (config.Default,), dict(d))
    return cls()


def save_checkpoint(path, ftn, optimizer, step, meta, curves):
    """Self-overwriting checkpoint: model + optimizer + step + curves + meta.
    Written atomically (tmp then os.replace) so a kill mid-save can't corrupt the
    live file."""
    payload = {
        "step": step,
        "model_state": ftn.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "curves": curves,
        **meta,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_progress_metrics(path, meta, step, curves, wall_s):
    """Lightweight metrics refreshed at each eval, so a long-running (possibly
    never-'completing') run always has a current summary on disk."""
    last = lambda key: (curves[key][-1] if curves.get(key) else None)
    fields = {
        "run_name": meta.get("run_name"),
        "config": meta.get("config_name"),
        "architecture": meta.get("architecture"),
        "seed": meta.get("seed"),
        "step": step,
        "target_steps": meta.get("target_steps"),
        "intermediate": meta.get("intermediate"),
        "intermediate_steps": meta.get("intermediate_steps"),
        "n_train_tasks": meta.get("n_train_tasks"),
        "n_holdout_tasks": meta.get("n_holdout_tasks"),
        "train_acc": last("train_acc"),
        "train_loss": last("train_loss"),
        "holdout_acc": last("hold_acc"),
        "holdout_loss": last("hold_loss"),
        "loss_ema": last("loss_ema"),
        "wall_time_s": wall_s,
    }
    metrics_mod.write_metrics(path, fields)


def _empty_curves():
    return {"step_x": [], "loss_ema": [], "eval_x": [],
            "train_acc": [], "train_loss": [], "hold_acc": [], "hold_loss": []}


# =============================================================================
# Training
# =============================================================================


def train(ftn, train_gen, eval_gen, train_pool, eval_train_tasks,
          eval_holdout_tasks, optimizer, Config, suffix, out_dir, k_steps=None,
          ckpt_meta=None, start_step=0, curves=None):
    """Shuffled multi-task training with per-batch cold-start reconfiguration and
    N-task gradient accumulation. Returns the recorded curves.

    Training draws batches from train_gen over the task indices in train_pool.
    When k_steps is set (intermediate / single-operator training), the model runs
    only k_steps recurrent steps per batch via ftn.forward_k; otherwise it runs
    the full unroll via ftn(x, y). Evaluation always uses eval_gen (the
    full-length compositions) over eval_train_tasks and eval_holdout_tasks.

    At every eval interval it (over)writes the outputs so a long run that is
    killed at any point still leaves a current checkpoint + curve plot + metrics:
    the self-overwriting checkpoint (model/optimizer/step/curves + ckpt_meta) is
    what eval_compose.py loads. start_step/curves let a --resume run continue an
    earlier checkpoint's step count and curves.
    """
    N = int(Config.SHUFFLE_TASKS_PER_STEP)
    n_steps = int(Config.SHUFFLE_STEPS)
    eval_every = int(Config.SHUFFLE_EVAL_EVERY)
    rng = np.random.default_rng(Config.SEED + start_step)

    if N > len(train_pool):
        print(
            f"  [warn] SHUFFLE_TASKS_PER_STEP={N} > {len(train_pool)} training "
            f"tasks; sampling {N} tasks WITH replacement per step."
        )
    replace = N > len(train_pool)

    if curves is None:
        curves = _empty_curves()
    loss_ema = curves["loss_ema"][-1] if curves["loss_ema"] else None

    tag = suffix or "run"
    ckpt_path = os.path.join(out_dir, f"checkpoint_{tag}.pt")
    metrics_path = os.path.join(out_dir, f"metrics_{tag}.txt")
    t_start = time.perf_counter()

    ftn.train()
    for step in range(start_step, n_steps):
        t0 = time.time()
        # N batches, each from a DIFFERENT random training task (without
        # replacement when possible), accumulated into one optimizer step.
        picks = rng.choice(train_pool, size=N, replace=replace)
        optimizer.zero_grad()
        step_loss = 0.0
        for task in picks:
            x, y = train_gen(int(task))
            # Cold-start reconfig -> unique mask(s); full unroll, or a k_steps
            # short unroll for intermediate (single-operator) training.
            pred = ftn(x, y) if k_steps is None else ftn.forward_k(x, y, k_steps)
            loss = ftn.loss(pred, y) / N          # mean over the N task-batches
            loss.backward()                       # ACCUMULATE weight gradients
            step_loss += loss.item()
        optimizer.step()

        loss_ema = step_loss if loss_ema is None else (
            Config.LOSS_EMA_SMOOTHING * loss_ema
            + (1 - Config.LOSS_EMA_SMOOTHING) * step_loss
        )
        curves["step_x"].append(step)
        curves["loss_ema"].append(loss_ema)

        if step == start_step:
            print(f"STEP {step} accumulated loss (mean over {N} tasks): {step_loss:.4f}")

        t_train = time.time()

        print(
            f"  [{suffix}] step {step:>5}: "
            f"loss_ema={loss_ema:.4f} | "
            f"{t_train-t0:.4f}s/train | "
        ,)

        if step % eval_every == 0 or step == n_steps - 1:
            tr = evaluate(ftn, eval_gen, eval_train_tasks, Config)
            ho = evaluate(ftn, eval_gen, eval_holdout_tasks, Config)
            t_eval = time.time()
            curves["eval_x"].append(step)
            curves["train_acc"].append(_mean(tr, "acc"))
            curves["train_loss"].append(_mean(tr, "loss"))
            curves["hold_acc"].append(_mean(ho, "acc"))
            curves["hold_loss"].append(_mean(ho, "loss"))
            print(
                f"  [{suffix}] step {step:>5}: "
                f"train loss={curves['train_loss'][-1]:.4f} acc={curves['train_acc'][-1]:.3f} | "
                f"holdout loss={curves['hold_loss'][-1]:.4f} acc={curves['hold_acc'][-1]:.3f} | "
                f"loss_ema={loss_ema:.4f} | "
                f"{np.round(100*(t_train-t0))/100}s/train | "
                f"{np.round(100*(t_eval-t_train))/100}s/eval | "
            ,)

            # Self-overwriting outputs at each eval interval so the run is useful
            # even if it never "completes" (just runs a long time and improves).
            save_curves_plot(curves, suffix, out_dir)
            if ckpt_meta is not None:
                save_checkpoint(ckpt_path, ftn, optimizer, step, ckpt_meta, curves)
                write_progress_metrics(metrics_path, ckpt_meta, step, curves,
                                       time.perf_counter() - t_start)

    return curves


def save_curves_plot(curves, suffix, out_dir):
    """Training-loss EMA + train-vs-holdout eval accuracy/loss over steps."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax0.plot(curves["step_x"], curves["loss_ema"], color="tab:blue")
    ax0.set_title("Training loss (EMA of accumulated step loss)")
    ax0.set_xlabel("gradient step")
    ax0.set_ylabel("loss")
    ax0.grid(alpha=0.3)

    ax1.plot(curves["eval_x"], curves["train_acc"], "-o", ms=3,
             color="tab:green", label="train acc")
    ax1.plot(curves["eval_x"], curves["hold_acc"], "-o", ms=3,
             color="tab:red", label="holdout acc")
    ax1.set_ylim(0, 1)
    ax1.set_title("Eval accuracy: train vs held-out tasks")
    ax1.set_xlabel("gradient step")
    ax1.set_ylabel("accuracy")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    fig.tight_layout()
    tag = f"_{suffix}" if suffix else ""
    path = os.path.join(out_dir, f"compose_curves{tag}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="auto",
                        help="Run name; outputs go to <out-root>/<name>/. 'auto' "
                        "derives it from the --config classes plus an index.")
    parser.add_argument("--config", nargs="+", default=["ComposeShuffle"],
                        help="config.py class names, combined left to right "
                        "(later overrides earlier). Defaults to ComposeShuffle.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override Config.SEED.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override Config.SHUFFLE_STEPS.")
    parser.add_argument("--out-root", default="artifacts",
                        help="Root dir for run outputs (default 'artifacts').")
    parser.add_argument("--noidx", action="store_true",
                        help="With --name auto, use <base>_0 and overwrite "
                        "instead of scanning for the lowest free index.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the run's self-overwriting checkpoint "
                        "(checkpoint_<name>.pt in the run dir) if present: load "
                        "model/optimizer/step/curves and continue. Use the same "
                        "--config/--name as the original run.")
    parser.add_argument("--resume-from", default=None,
                        help="Path to an existing run's artifact FOLDER; load its "
                        "checkpoint and continue. The run name, output folder, and "
                        "full config are taken from the checkpoint (so --config / "
                        "--name are not needed). Combine with --steps to extend "
                        "training. If the folder holds several checkpoints, pass "
                        "--name to pick one.")
    args = parser.parse_args()

    # --resume-from: load the checkpoint up front so the run name, output folder,
    # and config all come from it (self-contained continuation from a folder path).
    resume_ck = None
    if args.resume_from:
        import glob
        rdir = args.resume_from.rstrip("/")
        if args.name != "auto":
            ckpt_file = os.path.join(rdir, f"checkpoint_{args.name}.pt")
        else:
            found = sorted(glob.glob(os.path.join(rdir, "checkpoint_*.pt")))
            if not found:
                raise SystemExit(f"No checkpoint_*.pt found in {rdir}")
            if len(found) > 1:
                raise SystemExit(
                    f"Multiple checkpoints in {rdir}; pass --name to pick one: "
                    + ", ".join(os.path.basename(f) for f in found)
                )
            ckpt_file = found[0]
        if not os.path.exists(ckpt_file):
            raise SystemExit(f"Checkpoint not found: {ckpt_file}")
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        resume_ck = torch.load(ckpt_file, map_location=_dev, weights_only=False)
        args.name = resume_ck.get("run_name") or os.path.basename(rdir)
        args.resume = True
        print(f"--resume-from {rdir}: continuing run '{args.name}' from step "
              f"{resume_ck['step']}.")

    if args.name == "auto":
        base = "_".join(args.config)
        i = 0
        if not args.noidx:
            existing = set()
            if os.path.isdir(args.out_root):
                existing = {d for d in os.listdir(args.out_root)
                            if os.path.isdir(os.path.join(args.out_root, d))}
            while f"{base}_{i}" in existing:
                i += 1
        args.name = f"{base}_{i}"
        print(f"--name auto resolved to: {args.name}")

    RUN_TAG = f"_{args.name}" if args.name else ""
    # --resume-from writes back into the same folder it was given; otherwise the
    # run gets its own <out-root>/<name>/ directory.
    if args.resume_from:
        OUT_DIR = args.resume_from.rstrip("/")
    else:
        OUT_DIR = os.path.join(args.out_root, args.name) if args.name else args.out_root

    # Config: reconstructed from the checkpoint when resuming from a folder (so it
    # exactly matches the saved model), else composed from the --config classes.
    if resume_ck is not None:
        Config = config_from_dict(resume_ck["config_dict"])
    else:
        # Combine configs via MRO: bases reversed so the LAST listed wins conflicts.
        _bases = tuple(getattr(config, c) for c in reversed(args.config))
        Config = type("+".join(args.config), _bases, {})()
    if args.seed is not None:
        Config.SEED = args.seed
    if args.steps is not None:
        Config.SHUFFLE_STEPS = args.steps

    if getattr(Config, "TASK", "sine").lower() != "compose":
        print("[warn] run_compose.py is a composition experiment but TASK != "
              "'compose'. Add a compose config (e.g. --config ComposeShuffle).")

    J = int(Config.SHUFFLE_N_TRAIN)
    K = int(Config.SHUFFLE_N_HOLDOUT)
    n_tasks = J + K

    print("Running run_compose (shuffled multi-task composition):")
    print(f"Config: {type(Config).__name__}")
    print(f"Architecture: {getattr(Config, 'ARCHITECTURE', 'base')}")
    print(f"Tasks: J={J} training + K={K} held-out = {n_tasks}; "
          f"{Config.SHUFFLE_TASKS_PER_STEP} tasks/step, {Config.SHUFFLE_STEPS} steps")

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Record the effective config so the run directory is self-describing.
    with open(os.path.join(OUT_DIR, f"config{RUN_TAG}.txt"), "w") as f:
        f.write(f"# config: {type(Config).__name__}\n")
        f.write(f"# argv: {' '.join(sys.argv)}\n")
        for k in sorted(k for k in dir(Config) if k.isupper()):
            f.write(f"{k} = {getattr(Config, k)!r}\n")

    if Config.SEED is not None:
        torch.manual_seed(Config.SEED)

    # Eval generator: the full-length (COMP_STEPS) compositions, always used for
    # evaluation, split into J trained + K held-out tasks.
    eval_gen = make_compose_data_gen(Config, n_tasks, Config.SEED + 100)
    train_tasks = list(range(J))
    holdout_tasks = list(range(J, n_tasks))
    print("Eval training-split composition tasks:")
    for t in train_tasks:
        print(f"  task {t}: {eval_gen.task_description(t)}")
    print("Held-out composition tasks:")
    for t in holdout_tasks:
        print(f"  task {t}: {eval_gen.task_description(t)}")

    ftn = build_model(Config, eval_gen.total_input_dim, eval_gen.output_dim)
    optimizer = torch.optim.Adam(
        [p for p in ftn.parameters() if p.requires_grad], lr=Config.TRAIN_LR
    )

    # Training source. Default: train on the length-COMP_STEPS training
    # compositions (train_tasks) with the full unroll. Intermediate mode: train
    # the recurrent cell directly on a separate single-operator dataset
    # (COMPOSE_INTERMEDIATE_STEPS-length compositions) with a k_steps short
    # unroll (forward_k). Eval stays on the full-length compositions either way.
    intermediate = bool(getattr(Config, "COMPOSE_INTERMEDIATE", False))
    if intermediate:
        k_steps = int(Config.COMPOSE_INTERMEDIATE_STEPS)
        inter_n = int(Config.COMPOSE_INTERMEDIATE_N_TASKS)
        if not hasattr(ftn, "forward_k"):
            raise SystemExit(
                f"COMPOSE_INTERMEDIATE requires a recurrent architecture with "
                f"forward_k (rnn / rnn_residual / lorarnn); got "
                f"ARCHITECTURE={getattr(Config, 'ARCHITECTURE', 'base')!r}."
            )
        stack_layers = int(getattr(Config, "STACK_LAYERS", k_steps))
        if k_steps > stack_layers:
            raise SystemExit(
                f"COMPOSE_INTERMEDIATE_STEPS={k_steps} must be <= "
                f"STACK_LAYERS={stack_layers} (the recurrent step budget)."
            )
        # Single-operator (primitive) generator: same task shape but shorter.
        inter_pars = {**Config.COMP_PARS, "steps": k_steps}
        train_gen = task_comp.DataGenComposition(
            n_tasks=inter_n, n_samples=Config.T,
            rng=tasks.make_rng(Config.SEED + 200), seed=Config.SEED, **inter_pars,
        )
        train_pool = list(range(inter_n))
        print(f"\nINTERMEDIATE training: {inter_n} primitive tasks, "
              f"steps={k_steps} (short unroll), eval on length-{Config.COMP_STEPS} "
              f"compositions:")
        for t in train_pool:
            print(f"  primitive {t}: {train_gen.task_description(t)}")
    else:
        k_steps = None
        train_gen = eval_gen
        train_pool = train_tasks

    # Chance-level per-cell BCE baseline (compose is BINARY): binary entropy H(p).
    p = float(getattr(Config, "COMP_P", 0.5))
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p)) if 0.0 < p < 1.0 else 0.0
    print(f"Model: {utils.count_parameters(ftn)} trainable params, "
          f"hidden={Config.HIDDEN_SIZE} ({Config.SQUARE_DIM}x{Config.SQUARE_DIM})")
    print(f"Chance-level loss: {ent:.4f}")

    # Metadata saved inside every checkpoint so eval_compose.py can rebuild an
    # identical model and generator from the file alone.
    ckpt_meta = {
        "run_name": args.name,
        "config_name": (resume_ck.get("config_name") if resume_ck
                        else type(Config).__name__),
        "config_names": (resume_ck.get("config_names") if resume_ck
                         else args.config),
        "config_dict": effective_config_dict(Config),
        "architecture": getattr(Config, "ARCHITECTURE", "base"),
        "seed": Config.SEED,
        "input_size": eval_gen.total_input_dim,
        "output_size": eval_gen.output_dim,
        "matrix_dim": int(Config.COMP_MATRIX_DIM),
        "n_train_tasks": J,
        "n_holdout_tasks": K,
        "intermediate": intermediate,
        "intermediate_steps": (k_steps if intermediate else None),
        "target_steps": int(Config.SHUFFLE_STEPS),
    }

    # Resume: --resume-from already loaded the checkpoint (resume_ck); --resume
    # loads checkpoint_<name>.pt from the run dir if present. Either way, restore
    # model/optimizer/step/curves and continue.
    start_step, resume_curves = 0, None
    ckpt_path = os.path.join(OUT_DIR, f"checkpoint_{args.name}.pt")
    ck = resume_ck
    if ck is None and args.resume and os.path.exists(ckpt_path):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    if ck is not None:
        ftn.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_step = int(ck["step"]) + 1
        resume_curves = ck.get("curves")
        print(f"Resumed at step {start_step} (target {Config.SHUFFLE_STEPS}).")
        if start_step >= int(Config.SHUFFLE_STEPS):
            print(f"  [note] start step {start_step} >= SHUFFLE_STEPS "
                  f"{Config.SHUFFLE_STEPS}; pass --steps to extend training.")
    elif args.resume:
        print(f"--resume set but no checkpoint at {ckpt_path}; starting fresh.")

    print(f"\n{'=' * 60}\nShuffled training ({J} train / {K} holdout tasks)\n{'=' * 60}")
    t0 = time.perf_counter()
    curves = train(ftn, train_gen, eval_gen, train_pool, train_tasks,
                   holdout_tasks, optimizer, Config, args.name, OUT_DIR,
                   k_steps=k_steps, ckpt_meta=ckpt_meta, start_step=start_step,
                   curves=resume_curves)
    wall_s = time.perf_counter() - t0

    # Final eval + scalar metrics log (always on the full-length compositions).
    tr = evaluate(ftn, eval_gen, train_tasks, Config)
    ho = evaluate(ftn, eval_gen, holdout_tasks, Config)
    n_trainable, n_total = metrics_mod.count_params(ftn)
    with eval_reconfig_steps(ftn, Config):
        avg_mask_overlap_pct = metrics_mod.mean_pairwise_mask_overlap(
            ftn, eval_gen, train_tasks
        )
    metric_fields = {
        "run_name": args.name,
        "config": type(Config).__name__,
        "seed": Config.SEED,
        "architecture": getattr(Config, "ARCHITECTURE", "base"),
        "n_train_tasks": J,
        "n_holdout_tasks": K,
        "tasks_per_step": Config.SHUFFLE_TASKS_PER_STEP,
        "n_steps": Config.SHUFFLE_STEPS,
        "intermediate": intermediate,
        "intermediate_steps": (k_steps if intermediate else None),
        "n_params_trainable": n_trainable,
        "n_params_total": n_total,
        "train_acc_final": _mean(tr, "acc"),
        "train_loss_final": _mean(tr, "loss"),
        "holdout_acc_final": _mean(ho, "acc"),
        "holdout_loss_final": _mean(ho, "loss"),
        "chance_loss": ent,
        "avg_mask_overlap_pct": avg_mask_overlap_pct,
        "wall_time_train_s": wall_s,
    }
    metrics_path = os.path.join(OUT_DIR, f"metrics{RUN_TAG}.txt")
    metrics_mod.write_metrics(metrics_path, metric_fields)
    print(f"Metrics written to {metrics_path}")

    print(f"\n{'=' * 60}\nEXPERIMENT COMPLETE\n{'=' * 60}")
    print(f"Final train:   acc={_mean(tr, 'acc'):.3f} loss={_mean(tr, 'loss'):.4f}")
    print(f"Final holdout: acc={_mean(ho, 'acc'):.3f} loss={_mean(ho, 'loss'):.4f}")
    print(f"Chance-level loss: {ent:.4f}")
    print(f"\nOutputs saved to {OUT_DIR}/")
