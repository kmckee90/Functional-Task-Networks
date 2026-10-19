"""
eval_compose.py - Post-hoc generalization eval for a run_compose.py checkpoint.

Loads a self-overwriting checkpoint written by run_compose.py and re-evaluates
the (frozen) model on freshly generated composition tasks, letting you vary
independently:

  (1) --reconfig-steps    eval-time mask-search steps (cold-start thoroughness)
  (2) --comp-steps        composition length of the eval data generator
  (3) --recurrent-steps   recurrent steps (== masks) the model unrolls at eval
  (4) --n-tasks           number of tasks from the generator to eval on
  (5) --batch-size        samples per eval task batch

So you can probe generalization to composition length and its dependence on the
number of reconfiguration steps. Each of (1)(2)(3) accepts a single value or a
comma-separated list; the full grid is evaluated. --recurrent-steps defaults to
matching each --comp-steps value (one recurrent step per operator).

Only the mask search runs (weights are frozen). Recurrent architectures only
(rnn / rnn_residual / lorarnn): the weight-tied cell lets the model unroll a
different number of steps than it was trained with, which is the whole point of
the composition-length probe.

Usage:
    uv run python eval_compose.py --checkpoint artifacts/<run>/checkpoint_<run>.pt \
        --comp-steps 3,4,5,6,7 --reconfig-steps 10,60 --n-tasks 8 --batch-size 256

    uv run python eval_compose.py --checkpoint artifacts/ComposeShuffleLoRA_29/checkpoint_ComposeShuffleLoRA_29.pt \
        --comp-steps 3 --reconfig-steps 60 --n-tasks 8 --batch-size 256        
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import task_comp
import tasks
from run_compose import build_model, accuracy, config_from_dict
from stack_ftn import StackFTN
from lora_rnn_ftn import LoRARNNFTN
from wave_ftn import WaveFTN


def parse_int_list(s):
    """'10,60' -> [10, 60]; '' / None -> None; '5' -> [5]."""
    if s is None or str(s).strip() == "":
        return None
    return [int(v) for v in str(s).split(",") if v.strip() != ""]


# ------------------------------------------------------- arbitrary-step forward

def reconfig_k_masks(ftn, x, y, k):
    """Cold-start search of k recurrent-step masks (plus the LoRA I/O masks when
    io='ftn'), returning the mask list. Uses the simultaneous annealed-STE (or
    'fast') search directly so k may differ from the trained step count; the
    weight-tied cell makes any k valid. Match-stored recall is auto-disabled for
    k != trained length inside the search."""
    anneal = ftn.reconfig_algorithm != "fast"
    if isinstance(ftn, LoRARNNFTN):
        io_extra = 2 if ftn.io == "ftn" else 0
        return ftn._search_simultaneous(x, y, anneal=anneal, n_masks=k + io_extra)
    return ftn._search_simultaneous(x, y, anneal=anneal, n_steps=k)


def forward_with_masks(ftn, x, masks):
    """Forward the model under an explicit per-step mask list (len(masks) steps)."""
    if isinstance(ftn, LoRARNNFTN):
        return ftn._forward_masks(x, masks, detach=False)
    return ftn._forward_stack(x, masks, detach=False)


# ------------------------------------------------------------------ eval driver

def build_eval_gen(Config, comp_steps, n_tasks, batch_size, seed):
    """DataGenComposition at a chosen composition length. Clamps n_tasks down to
    the number of distinct length-comp_steps compositions that exist (matrix_dim
    and the reject flags are inherited from the checkpoint's config)."""
    pars = {**Config.COMP_PARS, "steps": comp_steps}
    req = int(n_tasks)
    while req >= 1:
        try:
            gen = task_comp.DataGenComposition(
                n_tasks=req, n_samples=batch_size,
                rng=tasks.make_rng(seed), seed=Config.SEED, **pars,
            )
            if req < n_tasks:
                print(f"  [warn] only {req} distinct length-{comp_steps} "
                      f"compositions exist; evaluating on {req} tasks.")
            return gen, req
        except ValueError:
            req -= 1
    raise SystemExit(f"No valid length-{comp_steps} compositions for eval.")


def eval_cell(ftn, Config, gen, n_used, k, eval_batches):
    """Mean loss/acc over n_used tasks: cold-start reconfigure k masks on a fresh
    batch, score on a second fresh batch, averaged over eval_batches draws."""
    losses, accs = [], []
    for t in range(n_used):
        for _ in range(eval_batches):
            xr, yr = gen(t)
            masks = reconfig_k_masks(ftn, xr, yr, k)   # runs its own grad
            xe, ye = gen(t)
            with torch.no_grad():
                pred = forward_with_masks(ftn, xe, masks)
                losses.append(ftn.loss(pred, ye).item())
                accs.append(accuracy(pred, ye, Config))
    return float(np.mean(losses)), float(np.mean(accs))


def run_grid(ftn, Config, reconfig_list, comp_list, rec_list, n_tasks,
             batch_size, eval_batches, seed):
    rows = []
    for cs in comp_list:
        gen, n_used = build_eval_gen(Config, cs, n_tasks, batch_size, seed)
        rec_values = rec_list if rec_list is not None else [cs]
        for rs in reconfig_list:
            ftn.reconfig_steps = int(rs)
            for k in rec_values:
                loss, acc = eval_cell(ftn, Config, gen, n_used, int(k), eval_batches)
                rows.append({
                    "comp_steps": cs, "reconfig_steps": int(rs),
                    "recurrent_steps": int(k), "n_tasks": n_used,
                    "batch_size": batch_size, "eval_batches": eval_batches,
                    "mean_acc": acc, "mean_loss": loss,
                })
                print(f"  comp={cs:>2} reconfig={rs:>3} recur={k:>2} "
                      f"tasks={n_used:>2}: acc={acc:.3f} loss={loss:.4f}")
    return rows


# ----------------------------------------------------------------- output files

def save_csv(rows, path):
    cols = ["comp_steps", "reconfig_steps", "recurrent_steps", "n_tasks",
            "batch_size", "eval_batches", "mean_acc", "mean_loss"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


def save_plot(rows, path):
    """Plot mean_acc against the primary swept axis (comp_steps, else
    reconfig_steps, else recurrent_steps), one line per remaining swept combo."""
    axes = ["comp_steps", "reconfig_steps", "recurrent_steps"]
    swept = [a for a in axes if len({r[a] for r in rows}) > 1]
    if not swept:
        return  # single point, nothing to plot
    x_key = swept[0]
    line_keys = [a for a in axes if a != x_key and len({r[a] for r in rows}) > 1]

    def line_label(r):
        return ", ".join(f"{k}={r[k]}" for k in line_keys) or "all"

    groups = {}
    for r in rows:
        groups.setdefault(line_label(r), []).append(r)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r[x_key])
        xs = [r[x_key] for r in rs]
        a0.plot(xs, [r["mean_acc"] for r in rs], "-o", ms=4, label=label)
        a1.plot(xs, [r["mean_loss"] for r in rs], "-o", ms=4, label=label)
    a0.set_ylim(0, 1)
    a0.set_ylabel("mean accuracy"); a0.set_title("Accuracy vs " + x_key)
    a1.set_ylabel("mean loss"); a1.set_title("Loss vs " + x_key)
    for a in (a0, a1):
        a.set_xlabel(x_key); a.grid(alpha=0.3)
        if line_keys:
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------------- main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a run_compose.py checkpoint (.pt).")
    parser.add_argument("--reconfig-steps", default=None,
                        help="(1) eval-time mask-search steps; single or CSV list "
                        "(default: the checkpoint's trained reconfig_steps).")
    parser.add_argument("--comp-steps", default=None,
                        help="(2) composition length of the eval data; single or "
                        "CSV list (default: the checkpoint's trained COMP_STEPS).")
    parser.add_argument("--recurrent-steps", default=None,
                        help="(3) recurrent steps / masks the model unrolls; "
                        "single or CSV list (default: match each --comp-steps).")
    parser.add_argument("--n-tasks", type=int, default=8,
                        help="(4) number of generator tasks to eval on "
                        "(clamped to how many distinct compositions exist).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="(5) samples per eval task batch (default: "
                        "checkpoint's T).")
    parser.add_argument("--eval-batches", type=int, default=1,
                        help="Batches averaged per task (default 1).")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for the eval data (default: checkpoint seed).")
    parser.add_argument("--tag", default=None,
                        help="Suffix for the output files (default derived from "
                        "the checkpoint name).")
    args = parser.parse_args()

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)

    Config = config_from_dict(ck["config_dict"])
    if args.seed is not None:
        Config.SEED = args.seed
    ftn = build_model(Config, ck["input_size"], ck["output_size"])
    ftn.load_state_dict(ck["model_state"])
    ftn.eval()

    recurrent = isinstance(ftn, (LoRARNNFTN, WaveFTN)) or (
        isinstance(ftn, StackFTN) and ftn.recurrent
    )
    if not recurrent:
        raise SystemExit(
            "eval_compose.py supports recurrent models (rnn / rnn_residual / "
            f"lorarnn); checkpoint architecture is {ck.get('architecture')!r}."
        )

    # Resolve the five knobs (defaults come from the checkpoint's config).
    reconfig_list = parse_int_list(args.reconfig_steps) or [int(ftn.reconfig_steps)]
    comp_list = parse_int_list(args.comp_steps) or [int(Config.COMP_STEPS)]
    rec_list = parse_int_list(args.recurrent_steps)   # None => match comp_steps
    batch_size = int(args.batch_size) if args.batch_size else int(Config.T)
    seed = int(Config.SEED) + 100

    # A shared readout (rnn/rnn_residual, lorarnn) can unroll any number of steps;
    # per-step readouts (stack_concat_readout) are indexed by step, so k is capped
    # at the trained n_stack there.
    if isinstance(ftn, StackFTN) and ftn._per_step_readouts:
        max_k = ftn.n_stack
        eff_ks = rec_list if rec_list is not None else comp_list
        if max(eff_ks) > max_k:
            raise SystemExit(
                f"This model uses per-step readouts, so recurrent steps are "
                f"capped at n_stack={max_k}; requested up to {max(eff_ks)}."
            )

    print(f"Loaded {args.checkpoint}")
    print(f"  architecture={ck.get('architecture')} step={ck.get('step')} "
          f"trained_comp_steps={Config.COMP_STEPS} trained_reconfig={ftn.reconfig_steps}")
    print(f"Grid: reconfig={reconfig_list} comp={comp_list} "
          f"recurrent={rec_list if rec_list is not None else '=comp'} "
          f"n_tasks={args.n_tasks} batch={batch_size} eval_batches={args.eval_batches}")
    print("Results:")
    rows = run_grid(ftn, Config, reconfig_list, comp_list, rec_list, args.n_tasks,
                    batch_size, args.eval_batches, seed)

    out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    base = args.tag or os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_csv(rows, os.path.join(out_dir, f"eval_{base}.csv"))
    save_plot(rows, os.path.join(out_dir, f"eval_{base}.png"))
    print("Done.")
