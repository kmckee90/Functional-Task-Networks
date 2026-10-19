"""
run.py — Driver for the LoRA-Mux LLM continual-learning experiment.

What it does:
    Loads a HuggingFace causal LM (default: distilgpt2), wraps every
    attention QKV projection (`c_attn`) with a LoRA-Mux pool of H components,
    then continually fine-tunes through n_tasks mutually-exclusive style
    transformations of the same source corpus (concept shift).

Compared methods:
    LoRA-Mux       — pool with SmoothKWTA configurer (the proposed method).
                     Reported under 'stored' (oracle) and 'reconfig'
                     (cold-start mask recovery, no task identity) protocols.
    LoRA-FixedMask — pool with deterministic disjoint-block mask per task.
                     Structural oracle; also runs both eval protocols.
    LoRA-Naive     — single shared LoRA (all-ones mask), sequentially
                     fine-tuned.  Catastrophic-forgetting baseline.

Usage:
    cd project_c_4
    uv run python experiments/lora/run.py
    # edit the constants at the top of this file to change methods/model/etc.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from lora_mux import (
    FixedMask,
    FullMask,
    LoRABank,
    MaskState,
    SmoothKWTA,
    apply_lora_mux,
    lora_mux_parameters,
)
from tasks import StyleTaskSuite
from trainer import TrainConfig, block_sequential_train, cl_metrics

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# --- Experiment ---
MODEL = "distilgpt2"
N_TASKS = 6
STEPS = 200  # training steps per task
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHODS = ["LoRA-Naive", "LoRA-FixedMask", "LoRA-FTN"]
OUT_PATH = os.path.join(HERE, "results", "lora_styles.json")

# --- Model / LoRA ---
N_COMPONENTS = 24 * 24  # pool size H per LoRA-Mux module (= GRID_DIM**2)
GRID_DIM = 24
RANK = 4
TARGET_NAMES = ["c_attn"]  # modules in each transformer block to wrap
LR = 5e-4
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
MAX_SEQ_LEN = 96

# --- Mask / configurer ---
K = (24 * 24) // 8  # active components per task (KWTA winners) = 72
RECONFIG_STEPS = 15
RECONFIG_LR = 0.5
KERNEL_SIZE = 3  # lateral smoothing kernel size for SmoothKWTA
LATERAL_STEPS = 15  # number of lateral smoothing iterations T
RECONFIG_BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_model(
    model_name: str, n_components: int, rank: int, target_names: tuple, device: str,
    random_B_init: float = 0.0,
):
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Freeze every parameter — LoRA wrappers will mark only A/B as trainable.
    for p in model.parameters():
        p.requires_grad = False
    mask_state = MaskState(n_components, device=device)
    wrappers = apply_lora_mux(
        model,
        n_components=n_components,
        rank=rank,
        mask_state=mask_state,
        target_names=target_names,
        random_B_init=random_B_init,
    )
    print(f"  wrapped {len(wrappers)} target modules", flush=True)
    n_lora_params = sum(p.numel() for p in lora_mux_parameters(wrappers))
    print(
        f"  LoRA-Mux trainable params: {n_lora_params / 1e6:.2f}M  "
        f"(H={n_components}, r={rank}, layers={len(wrappers)})",
        flush=True,
    )
    return tokenizer, model, mask_state, wrappers


# ---------------------------------------------------------------------------
# Method-specific configurers
# ---------------------------------------------------------------------------


def get_configurer(
    method: str,
    n_components: int,
    grid_dim: int,
    k: int,
    reconfig_steps: int,
    reconfig_lr: float,
    kernel_size: int = 3,
    lateral_steps: int = 15,
):
    if method == "LoRA-FTN":
        return SmoothKWTA(
            n_components=n_components,
            grid_dim=grid_dim,
            k=k,
            kernel_size=kernel_size,
            lateral_steps=lateral_steps,
            reconfig_steps=reconfig_steps,
            reconfig_lr=reconfig_lr,
        )
    if method == "LoRA-FixedMask":
        return FixedMask(n_components=n_components, k=k)
    if method == "LoRA-Naive":
        return FullMask(n_components=n_components)
    if method == "LoRA-Bank":
        return LoRABank(n_components=n_components)
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--out", type=str, default=OUT_PATH)
    parser.add_argument("--n-components", type=int, default=N_COMPONENTS)
    parser.add_argument("--grid-dim", type=int, default=GRID_DIM)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--random-B-init", type=float, default=0.0,
                        help="Std of Gaussian B init (>0 breaks symmetry; required for LoRA-Bank).")
    args = parser.parse_args()
    seed = args.seed
    methods = args.methods
    out_path = args.out
    n_components = args.n_components
    grid_dim = args.grid_dim
    rank = args.rank
    k_val = args.k
    random_B_init = args.random_B_init

    task_set = StyleTaskSuite(n_tasks=N_TASKS, seed=seed)
    print(f"styles: {task_set.style_names()}", flush=True)

    cfg = TrainConfig(
        n_steps_per_task=STEPS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        reconfig_batch_size=RECONFIG_BATCH_SIZE,
        lr=LR,
        max_seq_len=MAX_SEQ_LEN,
        device=DEVICE,
    )

    results: Dict = {
        "config": {
            "model": MODEL,
            "n_tasks": N_TASKS,
            "steps": STEPS,
            "seed": seed,
            "device": DEVICE,
            "methods": methods,
            "n_components": n_components,
            "grid_dim": grid_dim,
            "rank": rank,
            "target_names": TARGET_NAMES,
            "lr": LR,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "k": k_val,
            "reconfig_steps": RECONFIG_STEPS,
            "reconfig_lr": RECONFIG_LR,
            "kernel_size": KERNEL_SIZE,
            "lateral_steps": LATERAL_STEPS,
            "reconfig_batch_size": RECONFIG_BATCH_SIZE,
            "random_B_init": random_B_init,
        },
        "styles": task_set.style_names(),
        "methods": {},
    }

    for method in methods:
        print(f"\n{'=' * 70}\n  Method: {method}\n{'=' * 70}", flush=True)
        torch.manual_seed(seed)
        tokenizer, model, mask_state, wrappers = build_model(
            MODEL,
            n_components,
            rank,
            tuple(TARGET_NAMES),
            DEVICE,
            random_B_init=random_B_init,
        )
        configurer = get_configurer(
            method,
            n_components,
            grid_dim,
            k_val,
            RECONFIG_STEPS,
            RECONFIG_LR,
            kernel_size=KERNEL_SIZE,
            lateral_steps=LATERAL_STEPS,
        )

        t0 = time.time()
        out = block_sequential_train(
            model, mask_state, configurer, task_set, tokenizer, cfg, seed=seed
        )
        dt = time.time() - t0

        out["wallclock_sec"] = dt
        out["metrics_stored"] = cl_metrics(out["perf_matrix_stored"])
        out["metrics_reconfig"] = cl_metrics(out["perf_matrix_reconfig"])
        results["methods"][method] = out

        print(
            f"\n  [{method}] stored:   "
            f"avg_loss={out['metrics_stored']['avg_loss']:.4f}  "
            f"FM={out['metrics_stored']['FM']:.4f}  "
            f"BWT={out['metrics_stored']['BWT']:+.4f}",
            flush=True,
        )
        print(
            f"  [{method}] reconfig: "
            f"avg_loss={out['metrics_reconfig']['avg_loss']:.4f}  "
            f"FM={out['metrics_reconfig']['FM']:.4f}  "
            f"BWT={out['metrics_reconfig']['BWT']:+.4f}",
            flush=True,
        )
        print(f"  wallclock: {dt:.1f}s", flush=True)

        # Free GPU memory before the next method
        del model, tokenizer, mask_state, wrappers, configurer, out
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Final summary table
    print("\n" + "=" * 70)
    print("  Summary (per-token CE; lower is better; FM/BWT in nats)")
    print("=" * 70)
    print(
        f"{'Method':<18}  {'Protocol':<10}  {'avg_loss':>10}  {'FM':>10}  {'BWT':>10}"
    )
    for m, d in results["methods"].items():
        for proto in ("stored", "reconfig"):
            mt = d[f"metrics_{proto}"]
            print(
                f"{m:<18}  {proto:<10}  {mt['avg_loss']:>10.4f}  "
                f"{mt['FM']:>+10.4f}  {mt['BWT']:>+10.4f}"
            )


if __name__ == "__main__":
    main()
