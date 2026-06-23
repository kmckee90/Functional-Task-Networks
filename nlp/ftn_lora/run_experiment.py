"""Runner for FTN-LoRA on the TRACE-aligned task suite.

Usage:
    uv run python -m experiments.func_net_kt.ftn_lora.run_experiment \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --methods FTN FTN_router FTN_noreplay FixedMask FixedMask_router \
                  RandMask_router SeqLoRA SeqLoRA_ER \
        --seeds 0 \
        --train-steps-per-task 200 \
        --train-size 500
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lora_bank import LoRABankConfig, MaskController, install_lora_bank
from .tasks import DEFAULT_ORDER, build_task_suite
from .trainer import TrainConfig, block_sequential_train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--methods", nargs="+", default=["FTN"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--task-order", nargs="+", default=DEFAULT_ORDER)
    p.add_argument("--train-size", type=int, default=500)
    p.add_argument("--eval-size", type=int, default=100)
    p.add_argument("--support-size", type=int, default=16)
    p.add_argument("--train-steps-per-task", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--head-start-eval-size", type=int, default=64)
    p.add_argument("--cfg-inner-steps", type=int, default=5)
    p.add_argument("--cfg-lr", type=float, default=0.5)
    p.add_argument("--lora-H", type=int, default=128)
    p.add_argument("--lora-k", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-b-init", choices=["zero", "small"], default="zero",
                   help="zero: standard LoRA init; small: B init normal(0, 0.01)")
    p.add_argument("--results-dir", default="results/lora")
    p.add_argument("--results-suffix", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def make_model(model_name: str, lora_cfg: LoRABankConfig, device, bf16: bool):
    dtype = torch.bfloat16 if bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="eager",
    )
    model.to(device)
    model.gradient_checkpointing_enable()
    controller = MaskController(H=lora_cfg.H, k=lora_cfg.k, device=device)
    wrappers = install_lora_bank(model, lora_cfg, controller)
    # Move bank params to bf16 for efficiency; gradients are still bf16.
    for w in wrappers:
        w.to(device=device, dtype=dtype)
    n_lora = sum(p.numel() for w in wrappers for p in w.lora_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  bank size: {n_lora/1e6:.1f}M atoms, {len(wrappers)} layers")
    print(f"  total params (incl bank): {n_total/1e6:.1f}M")
    return model, tokenizer, controller, wrappers


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    lora_cfg = LoRABankConfig(
        H=args.lora_H, k=args.lora_k, rank=args.lora_rank,
        alpha=args.lora_alpha, dropout=args.lora_dropout,
        init_b_zero=(args.lora_b_init == "zero"),
    )

    print(f"Loading model {args.model} on {device}")
    model, tokenizer, controller, wrappers = make_model(args.model, lora_cfg, device, args.bf16)

    print(f"Building task suite: {args.task_order}")
    tasks = build_task_suite(
        args.task_order, seed=0,
        train_size=args.train_size, eval_size=args.eval_size,
        support_size=args.support_size,
    )
    for t in tasks:
        print(f"  task {t['spec'].name}: train={len(t['train'])} eval={len(t['eval'])} support={len(t['support'])}")

    for method in args.methods:
        for seed in args.seeds:
            torch.manual_seed(seed)
            print(f"\n=== method={method} seed={seed} ===")
            t0 = time.time()

            # Reset bank weights at the start of every (method, seed) run
            with torch.no_grad():
                import math
                for w in wrappers:
                    torch.nn.init.kaiming_uniform_(w.A, a=math.sqrt(5))
                    if lora_cfg.init_b_zero:
                        w.B.data.zero_()
                    else:
                        w.B.data.normal_(0.0, 0.01)

            train_cfg = TrainConfig(
                method=method,
                train_steps_per_task=args.train_steps_per_task,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                lr=args.lr,
                max_len=args.max_len,
                eval_batch_size=args.batch_size,
                head_start_eval_size=args.head_start_eval_size,
                cfg_inner_steps=args.cfg_inner_steps,
                cfg_lr=args.cfg_lr,
            )

            def log_fn(msg):
                print(f"  {msg}", flush=True)

            try:
                result = block_sequential_train(
                    model, tokenizer, wrappers, controller, tasks,
                    train_cfg, device, seed=seed, log_fn=log_fn,
                )
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

            method_label = method + (
                "_bnonzero" if args.lora_b_init != "zero" else ""
            )
            result["method"] = method_label
            result["seed"] = seed
            result["model"] = args.model
            result["lora_cfg"] = vars(lora_cfg)
            result["train_cfg"] = vars(train_cfg)
            result["wall_seconds"] = time.time() - t0

            out_path = Path(args.results_dir) / \
                f"trace_{method_label}_s{seed}{args.results_suffix}.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(f"  saved {out_path} ({result['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
