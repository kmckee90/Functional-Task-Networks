"""Compare GPU placements using unmodified BayesLoRA training steps.

Run with the default single-GPU placement, or set
FTN_BENCH_MODEL_PARALLEL=1 and vary CUDA_VISIBLE_DEVICES for sharding.
No checkpoint is loaded or written. The timed region calls train_block, so it
includes posterior inference, context distillation, replay bookkeeping, and
the optimizer step under the ordinary training settings.
"""
from __future__ import annotations

import json
import os
import time

import torch

torch.set_default_device("cuda")
torch.manual_seed(0)

from bayes_configurer import BayesianPerplexityConfigurer
from config import CFG
from data import build_examples, build_task, format_distill
from mask_replay import MaskReplayBuffer
from model import build, lora_params
from train import train_block

CFG.attn_implementation = os.environ.get("FTN_BENCH_ATTN", CFG.attn_implementation)
CFG.model_parallel = os.environ.get("FTN_BENCH_MODEL_PARALLEL") == "1"
CFG.batch_size = int(os.environ.get("FTN_BENCH_BATCH", CFG.batch_size))
if os.environ.get("FTN_BENCH_AUX") == "1":
    # Exercise both auxiliary backwards even in a two-step, one-task probe.
    CFG.source_ce_prob = 1.0
    CFG.replay_store_prob = 1.0
    CFG.replay_use_prob = 1.0
    CFG.replay_min_gap = 1
    CFG.replay_overlap_min = 0.0
    CFG.replay_overlap_max = 1.0


def sync():
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def main():
    started = time.perf_counter()
    model, tok, ctrl, wrappers, _ = build(CFG)
    bank_ids = {id(p) for p in lora_params(wrappers)}
    print("BACKBONE_TRAINABLE " + str(sum(p.numel() for p in model.parameters()
          if p.requires_grad and id(p) not in bank_ids)), flush=True)
    sync()
    built = time.perf_counter() - started
    placements = sorted({str(p.device) for p in model.parameters()})
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError(f"model is not wholly on visible GPUs: {placements}")

    for p in lora_params(wrappers):
        p.requires_grad = True
    opt = torch.optim.Adam(lora_params(wrappers), lr=CFG.lr, foreach=False)
    cfgr = BayesianPerplexityConfigurer(CFG, ctrl)
    replay = MaskReplayBuffer(CFG)
    gen = torch.Generator(device="cpu").manual_seed(CFG.seed)
    task = os.environ.get("FTN_BENCH_TASK", "base_ftn")
    data = build_task(CFG, tok, task)
    examples = build_examples(CFG, tok, task, data)
    examples["source"] = data.train
    source_gen = torch.Generator(device="cpu").manual_seed(CFG.seed + 104729)
    if os.environ.get("FTN_BENCH_SYNTHETIC_MAX") == "1":
        synthetic = format_distill("What is this?", " ".join(["answer"] * 1600),
                                   " ".join(["context"] * 1200), tok, CFG)
        examples["qa"] = [synthetic]
    if os.environ.get("FTN_BENCH_LONGEST") == "1":
        examples["qa"] = sorted(examples["qa"],
                                key=lambda e: max(len(e["ids"]), len(e["t_ids"])),
                                reverse=True)[:CFG.batch_size]
    print("BENCH_LENGTHS " + json.dumps({
        "student_max": max(len(e["ids"]) for e in examples["qa"]),
        "teacher_max": max(len(e["t_ids"]) for e in examples["qa"]),
    }), flush=True)
    ref_acc = torch.zeros(CFG.H, device="cpu")
    dev = next(model.parameters()).device

    n_steps = int(os.environ.get("FTN_BENCH_STEPS", "4"))
    sync()
    t0 = time.perf_counter()
    # Keep steps in one real training block: tensors surviving until the next
    # iteration contribute to the peak and must not be hidden by this harness.
    train_block(model, ctrl, cfgr, opt, task, examples, [], replay,
                steps=n_steps, rsteps=CFG.bayes_changepoint_reconfig_steps,
                gen=gen, source_gen=source_gen, dev=dev, tok=tok, t0=time.time(), epoch=0,
                ref_acc=ref_acc)
    sync()
    elapsed = time.perf_counter() - t0
    print("BENCH_RESULT " + json.dumps({
        "visible_gpus": torch.cuda.device_count(),
        "parameter_devices": placements,
        "batch_size": CFG.batch_size,
        "task": task,
        "attention": CFG.attn_implementation,
        "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
        "build_seconds": round(built, 3),
        "steps": n_steps,
        "mean_step_seconds": round(elapsed / n_steps, 3),
        "replay_entries": len(replay),
        "backbone_grad_elements": sum(p.numel() for p in model.parameters()
                                      if id(p) not in bank_ids and p.grad is not None),
        "peak_gb": [round(torch.cuda.max_memory_allocated(i) / 1e9, 2)
                    for i in range(torch.cuda.device_count())],
    }), flush=True)


if __name__ == "__main__":
    main()
