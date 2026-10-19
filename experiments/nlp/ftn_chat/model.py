"""Load the Qwen backbone, install the LoRA bank, and expose small helpers.

The bank attaches ONLY to the last `cfg.lora_layer_frac` fraction of the
decoder stack (default 25%): the LoCoMo layer-localization sweep showed
extractable semantic knowledge concentrates in the last ~25% of blocks while
mid-stack deltas are dead weight. Within a banked layer, EVERY nn.Linear is
wrapped (architecture-agnostic — covers Qwen3.6's Gated DeltaNet projections
as well as classic attention/MLP names).

Assumes torch.set_default_device("cuda") has been called at the entry point, so
we avoid per-tensor device plumbing.
"""
from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from lora_bank import (
    BankedLinear,
    LoRABankConfig,
    MaskController,
    install_lora_bank,
    lora_bank_state_dict,
    load_lora_bank_state_dict,
)


def banked_layer_range(n_layers: int, layer_frac: float) -> tuple[int, int]:
    """[start, end) decoder-layer indices that receive the bank."""
    n_banked = max(1, round(n_layers * layer_frac))
    return n_layers - n_banked, n_layers


def _shard_plan(cfg: Config):
    """Keep the stateful trainer on one GPU when it fits; shard if requested."""
    if torch.cuda.device_count() == 0:
        return None
    if not cfg.model_parallel:
        return 0
    return "balanced" if torch.cuda.device_count() > 1 else None


def build(cfg: Config):
    """Return (model, tokenizer, controller, wrappers, lora_cfg)."""
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = _shard_plan(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=dtype, attn_implementation=cfg.attn_implementation,
        device_map=device_map,
    )
    # Freeze the entire backbone before adding trainable atoms. Freezing only
    # the banked layers leaves the earlier transformer blocks accumulating
    # gradients that the LoRA optimizer neither uses nor clears.
    model.requires_grad_(False)
    if device_map is not None:
        devs = {str(p.device) for p in model.parameters()}
        print(f"  backbone placed on {len(devs)} GPU(s): {sorted(devs)}")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_cfg = LoRABankConfig(
        H=cfg.H, k=cfg.k, rank=cfg.rank, alpha=cfg.alpha,
        dropout=cfg.dropout, init_b_zero=cfg.init_b_zero,
    )
    controller = MaskController(H=cfg.H, k=cfg.k)

    layers = model.model.layers
    start, end = banked_layer_range(len(layers), cfg.lora_layer_frac)
    wrappers: list[BankedLinear] = []
    for i in range(start, end):
        wrappers += install_lora_bank(layers[i], lora_cfg, controller)
    for w in wrappers:
        w.to(dtype=dtype)

    n_lora = sum(p.numel() for w in wrappers for p in w.lora_parameters())
    print(f"  bank: H={cfg.H} ({cfg.grid_dim}x{cfg.grid_dim}) k={cfg.k} "
          f"rank={cfg.rank} -> {n_lora/1e9:.2f}B atom params over "
          f"{len(wrappers)} linears in decoder layers [{start}, {end}) of {len(layers)}")
    return model, tokenizer, controller, wrappers, lora_cfg


def reset_bank(wrappers, lora_cfg: LoRABankConfig):
    with torch.no_grad():
        for w in wrappers:
            torch.nn.init.kaiming_uniform_(w.A, a=math.sqrt(5))
            if lora_cfg.init_b_zero:
                w.B.data.zero_()
            else:
                w.B.data.normal_(0.0, 0.01)


def save_bank(wrappers, path):
    torch.save(lora_bank_state_dict(wrappers), path)


def load_bank(wrappers, path):
    load_lora_bank_state_dict(wrappers, torch.load(path, map_location="cpu"))


def lora_params(wrappers):
    return [p for w in wrappers for p in w.lora_parameters()]
