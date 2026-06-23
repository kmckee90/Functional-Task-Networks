"""Load the Qwen backbone, install the LoRA bank, and expose small helpers.

Assumes torch.set_default_device("cuda") has been called at the entry point, so
we avoid per-tensor device plumbing.
"""
from __future__ import annotations

import math

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from config import Config
from lora_bank import (
    BankedLinear,
    LoRABankConfig,
    MaskController,
    install_lora_bank,
    lora_bank_state_dict,
    load_lora_bank_state_dict,
)


def _backbone_gib(name: str) -> float:
    """Rough bf16 footprint (GiB) of the frozen backbone, from its config."""
    c = AutoConfig.from_pretrained(name)
    h, L, I, V = (c.hidden_size, c.num_hidden_layers,
                  c.intermediate_size, c.vocab_size)
    heads = c.num_attention_heads
    kvh = getattr(c, "num_key_value_heads", heads)
    hd = h // heads
    per_layer = (h * h + 2 * (h * kvh * hd) + h * h) + 3 * h * I   # attn + mlp
    embed = V * h * (1 if getattr(c, "tie_word_embeddings", False) else 2)
    total = embed + L * per_layer + h
    return total * 2 / 1024 ** 3


def _shard_plan(cfg: Config):
    """device_map + max_memory that force the backbone to SPREAD across all
    visible GPUs, so each decoder layer (and the bank atoms we attach to it,
    which inherit base.weight.device) lands on a different card. Returns
    (device_map, max_memory) or (None, None) for the single-GPU path."""
    n = torch.cuda.device_count()
    if n <= 1:
        return None, None
    gib = _backbone_gib(cfg.model_name)
    # ceil-divide forces use of all n GPUs (cap*(n-1) < gib <= cap*n), with a
    # little slack; the bank we add afterwards is NOT counted here (accelerate
    # only plans the backbone load) but co-locates onto these same devices.
    cap = max(2, math.ceil(gib * 1.05 / n))
    return "auto", {i: f"{cap}GiB" for i in range(n)}


def build(cfg: Config):
    """Return (model, tokenizer, controller, wrappers, lora_cfg)."""
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map, max_memory = _shard_plan(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=dtype, attn_implementation="eager",
        device_map=device_map, max_memory=max_memory,
    )
    if device_map is not None:
        devs = {str(p.device) for p in model.parameters()}
        print(f"  backbone sharded across {len(devs)} GPUs: {sorted(devs)}")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_cfg = LoRABankConfig(
        H=cfg.H, k=cfg.k, rank=cfg.rank, alpha=cfg.alpha,
        dropout=cfg.dropout, init_b_zero=cfg.init_b_zero,
    )
    controller = MaskController(H=cfg.H, k=cfg.k)
    wrappers = install_lora_bank(model, lora_cfg, controller)
    for w in wrappers:
        w.to(dtype=dtype)

    n_lora = sum(p.numel() for w in wrappers for p in w.lora_parameters())
    print(f"  bank: H={cfg.H} ({cfg.grid_dim}x{cfg.grid_dim}) k={cfg.k} "
          f"rank={cfg.rank} -> {n_lora/1e6:.1f}M atom params over {len(wrappers)} layers")
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
