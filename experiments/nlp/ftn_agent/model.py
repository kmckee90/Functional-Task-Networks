"""Load the backbone, install the LoRA bank, and expose small helpers.

Adapted from ftn_chat/model.py (design.md §2, §4, §6.3):
  - defaults to the cache-safe target set from lora_bank (skips k/v);
  - prints an EXACT bank-param count + a CUDA memory probe at build so we can
    confirm the config fits before committing (design.md §4.4).

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
    hd = getattr(c, "head_dim", h // heads)
    per_layer = (h * (heads * hd) + 2 * (h * kvh * hd) + (heads * hd) * h) + 3 * h * I
    embed = V * h * (1 if getattr(c, "tie_word_embeddings", False) else 2)
    total = embed + L * per_layer + h
    return total * 2 / 1024 ** 3


def _shard_plan(cfg: Config):
    """device_map + max_memory that force the backbone to SPREAD across all
    visible GPUs, so each decoder layer (and the bank atoms we attach to it,
    which inherit base.weight.device) lands on a different card."""
    n = torch.cuda.device_count()
    if n <= 1:
        return None, None
    gib = _backbone_gib(cfg.model_name)
    cap = max(2, math.ceil(gib * 1.05 / n))
    return "auto", {i: f"{cap}GiB" for i in range(n)}


def _memory_probe(prefix: str = "  ") -> None:
    """Print per-GPU allocated/reserved memory so we can see headroom."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / 1024 ** 3
        print(f"{prefix}GPU{i}: {used:.1f}/{total / 1024**3:.0f} GiB used", flush=True)


def build(cfg: Config):
    """Return (model, tokenizer, controller, wrappers, lora_cfg)."""
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map, max_memory = _shard_plan(cfg)
    print(f"  loading backbone {cfg.model_name} (~{_backbone_gib(cfg.model_name):.0f} GiB bf16)…",
          flush=True)
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
    targets = ",".join(lora_cfg.target_modules)
    print(f"  bank: H={cfg.H} ({cfg.grid_dim}x{cfg.grid_dim}) k={cfg.k} rank={cfg.rank} "
          f"targets=[{targets}]  ->  {n_lora / 1e9:.2f}B atom params "
          f"({n_lora * 2 / 1024**3:.1f} GiB bf16, "
          f"{n_lora * 8 / 1024**3:.1f} GiB fp32-Adam) over {len(wrappers)} layers")
    _memory_probe()
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


class Handles:
    """Everything the app / trainer / daemon share: one model load, reused for
    generation, reconfiguration, synthesis, and training."""

    def __init__(self, cfg: Config):
        import torch as _torch
        from perplexity_configurer import PerplexityConfigurer

        _torch.set_default_device("cuda" if _torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.model, self.tok, self.ctrl, self.wrappers, self.lora_cfg = build(cfg)
        for p in lora_params(self.wrappers):
            p.requires_grad = True
        self.cfgr = PerplexityConfigurer(cfg, self.ctrl)
        self.dev = next(self.model.parameters()).device
        # Load an existing bank checkpoint if present (design.md §7).
        bank_p = cfg.state_dir / "bank.pt"
        if bank_p.exists():
            load_bank(self.wrappers, bank_p)
            print(f"  loaded bank from {bank_p}")
        else:
            print("  no bank checkpoint yet — starting from fresh atoms")


def build_handles(cfg: Config) -> "Handles":
    return Handles(cfg)
