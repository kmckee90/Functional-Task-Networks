from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ftn_lora import (
    LoRABankConfig,
    MaskController,
    PerplexityConfigurer,
    ReconfigConfig,
    TARGET_MODULES_DEFAULT,
    install_lora_bank,
    load_lora_bank_state_dict,
    lora_bank_state_dict,
    lora_parameters,
)


def _dtype(name: str | None) -> torch.dtype:
    if name in (None, "auto"):
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    name = str(name).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def _device(name: str | None) -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(name) == "cuda" and not torch.cuda.is_available():
        print("warning: requested cuda but CUDA is unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(str(name))


def _load_tokenizer(model_cfg: dict[str, Any]):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tok.pad_token_id is None or int(tok.pad_token_id) < 0:
        tok.pad_token = tok.eos_token
    return tok


def _load_hf_causal_lm(model_cfg: dict[str, Any]):
    from transformers import AutoModelForCausalLM

    dtype = _dtype(model_cfg.get("dtype"))
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    if model_cfg.get("device_map") is not None:
        kwargs["device_map"] = model_cfg.get("device_map")
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **kwargs)
    if model_cfg.get("device_map") is None:
        model.to(_device(model_cfg.get("device")))
    return model


class ModelAdapter:
    kind = "base"

    def __init__(self, model: nn.Module, tokenizer: Any, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def before_batch(self, batch: dict[str, Any], train: bool) -> None:
        return None

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({"model": self.model.state_dict(), "kind": self.kind}, path)

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["model"])


class FrozenHFAdapter(ModelAdapter):
    kind = "hf_frozen"

    @classmethod
    def build(cls, cfg: dict[str, Any]) -> "FrozenHFAdapter":
        model_cfg = cfg["model"]
        tok = _load_tokenizer(model_cfg)
        model = _load_hf_causal_lm(model_cfg)
        model.config.pad_token_id = tok.pad_token_id
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        device = next(model.parameters()).device
        return cls(model, tok, device)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return []

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({"kind": self.kind}, path)

    def load_checkpoint(self, path: str | Path) -> None:
        return None


class FTNLoRAAdapter(ModelAdapter):
    kind = "ftn_lora"

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device,
        controller: MaskController,
        wrappers: list[Any],
        configurer: PerplexityConfigurer,
        train_reconfig_steps: int,
        eval_reconfig_steps: int,
    ):
        super().__init__(model, tokenizer, device)
        self.controller = controller
        self.wrappers = wrappers
        self.configurer = configurer
        self.train_reconfig_steps = train_reconfig_steps
        self.eval_reconfig_steps = eval_reconfig_steps

    @classmethod
    def build(cls, cfg: dict[str, Any]) -> "FTNLoRAAdapter":
        model_cfg = cfg["model"]
        ftn_cfg = dict(cfg.get("ftn_lora", {}))
        tok = _load_tokenizer(model_cfg)
        model = _load_hf_causal_lm(model_cfg)
        model.config.pad_token_id = tok.pad_token_id
        if model_cfg.get("gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        device = next(model.parameters()).device
        lora_cfg = LoRABankConfig(
            H=int(ftn_cfg.get("H", 256)),
            k=int(ftn_cfg.get("k", 32)),
            rank=int(ftn_cfg.get("rank", 2)),
            alpha=float(ftn_cfg.get("alpha", 16.0)),
            dropout=float(ftn_cfg.get("dropout", 0.05)),
            target_modules=tuple(ftn_cfg.get("target_modules", TARGET_MODULES_DEFAULT)),
            init_b_zero=bool(ftn_cfg.get("init_b_zero", False)),
        )
        controller = MaskController(lora_cfg.H, lora_cfg.k, device=device)
        wrappers = install_lora_bank(model, lora_cfg, controller)
        for wrapper in wrappers:
            wrapper.to(dtype=_dtype(model_cfg.get("dtype")))
        for p in lora_parameters(wrappers):
            p.requires_grad = True
        grid_dim = int(ftn_cfg.get("grid_dim", int(lora_cfg.H**0.5)))
        if grid_dim * grid_dim != lora_cfg.H:
            raise ValueError("ftn_lora.grid_dim ** 2 must equal ftn_lora.H")
        rcfg = ReconfigConfig(
            grid_dim=grid_dim,
            reconfig_steps=int(ftn_cfg.get("reconfig_steps", 1)),
            train_reconfig_steps=int(ftn_cfg.get("train_reconfig_steps", 1)),
            reconfig_lr=float(ftn_cfg.get("reconfig_lr", 1.0)),
            lateral_steps=int(ftn_cfg.get("lateral_steps", 0)),
            lateral_kernel=int(ftn_cfg.get("lateral_kernel", 3)),
            lateral_polyak=float(ftn_cfg.get("lateral_polyak", 0.0)),
            torus=bool(ftn_cfg.get("torus", False)),
        )
        configurer = PerplexityConfigurer(rcfg, controller)
        n_lora = sum(p.numel() for p in lora_parameters(wrappers))
        print(f"installed FTN-LoRA bank: {len(wrappers)} layers, {n_lora/1e6:.1f}M params")
        return cls(
            model,
            tok,
            device,
            controller,
            wrappers,
            configurer,
            train_reconfig_steps=rcfg.train_reconfig_steps,
            eval_reconfig_steps=rcfg.reconfig_steps,
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return lora_parameters(self.wrappers)

    def before_batch(self, batch: dict[str, Any], train: bool) -> None:
        steps = self.train_reconfig_steps if train else self.eval_reconfig_steps
        recfg_ids = batch["recfg_input_ids"].to(self.device)
        recfg_attn = batch["recfg_attention_mask"].to(self.device)
        self.configurer(self.model, recfg_ids, recfg_attn, steps=steps)

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({"kind": self.kind, "bank": lora_bank_state_dict(self.wrappers)}, path)

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu")
        load_lora_bank_state_dict(self.wrappers, state["bank"])


class GRULanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_token_id: int,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if tie_weights:
            if embedding_dim != hidden_dim:
                raise ValueError("tie_weights requires embedding_dim == hidden_dim")
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask
        h, _ = self.gru(self.embed(input_ids))
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


class GRUAdapter(ModelAdapter):
    kind = "gru"

    @classmethod
    def build(cls, cfg: dict[str, Any]) -> "GRUAdapter":
        model_cfg = cfg["model"]
        gru_cfg = cfg.get("gru", {})
        tok = _load_tokenizer(model_cfg)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        model = GRULanguageModel(
            vocab_size=len(tok),
            embedding_dim=int(gru_cfg.get("embedding_dim", 512)),
            hidden_dim=int(gru_cfg.get("hidden_dim", 1024)),
            num_layers=int(gru_cfg.get("num_layers", 2)),
            dropout=float(gru_cfg.get("dropout", 0.1)),
            pad_token_id=int(pad or 0),
            tie_weights=bool(gru_cfg.get("tie_weights", False)),
        )
        device = _device(model_cfg.get("device"))
        model.to(device)
        return cls(model, tok, device)


def build_model(kind: str, cfg: dict[str, Any]) -> ModelAdapter:
    kind = kind.lower()
    if kind == "hf_frozen":
        return FrozenHFAdapter.build(cfg)
    if kind == "ftn_lora":
        return FTNLoRAAdapter.build(cfg)
    if kind == "gru":
        return GRUAdapter.build(cfg)
    raise ValueError(f"unknown model kind: {kind}")
