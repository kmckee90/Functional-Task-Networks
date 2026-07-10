"""Plain frozen-backbone loader for baselines (floor/ceiling).

Deliberately NOT model.py's bank-installed build: no LoRA bank, sdpa attention
(model.py uses eager for the bank's reconfig backward — irrelevant here), KV
cache on. Much faster for the 26k-token ceiling contexts.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-32B"


def load_backbone(model_name: str = MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    return model, tok


@torch.no_grad()
def generate_answer(model, tok, system: str, user: str, max_new_tokens: int = 128) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True,
                                      enable_thinking=False)
    except TypeError:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    # Qwen3 may still emit an empty think block with enable_thinking=False.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
