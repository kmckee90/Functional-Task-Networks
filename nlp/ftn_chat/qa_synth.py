"""Synthesize context-distillation data from each task's source text.

Two stages, both with a LOCAL instruct model (no egress), and both BATCHED
across chunks (`CFG.synth_gen_batch` sequences per generate call) so the GPUs
stay saturated — single-stream decode was the synthesis bottleneck:

  1. QUESTIONS — with each chunk in context, generate `CFG.qa_per_chunk`
     diverse, self-contained questions (factual / conceptual / parameter-
     specific / applied / paraphrase), phrased as a user would ask an
     assistant about the project by name.
  2. TEACHER ANSWERS — the teacher answers each question WITH its chunk in
     context. Grounded answer + chunk are stored as (question, answer,
     context) triples.

train.py then distills: teacher = frozen backbone + context + zeroed mask;
student = masked bank, closed book; KL on the answer tokens. The student never
sees raw source text as a training target.

Cache: data_cache/qa/<task>.json. Run once before training:

    uv run python qa_synth.py [task ...]
"""

from __future__ import annotations

import json
import re
import sys
import time

import torch
from config import CFG
from data import build_task, teacher_user_msg
from transformers import AutoModelForCausalLM, AutoTokenizer

CTX = {
    "base_ftn": (
        "base FTN implementation — ftn.py (the FTN class: a masked ParallelMLP "
        "network that owns its task mask and reconfigures it with annealed "
        "straight-through KWTA, plus optional SmoothKWTA lateral mask smoothing "
        "and neurogenesis via expand()) and config.py (experiment configuration "
        "classes Default, Smoke, DomainShift, Replay, NoReplay, Reactive_Low/"
        "Med/High, PerTask, Smooth, PolyakSmooth, Fast, Large, with a glossary "
        "documenting every parameter)"
    ),
    "explorer": (
        "explorer agent model in agent_recode.py — RDDPG_Dual, a recurrent DDPG "
        "exploration agent with dual critics over an observation stream and a "
        "feedback stream, fixed (untrained) reservoir RNN state encoders, an "
        "ExplorerPolicy MLP actor, epsilon-greedy actions, prioritized/goal "
        "replay, and intrinsic novelty rewards from RECODE (recode.py), a "
        "cluster-memory density estimator with kNN / soft-visitation rewards"
    ),
    "globo": (
        "Globo game (globo_game) — a survival game built on the Luanti/Minetest "
        "engine as an RL training environment, derived from the Exile game. The "
        "focus is what Globo ADDS or CHANGES on top of the base engine: its "
        "custom mods (main, animals, climate, health, nodes_nature, minimal, "
        "std, respawn), survival stats (health, hunger, thirst, energy, body "
        "temperature), egg-laying as the score, sleeping, custom chat commands, "
        "and its globo_* settings; mobkit, naturalslopeslib, player_monoids, "
        "stairs, wielded_light and lightning are vendored third-party mods"
    ),
    "rl_mt10": (
        "FTN robotics MT10 scripts — train_mt10_bc.py (sequential behavior "
        "cloning of Meta-World MT10 scripted experts with an FTN actor whose "
        "mask is reconfigured by annealed-STE search on the BC loss itself, "
        "with mask-overlap-filtered BC replay) and train_mt10_ddpg.py "
        "(sequential DDPG with separate FTN actor and critic masks, critic "
        "reconfig objectives such as r_returns, cheat-init exploration, and "
        "mask-overlap-filtered chunked replay), plus their shared ftn.py "
        "(FTNActor, FTNCritic, anneal_ste_mask, kwta)"
    ),
}


def _batched(gen, gtok, prompts, max_new_tokens, do_sample, temperature, think,
             bs):
    """Run _generate over `prompts` in batches of bs; return all completions."""
    out = []
    for i in range(0, len(prompts), bs):
        out.extend(_generate(gen, gtok, prompts[i:i + bs], max_new_tokens,
                             do_sample, temperature, think))
    return out

Q_PROMPT = (
    "You are given an excerpt from the {ctx}.\n\n"
    "EXCERPT:\n{chunk}\n\n"
    "Write {n} diverse, self-contained questions about this material, as if a "
    "user asked an assistant about the project by name (no access to the "
    "excerpt).\n"
    "Requirements:\n"
    '- NEVER refer to "the excerpt", "the code above", "this file", or line '
    "numbers — each question must stand alone, naming the script, class, "
    "parameter, or concept it asks about.\n"
    "- Mix question types: factual (what does X do / what is the default of X), "
    "conceptual (why / how does X work), applied (what happens if / how would "
    "I), and comparative (how does X differ from Y).\n"
    "- Include a few paraphrases: the same underlying fact asked in a different "
    "way.\n"
    "- Only ask questions this excerpt can answer.\n"
    'Return ONLY a JSON array of question strings.\n\nJSON:'
)

# Quality filter: drop short, code-dump, or filename-only answers.
_CODEY = set("{}[]()=;<>|\\")


def _good_q(q: str) -> bool:
    return 12 <= len(q) <= 400 and ("?" in q)


def split_trace(response: str) -> tuple[str, str]:
    """Split a thinking-model response into (think_part, answer_part).
    Responses without a </think> marker are all answer."""
    if "</think>" in response:
        think, _, answer = response.partition("</think>")
        return think, answer.strip()
    return "", response.strip()


def _good_a(a: str) -> bool:
    """Judge the ANSWER part only (the think trace may be long and code-y)."""
    if len(a) < 40 or len(a) > 4000:
        return False
    if a.lower().endswith((".py", ".pdf", ".tex", ".json", ".ipynb", ".lua")):
        return False
    code_ratio = sum(c in _CODEY for c in a) / max(1, len(a))
    if code_ratio > 0.10:  # answer reads like a code dump
        return False
    return len(a.split()) >= 8


def extract_questions(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for q in arr:
        if isinstance(q, dict):
            q = q.get("question", "")
        q = str(q).strip()
        if _good_q(q):
            out.append(q)
    return out


@torch.no_grad()
def _generate(gen, gtok, user_msgs, max_new_tokens, do_sample, temperature=0.8,
              think=False):
    """Batched chat generation (left-padded). Returns decoded completions.
    `think` toggles the model's reasoning mode via the chat template."""
    texts = [
        gtok.apply_chat_template(
            [{"role": "user", "content": u}],
            add_generation_prompt=True, tokenize=False, enable_thinking=think,
        )
        for u in user_msgs
    ]
    enc = gtok(texts, return_tensors="pt", padding=True,
               add_special_tokens=False).to(gen.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                  pad_token_id=gtok.eos_token_id)
    if do_sample:
        kwargs.update(temperature=temperature, top_p=0.9)
    out = gen.generate(**enc, **kwargs)
    new = out[:, enc["input_ids"].shape[1]:]
    return [gtok.decode(row, skip_special_tokens=True).strip() for row in new]


def main():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    keys = sys.argv[1:] or list(CFG.domains)
    qa_dir = CFG.data_cache / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading generator {CFG.qa_gen_model} (gen_batch={CFG.synth_gen_batch}, "
          f"think={CFG.synth_think_answers})")
    gtok = AutoTokenizer.from_pretrained(CFG.qa_gen_model)
    if gtok.pad_token_id is None:
        gtok.pad_token = gtok.eos_token
    gtok.padding_side = "left"  # batched generation
    gen = AutoModelForCausalLM.from_pretrained(
        CFG.qa_gen_model, dtype=torch.bfloat16, device_map="balanced",
    )
    gen.eval()
    ctok = AutoTokenizer.from_pretrained(CFG.model_name)  # to decode the chunks
    bs = CFG.synth_gen_batch

    for key in keys:
        data = build_task(CFG, ctok, key)
        contexts = [ctok.decode(ch.tolist()) for ch in
                    (list(data.train) + list(data.support))]
        out_p = qa_dir / f"{key}.json"
        t0 = time.time()

        # Stage 1: questions for EVERY chunk, batched across chunks (sampled for
        # variety; thinking off — cheap and keeps the JSON parseable).
        q_prompts = [Q_PROMPT.format(ctx=CTX.get(key, key), n=CFG.qa_per_chunk,
                                     chunk=c) for c in contexts]
        q_out = _batched(gen, gtok, q_prompts, max_new_tokens=1400,
                         do_sample=True, temperature=0.8, think=False, bs=bs)
        # Flatten to (chunk_idx, question) pairs, dedup within each chunk.
        pairs = []
        for ci, txt in enumerate(q_out):
            qs = list(dict.fromkeys(extract_questions(txt)))[: CFG.qa_per_chunk]
            pairs.extend((ci, q) for q in qs)
        print(f"  {key}: {len(pairs)} questions over {len(contexts)} chunks "
              f"({time.time() - t0:.0f}s)", flush=True)

        # Stage 2: teacher answers WITH each question's chunk in context,
        # batched across ALL (chunk, question) pairs.
        a_prompts = [teacher_user_msg(contexts[ci], q) for ci, q in pairs]
        a_out = _batched(gen, gtok, a_prompts,
                         max_new_tokens=CFG.synth_answer_tokens,
                         do_sample=CFG.synth_think_answers, temperature=0.6,
                         think=CFG.synth_think_answers, bs=bs)

        triples = []
        for (ci, q), resp in zip(pairs, a_out):
            _, ans = split_trace(resp)  # think trace (if any) stripped for the filter
            if _good_a(ans):
                triples.append({"question": q, "answer": resp,
                                "context": contexts[ci]})
        out_p.write_text(json.dumps(triples, indent=2))
        print(f"  saved {len(triples)}/{len(pairs)} distillation triples -> "
              f"{out_p} ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
