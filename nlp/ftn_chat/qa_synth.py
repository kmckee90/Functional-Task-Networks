"""Synthesize context-distillation data from each task's source text.

Two stages, both with a LOCAL instruct model (no egress), and both BATCHED
across chunks (`CFG.synth_gen_batch` sequences per generate call) so the GPUs
stay saturated — single-stream decode was the synthesis bottleneck:

  1. QUESTIONS — with each chunk in context, generate `CFG.qa_per_chunk`
     diverse, self-contained questions (factual / conceptual / parameter-
     specific / applied), phrased as a user would ask an
     assistant about the project by name.
  2. TEACHER ANSWERS — the teacher answers each question WITH its chunk in
     context. Grounded answer + chunk are stored as (question, answer,
     context) triples.

train.py then distills: teacher = frozen backbone + context + zeroed mask;
student = masked bank, closed book; KL on the answer tokens. Optional source CE
can also train the student directly on training-split source chunks.

Cache: ~/ftn_chat_state/data_cache/qa/<task>.json on the pod (override with
FTN_DATA_CACHE_DIR). A normal run replaces the cache; --resume preserves it and
continues until the per-task total in CFG.qa_target_counts is met. It rejects
exact and close question/answer repeats and saves each completed answer batch.
Run once before training:

    uv run python qa_synth.py [task ...]
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
import re
import time
from collections import defaultdict

import torch
from config import CFG
from data import build_task, teacher_user_msg
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    "- Cover different facts and mechanisms; avoid paraphrases of one fact.\n"
    "- Only ask questions this excerpt can answer.\n"
    "{novelty}"
    'Return ONLY a JSON array of question strings.\n\nJSON:'
)

FOCUSES = (
    "specific defaults, parameters, and their effects",
    "implementation details and data flow",
    "causal explanations and design tradeoffs",
    "edge cases, failure modes, and diagnostics",
    "applied scenarios with concrete predictions",
    "comparisons among components in the material",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tasks", nargs="*", help="task keys; defaults to all configured tasks")
    p.add_argument("--resume", action="store_true",
                   help="preserve existing Q&A and generate until each configured target count")
    return p.parse_args()


def words(s):
    return re.findall(r"[a-z0-9_]+", s.casefold())


class NoveltyFilter:
    """Reject exact and close question or answer repeats across the whole cache."""

    def __init__(self, triples):
        self.questions = []
        self.q_exact = set()
        self.answers = []
        self.by_context = defaultdict(list)
        for row in triples:
            self.add(row)

    def add(self, row):
        q = row["question"]
        q_norm = " ".join(words(q))
        self.q_exact.add(q_norm)
        self.questions.append((q_norm, set(words(q))))
        _, answer = split_trace(row["answer"])
        self.answers.append(set(words(answer)))
        self.by_context[row["context"]].append(q)

    def question_is_new(self, q):
        norm = " ".join(words(q))
        if norm in self.q_exact:
            return False
        tokens = set(words(q))
        for old_norm, old_tokens in self.questions:
            overlap = len(tokens & old_tokens) / max(1, len(tokens | old_tokens))
            if overlap >= 0.83 or (overlap >= 0.55 and
                    SequenceMatcher(None, norm, old_norm).ratio() >= 0.78):
                return False
        return True

    def answer_is_new(self, answer):
        tokens = set(words(answer))
        for old in self.answers:
            if len(tokens & old) >= 8 and len(tokens & old) / max(1, len(tokens | old)) >= 0.88:
                return False
        return True


def save_atomic(path, rows):
    """Keep the previous cache intact if synthesis is interrupted."""
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(rows, indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_existing(path):
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or
        not all(isinstance(row.get(k), str) for k in ("question", "answer", "context"))
        for row in rows
    ):
        raise ValueError(f"invalid Q&A cache: {path}")
    return rows


def novelty_instruction(prior, round_idx):
    focus = FOCUSES[round_idx % len(FOCUSES)]
    # A short sample of covered questions steers the generator toward unused
    # details without consuming its full context window.
    samples = prior[-12:]
    seen = "" if not samples else (
        "Already covered questions (do not restate these facts):\n" +
        "\n".join(f"- {q}" for q in samples) + "\n"
    )
    return f"Focus this batch on {focus}. Seek details not yet covered.\n{seen}"

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
    # A generated question can accidentally refer to a fact outside its
    # excerpt. Do not train the student on an abstention as if it were an answer.
    if re.search(
        r"\b(?:no mention|no information|not mentioned|not specified|"
        r"not (?:explicitly )?stated|does not specify|does not contain|"
        r"cannot (?:answer|infer|determine|conclude)|cannot be answered|"
        r"cannot be determined|not possible to (?:answer|know|infer|determine)|"
        r"insufficient information|not enough information|"
        r"not (?:explicitly )?provided)\b",
        a.casefold(),
    ):
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
    args = parse_args()
    keys = args.tasks or list(CFG.domains)
    if args.resume:
        missing = [key for key in keys if CFG.qa_target_counts.get(key, 0) <= 0]
        if missing:
            raise SystemExit(f"set positive qa_target_counts for: {', '.join(missing)}")
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    qa_dir = CFG.data_cache / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for key in keys:
        path = qa_dir / f"{key}.json"
        existing = load_existing(path) if args.resume else []
        target = CFG.qa_target_counts.get(key) if args.resume else None
        if target is not None and len(existing) >= target:
            print(f"  {key}: already at {len(existing)}/{target}; skipping", flush=True)
            continue
        pending.append((key, path, existing, target))
    if not pending:
        return

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

    incomplete = []
    for key, out_p, triples, target in pending:
        data = build_task(CFG, ctok, key)
        contexts = [ctok.decode(ch.tolist()) for ch in
                    (list(data.train) + list(data.support))]
        if not contexts:
            raise RuntimeError(f"{key}: no train or support source chunks")
        t0 = time.time()
        novelty = NoveltyFilter(triples)
        initial = len(triples)
        print(f"  {key}: starting with {initial} accepted pairs; "
              f"target={target if target is not None else 'one pass'}; "
              f"{len(contexts)} source chunks", flush=True)
        stalled = 0
        max_rounds = CFG.qa_max_rounds if args.resume else 1
        for round_idx in range(max_rounds):
            if target is not None and len(triples) >= target:
                break
            before = len(triples)
            # One question batch per chunk. The focus rotates across rounds,
            # while the archive supplies examples of already covered material.
            q_prompts = [Q_PROMPT.format(
                ctx=key, n=CFG.qa_per_chunk, chunk=c,
                novelty=novelty_instruction(novelty.by_context[c], round_idx),
            ) for c in contexts]
            q_out = _batched(gen, gtok, q_prompts, max_new_tokens=1400,
                             do_sample=True, temperature=0.8, think=False, bs=bs)
            pairs = []
            pending_questions = set()
            rejected_q = 0
            for ci, txt in enumerate(q_out):
                for q in extract_questions(txt)[:CFG.qa_per_chunk]:
                    q_norm = " ".join(words(q))
                    if q_norm in pending_questions or not novelty.question_is_new(q):
                        rejected_q += 1
                        continue
                    pending_questions.add(q_norm)
                    pairs.append((ci, q))
            print(f"  {key} round {round_idx + 1}: {len(pairs)} new question "
                  f"candidates, {rejected_q} near/exact repeats", flush=True)

            rejected_a = 0
            for offset in range(0, len(pairs), bs):
                batch = pairs[offset:offset + bs]
                a_prompts = [teacher_user_msg(contexts[ci], q) for ci, q in batch]
                a_out = _generate(gen, gtok, a_prompts,
                                  max_new_tokens=CFG.synth_answer_tokens,
                                  do_sample=CFG.synth_think_answers,
                                  temperature=0.6, think=CFG.synth_think_answers)
                for (ci, q), resp in zip(batch, a_out):
                    _, ans = split_trace(resp)
                    if not _good_a(ans) or not novelty.question_is_new(q) or \
                            not novelty.answer_is_new(ans):
                        rejected_a += 1
                        continue
                    row = {"question": q, "answer": resp, "context": contexts[ci]}
                    triples.append(row)
                    novelty.add(row)
                    if target is not None and len(triples) >= target:
                        break
                if len(triples) > before:
                    save_atomic(out_p, triples)
                if target is not None and len(triples) >= target:
                    break
            print(f"  {key} round {round_idx + 1}: +{len(triples) - before} "
                  f"accepted ({len(triples)}/{target if target is not None else 'one pass'}), "
                  f"{rejected_a} invalid/redundant answers, "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
            if len(triples) == before:
                stalled += 1
            else:
                stalled = 0
            if stalled >= CFG.qa_stall_rounds:
                print(f"  {key}: stopped after {stalled} rounds without new pairs", flush=True)
                break
        if not args.resume:
            save_atomic(out_p, triples)
        if target is not None and len(triples) < target:
            incomplete.append(f"{key} {len(triples)}/{target}")
        print(f"  {key}: saved {len(triples)} total (+{len(triples) - initial}) "
              f"-> {out_p}", flush=True)
    if incomplete:
        raise SystemExit("Q&A targets not reached: " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
