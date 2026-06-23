"""Synthesize grounded Q&A pairs from each task's source text.

Uses a local instruct model (CFG.qa_gen_model) so no data leaves the box. For
each task we take its train+support chunks, ask the model to write self-contained
question/answer pairs answerable from that excerpt, and cache them to
data_cache/qa/<task>.json. Run once before training:

    uv run python qa_synth.py [task ...]
"""

from __future__ import annotations

import json
import re
import sys

import torch
from config import CFG
from data import build_task
from transformers import AutoModelForCausalLM, AutoTokenizer

CTX = {
    "cscg": "CSCG (clone-structured cognitive graph) model implementation in Python, cscg_he.py",
    "genetic_programming": "Genetic programming in JAX, jax_gp.py",
    "robotics": "FTN robotics / reinforcement-learning code in robotics/train.py",
    "sudoku": "Sudoku solving experiment with Gumbel Softmax Optimization (GSO) and Message Passing (MP)",
}

PROMPT = (
    "You are given an excerpt from the {ctx}. Write {n} self-contained "
    "question-answer pairs grounded in this excerpt.\n"
    "Requirements:\n"
    "- Questions are specific and natural, as if a user asked an assistant about "
    'this project script by name. Do NOT refer to "the excerpt", "the code above", or line numbers.\n'
    "- Answers are EXPLANATORY: 2 to 4 complete sentences in plain prose that "
    "explain the what and the why. Do NOT answer with a bare name, a filename, or "
    "a code dump; you may mention an identifier inline (like `foo`) but the answer "
    "must read as natural language a person would say out loud.\n"
    '- Mix factual ("what does X do") and conceptual ("why"/"how") questions.\n'
    'Return ONLY a JSON array of objects with keys "question" and "answer".\n\n'
    "EXCERPT:\n{chunk}\n\nJSON:"
)

# Quality filter: drop short, code-dump, or filename-only answers.
_CODEY = set("{}[]()=;<>|\\")


def _is_good(q, a):
    if len(a) < 60 or len(a) > 1400:
        return False
    if not q.rstrip().endswith("?") and "?" not in q:
        return False
    if a.lower().endswith((".py", ".pdf", ".tex", ".json", ".ipynb")):
        return False
    code_ratio = sum(c in _CODEY for c in a) / max(1, len(a))
    if code_ratio > 0.08:  # answer reads like a code dump
        return False
    if len(a.split()) < 10:  # too terse to be explanatory
        return False
    return True


def extract_pairs(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for d in arr:
        if not (isinstance(d, dict) and d.get("question") and d.get("answer")):
            continue
        q, a = str(d["question"]).strip(), str(d["answer"]).strip()
        if _is_good(q, a):
            out.append({"question": q, "answer": a})
    return out


def main():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    keys = sys.argv[1:] or list(CFG.domains)
    qa_dir = CFG.data_cache / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading generator {CFG.qa_gen_model}")
    gtok = AutoTokenizer.from_pretrained(CFG.qa_gen_model)
    gen = AutoModelForCausalLM.from_pretrained(CFG.qa_gen_model, dtype=torch.bfloat16)
    gen.eval()
    ctok = AutoTokenizer.from_pretrained(CFG.model_name)  # to decode the chunks

    for key in keys:
        data = build_task(CFG, ctok, key)
        chunks = list(data.train) + list(data.support)
        pairs = []
        for i, ch in enumerate(chunks):
            text = ctok.decode(ch.tolist())
            prompt = PROMPT.format(
                ctx=CTX.get(key, key), n=CFG.qa_per_chunk, chunk=text[:3000]
            )
            msgs = [{"role": "user", "content": prompt}]
            enc = gtok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(gen.device)
            with torch.no_grad():
                out = gen.generate(
                    **enc,
                    max_new_tokens=640,
                    do_sample=False,
                    pad_token_id=gtok.eos_token_id,
                )
            txt = gtok.decode(
                out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            pairs.extend(extract_pairs(txt))
            if i % 10 == 0:
                print(
                    f"  {key}: chunk {i + 1}/{len(chunks)} -> {len(pairs)} pairs",
                    flush=True,
                )
        (qa_dir / f"{key}.json").write_text(json.dumps(pairs, indent=2))
        print(f"saved {len(pairs)} Q&A pairs -> {qa_dir / (key + '.json')}", flush=True)


if __name__ == "__main__":
    main()
