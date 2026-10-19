"""Time one synthesis chunk (no-think) and extrapolate per-task synthesis time.

Loads the generator once, runs the real qa_synth code path (question gen +
batched teacher answers) on 2 chunks of base_ftn, and multiplies the median
per-chunk time by each task's chunk count.

    uv run python bench_synth.py
"""
from __future__ import annotations

import time

import torch

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

from config import CFG
from data import build_task, teacher_user_msg
from transformers import AutoModelForCausalLM, AutoTokenizer
from qa_synth import Q_PROMPT, CTX, extract_questions, split_trace, _generate, _good_a

print(f"gen model: {CFG.qa_gen_model}  think={CFG.synth_think_answers}  "
      f"qa_per_chunk={CFG.qa_per_chunk}  answer_tokens={CFG.synth_answer_tokens}")

gtok = AutoTokenizer.from_pretrained(CFG.qa_gen_model)
if gtok.pad_token_id is None:
    gtok.pad_token = gtok.eos_token
gtok.padding_side = "left"
gen = AutoModelForCausalLM.from_pretrained(
    CFG.qa_gen_model, dtype=torch.bfloat16, device_map="balanced")
gen.eval()
ctok = AutoTokenizer.from_pretrained(CFG.model_name)

data = build_task(CFG, ctok, "base_ftn")
chunks = (list(data.train) + list(data.support))[:3]  # warmup + 2 timed

per_chunk = []
for i, ch in enumerate(chunks):
    context = ctok.decode(ch.tolist())
    t0 = time.time()
    q_prompt = Q_PROMPT.format(ctx=CTX["base_ftn"], n=CFG.qa_per_chunk, chunk=context)
    questions = []
    for txt in _generate(gen, gtok, [q_prompt], max_new_tokens=1400, do_sample=True):
        questions.extend(extract_questions(txt))
    questions = list(dict.fromkeys(questions))[: CFG.qa_per_chunk]
    t_q = time.time() - t0
    t1 = time.time()
    responses = _generate(
        gen, gtok, [teacher_user_msg(context, q) for q in questions],
        max_new_tokens=CFG.synth_answer_tokens,
        do_sample=CFG.synth_think_answers, temperature=0.6,
        think=CFG.synth_think_answers)
    kept = sum(1 for r in responses if _good_a(split_trace(r)[1]))
    t_a = time.time() - t1
    tag = "warmup" if i == 0 else f"chunk{i}"
    print(f"  {tag}: {len(questions)}Q in {t_q:.1f}s + {len(responses)}A in "
          f"{t_a:.1f}s = {t_q+t_a:.1f}s ({kept} kept)", flush=True)
    if i > 0:
        per_chunk.append(t_q + t_a)

med = sorted(per_chunk)[len(per_chunk) // 2]
print(f"\nmedian per-chunk: {med:.1f}s")
# Chunk counts = train + support per task (what qa_synth iterates).
for key in CFG.domains:
    d = build_task(CFG, ctok, key)
    n = len(d.train) + len(d.support)
    tot = n * med
    print(f"  {key:<10} {n:>4} chunks -> {tot/60:.0f} min ({tot/3600:.1f} h)")
print("BENCH OK")
