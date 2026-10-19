"""Self-study synthesis for context distillation (Cartridges-style, adapted to
conversation sessions; recipe per the 2024-26 knowledge-injection literature —
see README).

For each session we generate (student_prompt, response) pairs:
  - the TEACHER (frozen backbone, mask zeroed) sees the session transcript in
    context and writes the response;
  - the STUDENT (masked bank, at train time) sees only the prompt — which names
    the speakers and the session date so the mask can route — and is distilled
    to the teacher's token distribution on the response.

Seed templates cover the SPA/Cartridges diversity axes: recall QA, temporal QA,
reversed QA, self-contained fact restatement, summary/timeline, aggregation,
implications. Focus hints (random turns) spread coverage across the session.
A held-out paraphrased QA set per session is generated for eval/early-stop and
NEVER trained on.

Everything is batched generation on the one loaded model (left-padded).
Datasets go to $FTN_AGENT_WORK/benchmarking/<conv>/<session>/ (PVC, prunable).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

from locomo_data import Conversation, Session, render_session, render_turn

WORK_ROOT = Path(os.environ.get("FTN_AGENT_WORK", "/home/dev/ftn_agent_work")) / "benchmarking"

# (name, instruction template, weight). {a},{b}: speakers; {date}: session
# datetime; {focus}: a quoted turn from the session to steer coverage.
SEEDS = [
    ("recall_qa",
     "Write 6 specific question/answer pairs about what {a} and {b} said and did "
     "in this session. Vary the question angle. Use full names, no pronouns "
     "without antecedents. Format each as 'Q: ...' / 'A: ...'.", 3),
    ("temporal_qa",
     "The session took place at {date}. Write 4 question/answer pairs about WHEN "
     "things happened or were said (resolve all relative dates like 'yesterday' "
     "or 'last week' to absolute dates). Format 'Q: ...' / 'A: ...'.", 2),
    ("reverse_qa",
     "Write 5 'Who ...?' or 'Which person ...?' question/answer pairs about this "
     "session — questions whose ANSWER is {a} or {b} or a person/thing they "
     "mentioned. Format 'Q: ...' / 'A: ...'.", 2),
    ("facts",
     "Rewrite the important content of this session as 10 standalone declarative "
     "facts. Each fact must be fully self-contained: full names, absolute dates "
     "(session date: {date}), no pronouns referring outside the sentence. One "
     "fact per line.", 3),
    ("facts_focus",
     "Here is one exchange from the session: {focus}\nState everything this "
     "exchange tells us, as 4-6 standalone facts with full names and absolute "
     "dates (session date: {date}). Then add 2 question/answer pairs about it.", 3),
    ("summary",
     "Summarize this session in detail: what each of {a} and {b} reported, "
     "decided, or planned, with absolute dates (session date: {date}).", 1),
    ("timeline",
     "Extract a dated timeline of every event mentioned in this session "
     "(including past events the speakers refer to). Resolve relative dates "
     "using the session date {date}. One event per line: 'DATE — EVENT'.", 1),
    ("aggregate",
     "Write 4 question/answer pairs that require combining several parts of "
     "this session (counts, lists of things a person mentioned, comparisons "
     "between {a} and {b}). Format 'Q: ...' / 'A: ...'.", 1),
    ("implications",
     "Write 4 question/answer pairs about what this session implies about "
     "{a}'s and {b}'s lives, relationships, values, or plans, with the "
     "supporting detail from the session in the answer. Format 'Q: ...' / 'A: ...'.", 1),
]

HELDOUT_SEED = (
    "Write 8 question/answer pairs testing memory of the concrete facts of this "
    "session, phrased differently from how the session phrases them. Use full "
    "names; resolve dates to absolute using {date}. Format 'Q: ...' / 'A: ...'."
)

TEACHER_SYS = (
    "You are studying a real conversation between {a} and {b} so it can be "
    "remembered later. Be precise and grounded: use only information from the "
    "transcript. Always use full names and absolute dates."
)
TEACHER_USER = "TRANSCRIPT:\n{session}\n\nTASK: {instruction}"

# What the student sees at train time (NO transcript). Names + date let the
# mask route; the phrasing matches closed-book QA time.
STUDENT_SYS = (
    "You are recalling, from memory, a conversation between {a} and {b} "
    "(session of {date})."
)


def session_dir(conv_id: str, session_index: int) -> Path:
    return WORK_ROOT / conv_id / f"session_{session_index}"


def _sample_focus(session: Session, gen: torch.Generator) -> str:
    turns = [t for t in session.turns if len(str(t.get("text", ""))) > 30] or session.turns
    i = int(torch.randint(0, len(turns), (1,), generator=gen, device="cpu"))
    return json.dumps(render_turn(turns[i]))


def build_jobs(conv: Conversation, session: Session, n_items: int, seed: int):
    """Expand seed templates (weighted, round-robin) into n_items generation
    jobs + the held-out job. Returns (jobs, heldout_job); each job is
    {seed_name, instruction, student_prompt}."""
    gen = torch.Generator().manual_seed(seed)
    weighted = [s for s in SEEDS for _ in range(s[2])]
    jobs = []
    for i in range(n_items):
        name, tmpl, _ = weighted[i % len(weighted)]
        instr = tmpl.format(a=conv.speaker_a, b=conv.speaker_b,
                            date=session.date_time,
                            focus=_sample_focus(session, gen) if "{focus}" in tmpl else "")
        jobs.append({"seed": name, "instruction": instr})
    heldout = {"seed": "heldout_qa",
               "instruction": HELDOUT_SEED.format(date=session.date_time)}
    return jobs, heldout


@torch.no_grad()
def _generate_batch(model, tok, prompts: list[list[dict]], max_new_tokens: int) -> list[str]:
    """Batched chat generation with left padding."""
    texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False) for m in prompts]
    prev_side = tok.padding_side
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    tok.padding_side = prev_side
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
    outs = []
    for row in out[:, enc["input_ids"].shape[1]:]:
        text = tok.decode(row, skip_special_tokens=True)
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        outs.append(text.strip())
    return outs


def parse_qa(text: str) -> list[dict]:
    pairs = re.findall(r"Q:\s*(.+?)\s*A:\s*(.+?)(?=\nQ:|\Z)", text, re.DOTALL)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in pairs
            if len(q.strip()) > 8 and len(a.strip()) > 1]


def synthesize_session(model, tok, ctrl, cfg_H: int, conv: Conversation,
                       session: Session, n_items: int = 48,
                       max_new_tokens: int = 420, batch_size: int = 8,
                       seed: int = 0, log=print) -> Path:
    """Generate + save one session's self-study dataset. Idempotent (skips if
    data.json exists). Returns the dataset dir."""
    d = session_dir(conv.sample_id, session.index)
    if (d / "data.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    sess_text = render_session(session)
    jobs, heldout = build_jobs(conv, session, n_items, seed + session.index)

    saved_mask = ctrl.mask.clone()
    ctrl.set_mask(torch.zeros(cfg_H, device=saved_mask.device))  # frozen backbone
    try:
        all_jobs = jobs + [heldout]
        for i in range(0, len(all_jobs), batch_size):
            chunk = all_jobs[i:i + batch_size]
            prompts = [[
                {"role": "system", "content": TEACHER_SYS.format(a=conv.speaker_a, b=conv.speaker_b)},
                {"role": "user", "content": TEACHER_USER.format(session=sess_text,
                                                                instruction=j["instruction"])},
            ] for j in chunk]
            for j, resp in zip(chunk, _generate_batch(model, tok, prompts, max_new_tokens)):
                j["response"] = resp
            log(f"    synth {conv.sample_id}/s{session.index}: "
                f"{min(i + batch_size, len(all_jobs))}/{len(all_jobs)}")
    finally:
        ctrl.set_mask(saved_mask)

    items = [{"seed": j["seed"], "instruction": j["instruction"],
              "response": j["response"],
              "student_sys": STUDENT_SYS.format(a=conv.speaker_a, b=conv.speaker_b,
                                                date=session.date_time)}
             for j in jobs if j.get("response")]
    heldout_qa = parse_qa(heldout.get("response", ""))
    n_tok = sum(len(i["response"].split()) for i in items)
    meta = {"conv": conv.sample_id, "session": session.index,
            "a": conv.speaker_a, "b": conv.speaker_b,
            "date_time": session.date_time, "n_items": len(items),
            "n_heldout": len(heldout_qa), "approx_words": n_tok,
            "session_words": len(sess_text.split())}
    (d / "data.json").write_text(json.dumps({"meta": meta, "items": items,
                                             "heldout_qa": heldout_qa,
                                             "session_text": sess_text}, indent=1))
    log(f"    synth {conv.sample_id}/s{session.index}: {len(items)} items "
        f"(~{n_tok}w vs session {meta['session_words']}w, "
        f"{len(heldout_qa)} heldout qa)")
    return d


def load_session_data(conv_id: str, session_index: int) -> dict:
    return json.loads((session_dir(conv_id, session_index) / "data.json").read_text())
