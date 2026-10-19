"""Floor + ceiling baselines (pod side).

floor   : closed-book, ZERO ingestion — the contamination probe. LoCoMo's
          personas are fictional, so a clean model should score ~0 outside the
          open-domain category; materially higher = leakage (README).
ceiling : the whole conversation (~26k tokens) in context.

Answer prompts here are THE canonical prompts — every system (baselines, FTN
closed-book, single-LoRA ablation) imports them so scores stay comparable.

    python benchmarking/run_baselines.py --mode floor
    python benchmarking/run_baselines.py --mode ceiling --limit 20   # smoke
Writes/resumes results/<mode>.jsonl (one record per question).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from locomo_data import apply_corrections, load_locomo, render_conversation

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ANSWER_SYSTEM = (
    "You are answering questions about a past multi-session conversation "
    "between {a} and {b}. Answer with a short, direct phrase — no explanation, "
    "no restating the question. For dates, give the specific date."
)
CEILING_USER = "{conversation}\n\nQuestion: {question}\nShort answer:"
FLOOR_USER = (
    "The conversation itself is not shown to you; answer from your own memory "
    "of it, if any. If you do not know, give your single best short guess.\n\n"
    "Question: {question}\nShort answer:"
)


def question_records(convs):
    for conv in convs:
        for i, qa in enumerate(conv.qa):
            yield {"sample_id": conv.sample_id, "q_idx": i, "question": qa.question,
                   "gold": qa.answer, "category": qa.category,
                   "category_name": qa.category_name, "corrected": qa.corrected,
                   "speaker_a": conv.speaker_a, "speaker_b": conv.speaker_b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["floor", "ceiling"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    convs = load_locomo()
    apply_corrections(convs)
    conv_text = {c.sample_id: render_conversation(c) for c in convs}
    records = list(question_records(convs))[: args.limit]

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.mode}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["sample_id"], r["q_idx"]))
        print(f"resuming: {len(done)} already answered in {out_path}")

    from backbone import generate_answer, load_backbone
    model, tok = load_backbone()

    t0, n = time.time(), 0
    with open(out_path, "a") as f:
        for r in records:
            if (r["sample_id"], r["q_idx"]) in done:
                continue
            system = ANSWER_SYSTEM.format(a=r["speaker_a"], b=r["speaker_b"])
            if args.mode == "ceiling":
                user = CEILING_USER.format(conversation=conv_text[r["sample_id"]],
                                           question=r["question"])
            else:
                user = FLOOR_USER.format(question=r["question"])
            pred = generate_answer(model, tok, system, user, args.max_new_tokens)
            f.write(json.dumps({**r, "mode": args.mode, "prediction": pred}) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                rate = n / (time.time() - t0)
                left = (len(records) - len(done) - n) / max(rate, 1e-9)
                print(f"  {n} answered ({rate:.2f} q/s, ~{left / 60:.0f} min left)", flush=True)
    print(f"done: {n} new answers -> {out_path}")


if __name__ == "__main__":
    main()
