"""Score a predictions .jsonl (laptop side — needs the `claude` CLI or an
OpenAI key; the pod never talks to the cloud).

    python benchmarking/score.py benchmarking/results/floor.jsonl --judge claude-cli
    python benchmarking/score.py benchmarking/results/ceiling.jsonl --judge none  # F1 only

Token-F1 (SQuAD-normalized) is always computed; the LLM judge adds the
headline "J" metric (Mem0-style CORRECT/WRONG). Writes <input>.scores.json.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

JUDGE_PROMPT = """You are grading answers to questions about a private multi-session conversation.
For each item decide if the model's answer is CORRECT with respect to the gold answer.
Rules: the answer must contain the key information of the gold answer; extra words are fine;
paraphrases and equivalent date formats are fine; vague, partial, evasive, or "I don't know"
answers are WRONG; for list answers, most of the items must be present.

Items:
{items}

Reply with ONLY a JSON array like [{{"id": 0, "verdict": "CORRECT"}}, ...] covering every item."""


# --- token F1 (SQuAD normalization) ------------------------------------------

def _norm(s: str) -> list[str]:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


# --- LLM judge ----------------------------------------------------------------

def _judge_claude_cli(batch: list[dict], model: str) -> list[bool]:
    items = "\n".join(
        f'{i}. Question: {r["question"]}\n   Gold answer: {r["gold"]}\n'
        f'   Model answer: {r["prediction"]}'
        for i, r in enumerate(batch))
    out = subprocess.run(
        ["claude", "-p", "--model", model, JUDGE_PROMPT.format(items=items)],
        capture_output=True, text=True, timeout=300)
    m = re.search(r"\[.*\]", out.stdout, re.DOTALL)
    if not m:
        raise RuntimeError(f"judge gave no JSON: {out.stdout[:500]} / {out.stderr[:200]}")
    verdicts = {int(d["id"]): str(d["verdict"]).upper().startswith("CORRECT")
                for d in json.loads(m.group(0))}
    return [verdicts.get(i, False) for i in range(len(batch))]


def _judge_openai(batch: list[dict], model: str) -> list[bool]:
    from openai import OpenAI
    items = "\n".join(
        f'{i}. Question: {r["question"]}\n   Gold answer: {r["gold"]}\n'
        f'   Model answer: {r["prediction"]}'
        for i, r in enumerate(batch))
    resp = OpenAI().chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(items=items)}])
    m = re.search(r"\[.*\]", resp.choices[0].message.content, re.DOTALL)
    verdicts = {int(d["id"]): str(d["verdict"]).upper().startswith("CORRECT")
                for d in json.loads(m.group(0))}
    return [verdicts.get(i, False) for i in range(len(batch))]


def run_judge(records: list[dict], backend: str, model: str, batch_size: int = 10,
              persist=None):
    fn = _judge_claude_cli if backend == "claude-cli" else _judge_openai
    todo = [r for r in records if r.get("judge_correct") is None]
    if len(todo) < len(records):
        print(f"  resuming: {len(records) - len(todo)} already judged, {len(todo)} left")
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        try:
            flags = fn(batch, model)
        except Exception as e:   # one retry, then mark unjudged
            print(f"  judge batch {i}: retrying ({e})")
            try:
                flags = fn(batch, model)
            except Exception as e2:
                print(f"  judge batch {i}: FAILED ({e2}); marking unjudged")
                flags = [None] * len(batch)
        for r, ok in zip(batch, flags):
            r["judge_correct"] = ok
        if persist is not None:
            persist()
        print(f"  judged {min(i + batch_size, len(todo))}/{len(todo)}", flush=True)


# --- aggregation ---------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r.get("category_name", "?")].append(r)
    out = {}
    for cat, rs in [("OVERALL", records)] + sorted(by_cat.items()):
        judged = [r for r in rs if r.get("judge_correct") is not None]
        out[cat] = {
            "n": len(rs),
            "f1": round(sum(token_f1(r["prediction"], r["gold"]) for r in rs) / max(len(rs), 1), 4),
            "judge": round(sum(bool(r["judge_correct"]) for r in judged) / len(judged), 4) if judged else None,
            "n_judged": len(judged),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--judge", choices=["claude-cli", "openai", "none"], default="claude-cli")
    ap.add_argument("--judge-model", default=None,
                    help="default: sonnet (claude-cli) / gpt-4o-mini (openai)")
    ap.add_argument("--batch-size", type=int, default=10)
    args = ap.parse_args()

    path = Path(args.predictions)
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"{path.name}: {len(records)} predictions")

    if args.judge != "none":
        model = args.judge_model or ("sonnet" if args.judge == "claude-cli" else "gpt-4o-mini")
        persist = lambda: path.write_text("".join(json.dumps(r) + "\n" for r in records))
        run_judge(records, args.judge, model, args.batch_size, persist=persist)
        persist()  # persist verdicts

    scores = aggregate(records)
    out = path.with_suffix(".scores.json")
    out.write_text(json.dumps(scores, indent=2))
    w = max(len(k) for k in scores)
    for cat, s in scores.items():
        j = f'{s["judge"]:.3f}' if s["judge"] is not None else "  -  "
        print(f'  {cat:<{w}}  n={s["n"]:>4}  F1={s["f1"]:.3f}  J={j}')
    print(f"-> {out}")


if __name__ == "__main__":
    main()
