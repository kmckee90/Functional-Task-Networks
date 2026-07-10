"""Closed-book QA eval for a trained run + routing diagnostics.

Uses the SAME canonical answer prompts as the baselines (run_baselines.py), so
the only difference from the floor is the weights. Routing: the question's
reconfigured mask is compared against every trained block's reference mask;
we report whether the top-overlap block belongs to the right conversation and
to one of the question's evidence sessions.

    python benchmarking/eval_ftn.py --run r1            # all questions of trained convs
    python benchmarking/eval_ftn.py --run r1 --limit 40 # smoke
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from locomo_data import apply_corrections, load_locomo
from run_baselines import ANSWER_SYSTEM, FLOOR_USER, RESULTS_DIR, question_records


def run_root(run: str) -> Path:
    import os
    return (Path(os.environ.get("FTN_AGENT_STATE", "/home/dev/ftn_agent_state"))
            / "benchmarking" / "runs" / run)


def make_run_config(run: str, **overrides):
    """Config pointed at the run dir (so train.py state I/O lands there)."""
    from config import Config
    saved = {}
    cfg_p = run_root(run) / "run_config.json"
    if cfg_p.exists():
        saved = json.loads(cfg_p.read_text())
    merged = {**saved, **overrides}
    fields = {k: v for k, v in merged.items()
              if k in Config.__dataclass_fields__
              and k not in ("state_dir", "work_dir", "inbox_dir")}
    cfg = Config(state_dir=run_root(run), **fields)
    return cfg, merged


@torch.no_grad()
def _generate(h, system: str, user: str, max_new_tokens: int = 128) -> str:
    tok, model = h.tok, h.model
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True,
                                      enable_thinking=False)
    except TypeError:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    enc = {k: v.to(h.dev) for k, v in enc.items()}
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def answer_closed_book(h, cfg, question: str, speakers: tuple[str, str],
                       rsteps: int = None, monolithic: bool = False,
                       max_new_tokens: int = 128):
    """Reconfigure on the bare question, generate under the mask. Returns
    (prediction, mask)."""
    if monolithic:
        mask = torch.ones(cfg.H, device=h.dev)
        h.ctrl.set_mask(mask)
    else:
        ids = torch.tensor(h.tok(question, add_special_tokens=False)["input_ids"],
                           dtype=torch.long, device=h.dev).unsqueeze(0)
        mask, _, _ = h.cfgr(h.model, ids,
                            steps=cfg.reconfig_steps if rsteps is None else rsteps)
        h.ctrl.set_mask(mask)
    pred = _generate(h, ANSWER_SYSTEM.format(a=speakers[0], b=speakers[1]),
                     FLOOR_USER.format(question=question), max_new_tokens)
    return pred, mask


def evidence_sessions(evidence: list) -> list[int]:
    out = []
    for e in evidence or []:
        m = re.match(r"D(\d+):", str(e))
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def route(mask: torch.Tensor, refs: dict, k: int):
    """Rank trained blocks by mask overlap. Block keys are '<conv>/s<idx>'."""
    scored = sorted(((float((mask.cpu() * ref).sum() / k), key)
                     for key, ref in refs.items()), reverse=True)
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--convs", nargs="*", default=None,
                    help="default: every conversation with >=1 trained block")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rsteps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg, meta = make_run_config(args.run)
    monolithic = bool(meta.get("monolithic"))
    from model import build_handles
    from train import load_state
    trained, refs, _ = load_state(cfg)
    if not trained:
        raise SystemExit(f"run '{args.run}' has no trained blocks")
    trained_convs = {k.split("/")[0] for k in trained}
    want = set(args.convs) if args.convs else trained_convs
    print(f"run {args.run}: {len(trained)} trained blocks over {sorted(trained_convs)}; "
          f"evaluating {sorted(want)} (monolithic={monolithic})")

    convs = [c for c in load_locomo() if c.sample_id in want]
    apply_corrections(convs, log=lambda *a: None)
    qa_evidence = {(c.sample_id, i): qa.evidence for c in convs for i, qa in enumerate(c.qa)}
    records = [r for r in question_records(convs)][: args.limit]

    h = build_handles(cfg)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{args.run}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["sample_id"], json.loads(l)["q_idx"])
                for l in out_path.read_text().splitlines() if l.strip()}
        print(f"resuming: {len(done)} already answered")

    t0, n = time.time(), 0
    with open(out_path, "a") as f:
        for r in records:
            if (r["sample_id"], r["q_idx"]) in done:
                continue
            pred, mask = answer_closed_book(
                h, cfg, r["question"], (r["speaker_a"], r["speaker_b"]),
                rsteps=args.rsteps, monolithic=monolithic)
            rec = {**r, "mode": f"ftn:{args.run}", "prediction": pred}
            if not monolithic:
                ranked = route(mask, refs, cfg.k)
                ev = evidence_sessions(qa_evidence.get((r["sample_id"], r["q_idx"])))
                top_conv, top_sess = ranked[0][1].split("/s")
                rec["route_top"] = ranked[0][1]
                rec["route_conv_hit"] = top_conv == r["sample_id"]
                rec["route_sess_hit"] = (top_conv == r["sample_id"]
                                         and int(top_sess) in ev) if ev else None
                rec["evidence_sessions"] = ev
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n} answered ({rate:.2f} q/s)", flush=True)

    lines = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    ch = [r for r in lines if "route_conv_hit" in r]
    sh = [r for r in lines if r.get("route_sess_hit") is not None]
    if ch:
        print(f"routing: conv-hit {sum(r['route_conv_hit'] for r in ch)}/{len(ch)}"
              + (f", session-hit {sum(r['route_sess_hit'] for r in sh)}/{len(sh)}" if sh else ""))
    print(f"done -> {out_path}  (score with score.py)")


if __name__ == "__main__":
    main()
