"""Sequential per-session ingestion driver.

    python benchmarking/ingest.py --run r1 --convs conv-26 \
        --grid 24 --rank 2 --k 192 --rsteps 8 --n-items 48 --epochs 2
    python benchmarking/ingest.py --run mono1 --convs conv-26 --monolithic
    python benchmarking/ingest.py --run full --convs all          # phase 2

For each conversation in order, for each session in order:
  synthesize self-study data (cached in $FTN_AGENT_WORK, shared across runs so
  the ftn and monolithic arms train on IDENTICAL data) -> distill the block
  into the bank -> store reference mask + replay buffer -> checkpoint every
  --ckpt-every blocks. A quick held-out-QA F1 over a sample of already-trained
  sessions runs every --heldout-every blocks (forgetting signal).

Resume: block keys already in state.json are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from distill_train import build_block_ref, make_examples, train_session_block
from eval_ftn import answer_closed_book, make_run_config, run_root
from locomo_data import load_locomo
from score import token_f1
from selfstudy import load_session_data, synthesize_session


def heldout_probe(h, cfg, convs_by_id, trained: list[str], monolithic: bool,
                  run_dir: Path, per_session: int = 2, max_sessions: int = 12,
                  log=print):
    """Token-F1 on held-out paraphrased QA sampled across trained sessions —
    cheap consolidation + forgetting signal (not the benchmark metric)."""
    keys = trained[-max_sessions:] if len(trained) > max_sessions else trained
    rows = []
    for key in keys:
        conv_id, s_idx = key.split("/s")
        try:
            data = load_session_data(conv_id, int(s_idx))
        except FileNotFoundError:
            continue
        conv = convs_by_id[conv_id]
        for qa in data["heldout_qa"][:per_session]:
            pred, _ = answer_closed_book(h, cfg, qa["question"],
                                         (conv.speaker_a, conv.speaker_b),
                                         monolithic=monolithic, max_new_tokens=64)
            rows.append({"block": key, "f1": token_f1(pred, qa["answer"]),
                         "question": qa["question"], "gold": qa["answer"], "pred": pred})
    if rows:
        f1 = sum(r["f1"] for r in rows) / len(rows)
        log(f"  [heldout] mean F1 {f1:.3f} over {len(rows)} qa "
            f"({len(keys)} sessions)")
        with open(run_dir / "heldout.jsonl", "a") as f:
            f.write(json.dumps({"after_block": trained[-1], "n": len(rows),
                                "mean_f1": f1, "rows": rows}) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--convs", nargs="+", required=True,
                    help="conversation sample_ids in ingestion order, or 'all'")
    ap.add_argument("--monolithic", action="store_true",
                    help="single-LoRA ablation: all-ones mask, no reconfig")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--rsteps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-items", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ce-weight", type=float, default=0.1)
    ap.add_argument("--replay-frac", type=float, default=0.3)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--heldout-every", type=int, default=5)
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="cap sessions per conversation (smoke)")
    args = ap.parse_args()

    cfg, meta = make_run_config(
        args.run, grid_dim=args.grid, rank=args.rank, k=args.k,
        reconfig_steps=args.rsteps, train_reconfig_steps=args.rsteps,
        lr=args.lr, monolithic=args.monolithic)
    run_dir = run_root(args.run)
    (run_dir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str))

    convs = load_locomo()   # ORIGINAL gold untouched — training never sees QA
    convs_by_id = {c.sample_id: c for c in convs}
    order = [c.sample_id for c in convs] if args.convs == ["all"] else args.convs

    from model import build_handles
    from train import load_state, make_optimizer, save_state
    h = build_handles(cfg)
    opt = make_optimizer(cfg, h.wrappers)
    trained, refs, replay_store = load_state(cfg)
    print(f"run {args.run}: {len(trained)} blocks already trained")

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    since_ckpt = 0
    for conv_id in order:
        conv = convs_by_id[conv_id]
        sessions = conv.sessions[: args.max_sessions]
        for sess in sessions:
            key = f"{conv_id}/s{sess.index}"
            if key in trained:
                continue
            t0 = time.time()
            synthesize_session(h.model, h.tok, h.ctrl, cfg.H, conv, sess,
                               n_items=args.n_items, seed=cfg.seed)
            data = load_session_data(conv_id, sess.index)
            examples = make_examples(h.tok, data)
            t_synth = time.time() - t0
            hist = train_session_block(
                cfg, h, opt, key, examples, replay_store,
                epochs=args.epochs, batch_size=args.batch_size,
                monolithic=args.monolithic, replay_frac=args.replay_frac,
                ce_weight=args.ce_weight, gen=gen)
            ref, buf = build_block_ref(cfg, h, examples, monolithic=args.monolithic)
            refs[key], replay_store[key] = ref, buf
            trained.append(key)
            since_ckpt += 1
            print(f"  ✓ {key}: {len(examples)} items, kl {hist[0]:.3f}->{hist[-1]:.3f}, "
                  f"synth {t_synth:.0f}s train {time.time() - t0 - t_synth:.0f}s "
                  f"({len(trained)} total)", flush=True)
            if since_ckpt >= args.ckpt_every:
                save_state(cfg, h.wrappers, trained, refs, replay_store)
                since_ckpt = 0
                print(f"  [ckpt] saved at {len(trained)} blocks", flush=True)
            if len(trained) % args.heldout_every == 0:
                heldout_probe(h, cfg, convs_by_id, trained, args.monolithic, run_dir)
        save_state(cfg, h.wrappers, trained, refs, replay_store)
        since_ckpt = 0
        print(f"[ckpt] conversation {conv_id} complete "
              f"({len(trained)} blocks)", flush=True)
    print("ingestion complete")


if __name__ == "__main__":
    main()
