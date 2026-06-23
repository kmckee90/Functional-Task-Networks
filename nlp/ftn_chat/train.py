"""Incremental continual-learning trainer for the FTN-Chat bank.

Block-sequential: each task is learned as one block ON TOP of whatever is already
in the bank. State persists across invocations, so you can keep adding datasets:

    uv run python train.py                      # train any not-yet-learned tasks
    uv run python train.py --tasks robotics     # add/extend a single task block
    uv run python train.py --fresh              # wipe and start over
    uv run python train.py --epochs 2 --steps 600   # deeper

Per training step (batch = 1 chunk): perplexity-reconfigure the mask on the
chunk (weights frozen), then update ONLY the selected atoms on its causal-LM
loss. A mask-overlap-filtered replay over PRIOR tasks' stored (chunk, mask)
buffers protects earlier blocks — this is the retention mechanism, and the
buffers persist to disk so replay works across separate runs.

Persisted state in checkpoints/:  bank.pt (atom weights), refs.pt (per-task
reference masks), replay.pt (per-task replay buffers), state.json (trained
order + bank shape).
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from config import CFG
from data import build_examples, build_task, load_rehearsal, report
from model import build, load_bank, lora_params, reset_bank, save_bank
from perplexity_configurer import PerplexityConfigurer
from registry import DEFAULT_ORDER, TASKS
from spatial import kwta


def overlap(a, b, k):
    return float((a * b).sum() / k)


def _paths():
    d = CFG.ckpt_dir
    return d / "bank.pt", d / "refs.pt", d / "replay.pt", d / "state.json"


def load_state(wrappers):
    """Return (trained_keys, refs, replay_store). Empty if no checkpoint."""
    bank_p, refs_p, replay_p, state_p = _paths()
    if not (bank_p.exists() and state_p.exists()):
        return [], {}, {}
    state = json.loads(state_p.read_text())
    if state.get("H") != CFG.H or state.get("k") != CFG.k:
        raise SystemExit(
            f"checkpoint bank shape {state.get('H')}/{state.get('k')} "
            f"!= config {CFG.H}/{CFG.k}; use --fresh or fix config."
        )
    load_bank(wrappers, bank_p)
    refs = {k: v.float() for k, v in torch.load(refs_p, map_location="cpu").items()}
    replay_store = {}
    if replay_p.exists():
        raw = torch.load(replay_p, map_location="cpu")
        replay_store = {
            k: [(d["ids"], d["lbl"], d["mask"]) for d in buf] for k, buf in raw.items()
        }
    return state["trained"], refs, replay_store


def save_state(wrappers, trained, refs, replay_store):
    bank_p, refs_p, replay_p, state_p = _paths()
    save_bank(wrappers, bank_p)
    torch.save({k: v.detach().cpu() for k, v in refs.items()}, refs_p)
    torch.save(
        {
            k: [{"ids": ids, "lbl": lbl, "mask": m} for ids, lbl, m in buf]
            for k, buf in replay_store.items()
        },
        replay_p,
    )
    state_p.write_text(
        json.dumps({"trained": trained, "H": CFG.H, "k": CFG.k}, indent=2)
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="task keys to (continue) training; default = registry order",
    )
    p.add_argument("--fresh", action="store_true", help="reset the bank and state")
    p.add_argument(
        "--retrain",
        action="store_true",
        help="re-run a block even if the task is already trained",
    )
    p.add_argument("--epochs", type=int, default=CFG.epochs)
    p.add_argument("--steps", type=int, default=CFG.steps_per_domain)
    p.add_argument("--reconfig-steps", type=int, default=CFG.train_reconfig_steps)
    return p.parse_args()


def _rand(gen, n):
    return int(torch.randint(0, n, (1,), generator=gen, device="cpu"))


def _collate(batch, pad_id, dev):
    """Right-pad a list of {ids, lbl} examples into a single batched tensor.

    Returns (input_ids, attention_mask, labels). Pads input_ids with `pad_id`
    (masked out via attention_mask) and labels with -100 (ignored by the loss),
    so padding never contributes to the gradient. Right padding is safe under
    causal attention: real tokens never attend to trailing pads.
    """
    n = len(batch)
    L = max(int(e["ids"].size(0)) for e in batch)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long, device=dev)
    labels = torch.full((n, L), -100, dtype=torch.long, device=dev)
    attn = torch.zeros((n, L), dtype=torch.long, device=dev)
    for i, e in enumerate(batch):
        li = int(e["ids"].size(0))
        input_ids[i, :li] = e["ids"].to(dev)
        labels[i, :li] = e["lbl"].to(dev)
        attn[i, :li] = 1
    return input_ids, attn, labels


def train_block(
    model,
    ctrl,
    cfgr,
    opt,
    key,
    ex,
    replay_store,
    rehearsal,
    epochs,
    steps,
    rsteps,
    gen,
    dev,
    tok,
    t0,
):
    """Learn one task block on a mix of raw text (LM) and answer-masked Q&A.

    Each step assembles a batch of `CFG.batch_size` examples, reconfigures on
    ONE randomly chosen member's `recfg`, and applies that single FTN mask to
    the whole batch in one padded forward. Replay over prior tasks' stored
    (ids, lbl, mask) protects earlier blocks.
    """
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    raw, qa = ex["raw"], ex["qa"]
    prior = [
        (ids, lbl, m)
        for tk, buf in replay_store.items()
        if tk != key
        for ids, lbl, m in buf
    ]
    model.train()
    running = None
    bs = max(1, CFG.batch_size)
    for epoch in range(epochs):
        for step in range(steps):
            # Assemble a batch of `bs` examples, then reconfigure on one random
            # member and apply that single mask to the whole batch.
            batch, tally = [], {}
            for _ in range(bs):
                r = float(torch.rand(1, generator=gen, device="cpu"))
                if rehearsal and r < CFG.rehearsal_frac:  # general-instruction rehearsal
                    e, tag = rehearsal[_rand(gen, len(rehearsal))], "reh"
                else:
                    use_qa = (
                        qa
                        and float(torch.rand(1, generator=gen, device="cpu"))
                        < CFG.qa_frac
                    )
                    pool = qa if use_qa else raw
                    e, tag = pool[_rand(gen, len(pool))], ("qa" if use_qa else "raw")
                batch.append(e)
                tally[tag] = tally.get(tag, 0) + 1

            recfg = batch[_rand(gen, len(batch))]["recfg"]
            m, _, _ = cfgr(model, recfg.to(dev).unsqueeze(0), steps=rsteps)
            ctrl.set_mask(m)

            input_ids, attn, labels = _collate(batch, pad_id, dev)
            opt.zero_grad()
            loss = model(
                input_ids=input_ids, attention_mask=attn, labels=labels
            ).loss
            loss.backward()

            # Replay a prior-task example with mid-band mask overlap (retention),
            # under its own stored mask.
            if (
                prior
                and float(torch.rand(1, generator=gen, device="cpu")) < CFG.replay_frac
            ):
                cand = [
                    (ids, lbl, mm)
                    for ids, lbl, mm in prior
                    if 0.1 < overlap(mm.to(dev), m, CFG.k) <= 0.8
                ]
                if cand:
                    ids_r, lbl_r, m_r = cand[_rand(gen, len(cand))]
                    ctrl.set_mask(m_r.to(dev))
                    model(
                        input_ids=ids_r.to(dev).unsqueeze(0),
                        labels=lbl_r.to(dev).unsqueeze(0),
                    ).loss.backward()
                    ctrl.set_mask(m)
            opt.step()

            lv = float(loss.detach())
            running = lv if running is None else 0.9 * running + 0.1 * lv
            if step % 50 == 0:
                mix = " ".join(f"{t}{n}" for t, n in sorted(tally.items()))
                print(
                    f"  {key:<10} e{epoch} step {step:>4} [{mix}] nll={lv:.3f} "
                    f"ema={running:.3f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )


def build_ref_and_buffer(model, cfgr, ex, dev):
    """Reference mask from raw support reconfigs; replay buffer seeded from both
    raw support and a few Q&A examples (each stored as (ids, lbl, mask))."""
    model.eval()
    buf, acc = [], torch.zeros(CFG.H)
    for e in ex["support"][:16]:
        mm, _, _ = cfgr(model, e["recfg"].to(dev).unsqueeze(0))
        acc += mm.float()
        buf.append((e["ids"], e["lbl"], mm.detach().cpu()))
    for e in ex["qa"][:16]:
        mm, _, _ = cfgr(model, e["recfg"].to(dev).unsqueeze(0))
        buf.append((e["ids"], e["lbl"], mm.detach().cpu()))
    return kwta(acc, CFG.k), buf


def main():
    args = parse_args()
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok, ctrl, wrappers, lora_cfg = build(CFG)

    if args.fresh:
        reset_bank(wrappers, lora_cfg)
        trained, refs, replay_store = [], {}, {}
        print("  fresh bank")
    else:
        trained, refs, replay_store = load_state(wrappers)
        if trained:
            print(f"  resumed; already trained: {trained}")
        else:
            reset_bank(wrappers, lora_cfg)
            print("  no checkpoint found -> fresh bank")
    for p in lora_params(wrappers):
        p.requires_grad = True

    requested = args.tasks or DEFAULT_ORDER
    unknown = [k for k in requested if k not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}; registered: {list(TASKS)}")
    to_run = [k for k in requested if k not in trained or args.retrain]
    print(f"  plan: train {to_run}" + (" (nothing new)" if not to_run else ""))

    cfgr = PerplexityConfigurer(CFG, ctrl)
    # foreach=False: the bank params are sharded across GPUs (device_map), and
    # the foreach/fused Adam path can't group optimizer tensors living on
    # different devices. Per-param stepping keeps each update on its own GPU.
    opt = torch.optim.Adam(lora_params(wrappers), lr=CFG.lr, foreach=False)
    gen = torch.Generator(device="cpu").manual_seed(CFG.seed)
    rehearsal = load_rehearsal(CFG, tok)
    print(
        f"  rehearsal: {len(rehearsal)} general-instruction examples "
        f"(frac={CFG.rehearsal_frac})",
        flush=True,
    )
    t0 = time.time()

    for key in to_run:
        data = build_task(CFG, tok, key)
        report({key: data})
        ex = build_examples(CFG, tok, key, data)
        print(
            f"  examples: {len(ex['raw'])} raw chunks, {len(ex['qa'])} Q&A pairs",
            flush=True,
        )
        train_block(
            model,
            ctrl,
            cfgr,
            opt,
            key,
            ex,
            replay_store,
            rehearsal,
            args.epochs,
            args.steps,
            args.reconfig_steps,
            gen,
            dev,
            tok,
            t0,
        )
        ref, buf = build_ref_and_buffer(model, cfgr, ex, dev)
        refs[key] = ref
        replay_store[key] = buf
        if key not in trained:
            trained.append(key)
        save_state(wrappers, trained, refs, replay_store)
        print(f"  ✓ checkpointed after '{key}' (trained: {trained})", flush=True)

    # Overlap matrix over everything learned so far.
    keys = trained
    if keys:
        print("\noverlap matrix (|mi ∩ mj| / k):")
        print(f"{'':<12}" + "".join(f"{k:>12}" for k in keys))
        for ki in keys:
            print(
                f"{ki:<12}"
                + "".join(f"{overlap(refs[ki], refs[kj], CFG.k):>12.2f}" for kj in keys)
            )


if __name__ == "__main__":
    main()
