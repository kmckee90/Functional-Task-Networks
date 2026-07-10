"""Stream-block continual-learning trainer for FTN-Agent.

Block-sequential (design.md §5.3): each synthesis run is one block, learned ON
TOP of the existing bank. Per step: perplexity-reconfigure the mask on a batch
member (weights frozen), train ONLY the selected atoms on the batch's causal-LM /
answer-masked loss, then an overlap-filtered replay over PRIOR blocks protects
earlier memory. State persists in state_dir (design.md §7): bank.pt, refs.pt,
replay.pt, state.json (+ blocks.json via registry).

Importable primitives (used by app.py / daemon.py) and a CLI:
    python train.py --list                 # blocks trained + pending datasets
    python train.py --all-pending          # train every un-trained dataset in work_dir
    python train.py --block <key>          # train one saved dataset
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import torch

import registry
from config import CFG, Config
from data import build_examples, load_block_dataset, load_rehearsal, prune_block_dataset
from model import build_handles, lora_params, save_bank
from spatial import kwta


# --- state I/O ---------------------------------------------------------------

def _paths(cfg: Config):
    d = cfg.state_dir
    return d / "bank.pt", d / "refs.pt", d / "replay.pt", d / "state.json"


def load_state(cfg: Config):
    """Return (trained_keys, refs, replay_store). Empty if no checkpoint."""
    bank_p, refs_p, replay_p, state_p = _paths(cfg)
    if not state_p.exists():
        return [], {}, {}
    state = json.loads(state_p.read_text())
    if state.get("H") not in (None, cfg.H) or state.get("k") not in (None, cfg.k):
        raise SystemExit(
            f"checkpoint bank shape {state.get('H')}/{state.get('k')} != config "
            f"{cfg.H}/{cfg.k}; wipe state_dir or fix config.")
    refs = {}
    if refs_p.exists():
        refs = {k: v.float() for k, v in torch.load(refs_p, map_location="cpu").items()}
    replay_store = {}
    if replay_p.exists():
        raw = torch.load(replay_p, map_location="cpu")
        replay_store = {k: [(d["ids"], d["lbl"], d["mask"]) for d in buf]
                        for k, buf in raw.items()}
    return state.get("trained", []), refs, replay_store


def save_state(cfg: Config, wrappers, trained, refs, replay_store):
    bank_p, refs_p, replay_p, state_p = _paths(cfg)
    # rotate: keep latest + one previous bank checkpoint (design.md §7.3)
    if bank_p.exists() and cfg.keep_checkpoints > 1:
        shutil.copy(bank_p, cfg.state_dir / "bank.prev.pt")
    save_bank(wrappers, bank_p)
    torch.save({k: v.detach().cpu() for k, v in refs.items()}, refs_p)
    torch.save({k: [{"ids": ids, "lbl": lbl, "mask": m} for ids, lbl, m in buf]
                for k, buf in replay_store.items()}, replay_p)
    state_p.write_text(json.dumps({"trained": trained, "H": cfg.H, "k": cfg.k}, indent=2))


def make_optimizer(cfg: Config, wrappers):
    params = lora_params(wrappers)
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            return bnb.optim.Adam8bit(params, lr=cfg.lr)
        except Exception as e:
            print(f"  8-bit Adam unavailable ({e}); falling back to fp32 Adam")
    # foreach=False: bank params are sharded across GPUs (device_map); the
    # foreach/fused path can't group optimizer tensors on different devices
    # (design.md CS1 — this is the exact bug the expert should warn about).
    return torch.optim.Adam(params, lr=cfg.lr, foreach=False)


# --- training ----------------------------------------------------------------

def overlap(a, b, k):
    return float((a * b).sum() / k)


def _rand(gen, n):
    return int(torch.randint(0, n, (1,), generator=gen, device="cpu"))


def _collate(batch, pad_id, dev):
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


def train_block(cfg, h, opt, key, ex, replay_store, rehearsal,
                epochs, steps, rsteps, gen, log=print):
    """Learn one block on a mix of answer-masked Q&A/insight + raw LM, with
    overlap-filtered replay over prior blocks. Mutates the bank in place."""
    model, ctrl, cfgr, tok, dev = h.model, h.ctrl, h.cfgr, h.tok, h.dev
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    raw, qa = ex["raw"], ex["qa"]
    if not raw and not qa:
        log(f"  {key}: no examples, skipping")
        return
    prior = [(ids, lbl, m) for tk, buf in replay_store.items() if tk != key
             for ids, lbl, m in buf]
    model.train()
    running, bs = None, max(1, cfg.batch_size)
    t0 = time.time()
    for epoch in range(epochs):
        for step in range(steps):
            batch, tally = [], {}
            for _ in range(bs):
                r = float(torch.rand(1, generator=gen, device="cpu"))
                if rehearsal and r < cfg.rehearsal_frac:
                    e, tag = rehearsal[_rand(gen, len(rehearsal))], "reh"
                else:
                    use_qa = qa and (float(torch.rand(1, generator=gen, device="cpu")) < cfg.qa_frac or not raw)
                    pool = qa if use_qa else (raw or qa)
                    e, tag = pool[_rand(gen, len(pool))], ("qa" if use_qa else "raw")
                batch.append(e)
                tally[tag] = tally.get(tag, 0) + 1

            recfg = batch[_rand(gen, len(batch))]["recfg"]
            m, _, _ = cfgr(model, recfg.to(dev).unsqueeze(0), steps=rsteps)
            ctrl.set_mask(m)

            input_ids, attn, labels = _collate(batch, pad_id, dev)
            opt.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            loss.backward()

            if prior and float(torch.rand(1, generator=gen, device="cpu")) < cfg.replay_frac:
                cand = [(ids, lbl, mm) for ids, lbl, mm in prior
                        if 0.1 < overlap(mm.to(dev), m, cfg.k) <= 0.8]
                if cand:
                    ids_r, lbl_r, m_r = cand[_rand(gen, len(cand))]
                    ctrl.set_mask(m_r.to(dev))
                    model(input_ids=ids_r.to(dev).unsqueeze(0),
                          labels=lbl_r.to(dev).unsqueeze(0)).loss.backward()
                    ctrl.set_mask(m)
            opt.step()

            lv = float(loss.detach())
            running = lv if running is None else 0.9 * running + 0.1 * lv
            if step % 25 == 0:
                mix = " ".join(f"{t}{n}" for t, n in sorted(tally.items()))
                log(f"  {key} e{epoch} step {step:>4} [{mix}] nll={lv:.3f} "
                    f"ema={running:.3f} ({time.time() - t0:.0f}s)")


def build_ref_and_buffer(cfg, h, ex):
    """Reference mask from support reconfigs + a bounded replay buffer for this
    block (design.md §7.3 retention bounds)."""
    model, ctrl, cfgr, dev = h.model, h.ctrl, h.cfgr, h.dev
    model.eval()
    buf, acc = [], torch.zeros(cfg.H, device=dev)  # explicit: worker-thread safe
    pool = (ex["support"] + ex["qa"])[: cfg.replay_per_block]
    for e in pool:
        mm, _, _ = cfgr(model, e["recfg"].to(dev).unsqueeze(0))
        acc += mm.float()
        buf.append((e["ids"], e["lbl"], mm.detach().cpu()))
    ref = kwta(acc, cfg.k) if float(acc.sum()) > 0 else torch.zeros(cfg.H)
    return ref, buf


def _trim_replay(cfg, replay_store, trained):
    """Enforce max_blocks_retained: drop oldest blocks' buffers."""
    if len(trained) <= cfg.max_blocks_retained:
        return
    for old in trained[: len(trained) - cfg.max_blocks_retained]:
        replay_store.pop(old, None)


# --- high-level: train one block from a saved dataset ------------------------

def train_saved_block(cfg, h, opt, key, rehearsal, epochs=None, steps=None,
                      rsteps=None, log=print) -> bool:
    """Load a block's dataset from work_dir, train it on top of the bank,
    checkpoint, register the block, and PRUNE the dataset (design.md §7.3).
    Returns True if trained."""
    import time as _t
    pairs, raw_text = load_block_dataset(cfg, key)
    ex = build_examples(cfg, h.tok, pairs, raw_text)
    if not ex["qa"] and not ex["raw"]:
        log(f"  {key}: empty dataset, skipping")
        return False
    log(f"  {key}: {len(ex['qa'])} qa/insight + {len(ex['raw'])} raw chunks")

    trained, refs, replay_store = load_state(cfg)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + len(trained))
    # Training a 32B backward needs gradient checkpointing (cache off) or it OOMs;
    # generation wants the opposite. Toggle for the duration, then restore.
    model = h.model
    prev_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass
    try:
        train_block(cfg, h, opt, key, ex, replay_store, rehearsal,
                    epochs or cfg.epochs, steps or cfg.steps_per_block,
                    rsteps or cfg.train_reconfig_steps, gen, log)
        ref, buf = build_ref_and_buffer(cfg, h, ex)
    finally:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        model.config.use_cache = prev_cache
        model.eval()
    refs[key] = ref
    replay_store[key] = buf
    if key not in trained:
        trained.append(key)
    _trim_replay(cfg, replay_store, trained)
    save_state(cfg, h.wrappers, trained, refs, replay_store)
    registry.register_block(cfg.state_dir, key, _label_for(key, pairs),
                            created_ts=_t.time(), n_examples=len(ex["qa"]) + len(ex["raw"]))
    prune_block_dataset(cfg, key)
    log(f"  ✓ trained + checkpointed '{key}'; pruned dataset (trained: {len(trained)})")
    return True


def _label_for(key: str, pairs: list[dict]) -> str:
    n_ins = sum(p.get("kind") == "insight" for p in pairs)
    return f"{key} ({len(pairs)} ex, {n_ins} insights)"


def pending_blocks(cfg) -> list[str]:
    """Datasets sitting in work_dir that aren't in state.trained yet."""
    from pathlib import Path
    trained, _, _ = load_state(cfg) if (cfg.state_dir / "state.json").exists() else ([], {}, {})
    have = {p.name for p in Path(cfg.work_dir).iterdir() if p.is_dir()} if Path(cfg.work_dir).exists() else set()
    return [k for k in sorted(have) if k not in trained]


def train_pending(cfg, h, opt=None, log=print) -> int:
    """Train every pending dataset. Returns #blocks trained."""
    opt = opt or make_optimizer(cfg, h.wrappers)
    rehearsal = load_rehearsal(cfg, h.tok)
    n = 0
    for key in pending_blocks(cfg):
        if train_saved_block(cfg, h, opt, key, rehearsal, log=log):
            n += 1
    return n


# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all-pending", action="store_true")
    ap.add_argument("--block", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    if args.list:
        trained, _, _ = load_state(CFG) if (CFG.state_dir / "state.json").exists() else ([], {}, {})
        print("trained blocks:", trained)
        print("pending datasets:", pending_blocks(CFG))
        return

    h = build_handles(CFG)
    opt = make_optimizer(CFG, h.wrappers)
    rehearsal = load_rehearsal(CFG, h.tok)
    if args.block:
        train_saved_block(CFG, h, opt, args.block, rehearsal, args.epochs, args.steps)
    elif args.all_pending:
        n = train_pending(CFG, h, opt)
        print(f"trained {n} pending block(s)")
    else:
        print("nothing to do; use --list / --all-pending / --block <key>")


if __name__ == "__main__":
    main()
