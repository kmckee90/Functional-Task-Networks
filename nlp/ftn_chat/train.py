"""Incremental continual-learning trainer for the FTN-Chat bank.

Each epoch is one interleaved pass over all requested tasks — with --epochs 2
and two tasks the order is base_ftn, explorer, base_ftn, explorer — each task
learned as a block ON TOP of whatever is already in the bank, checkpointing
after every epoch. State persists across invocations, so you can keep adding
datasets:

    uv run python train.py                      # train any not-yet-learned tasks
    uv run python train.py --tasks globo        # add/extend a single task block
    uv run python train.py --fresh              # wipe and start over
    uv run python train.py --epochs 2 --steps 600   # two interleaved passes, deeper

BayesLoRA mask inference: each training batch makes a small posterior update
on one bare question's NLL. The EMA of prompt NLL under the preceding mask
detects changepoints; those get a longer reset/refit. The detector and logits
persist across task blocks and train.py invocations.

Mask-filtered replay stores sparse, time-spaced Q&A examples with their hard
training masks. Partially overlapping stored masks get occasional distillation
updates under their own masks. The live posterior and detector are untouched.

  Exact-k Gumbel top-k masks with a sigmoid straight-through gradient.

OBJECTIVE — context distillation, with optional source-text CE: per batch the teacher
(the SAME frozen backbone with the source excerpt in context and the mask
zeroed) provides answer-token distributions; the student (masked bank, no
context) minimizes KL to them over the answer span. Teacher logits are
mask-independent; posterior search precedes the teacher/student passes. When
enabled, a sampled training-only source chunk adds next-token CE under the
same question-selected mask.

Persisted state in checkpoints/: bank.pt, refs.pt, mask.pt, posterior.pt,
replay.pt, state.json. Legacy checkpoints load with an empty replay buffer.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from config import CFG
from data import build_examples, build_task, load_rehearsal, report
from model import build, load_bank, lora_params, reset_bank, save_bank
from bayes_configurer import BayesianPerplexityConfigurer
from mask_replay import MaskReplayBuffer
from input_routing import InputRoutingArchive
from route_repair import repair_routes
from registry import DEFAULT_ORDER, TASKS
from spatial import kwta


def overlap(a, b, k):
    return float((a * b).sum() / k)


def _paths():
    d = CFG.ckpt_dir
    return (d / "bank.pt", d / "refs.pt", d / "state.json", d / "mask.pt",
            d / "posterior.pt", d / "replay.pt", d / "routing.pt")


def load_state(wrappers, ctrl):
    """Return trained tasks, refs, detector, posterior, replay state."""
    bank_p, refs_p, state_p, mask_p, posterior_p, replay_p, routing_p = _paths()
    if not (bank_p.exists() and state_p.exists()):
        return [], {}, None, None, None, None
    state = json.loads(state_p.read_text())
    for key, want in (("H", CFG.H), ("k", CFG.k), ("rank", CFG.rank),
                      ("n_wrappers", len(wrappers))):
        got = state.get(key)
        if got is not None and got != want:
            raise SystemExit(
                f"checkpoint {key}={got} != config {want}; use --fresh or fix config."
            )
    load_bank(wrappers, bank_p)
    refs = {k: v.float() for k, v in torch.load(refs_p, map_location="cpu").items()}
    if mask_p.exists():
        ctrl.set_mask(torch.load(mask_p, map_location="cpu").float())
    posterior = torch.load(posterior_p, map_location="cpu", weights_only=True) if posterior_p.exists() else None
    detector = state.get("reactive") if state.get("detector_objective") == "prompt_nll" and posterior is not None else None
    replay = torch.load(replay_p, map_location="cpu", weights_only=True) if replay_p.exists() else None
    routing = torch.load(routing_p, map_location="cpu", weights_only=True) if routing_p.exists() else None
    return state["trained"], refs, detector, posterior, replay, routing


def save_state(wrappers, trained, refs, cfgr, ctrl, replay, routing):
    bank_p, refs_p, state_p, mask_p, posterior_p, replay_p, routing_p = _paths()
    save_bank(wrappers, bank_p)
    torch.save({k: v.detach().cpu() for k, v in refs.items()}, refs_p)
    torch.save(ctrl.mask.detach().cpu(), mask_p)
    torch.save(cfgr.state_dict(), posterior_p)
    torch.save(replay.state_dict(), replay_p)
    torch.save(routing.state_dict(), routing_p)
    state_p.write_text(
        json.dumps(
            {
                "trained": trained,
                "H": CFG.H,
                "k": CFG.k,
                "rank": CFG.rank,
                "layer_frac": CFG.lora_layer_frac,
                "n_wrappers": len(wrappers),
                "lateral": {name: getattr(CFG, name) for name in (
                    "lateral_steps", "lateral_kernel", "lateral_polyak",
                    "lateral_noise_sd", "torus")},
                "reactive": cfgr.detector.to_dict(),
                "detector_objective": "prompt_nll",
            },
            indent=2,
        )
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
    p.add_argument("--input-routing", action="store_true",
                   help="maintain a repaired question-conditioned routing archive")
    p.add_argument("--build-routes", action="store_true",
                   help="fit routes for already-trained tasks without changing bank weights")
    p.add_argument(
        "--retrain",
        action="store_true",
        help="re-run a block even if the task is already trained",
    )
    p.add_argument("--epochs", type=int, default=CFG.epochs)
    p.add_argument("--steps", type=int, default=CFG.steps_per_domain)
    p.add_argument("--reconfig-steps", type=int, default=CFG.train_reconfig_steps)
    p.add_argument("--source-ce-prob", type=float, default=None,
                   help="fraction of Q&A steps that also train on a source chunk")
    p.add_argument("--source-ce-weight", type=float, default=None,
                   help="weight of source next-token CE on selected steps")
    p.add_argument(
        "--seed",
        type=int,
        default=CFG.seed,
        help="RNG seed for torch + data sampling (default: config seed). Use a "
        "different value to draw a fresh batch sequence when continuing a run.",
    )
    return p.parse_args()


def _rand(gen, n):
    return int(torch.randint(0, n, (1,), generator=gen, device="cpu"))


def _collate(seqs, pad_id, dev):
    """Right-pad a list of (ids, lbl) pairs into batched tensors.

    Returns (input_ids, attention_mask, labels). Pads input_ids with `pad_id`
    (masked out via attention_mask) and labels with -100, so padding never
    contributes. Right padding is safe under causal attention."""
    n = len(seqs)
    L = max(int(ids.size(0)) for ids, _ in seqs)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long, device=dev)
    labels = torch.full((n, L), -100, dtype=torch.long, device=dev)
    attn = torch.zeros((n, L), dtype=torch.long, device=dev)
    for i, (ids, lbl) in enumerate(seqs):
        li = int(ids.size(0))
        input_ids[i, :li] = ids.to(dev)
        labels[i, :li] = lbl.to(dev)
        attn[i, :li] = 1
    return input_ids, attn, labels


def _gather_answer_logits(logits, labels):
    """Per-row logits at the positions that PREDICT the answer tokens, plus the
    answer token ids, concatenated over the batch: (N_ans, V), (N_ans,).

    labels[b, t] != -100 marks answer token t; it is predicted by logits at
    t-1. Row order is (batch, position), identical for teacher and student
    because both carry the same answer token ids per example."""
    sel_logits, sel_targets = [], []
    for b in range(labels.size(0)):
        pos = (labels[b] != -100).nonzero(as_tuple=True)[0]
        pos = pos[pos > 0]
        sel_logits.append(logits[b, pos - 1])
        sel_targets.append(labels[b, pos])
    return torch.cat(sel_logits, 0), torch.cat(sel_targets, 0)


def distill_loss(model, ctrl, mask, s_batch, teacher_sel, cfg):
    """Student forward under `mask`; KL to the cached teacher answer-token
    distributions (+ optional CE on the teacher's answer tokens)."""
    s_ids, s_attn, s_lbl = s_batch
    ctrl.set_mask(mask)
    logits = model(input_ids=s_ids, attention_mask=s_attn).logits
    s_sel, targets = _gather_answer_logits(logits, s_lbl)
    T = cfg.kl_temp
    s_logp = F.log_softmax(s_sel.float() / T, dim=-1)
    t_logp = F.log_softmax(teacher_sel.float().to(s_sel.device) / T, dim=-1)
    # Per-token forward KL, clamped (OPSD-style clipping) so a few stylistic /
    # high-surprise positions can't dominate the batch gradient.
    kl_per_tok = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    if cfg.kl_clip > 0:
        kl_per_tok = kl_per_tok.clamp(max=cfg.kl_clip)
    loss = kl_per_tok.mean() * (T * T)
    if cfg.ce_weight > 0:
        loss = loss + cfg.ce_weight * F.cross_entropy(s_sel.float(), targets)
    return loss


def source_ce_loss(model, ctrl, mask, chunk, dev):
    """Next-token CE on a training-only source chunk under the Q&A mask."""
    ids = chunk.to(dev).unsqueeze(0)
    ctrl.set_mask(mask)
    return model(input_ids=ids, labels=ids).loss


@torch.no_grad()
def teacher_answer_logits(model, ctrl, t_batch):
    """Teacher pass: same backbone, mask zeroed (bank inert), source excerpt in
    context. Returns the answer-position logits, detached."""
    t_ids, t_attn, t_lbl = t_batch
    ctrl.set_mask(torch.zeros(ctrl.H, device=ctrl.mask.device))
    logits = model(input_ids=t_ids, attention_mask=t_attn).logits
    sel, _ = _gather_answer_logits(logits, t_lbl)
    return sel.detach()


def stored_mask_loss(model, ctrl, item, pad_id, dev):
    """Distill one stored example under its original mask.

    Teacher predictions remain available because the backbone is frozen. The
    live mask is restored before returning, so the next prompt's changepoint
    probe observes the actual persistent state rather than a replay mask.
    """
    live_mask = ctrl.mask.detach().clone()
    ex = item["example"]
    try:
        t_batch = _collate([(ex["t_ids"], ex["t_lbl"])], pad_id, dev)
        teacher_sel = teacher_answer_logits(model, ctrl, t_batch)
        s_batch = _collate([(ex["ids"], ex["lbl"])], pad_id, dev)
        return distill_loss(model, ctrl, item["mask"].to(dev),
                            s_batch, teacher_sel, CFG)
    finally:
        ctrl.set_mask(live_mask)


def train_block(
    model, ctrl, cfgr, opt, key, ex, rehearsal, replay,
    steps, rsteps, gen, source_gen, dev, tok, t0, epoch, ref_acc,
):
    """Train one task for a SINGLE epoch by context distillation with reactive
    reconfig. main() owns the epoch loop and interleaves the tasks, so one call
    is one pass of `steps` steps over `key` (`epoch` is passed only for logging).

    Each step draws `CFG.batch_size` (question, answer, context) examples,
    computes the teacher's answer-token distributions once (mask-independent),
    and trains the selected atoms on the student KL. The mask is only
    updated on one random member's bare question, the same signal inference
    uses. Stable updates are short; changepoints use the longer reset/refit.
    Optional source CE uses the resulting mask without another reconfiguration.

    Returns the task's reference mask (kwta over the dwell-weighted sum of every
    step's training mask) — used ONLY for GUI color-coding, harvested for free
    from the masks training already produced (no extra reconfiguration searches;
    the polished usage-weighted refs come from rebuild_refs_usage.py offline).
    `ref_acc` is the caller-owned (H,) accumulator for this task; it persists
    across epochs, so the returned mask reflects usage over ALL epochs so far."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    qa, source = ex["qa"], ex["source"]
    if not qa:
        raise SystemExit(f"no distillation triples for '{key}'; run qa_synth.py first")
    if CFG.source_ce_prob > 0 and CFG.source_ce_weight > 0 and not source:
        raise SystemExit(f"source CE enabled but '{key}' has no training chunks")
    model.train()
    running, source_running = None, None
    n_reconfigs, n_replays, n_stored, n_source = 0, 0, 0, 0

    for step in range(steps):
        batch, task_examples, tally = [], [], {}
        for _ in range(CFG.batch_size):
            if rehearsal and float(
                torch.rand(1, generator=gen, device="cpu")
            ) < CFG.rehearsal_frac:
                e, tag = rehearsal[_rand(gen, len(rehearsal))], "reh"
            else:
                e, tag = qa[_rand(gen, len(qa))], "qa"
            batch.append(e)
            if tag == "qa":
                task_examples.append(e)
            tally[tag] = tally.get(tag, 0) + 1

        s_batch = _collate([(e["ids"], e["lbl"]) for e in batch], pad_id, dev)
        t_batch = _collate([(e["t_ids"], e["t_lbl"]) for e in batch], pad_id, dev)
        # One support question per batch, just as in the previous trainer.
        # Every batch gets a cheap posterior update; changes get a longer fit.
        e = batch[_rand(gen, len(batch))]
        m, _, _ = cfgr(model, e["recfg"].to(dev).unsqueeze(0),
                        changepoint_steps=rsteps)
        n_reconfigs += int(cfgr.last_stats["changepoint"])
        teacher_sel = teacher_answer_logits(model, ctrl, t_batch)
        loss = distill_loss(model, ctrl, m, s_batch, teacher_sel, CFG)

        # Harvest the mask this step trained under for the task's reference mask
        # (GUI color-coding only). m is k-hot each step and persists until a
        # reconfig, so summing every step dwell-weights by how long each mask
        # was active; kwta at block end picks the atoms the task used most. No
        # extra reconfiguration passes — this replaces the old build_ref sweep.
        ref_acc += m.detach().float().cpu()

        # Draw replay from the previously stored buffer before considering this
        # batch. Only partially overlapping stored masks are eligible.
        replay_item = None
        if (len(replay) and CFG.replay_weight > 0 and
                float(torch.rand((), generator=gen, device="cpu")) < CFG.replay_use_prob):
            replay_item = replay.sample(m, gen)
        n_stored += int(replay.consider(key, task_examples, m, gen))

        opt.zero_grad()
        loss.backward()
        if (CFG.source_ce_prob > 0 and CFG.source_ce_weight > 0 and
                float(torch.rand((), generator=source_gen, device="cpu"))
                < CFG.source_ce_prob):
            chunk = source[_rand(source_gen, len(source))]
            source_loss = source_ce_loss(model, ctrl, m, chunk, dev)
            (CFG.source_ce_weight * source_loss).backward()
            sv = float(source_loss.detach())
            source_running = (sv if source_running is None else
                              0.9 * source_running + 0.1 * sv)
            n_source += 1
        if replay_item is not None:
            old_loss = stored_mask_loss(model, ctrl, replay_item, pad_id, dev)
            (CFG.replay_weight * old_loss).backward()
            n_replays += 1
        opt.step()

        lv = float(loss.detach())
        running = lv if running is None else 0.9 * running + 0.1 * lv
        if step % 50 == 0:
            mix = " ".join(f"{t}{n}" for t, n in sorted(tally.items()))
            source_display = "--" if source_running is None else f"{source_running:.3f}"
            print(
                f"  {key:<10} e{epoch} step {step:>4} [{mix}] kl={lv:.3f} "
                f"ema={running:.3f} reconfigs={n_reconfigs} "
                f"source_ce={n_source} source_ema={source_display} "
                f"replay={n_replays} stored={n_stored}/{len(replay)} "
                f"(prompt_ema={cfgr.detector.mean:.3f} sd={cfgr.detector.sd():.3f}) "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    print(
        f"  [{key}] e{epoch} done: reactive_reconfigs={n_reconfigs} "
        f"source_ce={n_source} replay={n_replays} stored={n_stored}/{len(replay)} "
        f"prompt_ema={cfgr.detector.mean:.4f} sd={cfgr.detector.sd():.4f}",
        flush=True,
    )
    return kwta(ref_acc, CFG.k)


def main():
    args = parse_args()
    CFG.seed = args.seed  # feeds both torch.manual_seed below and the data gen
    if args.source_ce_prob is not None:
        CFG.source_ce_prob = args.source_ce_prob
    if args.source_ce_weight is not None:
        CFG.source_ce_weight = args.source_ce_weight
    if not 0 <= CFG.source_ce_prob <= 1 or CFG.source_ce_weight < 0:
        raise SystemExit("source CE probability must be in [0, 1] and weight nonnegative")
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok, ctrl, wrappers, lora_cfg = build(CFG)

    if args.fresh:
        reset_bank(wrappers, lora_cfg)
        trained, refs, reactive_d, posterior, replay_d, routing_d = [], {}, None, None, None, None
        print("  fresh bank")
    else:
        trained, refs, reactive_d, posterior, replay_d, routing_d = load_state(wrappers, ctrl)
        if trained:
            print(f"  resumed; already trained: {trained}")
        else:
            reset_bank(wrappers, lora_cfg)
            print("  no checkpoint found -> fresh bank")
    for p in lora_params(wrappers):
        p.requires_grad = True
    cfgr = BayesianPerplexityConfigurer(CFG, ctrl, reactive_d)
    if posterior is not None:
        cfgr.load_state_dict(posterior)
    replay = MaskReplayBuffer(CFG)
    if replay_d is not None:
        replay.load_state_dict(replay_d)
    routing = InputRoutingArchive(CFG, tok)
    if routing_d is not None:
        routing.load_state_dict(routing_d)
    routing_enabled = args.input_routing or args.build_routes or bool(routing.routes)

    if args.build_routes:
        if not trained:
            raise SystemExit("--build-routes needs an existing trained checkpoint")
        print(f"  fitting input routes for {trained}", flush=True)
        stats = repair_routes(model, ctrl, cfgr, routing, trained, tok, CFG)
        print(f"  routing repair: {stats}", flush=True)
        torch.save(routing.state_dict(), CFG.ckpt_dir / "routing.pt")
        return

    requested = args.tasks or DEFAULT_ORDER
    unknown = [k for k in requested if k not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}; registered: {list(TASKS)}")
    to_run = [k for k in requested if k not in trained or args.retrain]
    print(f"  plan: train {to_run}" + (" (nothing new)" if not to_run else ""))
    print(
        f"  BayesLoRA: stable={CFG.bayes_reconfig_steps} steps T={CFG.bayes_temperature} "
        f"KL={CFG.bayes_kl_weight}; changepoint={args.reconfig_steps} steps "
        f"T={CFG.bayes_changepoint_temperature}; detector sd={CFG.reactive_sd} "
        f"ema={CFG.reactive_ema}; replay: {CFG.replay_use_prob:.0%}/step "
        f"from {len(replay)} stored (store {CFG.replay_store_prob:.0%}/step, "
        f"gap {CFG.replay_min_gap}, cap {CFG.replay_per_task}/task)"
    )

    # foreach=False: the bank params are sharded across GPUs (device_map), and
    # the foreach/fused Adam path can't group optimizer tensors living on
    # different devices. Per-param stepping keeps each update on its own GPU.
    opt = torch.optim.Adam(lora_params(wrappers), lr=CFG.lr, foreach=False)
    gen = torch.Generator(device="cpu").manual_seed(CFG.seed)
    source_gen = torch.Generator(device="cpu").manual_seed(CFG.seed + 104729)
    rehearsal = load_rehearsal(CFG, tok)
    if rehearsal:
        print(f"  rehearsal: {len(rehearsal)} examples (frac={CFG.rehearsal_frac})")
    t0 = time.time()

    # Build each requested task's examples once, then interleave: every epoch
    # is a full pass over all tasks (base_ftn, explorer, base_ftn, ... for
    # --epochs 2), not `epochs` back-to-back passes of a single task.
    task_ex = {}
    for key in to_run:
        data = build_task(CFG, tok, key)
        report({key: data})
        ex = build_examples(CFG, tok, key, data)
        ex["source"] = data.train
        print(f"  examples[{key}]: {len(ex['qa'])} distillation triples", flush=True)
        if CFG.source_ce_prob > 0 and CFG.source_ce_weight > 0:
            print(f"  source CE: {len(data.train)} train chunks, "
                  f"{CFG.source_ce_prob:.0%} of Q&A steps, weight {CFG.source_ce_weight:g}",
                  flush=True)
        task_ex[key] = ex

    # Per-task reference-mask accumulators, owned here so they persist across
    # epochs: each task's mask usage is summed over every epoch, not reset per
    # pass (see train_block).
    ref_acc = {k: torch.zeros(CFG.H, device="cpu") for k in to_run}
    for epoch in range(args.epochs):
        if not to_run:
            break
        for key in to_run:
            ex = task_ex[key]
            # Reset Adam state at each task switch (base resets per task) so
            # optimizer momentum from the previous task can't drift atoms the
            # new mask excludes.
            opt.state = defaultdict(dict)
            # train_block returns the task's reference mask, harvested from the
            # masks it actually trained under — no extra reconfiguration sweep.
            refs[key] = train_block(
                model, ctrl, cfgr, opt, key, ex, rehearsal, replay,
                args.steps, args.reconfig_steps, gen, source_gen, dev, tok, t0, epoch,
                ref_acc[key],
            )
            if key not in trained:
                trained.append(key)
            if routing_enabled:
                stats = repair_routes(model, ctrl, cfgr, routing, trained, tok,
                                      CFG, current_task=key)
                print(f"  routing repair after {key}: {stats}", flush=True)
        # Persist once per epoch: bank + refs for every task + reactive EMA.
        save_state(wrappers, trained, refs, cfgr, ctrl, replay, routing)
        print(f"  ✓ checkpointed after epoch {epoch} (trained: {trained})", flush=True)

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
