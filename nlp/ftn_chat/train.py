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

Design mirrors base/ `run_experiments.py --config Large NoReplay Reactive_Low`:

  REACTIVE RECONFIGURATION (Reactive_Low): the mask is reconfigured once on
  the very first training batch, then only when the batch loss under the
  current mask exceeds an EMA mean by `reactive_sd` standard deviations
  (EMA factor `reactive_ema`; triggers suppressed for ~1/(1-ema) batches
  after each reconfig while the statistics re-form). The EMA state is
  boundary-free — it never resets at task switches and persists across
  train.py invocations via state.json.

  NO REPLAY (NoReplay): no replay buffers, no rehearsal of prior blocks.

  PLAIN KWTA (Default): annealSTE mask search, no lateral smoothing.

OBJECTIVE — context distillation, no raw-text training: per batch the teacher
(the SAME frozen backbone with the source excerpt in context and the mask
zeroed) provides answer-token distributions; the student (masked bank, no
context) minimizes KL to them over the answer span. Teacher logits are
mask-independent, so a reactive re-reconfig only recomputes the student pass.

Persisted state in checkpoints/: bank.pt (atom weights), refs.pt (per-task
reference masks), state.json (trained order + bank shape + reactive EMA).
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
from perplexity_configurer import PerplexityConfigurer
from registry import DEFAULT_ORDER, TASKS
from spatial import kwta


def overlap(a, b, k):
    return float((a * b).sum() / k)


def _paths():
    d = CFG.ckpt_dir
    return d / "bank.pt", d / "refs.pt", d / "state.json", d / "mask.pt"


def load_state(wrappers, ctrl):
    """Return (trained_keys, refs, reactive_state). Empty if no checkpoint.
    Restores the last training mask (boundary-free reactive resume) if saved."""
    bank_p, refs_p, state_p, mask_p = _paths()
    if not (bank_p.exists() and state_p.exists()):
        return [], {}, None
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
    return state["trained"], refs, state.get("reactive")


def save_state(wrappers, trained, refs, reactive_state, ctrl):
    bank_p, refs_p, state_p, mask_p = _paths()
    save_bank(wrappers, bank_p)
    torch.save({k: v.detach().cpu() for k, v in refs.items()}, refs_p)
    torch.save(ctrl.mask.detach().cpu(), mask_p)
    state_p.write_text(
        json.dumps(
            {
                "trained": trained,
                "H": CFG.H,
                "k": CFG.k,
                "rank": CFG.rank,
                "layer_frac": CFG.lora_layer_frac,
                "n_wrappers": len(wrappers),
                "reactive": reactive_state,
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
    p.add_argument(
        "--retrain",
        action="store_true",
        help="re-run a block even if the task is already trained",
    )
    p.add_argument("--epochs", type=int, default=CFG.epochs)
    p.add_argument("--steps", type=int, default=CFG.steps_per_domain)
    p.add_argument("--reconfig-steps", type=int, default=CFG.train_reconfig_steps)
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


@torch.no_grad()
def teacher_answer_logits(model, ctrl, t_batch):
    """Teacher pass: same backbone, mask zeroed (bank inert), source excerpt in
    context. Returns the answer-position logits, detached."""
    t_ids, t_attn, t_lbl = t_batch
    ctrl.set_mask(torch.zeros(ctrl.H, device=ctrl.mask.device))
    logits = model(input_ids=t_ids, attention_mask=t_attn).logits
    sel, _ = _gather_answer_logits(logits, t_lbl)
    return sel.detach()


class ReactiveState:
    """EMA of the training-loss mean/variance for loss-triggered reconfig.
    Serialized into state.json so the mechanism stays boundary-free across
    train.py invocations as well as across task blocks."""

    def __init__(self, d=None):
        d = d or {}
        self.mean = d.get("mean")  # None => reconfig on the first batch ever
        self.var = d.get("var", 0.0)
        self.since = d.get("since", 0)

    def sd(self):
        return max(self.var, 0.0) ** 0.5

    def triggered(self, loss):
        return (
            self.mean is not None
            and self.since >= CFG.reactive_warmup
            and loss > self.mean + CFG.reactive_sd * self.sd()
        )

    def update(self, loss):
        if self.mean is None:
            self.mean = loss
        else:
            a = CFG.reactive_ema
            diff = loss - self.mean
            self.mean = a * self.mean + (1 - a) * loss
            self.var = a * self.var + (1 - a) * diff * diff
        self.since += 1

    def to_dict(self):
        return {"mean": self.mean, "var": self.var, "since": self.since}


def train_block(
    model, ctrl, cfgr, opt, key, ex, rehearsal, reactive,
    steps, rsteps, gen, dev, tok, t0, epoch, ref_acc,
):
    """Train one task for a SINGLE epoch by context distillation with reactive
    reconfig. main() owns the epoch loop and interleaves the tasks, so one call
    is one pass of `steps` steps over `key` (`epoch` is passed only for logging).

    Each step draws `CFG.batch_size` (question, answer, context) examples,
    computes the teacher's answer-token distributions once (mask-independent),
    and trains the selected atoms on the student KL. The mask is only
    re-searched when the reactive heuristic fires (or on the first batch ever),
    reconfiguring on one random member's bare question — the same signal
    inference uses.

    Returns the task's reference mask (kwta over the dwell-weighted sum of every
    step's training mask) — used ONLY for GUI color-coding, harvested for free
    from the masks training already produced (no extra reconfiguration searches;
    the polished usage-weighted refs come from rebuild_refs_usage.py offline).
    `ref_acc` is the caller-owned (H,) accumulator for this task; it persists
    across epochs, so the returned mask reflects usage over ALL epochs so far."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    qa = ex["qa"]
    if not qa:
        raise SystemExit(f"no distillation triples for '{key}'; run qa_synth.py first")
    model.train()
    running, n_reconfigs = None, 0

    def reconfigure(batch):
        e = batch[_rand(gen, len(batch))]
        m, _, _ = cfgr(model, e["recfg"].to(dev).unsqueeze(0), steps=rsteps)
        reactive.since = 0
        return m

    for step in range(steps):
        batch, tally = [], {}
        for _ in range(CFG.batch_size):
            if rehearsal and float(
                torch.rand(1, generator=gen, device="cpu")
            ) < CFG.rehearsal_frac:
                e, tag = rehearsal[_rand(gen, len(rehearsal))], "reh"
            else:
                e, tag = qa[_rand(gen, len(qa))], "qa"
            batch.append(e)
            tally[tag] = tally.get(tag, 0) + 1

        s_batch = _collate([(e["ids"], e["lbl"]) for e in batch], pad_id, dev)
        t_batch = _collate([(e["t_ids"], e["t_lbl"]) for e in batch], pad_id, dev)
        # Save the current training mask BEFORE the teacher pass — the
        # teacher zeroes the shared controller mask to silence the bank.
        m = ctrl.mask.detach().clone()
        teacher_sel = teacher_answer_logits(model, ctrl, t_batch)

        # Reactive reconfiguration (base run_experiments.py logic): once on
        # the first batch ever, then loss-triggered. Also reconfigure if no
        # mask is active (resumed run whose mask.pt was missing).
        if reactive.mean is None or float(m.sum()) == 0:
            m = reconfigure(batch)
            n_reconfigs += 1

        loss = distill_loss(model, ctrl, m, s_batch, teacher_sel, CFG)
        if reactive.triggered(float(loss.detach())):
            m = reconfigure(batch)
            n_reconfigs += 1
            loss = distill_loss(model, ctrl, m, s_batch, teacher_sel, CFG)
        # EMA tracks the loss under the trained mask (the pre-reconfig
        # spike is never recorded).
        reactive.update(float(loss.detach()))

        # Harvest the mask this step trained under for the task's reference mask
        # (GUI color-coding only). m is k-hot each step and persists until a
        # reconfig, so summing every step dwell-weights by how long each mask
        # was active; kwta at block end picks the atoms the task used most. No
        # extra reconfiguration passes — this replaces the old build_ref sweep.
        ref_acc += m.detach().float().cpu()

        opt.zero_grad()
        loss.backward()
        opt.step()

        lv = float(loss.detach())
        running = lv if running is None else 0.9 * running + 0.1 * lv
        if step % 50 == 0:
            mix = " ".join(f"{t}{n}" for t, n in sorted(tally.items()))
            print(
                f"  {key:<10} e{epoch} step {step:>4} [{mix}] kl={lv:.3f} "
                f"ema={running:.3f} reconfigs={n_reconfigs} "
                f"(loss_ema={reactive.mean:.3f} sd={reactive.sd():.3f}) "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    print(
        f"  [{key}] e{epoch} done: reactive_reconfigs={n_reconfigs} "
        f"loss_ema={reactive.mean:.4f} sd={reactive.sd():.4f}",
        flush=True,
    )
    return kwta(ref_acc, CFG.k)


def main():
    args = parse_args()
    CFG.seed = args.seed  # feeds both torch.manual_seed below and the data gen
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok, ctrl, wrappers, lora_cfg = build(CFG)

    if args.fresh:
        reset_bank(wrappers, lora_cfg)
        trained, refs, reactive_d = [], {}, None
        print("  fresh bank")
    else:
        trained, refs, reactive_d = load_state(wrappers, ctrl)
        if trained:
            print(f"  resumed; already trained: {trained}")
        else:
            reset_bank(wrappers, lora_cfg)
            print("  no checkpoint found -> fresh bank")
    for p in lora_params(wrappers):
        p.requires_grad = True
    reactive = ReactiveState(reactive_d)

    requested = args.tasks or DEFAULT_ORDER
    unknown = [k for k in requested if k not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}; registered: {list(TASKS)}")
    to_run = [k for k in requested if k not in trained or args.retrain]
    print(f"  plan: train {to_run}" + (" (nothing new)" if not to_run else ""))
    print(
        f"  reconfig: REACTIVE sd={CFG.reactive_sd} ema={CFG.reactive_ema} "
        f"(warmup {CFG.reactive_warmup}) | replay: OFF | mask: plain KWTA "
        f"(lateral_steps={CFG.lateral_steps})"
    )

    cfgr = PerplexityConfigurer(CFG, ctrl)
    # foreach=False: the bank params are sharded across GPUs (device_map), and
    # the foreach/fused Adam path can't group optimizer tensors living on
    # different devices. Per-param stepping keeps each update on its own GPU.
    opt = torch.optim.Adam(lora_params(wrappers), lr=CFG.lr, foreach=False)
    gen = torch.Generator(device="cpu").manual_seed(CFG.seed)
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
        print(f"  examples[{key}]: {len(ex['qa'])} distillation triples", flush=True)
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
                model, ctrl, cfgr, opt, key, ex, rehearsal, reactive,
                args.steps, args.reconfig_steps, gen, dev, tok, t0, epoch,
                ref_acc[key],
            )
            if key not in trained:
                trained.append(key)
        # Persist once per epoch: bank + refs for every task + reactive EMA.
        save_state(wrappers, trained, refs, reactive.to_dict(), ctrl)
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
