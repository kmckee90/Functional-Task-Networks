"""Fit and validate archived masks after the shared LoRA bank changes.

Only the mask logits move. The backbone and LoRA atoms are frozen during this
maintenance pass, and the live posterior/mask/detector are restored afterward.
"""
from __future__ import annotations

import torch

from data import format_distill, load_qa
from spatial import kwta, lateral_kwta


def _answer_ce(model, ctrl, example, mask):
    ctrl.set_mask(mask)
    ids = example["ids"].to(ctrl.mask.device).unsqueeze(0)
    labels = example["lbl"].to(ctrl.mask.device).unsqueeze(0)
    return model(input_ids=ids, labels=labels).loss.float()


@torch.no_grad()
def _validate(model, ctrl, examples, mask):
    return sum(float(_answer_ce(model, ctrl, ex, mask)) for ex in examples) / len(examples)


def _examples(cfg, tok, task, repair_round):
    triples = load_qa(cfg, task)
    if len(triples) < cfg.input_route_fit_rows + cfg.input_route_validation_rows:
        return None, None, None, None
    # Deterministic, disjoint fit/validation samples; existing held-out data
    # never enters route repair. Use a spread through the Q&A cache, not its
    # potentially correlated prefix.
    g = torch.Generator(device="cpu").manual_seed(
        cfg.seed + sum(map(ord, task)) + 104729 * repair_round)
    order = torch.randperm(len(triples), generator=g, device="cpu").tolist()
    needed = cfg.input_route_fit_rows + cfg.input_route_validation_rows
    selected = [triples[i] for i in order[:needed]]
    examples = [format_distill(t["question"], t["answer"], None, tok, cfg)
                for t in selected]
    names = [triples[i]["question"] for i in order[:cfg.input_route_archive_questions]]
    return (examples[:cfg.input_route_fit_rows], examples[cfg.input_route_fit_rows:],
            names, [t["question"] for t in triples])


def repair_routes(model, ctrl, cfgr, archive, tasks, tok, cfg, current_task=None):
    """Cold-search each route; accept a new mask only if held-out CE improves."""
    if not tasks:
        return {}
    live_mask, live_logits = ctrl.mask.detach().clone(), cfgr.mask_logits.detach().clone()
    old_training = model.training
    ckpt_on = getattr(model, "is_gradient_checkpointing", False)
    params = [(p, p.requires_grad) for p in model.parameters()]
    stats = {}
    archive.repair_round += 1
    try:
        for p, _ in params:
            p.requires_grad_(False)
        if ckpt_on:
            model.gradient_checkpointing_disable()
        model.eval()
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for task in tasks:
                fit, validation, questions, lexical_questions = _examples(
                    cfg, tok, task, archive.repair_round)
                if fit is None:
                    stats[task] = {"skipped": "insufficient Q&A"}
                    continue
                old = archive.routes.get(task)
                baseline = old["logits"].to(ctrl.mask.device) if old else (
                    live_logits if task == current_task else torch.zeros_like(live_logits))
                # Archived logits encode an already-smoothed, validated mask;
                # do not smooth it a second time during baseline evaluation.
                baseline_mask = (kwta(baseline, cfg.k) if old else
                                 lateral_kwta(baseline, cfg.k, cfg))
                before = _validate(model, ctrl, validation, baseline_mask)
                best_loss = before
                best_logits = torch.where(baseline_mask > 0.5,
                    torch.full_like(baseline, cfg.input_route_logit_magnitude),
                    torch.full_like(baseline, -cfg.input_route_logit_magnitude))
                for restart in range(cfg.input_route_restarts):
                    # A cold search can escape stale masks after weight drift.
                    # The live posterior is also tried as the initial candidate
                    # above for the just-trained task.
                    logits = torch.zeros_like(baseline).requires_grad_(True)
                    if restart:
                        logits.data.normal_(std=0.02)
                    optimizer = torch.optim.Adam([logits], lr=cfg.input_route_lr)
                    for step in range(cfg.input_route_steps):
                        ex = fit[step % len(fit)]
                        hard = lateral_kwta(logits, cfg.k, cfg)
                        soft = torch.sigmoid(logits / cfg.input_route_temperature)
                        mask = hard + soft - soft.detach()
                        loss = _answer_ce(model, ctrl, ex, mask)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    candidate = lateral_kwta(logits.detach(), cfg.k, cfg)
                    candidate_loss = _validate(model, ctrl, validation, candidate)
                    if candidate_loss + cfg.input_route_min_improvement < best_loss:
                        best_loss = candidate_loss
                        # Direct routing uses hard masks. Saturated logits keep
                        # exact top-k selection stable at inference.
                        best_logits = torch.where(candidate > 0.5,
                            torch.full_like(candidate, cfg.input_route_logit_magnitude),
                            torch.full_like(candidate, -cfg.input_route_logit_magnitude))
                if old is None or best_loss < before:
                    archive.upsert(task, questions if old is None else old["questions"],
                                   best_logits, lexical_questions)
                elif "terms" not in old:
                    archive.upsert(task, old["questions"], old["logits"], lexical_questions)
                stats[task] = {"before_ce": before, "after_ce": best_loss,
                               "accepted": old is None or best_loss < before}
    finally:
        ctrl.set_mask(live_mask)
        cfgr.mask_logits.copy_(live_logits)
        for p, requires_grad in params:
            p.requires_grad_(requires_grad)
        if ckpt_on:
            model.gradient_checkpointing_enable()
        model.train(old_training)
    return stats
