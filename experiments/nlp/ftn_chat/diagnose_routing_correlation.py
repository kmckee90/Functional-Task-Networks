"""Measure whether prompt NLL ranks masks by answer quality.

Read-only with respect to model/checkpoints. Runs on the actl pod and saves a
small JSON of per-question measurements under ~/ftn_chat_state/diagnostics/.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_device("cuda")

from config import CFG
from data import format_distill, load_qa
from model import build, load_bank
from spatial import kwta

TASKS = ("base_ftn", "explorer", "rl_mt10")
QUESTIONS_PER_TASK = 8
MASKS_PER_QUESTION = 16
ANSWER_TOKENS = 64


def sampled_mask(logits, k, temperature):
    u = torch.rand_like(logits).clamp_(1e-7, 1 - 1e-7)
    return kwta(logits / temperature - torch.log(-torch.log(u)), k)


def answer_logits(model, ids, first_answer_position, n_answer_tokens):
    logits = model(input_ids=ids).logits[0]
    start = first_answer_position
    return logits[start - 1:start + n_answer_tokens - 1].float()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_mean_ci(values, seed=1234):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = a[rng.integers(0, len(a), size=(4000, len(a)))].mean(axis=1)
    return [float(a.mean()), *np.quantile(boot, [0.025, 0.975]).tolist()]


def summarize(rows, key):
    subset = [r for r in rows if key == "all" or r["task"] == key]
    out = {"n_questions": len(subset)}
    for metric in ("ce8", "ce64", "kl8", "kl64"):
        rho = [r[f"rho_{metric}"] for r in subset]
        advantage = [r[f"nll_choice_{metric}"] - r[f"mean_{metric}"] for r in subset]
        oracle = [r[f"best_{metric}"] - r[f"mean_{metric}"] for r in subset]
        out[metric] = {
            "spearman_mean_ci95": bootstrap_mean_ci(rho),
            "nll_selection_minus_random_mean_ci95": bootstrap_mean_ci(advantage),
            "oracle_minus_random_mean_ci95": bootstrap_mean_ci(oracle),
            "nll_selection_rank_mean": float(np.mean([r[f"nll_choice_rank_{metric}"] for r in subset])),
            "nll_selection_best_count": sum(r[f"nll_choice_rank_{metric}"] == 1 for r in subset),
        }
    return out


def main():
    torch.manual_seed(20260924)
    rng = random.Random(20260924)
    model, tok, ctrl, wrappers, _ = build(CFG)
    load_bank(wrappers, CFG.ckpt_dir / "bank.pt")
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.eval()
    posterior = torch.load(CFG.ckpt_dir / "posterior.pt", map_location="cpu", weights_only=True)
    prior_logits = posterior["mask_logits"].to(ctrl.mask.device).float()
    saved_mask = posterior["mask"].to(ctrl.mask.device).float()
    print("checkpoint", CFG.ckpt_dir, "mask T", CFG.bayes_temperature,
          "question/task", QUESTIONS_PER_TASK, "masks/question", MASKS_PER_QUESTION, flush=True)

    rows = []
    with torch.inference_mode():
        for task in TASKS:
            triples = load_qa(CFG, task)
            valid = [i for i, t in enumerate(triples) if len(t["answer"]) >= 40
                     and not any(s in t["answer"].lower() for s in
                                 ("not possible", "cannot determine", "insufficient information"))]
            indices = rng.sample(valid, QUESTIONS_PER_TASK)
            for index in indices:
                t = triples[index]
                ex = format_distill(t["question"], t["answer"], t["context"], tok, CFG)
                s_start = int((ex["lbl"] != -100).nonzero()[0].item())
                t_start = int((ex["t_lbl"] != -100).nonzero()[0].item())
                n = min(ANSWER_TOKENS, ex["lbl"].numel() - s_start)
                s_ids = ex["ids"][:s_start + n].unsqueeze(0).to("cuda")
                t_ids = ex["t_ids"][:t_start + n].unsqueeze(0).to("cuda")
                target = ex["lbl"][s_start:s_start + n].to("cuda")
                q_ids = ex["recfg"].unsqueeze(0).to("cuda")
                ctrl.set_mask(torch.zeros_like(saved_mask))
                teacher = answer_logits(model, t_ids, t_start, n)
                teacher_logp = F.log_softmax(teacher, dim=-1)
                teacher_prob = teacher_logp.exp()
                del teacher

                masks = [saved_mask] + [sampled_mask(prior_logits, ctrl.k, CFG.bayes_temperature)
                                        for _ in range(MASKS_PER_QUESTION - 1)]
                observations = []
                for mask in masks:
                    ctrl.set_mask(mask)
                    q_nll = float(model(input_ids=q_ids, labels=q_ids).loss)
                    student = answer_logits(model, s_ids, s_start, n)
                    logp = F.log_softmax(student, dim=-1)
                    ce = -logp.gather(1, target[:, None]).squeeze(1)
                    kl = (teacher_prob * (teacher_logp - logp)).sum(-1)
                    observations.append({"q_nll": q_nll, "ce8": float(ce[:8].mean()),
                                         "ce64": float(ce.mean()), "kl8": float(kl[:8].mean()),
                                         "kl64": float(kl.mean())})
                    del student, logp, ce, kl

                q = np.array([o["q_nll"] for o in observations])
                choice = int(q.argmin())
                row = {"task": task, "index": index, "question": t["question"],
                       "answer_tokens": n, "nll_choice": choice,
                       "q_nll_range": [float(q.min()), float(q.max())],
                       "observations": observations}
                for metric in ("ce8", "ce64", "kl8", "kl64"):
                    a = np.array([o[metric] for o in observations])
                    row[f"rho_{metric}"] = spearman(q, a)
                    row[f"mean_{metric}"] = float(a.mean())
                    row[f"best_{metric}"] = float(a.min())
                    row[f"nll_choice_{metric}"] = float(a[choice])
                    row[f"nll_choice_rank_{metric}"] = int(np.argsort(np.argsort(a))[choice]) + 1
                rows.append(row)
                print(task, index, "rho CE64", round(row["rho_ce64"], 3),
                      "rho KL64", round(row["rho_kl64"], 3),
                      "CE64 rank of NLL choice", row["nll_choice_rank_ce64"], flush=True)
            print("completed", task, flush=True)

    summary = {key: summarize(rows, key) for key in ("all", *TASKS)}
    print(json.dumps(summary, indent=2), flush=True)
    outdir = CFG.ckpt_dir.parent / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"routing_mismatch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps({"checkpoint": str(CFG.ckpt_dir), "summary": summary,
                                  "questions": rows}, indent=2))
    print("saved", output, flush=True)


if __name__ == "__main__":
    main()
