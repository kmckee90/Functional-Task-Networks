"""Headless routing de-risk for FTN-Chat (the crux gate).

Builds a reference mask per domain (reconfigure on that domain's support pool),
then for every held-out chunk reconfigures and asks which domain reference it
overlaps most. Reports routing accuracy + the cross-domain mask-overlap matrix.

Run on an UNTRAINED bank first to confirm plumbing + see the random baseline:
    cd ftn_chat && uv run python routing_check.py --untrained
After training:
    cd ftn_chat && uv run python routing_check.py
"""
from __future__ import annotations

import argparse

import torch

from config import CFG, DOMAIN_LABELS
from data import build_domains
from model import build, load_bank
from perplexity_configurer import PerplexityConfigurer


def overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    return float((a * b).sum() / k)


def ref_mask(cfgr, model, chunks, k):
    """Domain signature: sum of per-chunk reconfigured masks, top-k."""
    acc = torch.zeros(CFG.H)
    for ch in chunks:
        m, _, _ = cfgr(model, ch.unsqueeze(0))
        acc += m.float()
    idx = acc.topk(k).indices
    out = torch.zeros(CFG.H)
    out[idx] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--untrained", action="store_true",
                    help="skip loading the trained bank; test the random baseline")
    ap.add_argument("--ckpt", default=str(CFG.ckpt_dir / "bank.pt"))
    ap.add_argument("--max-heldout", type=int, default=16)
    args = ap.parse_args()

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG.seed)

    model, tok, ctrl, wrappers, lora_cfg = build(CFG)
    if not args.untrained:
        load_bank(wrappers, args.ckpt)
        print(f"  loaded trained bank from {args.ckpt}")
    else:
        print("  UNTRAINED bank (random baseline)")
    model.eval()

    doms = build_domains(CFG, tok)
    cfgr = PerplexityConfigurer(CFG, ctrl)
    keys = list(doms.keys())

    print("\nBuilding domain reference masks...")
    refs = {k: ref_mask(cfgr, model, doms[k].support, CFG.k) for k in keys}

    # Cross-domain overlap matrix.
    print("\nCross-domain mask-overlap matrix (|mi ∩ mj| / k):")
    print(f"{'':<12}" + "".join(f"{k:>12}" for k in keys))
    for ki in keys:
        row = "".join(f"{overlap(refs[ki], refs[kj], CFG.k):>12.2f}" for kj in keys)
        print(f"{ki:<12}{row}")

    # Route held-out chunks.
    print("\nRouting held-out chunks to nearest domain reference:")
    confusion = {ki: {kj: 0 for kj in keys} for ki in keys}
    for ki in keys:
        for ch in doms[ki].heldout[: args.max_heldout]:
            m, _, _ = cfgr(model, ch.unsqueeze(0))
            scores = {kj: overlap(m, refs[kj], CFG.k) for kj in keys}
            pred = max(scores, key=scores.get)
            confusion[ki][pred] += 1

    print(f"{'true/pred':<12}" + "".join(f"{k:>12}" for k in keys) + f"{'acc':>8}")
    total_correct = total = 0
    for ki in keys:
        n = sum(confusion[ki].values())
        correct = confusion[ki][ki]
        total_correct += correct
        total += n
        row = "".join(f"{confusion[ki][kj]:>12}" for kj in keys)
        acc = correct / n if n else 0.0
        print(f"{ki:<12}{row}{acc:>8.2f}")
    print(f"\noverall routing accuracy: {total_correct}/{total} "
          f"= {total_correct/max(1,total):.2f}")


if __name__ == "__main__":
    main()
