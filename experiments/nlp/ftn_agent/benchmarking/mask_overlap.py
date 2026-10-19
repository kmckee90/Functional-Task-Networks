"""Did the bank's atoms specialize by conversation? Direct check on refs.pt.

For a run's per-block reference masks (key '<conv>/s<idx>', binary over H,
k active), report mean pairwise overlap |A∩B|/k within vs between
conversations (random expectation = k/H), plus atom-usage concentration.

    uv run python mask_overlap.py --run p2-ftn
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from eval_ftn import run_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p2-ftn")
    args = ap.parse_args()

    refs = torch.load(run_root(args.run) / "refs.pt", map_location="cpu")
    keys = sorted(refs)
    masks = torch.stack([refs[k].float() for k in keys])          # (B, H)
    convs = [k.split("/")[0] for k in keys]
    B, H = masks.shape
    ks = masks.sum(1)
    k = int(ks.mode().values)
    print(f"{args.run}: {B} blocks, H={H}, k={k} "
          f"(active counts min={int(ks.min())} max={int(ks.max())}), "
          f"random-overlap expectation k/H = {k / H:.4f}\n")

    ov = (masks @ masks.T) / k                                    # (B, B)
    same = torch.tensor([[a == b for b in convs] for a in convs])
    eye = torch.eye(B, dtype=torch.bool)
    within = ov[same & ~eye]
    between = ov[~same]
    print(f"pairwise overlap |A∩B|/k:")
    print(f"  within-conv  mean={within.mean():.4f}  std={within.std():.4f}  n={within.numel()}")
    print(f"  between-conv mean={between.mean():.4f}  std={between.std():.4f}  n={between.numel()}")

    per_conv = {}
    for c in sorted(set(convs)):
        idx = [i for i, cc in enumerate(convs) if cc == c]
        if len(idx) > 1:
            sub = ov[idx][:, idx]
            m = sub[~torch.eye(len(idx), dtype=torch.bool)].mean()
            per_conv[c] = float(m)
    print("  per-conv within means: "
          + "  ".join(f"{c}={v:.3f}" for c, v in per_conv.items()))

    usage = masks.mean(0)                                         # (H,)
    print(f"\natom usage (fraction of blocks selecting each atom; uniform = {k / H:.3f}):")
    print(f"  mean={usage.mean():.4f}  std={usage.std():.4f}  "
          f"min={usage.min():.4f}  max={usage.max():.4f}")
    for thresh in (0.9, 0.99):
        print(f"  atoms in >{int(thresh * 100)}% of blocks: {(usage > thresh).sum().item()}/{H}")
    print(f"  atoms never selected: {(usage == 0).sum().item()}/{H}")
    p = usage / usage.sum()
    ent = -(p[p > 0] * p[p > 0].log()).sum()
    print(f"  usage entropy: {ent:.3f} nats (uniform over H = {torch.log(torch.tensor(float(H))):.3f})")


if __name__ == "__main__":
    main()
