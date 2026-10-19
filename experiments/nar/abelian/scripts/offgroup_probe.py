"""Off-group reachability probe: does the trained bank encode THE GROUP, or
the generic skill of one-hot routing?

Every probe is a bijective map on token sequences with the same I/O format and
description-length ballpark as the trained tasks; what varies is distance from
the trained abelian group Z_p x Z_L:

  tier 0: held-out cells of the trained group (reference)
  tier 1: product-structure S_L x S_p maps OUTSIDE the abelian subgroup
          (position reversal, value negation, multiply mod p, transpositions,
          random position+value permutations)
  tier 2: non-product bit-permutations (independent random value permutation
          at each position) — still a permutation of the 40 input bits, but
          with no shared value map across positions

All probes get the identical reconfig protocol on the FROZEN trained bank
(and, for reference, the random bank): same support, same steps, best of R
restarts. Sharp tier-0 >> tier-1 favors "algorithm/group learned"; flat
tiers favor "generic routing basis" (compositional function approximation).
"""

import argparse
import json
from pathlib import Path

import torch

from ftn_nar.config import full
from ftn_nar.configurer import AnnealSTEConfigurer
from ftn_nar.data import TaskGrid
from ftn_nar.model import FTN2, sequence_ce


class ProbeTask:
    """y_i = val_maps[i][ x[pos_map[i]] ] — covers every tier."""

    def __init__(self, name, tier, pos_map, val_maps):
        self.name, self.tier = name, tier
        self.pos_map = torch.tensor(pos_map)
        self.val_maps = torch.stack([torch.tensor(v) for v in val_maps])

    def apply(self, seqs):
        pos = seqs[:, self.pos_map.to(seqs.device)]
        vm = self.val_maps.to(seqs.device)
        return torch.gather(vm.unsqueeze(0).expand(seqs.shape[0], -1, -1),
                            2, pos.unsqueeze(-1)).squeeze(-1)


def build_probes(p, L, held_cells, seed=7):
    g = torch.Generator().manual_seed(seed)
    ident_v = list(range(p))
    ident_p = list(range(L))

    def shift(a):
        return [(v + a) % p for v in range(p)]

    probes = []
    # tier 0: held-out group cells (reference), as ProbeTasks for identical code path
    for (a, r) in held_cells:
        pos_map = [(i - r) % L for i in range(L)]
        probes.append(ProbeTask(f"group({a},{r})", 0, pos_map, [shift(a)] * L))

    # tier 1: product maps outside the abelian subgroup
    probes.append(ProbeTask("pos-reversal", 1, list(reversed(ident_p)),
                            [ident_v] * L))
    probes.append(ProbeTask("val-negation", 1, ident_p,
                            [[(-v) % p for v in range(p)]] * L))
    probes.append(ProbeTask("val-mult3", 1, ident_p,
                            [[(3 * v) % p for v in range(p)]] * L))
    probes.append(ProbeTask("pos-swap01", 1, [1, 0, 2, 3], [ident_v] * L))
    probes.append(ProbeTask("val-swap(0,1)", 1, ident_p,
                            [[1, 0] + list(range(2, p))] * L))
    for j in range(3):
        sp = torch.randperm(L, generator=g).tolist()
        vp = torch.randperm(p, generator=g).tolist()
        probes.append(ProbeTask(f"rand-SLxSp-{j}", 1, sp, [vp] * L))
    # composite: group op composed with one off-group op
    probes.append(ProbeTask("reversal+shift2", 1, list(reversed(ident_p)),
                            [shift(2)] * L))

    # tier 2: non-product permutations of the 40 bits
    for j in range(4):
        vms = [torch.randperm(p, generator=g).tolist() for _ in range(L)]
        probes.append(ProbeTask(f"nonprod-{j}", 2, ident_p, vms))
    return probes


@torch.no_grad()
def _acc(model, m, x, y, L, p):
    model.eval()
    pred = model(x, m).reshape(-1, L, p).argmax(-1)
    model.train()
    return (pred == y).float().mean().item(), (pred == y).all(-1).float().mean().item()


def probe_model(model, grid, probes, cfg, restarts=3):
    configurer = AnnealSTEConfigurer(cfg.k1, cfg.k2, cfg.configurer_lr, cfg.L, cfg.p)
    x_test = grid.encode(grid.test_seqs)
    rows = []
    for t in probes:
        y_test = t.apply(grid.test_seqs)
        best = (0.0, 0.0)
        for _ in range(restarts):
            idx = torch.randint(0, grid.train_seqs.shape[0], (cfg.support_size,),
                                device=grid.train_seqs.device)
            xs = grid.encode(grid.train_seqs[idx])
            ys = t.apply(grid.train_seqs[idx])
            m = configurer(model, (xs, ys), cfg.s_eval)
            a = _acc(model, m, x_test, y_test, cfg.L, cfg.p)
            if a[0] > best[0]:
                best = a
        rows.append(dict(name=t.name, tier=t.tier, pos=best[0], exact=best[1]))
        print(f"  tier {t.tier} {t.name:>16}: pos={best[0]:.3f} exact={best[1]:.3f}",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2")
    args = ap.parse_args()
    cfg = full()
    torch.manual_seed(321)
    out = Path("results") / args.run
    ckpt = torch.load(out / "checkpoint.pt", weights_only=False)
    grid = TaskGrid(cfg.p, cfg.L, cfg.input_train_frac, ckpt["cfg"]["seed"], cfg.device)
    held_cells = [tuple(c) for c in ckpt["held_cells"]]

    model = FTN2(cfg.L * cfg.p, cfg.L * cfg.p, cfg.H1, cfg.H2, cfg.d_mid,
                 cfg.neuron_depth, cfg.neuron_width, cfg.dropout).to(cfg.device)
    model.load_state_dict(ckpt["model"])

    probes = build_probes(cfg.p, cfg.L, held_cells)
    print("=== trained bank ===", flush=True)
    trained_rows = probe_model(model, grid, probes, cfg)

    print("=== random bank (reference) ===", flush=True)
    rand_model = FTN2(cfg.L * cfg.p, cfg.L * cfg.p, cfg.H1, cfg.H2, cfg.d_mid,
                      cfg.neuron_depth, cfg.neuron_width, cfg.dropout).to(cfg.device)
    off_probes = [t for t in probes if t.tier > 0]
    rand_rows = probe_model(rand_model, grid, off_probes, cfg, restarts=1)

    with open(out / "offgroup_probe.json", "w") as f:
        json.dump(dict(trained=trained_rows, random=rand_rows), f, indent=2)

    import statistics as st
    for tier in (0, 1, 2):
        vals = [r["pos"] for r in trained_rows if r["tier"] == tier]
        if vals:
            print(f"TIER {tier}: mean pos={st.mean(vals):.3f} "
                  f"min={min(vals):.3f} max={max(vals):.3f} (n={len(vals)})",
                  flush=True)
    print("probe results written")


if __name__ == "__main__":
    main()
