"""Evaluation protocols.

Trained cells:   stored-mask accuracy, reconfig-protocol accuracy (both on
                 held-out input sequences).
Held-out cells:  zero-shot additive synthesis (headline), majority-union,
                 reconfig protocol (existence of a mask in the frozen bank),
                 mu-only / nearest-neighbor / random floors.
Consistency:     temporal route (iterate trained cell (a, r) x2) vs spatial
                 route (one-shot synthesized mask at (2a, 2r)) vs generator.
"""

import torch

from .configurer import AnnealSTEConfigurer
from .data import TaskGrid
from .model import FTN2, accuracy


def eval_trained_cells(model, grid: TaskGrid, cells, stored_masks, cfg,
                       final_masks=None):
    configurer = AnnealSTEConfigurer(cfg.k1, cfg.k2, cfg.configurer_lr, cfg.L, cfg.p)
    rows = []
    for cell in cells:
        x, y = grid.eval_tensors(cell, "test")
        m = stored_masks[cell].to(cfg.device)
        pos_s, ex_s = accuracy(model, m, x, y, cfg.L, cfg.p)

        xs, ys = grid.batch(cell, cfg.support_size, "train")
        m_rc = configurer(model, (xs, ys), cfg.s_eval)
        pos_r, ex_r = accuracy(model, m_rc, x, y, cfg.L, cfg.p)
        row = dict(cell=cell, stored=(pos_s, ex_s), reconfig=(pos_r, ex_r))
        if final_masks is not None:
            row["final"] = accuracy(model, final_masks[cell].to(cfg.device),
                                    x, y, cfg.L, cfg.p)
        rows.append(row)
    return rows


def eval_heldout_cells(model, grid: TaskGrid, cells, synthesizer, cfg):
    configurer = AnnealSTEConfigurer(cfg.k1, cfg.k2, cfg.configurer_lr, cfg.L, cfg.p)
    g = torch.Generator().manual_seed(cfg.seed + 4)
    rows = []
    for cell in cells:
        a, r = cell
        if not synthesizer.covers(a, r):
            continue  # factor level absent from training: unidentifiable
        x, y = grid.eval_tensors(cell, "test")
        entry = dict(cell=cell)

        for name, m in (
            ("additive", synthesizer.additive(a, r)),
            ("assigned", synthesizer.assigned(a, r)),
            ("assigned_swap", synthesizer.assigned(a, r, swap=True)),
            ("majority_union", synthesizer.majority_union(a, r)),
            ("mu_only", synthesizer.mu_only()),
            ("nearest_neighbor", synthesizer.nearest_neighbor(a, r)),
            ("random", synthesizer.random_mask(g)),
        ):
            entry[name] = accuracy(model, m.to(cfg.device), x, y, cfg.L, cfg.p)

        # Reconfig protocol: labeled support from the cell, theta frozen.
        xs, ys = grid.batch(cell, cfg.support_size, "train")
        m_rc = configurer(model, (xs, ys), cfg.s_eval)
        entry["reconfig"] = accuracy(model, m_rc, x, y, cfg.L, cfg.p)
        rows.append(entry)
    return rows


@torch.no_grad()
def consistency_check(model, grid: TaskGrid, train_cells, held_cells,
                      stored_masks, synthesizer, cfg, k_max: int = 4):
    """Two routes to the same composite.

    For trained cells (a, r) whose k-fold composite (ka, kr) is a held-out
    cell: temporal route = apply the trained mask k times autoregressively
    (argmax re-encode between steps); spatial route = one-shot synthesized
    mask at (ka, kr). Both scored against the generator; agreement rate
    between routes also reported.
    """
    model.eval()
    rows = []
    x, _ = grid.eval_tensors((0, 0), "test")
    seq0 = grid.test_seqs
    for cell in train_cells:
        a, r = cell
        for k in range(2, k_max + 1):
            comp = ((k * a) % cfg.p, (k * r) % cfg.L)
            if comp not in held_cells or not synthesizer.covers(*comp):
                continue
            m = stored_masks[cell].to(cfg.device)

            seq = seq0
            step_ok = []
            for j in range(1, k + 1):
                logits = model(grid.encode(seq), m)
                seq = grid.decode(logits)
                truth = grid.apply(seq0, (j * a) % cfg.p, (j * r) % cfg.L)
                step_ok.append((seq == truth).float().mean().item())
            temporal_exact = (seq == grid.apply(seq0, *comp)).all(-1).float().mean().item()

            ms = synthesizer.additive(*comp).to(cfg.device)
            logits_s = model(grid.encode(seq0), ms)
            spat = grid.decode(logits_s)
            spatial_pos = (spat == grid.apply(seq0, *comp)).float().mean().item()
            spatial_exact = (spat == grid.apply(seq0, *comp)).all(-1).float().mean().item()
            agree = (spat == seq).float().mean().item()

            rows.append(dict(cell=cell, k=k, composite=comp,
                             temporal_step_pos=step_ok,
                             temporal_exact=temporal_exact,
                             spatial_pos=spatial_pos,
                             spatial_exact=spatial_exact,
                             route_agreement=agree))
    model.train()
    return rows
