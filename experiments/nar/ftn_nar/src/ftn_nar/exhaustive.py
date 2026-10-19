"""Tiny-bank exhaustive mask enumeration diagnostic (design section 13).

Tiny model: H=12/12, k=2/2 -> C(12,2)^2 = 4,356 joint mask pairs, enumerable exactly.
Tests directly whether overlapping approximate solutions create a pathway to an exact
solution, and where the gradient configurer's choice ranks.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import algorithms as A
from . import support as SUP
from .config import load_config
from .evaluator import EvalCache, build_support_batch, evaluate_program
from .configurer import configure_with_restarts, params_frozen
from .losses import support_loss
from .masks import mask_from_indices
from .model import Program
from .trainer import Trainer


def _all_masks(h: int, k: int, device) -> list[tuple[tuple[int, ...], torch.Tensor]]:
    return [(c, mask_from_indices(h, c, device)) for c in itertools.combinations(range(h), k)]


@torch.no_grad()
def _support_loss_of(model, batch, cfg, prog) -> float:
    out = model(batch, prog, edge_mode="hard", return_trace=False)
    return float(support_loss(out, batch, cfg.loss))


def exhaustive_run(run_dir: str, screen_count: int = 64, top_k_eval: int = 100,
                   near_exact_screen: float = 0.95, epistasis: bool = True) -> dict:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    tr = Trainer(cfg, resume=True, run_dir=str(run_dir))
    model = tr.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = tr.device
    held = cfg.held_out
    mcfg = tr.mcfg
    t0 = time.time()

    ext_masks = _all_masks(mcfg.extender_atoms, mcfg.active_extender_atoms, device)
    red_masks = _all_masks(mcfg.reducer_atoms, mcfg.active_reducer_atoms, device)

    sup_graphs = SUP.greedy_identifiable_support(
        cfg.seed, held, held, A.ALL_ALGORITHMS, support_id=0,
        size=cfg.data.support_size, pool_size=cfg.data.support_candidate_pool)
    sup_b = build_support_batch(sup_graphs, held, device)

    cache = EvalCache(cfg.seed, cfg)
    screen_batches = cache.batches(held, "id_val", screen_count)  # small fixed query set

    # 1-4. support loss + screen exact accuracy for every pair
    pairs = []
    for ei, (ec, em) in enumerate(ext_masks):
        for ri, (rc, rm) in enumerate(red_masks):
            prog = Program(gate_ext=em, gate_red=rm)
            sl = _support_loss_of(model, sup_b, cfg, prog)
            scr = evaluate_program(model, prog, screen_batches, cfg, device)
            pairs.append({"ext": ec, "red": rc, "support_loss": sl,
                          "screen_exact": scr["exact_graph_value_acc"],
                          "screen_node": scr["node_acc"]})
    order = np.argsort([p["support_loss"] for p in pairs])
    for rank, i in enumerate(order):
        pairs[i]["support_rank"] = int(rank)

    # 5. deep evaluation: top-100 by support loss + near-exact screens
    deep_idx = set(order[:top_k_eval].tolist())
    deep_idx |= {i for i, p in enumerate(pairs) if p["screen_exact"] >= near_exact_screen}
    deep_conditions = ["id_val", "ood_width_32", "ood_depth_8"]
    for i in sorted(deep_idx):
        p = pairs[i]
        prog = Program(gate_ext=mask_from_indices(mcfg.extender_atoms, p["ext"], device),
                       gate_red=mask_from_indices(mcfg.reducer_atoms, p["red"], device))
        p["deep"] = {c: evaluate_program(model, prog, cache.batches(held, c), cfg, device)
                     for c in deep_conditions}

    deep = [(i, pairs[i]) for i in sorted(deep_idx)]
    best_i, best_p = max(deep, key=lambda ip: (ip[1]["deep"]["id_val"]["exact_graph_value_acc"],
                                               -ip[1]["support_loss"]))
    exact_exists = best_p["deep"]["id_val"]["exact_graph_value_acc"] >= 1.0
    near = [p for _, p in deep if p["deep"]["id_val"]["exact_graph_value_acc"] >= 0.95]

    # 8. gradient configurer's selection and its rank
    best_cfg, _ = configure_with_restarts(model, sup_b, cfg,
                                          (cfg.seed, held, "exhaustive"),
                                          cfg.configurer.eval_restarts, mode="ftn")
    ce = tuple(np.flatnonzero(best_cfg.masks_np()[0] > 0.5).tolist())
    cr = tuple(np.flatnonzero(best_cfg.masks_np()[1] > 0.5).tolist())
    conf_i = next((i for i, p in enumerate(pairs) if p["ext"] == ce and p["red"] == cr), None)

    out = {
        "run": run_dir.name, "held_out": A.ALGORITHM_NAMES[held], "seed": cfg.seed,
        "n_pairs": len(pairs),
        "best_support_loss": pairs[order[0]]["support_loss"],
        "best_pair": {"ext": list(best_p["ext"]), "red": list(best_p["red"]),
                      "support_rank": best_p["support_rank"],
                      "deep": best_p["deep"], "support_loss": best_p["support_loss"]},
        "exact_mask_exists_id": bool(exact_exists),
        "best_id_exact": best_p["deep"]["id_val"]["exact_graph_value_acc"],
        "best_ood_width_32_exact": max(p["deep"]["ood_width_32"]["exact_graph_value_acc"]
                                       for _, p in deep),
        "best_ood_depth_8_exact": max(p["deep"]["ood_depth_8"]["exact_graph_value_acc"]
                                      for _, p in deep),
        "n_near_exact_95": len(near),
        "configurer_pair": {"ext": list(ce), "red": list(cr),
                            "support_rank": pairs[conf_i]["support_rank"] if conf_i is not None else None,
                            "selection_loss": best_cfg.selection_loss,
                            "deep": None},
        # correlation between support loss and screen exact accuracy over all pairs
        "spearman_support_vs_screen": _spearman(
            [p["support_loss"] for p in pairs], [-p["screen_exact"] for p in pairs]),
    }
    if conf_i is not None and "deep" not in pairs[conf_i]:
        prog = Program(gate_ext=mask_from_indices(mcfg.extender_atoms, ce, device),
                       gate_red=mask_from_indices(mcfg.reducer_atoms, cr, device))
        pairs[conf_i]["deep"] = {c: evaluate_program(model, prog, cache.batches(held, c), cfg, device)
                                 for c in deep_conditions}
    if conf_i is not None:
        out["configurer_pair"]["deep"] = pairs[conf_i]["deep"]

    if epistasis:
        out["epistasis"] = _epistasis(model, cfg, mcfg, sup_b, cache, held, best_p, device)

    out["wall_s"] = time.time() - t0
    ex_dir = run_dir / "eval"
    ex_dir.mkdir(exist_ok=True)
    # full ranking table (compact): support loss + screen exact per pair
    table = [{"ext": list(p["ext"]), "red": list(p["red"]),
              "support_loss": round(p["support_loss"], 6),
              "support_rank": p["support_rank"],
              "screen_exact": round(p["screen_exact"], 4)} for p in pairs]
    (ex_dir / "exhaustive_table.json").write_text(json.dumps(table))
    (ex_dir / "exhaustive.json").write_text(json.dumps(out, indent=1))
    return out


def _spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def _epistasis(model, cfg, mcfg, sup_b, cache: EvalCache, held: int, best_p: dict,
               device) -> dict:
    """Single-swap landscape + partial-subset marginals around the best mask
    (design 13.6). Partial subsets are evaluated with mean-over-active scaling."""
    e_act, r_act = list(best_p["ext"]), list(best_p["red"])
    base = best_p["support_loss"]
    ood_batches = cache.batches(held, "ood_width_32")

    def ev(ec, rc):
        prog = Program(gate_ext=mask_from_indices(mcfg.extender_atoms, ec, device),
                       gate_red=mask_from_indices(mcfg.reducer_atoms, rc, device))
        sl = _support_loss_of(model, sup_b, cfg, prog)
        ood = evaluate_program(model, prog, ood_batches, cfg, device)["exact_graph_value_acc"]
        return sl, ood

    swaps = []
    for bank, act in (("ext", e_act), ("red", r_act)):
        h = mcfg.extender_atoms if bank == "ext" else mcfg.reducer_atoms
        for a in act:
            for b in range(h):
                if b in act:
                    continue
                cand = sorted(set(act) - {a} | {b})
                ec = cand if bank == "ext" else e_act
                rc = cand if bank == "red" else r_act
                sl, ood = ev(tuple(ec), tuple(rc))
                swaps.append({"bank": bank, "out": a, "in": b,
                              "d_support_loss": sl - base, "ood_exact": ood})

    # one- and two-atom partial subsets of the best joint mask
    partials = []
    for ec in _subsets(e_act):
        for rc in _subsets(r_act):
            if len(ec) == len(e_act) and len(rc) == len(r_act):
                continue
            sl, ood = ev(tuple(ec), tuple(rc))
            partials.append({"ext": list(ec), "red": list(rc),
                             "support_loss": sl, "ood_exact": ood})
    improving = sum(1 for s in swaps if s["d_support_loss"] < 0)
    return {"base_support_loss": base, "swaps": swaps,
            "n_improving_swaps": improving,
            "partials": partials}


def _subsets(act: list[int]) -> list[tuple]:
    out = []
    for r in range(1, len(act) + 1):
        out.extend(itertools.combinations(act, r))
    return out
