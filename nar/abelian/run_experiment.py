"""Orchestrate the FTN abelian-grid experiment.

Stages: train -> eval -> baseline -> consistency -> figures. Artifacts land
in results/<run_name>/ (model state, masks, results.json, figures).

Usage (on the pod, from /home/dev/workspace):
    uv run python run_experiment.py --config small --run smoke
    uv run python run_experiment.py --config full --run main --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import torch

from ftn_nar import config as config_mod
from ftn_nar.baselines import train_joint_baseline
from ftn_nar.data import TaskGrid, split_grid
from ftn_nar.evals import consistency_check, eval_heldout_cells, eval_trained_cells
from ftn_nar.model import FTN2
from ftn_nar.synthesis import MaskSynthesizer, anova_decomposition, overlap_matrix
from ftn_nar.train import train_sequential


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["small", "full"], default="small")
    ap.add_argument("--run", default="smoke")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    cfg = getattr(config_mod, args.config)()
    if args.seed is not None:
        cfg.seed = args.seed
    torch.manual_seed(cfg.seed)

    out = Path("results") / args.run
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    t0 = time.time()
    log(f"config={args.config} seed={cfg.seed} device={cfg.device}")
    log(json.dumps(cfg.to_dict()))

    grid = TaskGrid(cfg.p, cfg.L, cfg.input_train_frac, cfg.seed, cfg.device)
    train_cells, held_cells = split_grid(
        cfg.p, cfg.L, cfg.n_train_cells, cfg.min_per_a, cfg.min_per_r, cfg.seed)
    log(f"train cells ({len(train_cells)}): {train_cells}")
    log(f"held-out cells ({len(held_cells)}): {held_cells}")

    model = FTN2(cfg.L * cfg.p, cfg.L * cfg.p, cfg.H1, cfg.H2, cfg.d_mid,
                 cfg.neuron_depth, cfg.neuron_width, cfg.dropout).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"FTN^2 params: {n_params:,}")

    # ---- Stage 1: block-sequential FTN training ----
    stored_masks, mask_emas, history, order = train_sequential(
        model, grid, train_cells, cfg, log_fn=log)
    torch.save(dict(model=model.state_dict(),
                    stored_masks=stored_masks, mask_emas=mask_emas,
                    train_cells=train_cells, held_cells=held_cells,
                    order=order, cfg=cfg.to_dict()),
               out / "checkpoint.pt")
    log(f"training done in {time.time() - t0:.0f}s")

    # ---- Stage 2: post-stream consensus masks (reconfig protocol against the
    # FINAL weights; block-end masks go stale under later-block drift) ----
    from ftn_nar.configurer import AnnealSTEConfigurer, kwta
    configurer = AnnealSTEConfigurer(cfg.k1, cfg.k2, cfg.configurer_lr, cfg.L, cfg.p)
    final_scores, final_masks = {}, {}
    for cell in train_cells:
        acc = torch.zeros(cfg.H1 + cfg.H2)
        for _ in range(cfg.final_restarts):
            xs, ys = grid.batch(cell, cfg.support_size, "train")
            acc += configurer(model, (xs, ys), cfg.s_eval).cpu()
        score = acc / cfg.final_restarts
        final_scores[cell] = score
        final_masks[cell] = torch.cat([
            kwta(score[:cfg.H1], cfg.k1), kwta(score[cfg.H1:], cfg.k2)])
    torch.save(dict(final_scores=final_scores, final_masks=final_masks),
               out / "final_masks.pt")
    log("post-stream consensus masks built")

    synthesizer = MaskSynthesizer(final_scores, train_cells, cfg.H1, cfg.k1, cfg.k2,
                                  cfg.p, cfg.L)
    trained_rows = eval_trained_cells(model, grid, train_cells, stored_masks, cfg,
                                      final_masks=final_masks)
    held_rows = eval_heldout_cells(model, grid, held_cells, synthesizer, cfg)
    log("trained cells (stored): mean pos acc = " + str(
        sum(r["stored"][0] for r in trained_rows) / len(trained_rows)))
    log("held-out cells (additive synthesis): mean pos acc = " + str(
        sum(r["additive"][0] for r in held_rows) / len(held_rows)))

    # ---- Stage 3: joint task-conditioned baseline ----
    joint_train, joint_held = train_joint_baseline(
        grid, train_cells, held_cells, cfg, log_fn=log)

    # ---- Stage 4: consistency + mask structure ----
    consistency = consistency_check(model, grid, train_cells, held_cells,
                                    final_masks, synthesizer, cfg)
    anova_stored = anova_decomposition(final_scores, train_cells, cfg.H1)
    anova_ema = anova_decomposition(mask_emas, train_cells, cfg.H1)

    all_cells = train_cells + held_cells
    results = dict(
        cfg=cfg.to_dict(), train_cells=train_cells, held_cells=held_cells,
        order=order, history=history,
        trained_rows=trained_rows, held_rows=held_rows,
        joint_train=joint_train, joint_held=joint_held,
        consistency=consistency,
        anova_stored=anova_stored, anova_ema=anova_ema,
        elapsed_s=time.time() - t0,
    )
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"results written to {out / 'results.json'} "
        f"({time.time() - t0:.0f}s total)")

    # ---- Stage 5: figures ----
    if not args.skip_figures:
        from ftn_nar.viz import make_all_figures
        make_all_figures(out, model, grid, results, final_masks, final_scores,
                         synthesizer, cfg)
        log("figures written")


if __name__ == "__main__":
    main()
