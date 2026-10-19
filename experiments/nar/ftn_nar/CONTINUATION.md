# FTN-NAR continuation state (updated 2026-07-18)

Design doc `ftn_neural_algorithmic_reasoning_experiment_design.txt` = complete spec.
Cold-resume handoff.

## Where everything runs
- Pod `ftn-nar` (4×H100, ns obelisk) → `/home/dev/workspace`. EDIT LOCAL, RUN REMOTE
  (`actl pod exec ftn-nar -- bash -c 'cd /home/dev/workspace && ...'`). Never run local.
- Runs on pod under `/home/dev/*` (not synced; PVC-persistent). torch 2.8. uv project.
- Sync dies when laptop sleeps → restart with `actl pod sync ftn-nar --no-tty` in bg,
  then wait for a local edit to appear on pod before running. `actl pod exec` works
  without sync (kubectl exec) but does NOT propagate file edits.
- Detached pod launches survive laptop sleep: `setsid nohup ... >log 2>&1 </dev/null &`.

## Task status
1–6 DONE. Executor gate assessed (task 6): passes ID+width, fails OOD-depth (accepted
  as documented limitation per user decision). 7 IN PROGRESS. 8 PENDING.

## Executor — RESOLVED understanding (see memory ftn-nar-executor-findings)
Correct code now. Key fixes landed: register readout (not tanh head); loss beta 1/16384
+ value weights; round_register=True; steps_per_task=8000. Two BUGS fixed this session:
(a) frozen-hu — `hu=h` was bound once before the recurrent loop so banks saw the initial
latent; symptom was boundary R-row != reload eval. (b) config_hash shifts when adding
ModelCfg fields broke resume dir lookup → Trainer now takes explicit run_dir=.
ALWAYS assert boundary==reload after model changes.
Rejected ablations (flags exist, default off): op_latent=False (breaks learning),
extender_residual (neutral, doesn't fix depth).
Executor reality: seen ID 0.98, OOD-width N=64 0.88 (good); OOD-depth is a hard NAR wall
(seen fails D=16). Held-out composition gap: node 0.92-0.97 vs seen 0.99.

## RUNNING NOW (task 7): dev-fold method sweep
8 methods on MIN_ADD fold seed 0, discovery.yaml, in `/home/dev/matrix/`, launcher log
`/home/dev/matrix_devfold.log`, 2 slots/GPU. Methods: FTN_FULL, FTN_NO_REPLAY,
FTN_RANDOM_MASK_TRAIN, DENSE_ACTIVE_MATCHED, SOFT_MIXTURE, DISJOINT_TASK_MODULES,
JOINT_MULTITASK_FTN, ORACLE_COMPOSITION. FTN_RETRIEVAL derived free at eval from FTN_FULL.
FTN_FULL ~304 ms/step (config = 77%), ~4.7 hr/fold; sweep ~6-9 hr wall.

### ON RESUME
```bash
actl pod exec ftn-nar -- bash -c 'grep -c "ALL DONE" /home/dev/matrix_devfold.log; ls /home/dev/matrix/*/eval/eval.json | wc -l'
```
When done (8 evals): aggregate the dev fold:
```bash
actl pod exec ftn-nar -- bash -c 'cd /home/dev/workspace && uv run python -m ftn_nar.cli aggregate --runs-dir /home/dev/matrix --out outputs/reports_devfold'
actl pod exec ftn-nar -- bash -c 'cd /home/dev/workspace && CUDA_VISIBLE_DEVICES="" uv run python scripts/show_gate.py /home/dev/matrix'  # oracle gate view
```
Inspect report.json comparisons: FTN_vs_RETRIEVAL / FTN_vs_DENSE / FTN_vs_SOFT at
id_test and ood_width_64 (the design's primary N=64 metric §16.3). H2 success = FTN
beats retrieval by ≥15pp at N=64, CI excludes 0.

## Then (task 8)
- Tiny-bank exhaustive diagnostic (cheap, tiny model): train `--tiny` FTN_FULL on a fold,
  then `cli exhaustive --run-dir <dir>`. Also the scaffolding comparison
  (per-minibatch vs FTN_PER_TASK_MASK vs FTN_RANDOM_MASK_TRAIN vs FTN_NO_REPLAY).
- Expand folds/seeds as compute allows (discovery seeds 0-2). Full 8×5 matrix = design's
  1500-2500 GPU-hr (weeks) — run discovery scale, document extrapolation.
- Aggregate all → tables + figures (cli aggregate builds both) → write RESULTS.md with
  H1-H5 verdicts. Primary compositional metric is at N=64 WIDTH (executor handles it).

## Diagnostic scripts (scripts/): show_gate, diag_seen_vs_heldout, diag_heldout,
diag_ckpt, cmp_block0, probe_overfit/lossgap/evalgap. Consistency invariant: boundary
R-row == fresh reload evaluate_program (assert after any model change).
