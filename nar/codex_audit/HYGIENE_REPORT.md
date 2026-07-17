# Hygiene Audit Report: FTN NAR Abelian-Grid Experiment

Date: 2026-07-16

Scope: read-only audit of `RESULTS.md`, the `ftn_nar/` source, experiment scripts, and saved artifacts under `results/`. I did not rerun the full experiment or modify experiment files. This report focuses on bad conclusions, confounds, leakage, cheating risk, hallucinated details, and reproducibility hygiene.

## Executive Verdict

The core artifact-level result is real in a narrow sense: for `results/main2`, the trained FTN2 checkpoint plus support-conditioned mask search reaches about 0.979 per-position / 0.918 exact-match accuracy on the 20 held-out task cells, and a random untrained bank with the same configurer is near chance. I did not find evidence that evaluation input sequences are directly used in gradient updates.

The writeup is not clean enough for the strongest conclusions in `RESULTS.md`. The biggest issues are interpretive: the headline held-out result is few-shot support-conditioned mask fitting, not data-free composition; the main baseline comparisons are not information-matched; the protocol was tuned using held-out-cell performance; simple support-only permutation baselines are missing; and several claims are either unsupported by artifacts or numerically wrong.

Recommended corrected headline:

> A two-stage FTN trained sequentially on 20 of 40 fixed permutation tasks contains frozen weights from which high-accuracy masks for held-out cells can be found using labeled support examples and mask-logit optimization. Direct data-free mask algebra fails; temporal iteration of trained masks works for cyclic-subgroup composites. The result is promising evidence of learned reusable routing competence, but not yet clean evidence of data-free compositional program induction or of superiority over matched few-shot baselines.

## What Checks Out

1. The main `main2` metrics in `RESULTS.md` mostly match saved artifacts:
   - Held-out reconfig: 0.978545 per-position / 0.91794 exact-match.
   - Trained consensus/final masks: 0.9911325 per-position / 0.96526 exact-match.
   - Random untrained bank control in `controls.json`: 0.1149475 mean per-position, max 0.1436.
   - Off-group trained-bank means in `offgroup_probe.json`: tier 0 = 0.982465, tier 1 = 0.732650, tier 2 = 0.496737.

2. The global input train/test split is real:
   - `TaskGrid` enumerates all 10^4 sequences, splits them into `train_seqs` and `test_seqs`, and `eval_tensors(..., "test")` uses the disjoint test split.
   - Training and support batches use `grid.batch(..., "train")`; evaluation uses `grid.eval_tensors(..., "test")`.

3. I did not see direct gradient leakage from held-out task evaluation inputs:
   - Held-out-cell reconfiguration samples labeled support from the training input split, then evaluates on the test input split.
   - Weights are frozen during held-out-cell mask search; only mask logits are optimized.

4. The negative mask-algebra result is supported:
   - `results/main2/results.json` shows additive synthesis at 0.087115, assigned at 0.054715, assigned-swapped at 0.04078, majority union at 0.0809325, and mu-only at 0.0581575.

## Major Findings

### 1. The headline held-out result is few-shot mask fitting, not data-free composition

`RESULTS.md` says the held-out result is "composition by spatial selection" and emphasizes that held-out cells are never trained. That is true for weights, but the held-out protocol gives the configurer 128 labeled examples from the new algorithm and 60 optimization steps on mask logits.

Evidence:
 - `RESULTS.md` states this at lines 133-138.
 - `ftn_nar/evals.py` lines 61-64 explicitly sample `xs, ys = grid.batch(cell, cfg.support_size, "train")` for held-out cells and run the configurer.
 - `ftn_nar/configurer.py` optimizes mask logits against those labels.

This is a valid few-shot protocol, but the writeup should not let readers infer that the held-out algorithms are solved without task data. The clean zero-data result is only the temporal iteration result for reachable cyclic-subgroup composites, not the main all-held-out-cells result.

### 2. The support set almost identifies the task/function by itself

The task family consists of fixed permutations of 40 one-hot input bits. With 128 random support sequences of length 4 over 10 values, each position-value input bit is expected to appear about 12.8 times; the probability a particular bit is unseen is roughly 0.9^128, about 1.4e-6. So the support set is usually enough to infer the entire bit permutation by simple co-occurrence, without FTN machinery.

This does not make the FTN result fake. It means the correct comparison is not only "random FTN bank plus same mask search." A very simple support-only permutation learner, a finite-family enumerator over the 40 `(a,r)` cells, or a linear/logistic model fit on support could likely solve the held-out group tasks at or near 100% with 128 examples. For off-group probes, a co-occurrence bit-permutation learner would also be the relevant sanity baseline.

Missing baseline: support-only bit-permutation inference from the 128 labeled examples, evaluated on the same test split.

### 3. The joint task-conditioned MLP baseline is not information-matched

`RESULTS.md` uses the joint MLP failure to argue that the task is not solved by ordinary representation learning. That comparison is not fair to the claim being made:

 - FTN held-out reconfig gets labeled support examples from the held-out task.
 - The joint MLP gets no held-out support examples; it is evaluated zero-shot from one-hot `a` and `r` task codes.

Evidence:
 - FTN held-out support: `ftn_nar/evals.py` lines 61-64.
 - Joint MLP training/evaluation: `ftn_nar/baselines.py` lines 35-69. It trains only on training cells and directly evaluates held-out cells with task codes.

The baseline does show that this particular one-hot-conditioned MLP does not interpolate unseen task-code pairs. It does not establish that "standard representation learning" or a matched few-shot learner fails. The wording in `RESULTS.md` lines 150-155 and 273-276 is too strong.

Needed baselines:
 - Joint/task-conditioned MLP fine-tuned or adapter-tuned on the same 128 support examples.
 - Linear probe or logistic regression from one-hot inputs using support examples.
 - Co-occurrence bit-permutation support baseline.
 - Meta-learning or hypernetwork baseline if the claim is few-shot program induction.

### 4. Hyperparameters and protocol were selected with held-out-cell feedback

`RESULTS.md` presents iteration history with held-out reconfig scores for `main0`, `main1`, and `main2`, and the final protocol was chosen after seeing those held-out outcomes.

Evidence:
 - `RESULTS.md` lines 96-102 list held-out reconfig performance for successive runs and the protocol changes that led to `main2`.

This is not cheating inside a single run, but it means `main2` seed 0 is not an unbiased final holdout evaluation. The seed-replication table helps, because seeds 1 and 2 redraw grid/input splits, but those are still post-selection replications of a protocol selected using earlier held-out feedback. Three seeds is thin for a final claim.

Needed hygiene fix: freeze the protocol, then run a fresh locked evaluation on new seeds/grid splits not used during development, with no further tuning.

### 5. The parameter count in `RESULTS.md` is wrong

`RESULTS.md` says "7.4M parameters total" at line 66.

The run log and code show 2,871,296 parameters:
 - `results/main2/train_log.txt` line 5: `FTN^2 params: 2,871,296`.
 - `run_experiment.py` lines 59-62 constructs the model and logs `sum(p.numel() for p in model.parameters())`.
 - The model dimensions in `ftn_nar/model.py` lines 58-68 and `ftn_nar/config.py` lines 18-26 are consistent with about 2.87M parameters.

This is a straightforward hallucinated/arithmetic error in the writeup.

### 6. The single-cell sweep claim is not auditable from saved artifacts

`RESULTS.md` lines 104-108 claim a single-cell sweep showed large `k` was decisive, taking one-cell accuracy from 0.62 to 1.00. I found no saved sweep results under `results/`.

The only sweep file, `scripts/sweep_single_cell.py`, prints to stdout and does not save outputs. Also, in the committed code, `small()` already sets `k1=k2=128`, and the `k128` variant in `scripts/sweep_single_cell.py` lines 12-20 is therefore a no-op relative to the base small config. That makes the exact reported sweep claim unreproducible from the current repository.

Needed hygiene fix: save sweep outputs, exact configs, seeds, and command logs, or remove the claim.

### 7. The off-group table mixes sources and overstates what is ruled out

`RESULTS.md` line 175 gives a random-bank value for tier 0 in the off-group table. But `scripts/offgroup_probe.py` excludes tier 0 from the random-bank probe:

 - Tier 0 probes are created at lines 57-61.
 - The trained model evaluates all probes at lines 133-135.
 - The random model evaluates only `off_probes = [t for t in probes if t.tier > 0]` at lines 137-141.

So the tier-0 random value in the table is presumably borrowed from `controls.json`, not `offgroup_probe.json`. That should be stated explicitly, because the table claims to summarize `scripts/offgroup_probe.py` / `offgroup_probe.json`.

The interpretation also goes too far. The tier staircase is suggestive, but it does not "rule out" a generic routing-basis explanation by itself, because:

 - Tier 1 has only 9 probes and tier 2 has only 4 probes.
 - Tier 1 includes easy near-identity or low-support changes, which the writeup correctly acknowledges.
 - No support-only bit-permutation baseline is included.
 - No regression on distance/proximity to trained cells is actually run, despite being proposed in the caveat.

The supported conclusion is weaker: the trained FTN bank is much more reachable for in-group tasks than for these few off-group probes, and it has partial reachability for product-structured probes.

### 8. "FTN2 is forced/minimal" is not established

`RESULTS.md` lines 69-78 argue that a single additive readout cannot express the composites and therefore FTN2 is the minimal forced extension.

The math argues against an additive factor decomposition of the form `f(x;a)+g(x;r)`. It does not prove a single-bank FTN with task-specific masks cannot memorize or represent all 40 permutation cells, nor does the repo contain a matched single-bank/no-depth ablation. Since every task cell is itself a fixed 40-bit permutation, expressivity claims should be much more careful.

Needed ablations:
 - Single-bank FTN with matched parameter count and active-mask budget.
 - Two-bank model without curriculum/replay.
 - Explicit factorized architecture baseline if the claim is about compositional factorization.

### 9. The ablation story is confounded

The iteration table changes multiple things at once: learning rate, steps per cell, replay samples, post-stream consensus masks, multiple cycles, and curriculum. Therefore the improvement from `main1` to `main2` cannot be attributed cleanly to "3 cycles x 300 steps + primitives-first curriculum."

Needed hygiene fix: run isolated ablations for cycles, total updates, curriculum, replay settings, and final consensus masks.

### 10. Some wording overclaims accuracy and generalization

Examples:

 - "full OOD correctness" should be "evaluated OOD against the generator"; the main held-out result is 0.918 exact-match, not full correctness.
 - "from as few as ~16 labeled examples" is misleading if attached to the 0.98 headline. The support sweep gives 16 examples -> 0.876, 32 -> 0.937, 128 -> about 0.977.
 - "composition was learned" is too broad. The random-bank control shows the result is not due to configurer search alone in an untrained FTN2, but it does not rule out support-only learners or other trained architectures.

## Leakage and Cheating Assessment

I found no obvious direct cheating in the saved experiment:

 - Held-out task cells are not included in the weight-training cell list.
 - Evaluation input sequences are disjoint from training/support input sequences.
 - Held-out reconfiguration freezes weights and updates only mask logits.
 - The random-bank control uses the same support/search mechanism on untrained weights.

The main leakage-like issue is selection bias:

 - The development loop used held-out-cell scores to choose the final protocol.
 - Training logs evaluate trained cells on test input sequences during training; this is not gradient leakage, but those metrics can influence human/agent decisions.

The main confound is support information:

 - The held-out task labels supplied to the configurer are rich enough to identify the task/function for this finite permutation family.
 - Baselines that receive no held-out support are not matched to the FTN reconfig protocol.

## Reproducibility Gaps

1. No commit hash or environment snapshot is recorded in `results/main2/results.json`.
2. Control/off-group scripts do not save full config, command invocation, git state, or hardware metadata into their JSON outputs.
3. The single-cell sweep has no saved artifacts.
4. `scripts/refigure.py` uses `eval()` to parse stringified tuples from JSON. This is not an experimental validity issue, but it is brittle and unnecessary.
5. GPU determinism is not controlled beyond manual seeds; exact reruns may differ.

## Recommended Next Steps

1. Freeze the protocol and run a fresh final evaluation on new seeds/grid splits, with no further protocol changes after seeing results.
2. Add support-matched baselines:
   - Co-occurrence bit-permutation inference from support.
   - Finite-family enumerator over all 40 `(a,r)` cells using support, reported as an oracle-family baseline.
   - Logistic regression or linear model fit from support.
   - MLP/adapter fine-tuning on the same support budget.
3. Add architecture ablations:
   - Single-bank FTN, matched parameters.
   - FTN2 without replay.
   - FTN2 without curriculum.
   - Same total updates but different cycle structure.
4. Redo off-group probes with many random probes per tier, random-bank tier 0 included in the same JSON, and regression against nearest trained-cell agreement.
5. Fix `RESULTS.md` before sharing:
   - Change 7.4M params to 2.871M.
   - Reword held-out reconfig as few-shot support-conditioned mask search.
   - Remove or substantiate the single-cell sweep claim.
   - Temper claims about standard representation learning and generic routing bases.
   - Report exact-match alongside per-position accuracy wherever "correctness" is discussed.

## Bottom Line

The experiment is a useful signal that sequential FTN training creates frozen routing structure that mask search can exploit for unseen cells. It is not clean evidence yet for data-free compositional program induction, symbolic mask algebra, or failure of ordinary representation learning. The strongest current claim should be framed as few-shot support-conditioned retrieval/adaptation over a trained FTN bank, with promising but not yet decisive evidence of compositional structure.
