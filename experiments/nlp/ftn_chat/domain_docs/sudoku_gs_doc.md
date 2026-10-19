# Documentation: sudoku_gs.py — Gradient-search Sudoku solving via relaxed one-hot assignments in JAX

## Overview

`sudoku_gs.py` solves Sudoku puzzles by **gradient search** (the "gs" variant,
as opposed to a message-passing solver `sudoku_mp.py`): instead of backtracking
or constraint propagation, it treats each empty cell as a categorical variable
and *optimizes a continuous relaxation of the whole grid* with a JAX optimizer
until the assignment satisfies the Sudoku rules. Each of the 81 cells carries a
vector of logits over the 9 possible digits; a differentiable loss penalizes any
row, column, or 3×3 block in which a digit fails to appear exactly once, and
gradient descent (Adam by default) pushes the logits toward a valid, fully
constrained solution.

The spirit of the code is **constraint satisfaction as differentiable
optimization**: the discrete "every digit appears once per group" rule is turned
into a smooth, almost-everywhere-differentiable objective using a Gumbel-softmax
relaxation with a straight-through estimator, and many independent candidate
solutions ("lanes") are optimized in parallel per puzzle so the best one can be
selected at the end. The whole inner loop is compiled and vectorized with JAX
(`jax.jit`, `jax.lax.scan`), so a puzzle's full optimization runs as one fused
GPU kernel.

Board constants come from the sibling module `sudoku_mp`: `GRID_SIZE = 9`,
`BLOCK_SIZE = 3`, `N_CELLS = 81`, and `N_CATEGORIES = 9` (the digits 1..9). That
module also supplies the data/IO and checking helpers `count_sudokus`,
`iter_sudoku_batches`, and `verify_sudoku`.

## Cell representation and clue locking

A puzzle is an integer array of shape `(batch, 9, 9)` where `0` marks an empty
cell and `1..9` marks a given clue. The solver works in logit space: a tensor of
shape `(total_lanes, N_CELLS, N_CATEGORIES)` holding, per cell, an unnormalized
score for each digit. `argmax` over the last axis (plus one) reads out the digit.

Given clues must never change, so they are *locked* rather than optimized:

- `_build_clue_masks(puzzles)` returns `(known_mask, known_assignments,
  locked_logits)`. `known_mask` is the boolean "this cell is a clue" map;
  `known_assignments` is the one-hot of each clue digit; `locked_logits` is a
  saturated logit field built as `-LOCKED_LOGIT + 2*LOCKED_LOGIT *
  known_assignments` (with `LOCKED_LOGIT = 30.0`), i.e. a very large positive
  logit on the clue's digit and a very large negative logit elsewhere.
- `_apply_clues(logits, known_mask, locked_logits)` overwrites the logits at clue
  cells with those locked values via `jnp.where`. It is applied at
  initialization, inside the loss, and after every optimizer update, so clue
  cells are effectively frozen throughout the search.

## Parallel candidate lanes

For each puzzle the solver runs `n_samples` (default 180) **independent candidate
solutions in parallel**, called *lanes* — note these are separate restarts of the
optimization, not Monte-Carlo samples within a step. With a batch of puzzles the
working tensor is flattened to `total_lanes = batch_size * n_lanes`. The
per-puzzle clue tensors are broadcast across lanes by `_repeat_lane_axis`, which
repeats along a new lane axis and folds it into the leading dimension. Running
many lanes hedges against the non-convexity of the relaxed objective: different
random initializations land in different local optima, and the best lane is
chosen at the end.

## Relaxed sampling with a straight-through estimator

`_sample_relaxed_assignments(logits, tau, key)` converts logits into an
approximately one-hot assignment that is still differentiable:

- Add Gumbel noise and take a temperature-scaled softmax: `soft =
  softmax((logits + gumbel) / tau)`.
- Compute the hard one-hot `hard = one_hot(argmax(soft))`.
- Return `soft + stop_gradient(hard - soft)` — the **straight-through estimator**:
  the forward pass uses the crisp one-hot `hard`, while gradients flow through the
  smooth `soft`. Lower temperature `tau` makes `soft` sharper.

## Differentiable uniqueness constraint

The core trick is making "each digit appears exactly once in this group of 9
cells" differentiable. For a group of 9 one-hot(ish) cells:

- `_one_hot_group_match_prob(groups)` computes, for each digit, a *leave-one-out*
  probability that exactly one cell in the group carries it. Using
  `one_minus = 1 - groups`, it forms exclusive prefix and suffix cumulative
  products (`cumprod`) so that, for each position, it multiplies the
  "nobody-else-has-it" mass of all *other* cells; summing `groups * leave_one_out`
  gives the probability that precisely one cell holds the digit.
- `_one_hot_log_loss_per_lane(groups, eps)` softens that probability away from 0
  and 1 by `eps` (`DEFAULT_CONSTRAINT_EPS = 1e-4`), clips it, and returns the mean
  negative log `-log(p)` over the group — a per-lane scalar that is 0 only when
  every digit is uniquely placed.
- `_per_digit_constraint_loss_per_lane(assignments, eps)` reshapes/transposes the
  `(lane, 9, 9, 9)` assignment tensor into row groups, column groups, and 3×3
  block groups, scores each with the log-loss above, and returns
  `row_loss + col_loss + block_loss`. Minimizing this drives every row, column,
  and block toward a permutation of 1..9.

## Loss, optimizer, and the JIT training loop

`solve_sudoku_gs(...)` ties it together:

- `_loss_per_lane` applies clues, samples a relaxed assignment, splices the locked
  clue one-hots back in via `jnp.where`, reshapes to `(lane, 9, 9, 9)`, and
  returns the per-lane constraint loss; `loss_fn` averages it to a scalar.
- The optimizer is built by `_build_optimizer` (`adam` default, also `rmsprop` /
  `sgd`) at `DEFAULT_LEARNING_RATE = 0.1`. Gradients come from
  `value_and_grad(loss_fn)`.
- `train_step` does one optax update and re-applies clues to the result.
- A **temperature schedule** `_temperature_schedule` linearly anneals `tau` from
  `temperature_init = 1.0` to `temperature_final = 0.6` over `n_iters`
  (default 40000) steps, so sampling starts soft/exploratory and ends sharp.
- `run_train_loop` runs the whole schedule with `lax.scan` over the `tau` values
  (one fused, jitted loop), threading `(logits, opt_state, key)` as carry and
  splitting the PRNG key each step. A `verbose` path instead runs the steps in
  Python and periodically prints `loss` and the number of satisfied rules.

## Selecting and verifying the solution

After optimization, logits are read out per lane and the best lane per puzzle is
chosen:

- `_count_satisfied_rules(solved)` sorts each row/column/block and counts how many
  of the `3 * GRID_SIZE = 27` groups are exactly `1..9` — a hard, discrete score
  of how close a lane is to a valid grid.
- The winning lane is the one with the **most satisfied rules**, breaking ties by
  **lowest constraint loss** (`masked_losses` sets non-best-rule lanes to `inf`
  before `argmin`).
- The selected grid is checked to ensure it still respects the original clues; a
  violation raises `ValueError`.
- `solve_sudoku_gs` returns `(solved, selected_logits, selected_loss)` for the
  chosen lane per puzzle.

## Command-line entry point

`main()` parses arguments (`--csv-path`, `--num-iterations`, `--temperature-init/
final`, `--constraint-eps`, `--learning-rate`, `--n-samples`, `--optimizer`,
`--seed`, `--batch-size`, `--limit`, `--verbose`), then streams puzzles from a CSV
with `iter_sudoku_batches`, solves each batch with `solve_sudoku_gs`, verifies the
result with `verify_sudoku`, and reports a running correctness rate via `tqdm`.
The seed is offset per batch so different batches get independent initializations.
