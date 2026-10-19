# composition/ — noncommutative function-composition testbed

A rigorous testbed for whether an FTN (and baselines) learn and *compose*
primitive functions, and generalize the compositions out of distribution.

## Task (GF(2) linear + interleaved reversible S-box)

State = `d = matrix_dim**2` bits. `n_ops` primitive bijections
`f_i(v) = S(M_i · v)` over GF(2): `M_i` a random **invertible** GF(2) matrix
(sparsity tunable), `S` a fixed reversible block S-box (interleaved after every
linear step). A task is a length-`steps` word; its function is the composition.

Why this design (see the session discussion):
- **Lookup-proof**: no output bit copies one input bit, so per-cell column
  matching (which trivially solves the permutation task) is useless.
- **Solver-proof**: `S` interleaved between linear steps means the composition is
  not a single matrix — no closed-form linear solver.
- **Zero redundancy**: distinct words are distinct bijections (verified on a probe
  batch); the catalog is the exact hypothesis space.
- **Clean claim**: a structureless nonlinear bijection admits **no** OOD
  generalization except through the known compositional structure, so any
  above-chance held-out / length-generalized accuracy is direct evidence of
  compositional learning.

Difficulty knobs (in `compose_config.py`, defaulting to the hardest discussed
setting — dense linear + S-box): `GF2_M_DENSITY` (sparsity), `GF2_NONLINEAR` /
`GF2_SBOX_BITS` (nonlinearity), `GF2_MATRIX_DIM` (state size).

## Files

- `compose_config.py` — **one** consolidated config: GF(2) task + difficulty +
  baselines + sweep/gate schedule + inherited FTN `*_PARS`. Subclasses
  base `config.Default`, so any base architecture is available.
- `task_gf2.py` — the GF(2)+S-box generator (`DataGenGF2`), parity with
  `task_comp.DataGenComposition`.
- `oracle.py` — exact ceilings/floors: catalog oracle (Bayes cap), identity /
  marginal / memorization floors.
- `models_baseline.py` — MLP, multi-step RNN, in-context DeepSet (+ param match).
- `models_ftn.py` — base-FTN builder (dispatch mirror of run_compose.build_model).
- `capacity_gate.py` — **run first**: can the arch represent one primitive at
  1 step/operator? Finds a viable difficulty.
- `run_compose.py` / `eval_compose.py` — FTN training + length-gen (added next
  pass, with GF(2) generator dispatch).
- `bench.py` / `run_baselines.py` — support-size sweep: model vs cap vs baselines
  (added next pass).

## Running (pod `base`, 2×GPU, namespace obelisk)

Edit locally; run remotely. Never touch the pod venv.

```sh
# capacity gate at the reference difficulty
actl pod exec -- bash -c 'cd /home/dev/workspace/composition && uv run python capacity_gate.py --config GF2Compose'

# dial difficulty down to find a representable regime
actl pod exec -- bash -c 'cd /home/dev/workspace/composition && uv run python capacity_gate.py --config GF2Compose --m-density 0.2 --sbox-bits 2'
```
