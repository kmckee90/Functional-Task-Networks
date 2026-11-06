# BayesLoRA experiment

Self-contained extraction of `ftn/projects/bayesian_adv` restricted to its
feed-forward `BayesLoRA` model. The default preserves that preset's backbone,
posterior updates, reactive change-point refits, replay, and evaluation settings.
No recurrent LoRA, lateral feature/mask kernels, or overlap search penalties are
included. Replay overlap filtering and overlap metrics remain.

Run from this directory in the workspace pod:

```sh
uv run python run_experiments.py
uv run python run_experiments.py --config Smoke --name smoke
uv run python run_experiments.py --config BayesLoRA DomainShift NoConceptShift InputConditioned
uv run python run_experiments.py --config Smoke DomainShift NoConceptShift InputConditioned InputConditionedSmoke --name smoke_ic
```

Presets compose left to right; later presets override earlier ones. `Default`
and `BayesLoRA` select the same model. `Smoke` reduces model and run sizes while
retaining all visualizations and entropy probes. Package execution also works:
`python -m ftn_lora.run_experiments` from the repository root (use
`--out-root ftn_lora/artifacts` there).

`SpatialPermuteRandomCenter` is already enabled by default, exactly as in
`ftn/config.py`: random blobs, center attraction 0.25, roundness 2.0, radius 7,
eight sweeps, temperature 0.05, seed 0. The implementation is copied from
`ftn/spatial_permute.py`. Every mask has `SQUARE_DIM**2` atoms and renders as one
square. At task-block boundaries the display layout refines its previous state;
training atom IDs, replay masks, intersections, and training RNG are unchanged.
The neighborhood kernel in this display optimization does not smooth model
features or masks. Categorical colors retain all tasks and blend intersections.

`InputConditioned` retains input reconstruction routing, optional novelty
triggers and retrieved training priors, direct/refined evaluation, and archived
mask repair from labeled training replay. It is opt-in, as in the source.
Use `DomainShift NoConceptShift` for distinguishable input domains; identical
input distributions cannot identify different tasks from inputs alone. Router
settings are in `config.py`; statistics are included in the metrics file.

Outputs retain `artifacts/<name>/<prefix>_<name>.<extension>`:

- `config` and `metrics` text files
- `bw` and `rgb` GIFs, plus the final `rgb` PNG
- `loss`, `task_vis`, and `xfer` PNGs (per-epoch `xfer` names for multiple epochs)
- `entropy_acc`, `entropy_std`, and `entropy_share` PNGs

The three entropy probes are enabled by default. `--render-steps 0` suppresses
mask animations and task surfaces; `--metrics-only` suppresses all visual output
and entropy probes while retaining evaluation, metrics, and config output.
Run names auto-increment unless explicitly supplied or `--noidx` is used.
`gpurun.sh`, `run_seeds.py`, and `analyze_sweep.py` retain launch/sweep support.

Validation on the pod:

```sh
python -m unittest discover -s . -p 'test_*.py' -v
```

The tests cover posterior/weight separation, cold evaluation state, square
layout invariants, routing, and complete smoke-run artifact manifests.

## Entropy-based evaluation reconfiguration

`EntropyReconfig` searches masks using only support inputs, minimizing the
batch mean of per-example softmax entropy, `-sum(p * log(p))` (in nats).
During evaluation, targets are used only to score accuracy/loss after mask search.
Training mask updates remain supervised by default.

```sh
./gpurun.sh BayesLoRA EntropyReconfig
./gpurun.sh BayesLoRA EntropyReconfig -- --eval-same-support 1
```

`EVAL_SAME_SUPPORT=False` (or `--eval-same-support 0`) scores a fresh query
batch. Set it to `True` (or pass `--eval-same-support 1`) to score the exact
support rows used by the search, capped by `EVAL_RECONFIG_SAMPLE_SIZE`.
The flag applies to backwards/prospective evaluation, final evaluation, and
entropy-test matrices. In the matrices, column j reuses task j's support set;
off-diagonal cells still evaluate it under task i's mask.
Both choices are recorded in config and metrics artifacts.

With `InputConditioned`, router misses use cold entropy search; router hits
refine the retrieved mask with entropy plus the existing prior KL. Entropy
refinement also runs when `IC_EVAL_STRATEGY='direct'`, so the entropy preset
always performs a search. Support labels never enter either path. This objective
optimizes prediction confidence, which need not correspond to accuracy.

Set `TRAIN_RECONFIG_ENTROPY=True`, or pass `--train-reconfig-entropy 1`, to use
entropy for training mask updates as well:

```sh
./gpurun.sh BayesLoRA EntropyReconfig -- --train-reconfig-entropy 1
```

This flag covers scheduled updates, stable-batch Bayesian updates, and
change-point refits, including input-conditioned retrieved priors. Existing
Bayesian KL terms remain. It is independent of the evaluation preset; set it
back to `0` for supervised training mask search. Weight training, replay repair,
and the reactive change-point detector still use their supervised losses.
The flag is recorded in both config and metrics artifacts.
