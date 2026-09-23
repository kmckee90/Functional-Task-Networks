# Wave FTN architecture

Run these commands from `/home/dev/workspace` in the **actl `base` pod**.

```bash
python run_experiments.py --config Wave
python run_experiments.py --config Wave WaveSharedMask WaveDetach
python animate_wave.py --config WaveDemo --samples 3 --output projects/wave/ftn_wave.gif
```

The first command uses independent per-step masks and full mask-search BPTT.
The second reuses one mask and differentiates only its final use during search.
All four combinations are supported: the two mixins are independent.

## Computation

Each sample starts with a zero two-channel reservoir on the `SQUARE_DIM` grid.
At each of `STACK_LAYERS` processing steps:

1. Each FTN neuron reads the task input concatenated with its centered J×J patch
   of the primary reservoir layer (`WAVE_INPUT_KERNEL_SIZE`, default 3).
   Patches use `WAVE_PADDING` at the edges. The secondary layer is internal wave memory.
2. Apply the current step mask to the shared `ParallelMLP` output to obtain `z`.
3. Add `WAVE_INPUT_SCALE * z` directly to the corresponding primary-layer pixels.
4. Apply the fixed `WaveTanh` kernel `WAVE_SUBSTEPS` times without further injection.

The output is the shared linear readout of the **last masked `z`**, not of `r`.
Input shape `(batch, D)` repeats the task input at each processing step; explicit
`(batch, steps, D)` inputs are also supported. No state persists between calls
or leaks across samples. Increasing processing steps adds no learned parameters.
Set `WAVE_INPUT_FIRST_STEP_ONLY = True` to zero the task-input slice after the
first processing step (for either input shape). The primary reservoir field
continues to feed the neurons at every step. This also applies during mask
search and animation.

The current `ParallelMLP` ends in tanh, so both injected values and reservoir
values can be positive or negative.

`WAVE_INPUT_KERNEL_SIZE` must be a positive odd integer with `J // 2 < SQUARE_DIM`.
It is explicitly available in both `Default` and `Wave2`. Each neuron's first
layer now has `input_size + J²` inputs, reducing reservoir-connected weights
from `SQUARE_DIM²` to `J²` per hidden unit. Checkpoints from the former dense
reservoir input have incompatible first-layer weight shapes.

## Configuration

| Setting | Default for `Wave` | Meaning |
|---|---:|---|
| `STACK_LAYERS` | 6 | FTN processing steps |
| `WAVE_STEP_MASKS` | `True` | Independent per-step masks; `False` shares one |
| `WAVE_RECONFIG_BPTT` | `True` | Backpropagate mask gradients through history |
| `WAVE_INPUT_KERNEL_SIZE` | 3 | Centered J×J primary-field input patch |
| `WAVE_RHO` | .99 | Linearized radius per wave microstep |
| `WAVE_C` | .65 | Spatial coupling |
| `WAVE_OMEGA` | .15 | Oscillation parameter |
| `WAVE_PADDING` | `"circular"` | Also `"zeros"` or `"reflect"` |
| `WAVE_SUBSTEPS` | 4 | Wave updates after each injection |
| `WAVE_INPUT_SCALE` | .025 | Fixed injection gain |
| `WAVE_INPUT_THRESHOLD` | 0.0 | Admit only masked values with `abs(z) > threshold`, before scaling |
| `WAVE_INPUT_FIRST_STEP_ONLY` | `False` | Give task input only at step zero when enabled |

For example, `WAVE_INPUT_THRESHOLD = 0.6` admits only values with magnitude
strictly above 0.6. Admitted values keep their sign and amplitude; this is a
hard admission gate, not an amplitude clamp. It affects only new reservoir
injections: existing ripples continue and the readout still receives all masked
`z` values. Positive thresholds give rejected values zero injection gradient,
including during mask search. Zero disables the gate and preserves the original
straight-through mask gradients. The animation uses the same model path and
therefore reflects the threshold automatically.

`LAYERS`, `INNER_DIM`, and `DROPOUT` configure the shared neuron bank. Existing
`K`, `RECONFIG_*`, and `MASK_*` knobs configure mask search. Independent masks
may overlap; no disjointness constraint is imposed. Replay stores either all
step masks or one shared mask, according to the selected mode.

`WAVE_RECONFIG_BPTT=False` freezes reservoir history before the final step in
**mask search only**. Earlier independent masks stay fixed, including during
stored-mask recall. A fresh model initializes them as random K-hot masks. With
a shared mask, only its final use gets search gradient. Ordinary weight training
always backpropagates through the full recurrence in both modes.

## Animation and validation

`animate_wave.py` traces actual model states. Current masks appear as white
pixels; positive/negative primary-field values are blue/red, and zero is black.
The signed-log color scale is fixed across the whole animation. Each sample
begins with a blank reset frame. Each processing step shows the injection and
every intermediate wave update. A PNG strip and JSON metadata accompany the GIF.

`WaveDemo` uses a 32x32 grid, K=16, six processing steps, and eight wave microsteps.
The default demo searches masks on a support batch of an untrained model and
animates fresh samples; its header says so. Use `--train-batches N` for a small
single-task training run, or `--checkpoint PATH` to load a `run_compose.py`
checkpoint (its mask is reconfigured for `--task` on a fresh support batch).

```bash
python -m unittest test_wave_ftn.py
python run_experiments.py --config WaveSmoke --metrics-only
python run_experiments.py --config WaveSmoke WaveSharedMask WaveDetach --metrics-only
```

The architecture is dispatched by both `run_experiments.py` and `run_compose.py`;
`eval_compose.py` also accepts its checkpoints and alternate recurrent depths.
