# Functional Task Networks

Dynamic sparse masking with lateral cortical dynamics for multi-task and continual learning.

A single shared MLP is partitioned into task-specific subnetworks via binary masks over hidden units. Masks are determined online through gradient-based optimization followed by lateral smoothing on a 2D cortical grid and k-winner-take-all (KWTA) sparsification. An offline replay buffer with mask-correlation filtering stabilizes shared representations across tasks.

## Conceptual Overview

### Core Idea

Instead of training separate models per task or relying on fixed task-routing, this approach treats the hidden layer as a 2D grid of neurons (like a cortical sheet) and dynamically selects which neurons participate in each task. The selection process has three stages:

1. **Gradient-based mask optimization** -- The mask is treated as a continuous differentiable parameter and optimized (via RMSprop) to minimize task loss. This identifies which neurons are informative for the current data. The mask is passed through a sigmoid and stochastic Bernoulli sampling with a straight-through estimator.

2. **Lateral dynamics** -- The continuous mask is reshaped to a 2D grid and convolved with a smoothing kernel, enforcing spatial coherence so that nearby neurons tend to co-activate, producing contiguous subnetwork regions.

3. **KWTA sparsification** -- The top-k mask values are kept and the rest zeroed out, enforcing a fixed capacity budget per task.

### Why It Works

- **Catastrophic forgetting protection**: Only masked (active) neurons contribute to the forward pass, so gradient updates only affect the active subnetwork. Inactive neurons' weights are left untouched.
- **Spatial separation**: Lateral dynamics on the 2D grid push different tasks toward spatially separated regions, reducing overlap and interference.
- **Selective replay**: The replay buffer only rehearses examples whose masks *partially* overlap with the current mask (correlation between 0.2 and 0.8), targeting exactly the shared neurons that are at risk of interference.

### Experiment Phases

The experiment pipeline tests these properties through a sequence of phases:

| Phase | Name | Purpose |
|-------|------|---------|
| 0 | Pre-train task 0 | Establish a baseline for one task before multi-task training |
| 1 | Multi-task learning | Train all N tasks block-sequentially |
| 2a | OOD neuron combos | Evaluate (no training) on out-of-distribution mask configurations |
| 2b | OOD data generator | Evaluate on data from a different generator seed |
| 3 | Task 0 recovery | Report task 0 performance across all prior phases |
| 4 | Neurogenesis | Expand the hidden layer (2x), retrain, and analyze mask allocation |

### Data Generation

Tasks are synthetic. The `DataGenSine` generator creates classification or regression tasks by:
1. Projecting random inputs through a fixed linear encoder
2. Applying a task-specific binary mask to select which hidden dimensions are active
3. Passing through a sinusoidal nonlinearity (`sin(frequency * z)`)
4. Decoding to outputs (thresholded sum for classification, scaled sum for regression)

Tasks can be **orthogonal** (disjoint hidden dimensions) or **correlated** (shared dimensions via pairwise unions). This controls inter-task interference at the data level.

### Model Architecture

`MaskedMLPSpatial` is a multi-layer MLP where every hidden layer's activations are element-wise multiplied by the mask. It also includes:
- A parallel linear skip connection (`self.lin`) that bypasses the MLP layers
- Lateral smoothing via a learned-free Conv2d kernel applied to hidden activations reshaped as a 2D grid
- Dropout applied through the mask (Bernoulli sampling gated by the mask)

The final output combines the MLP path and the linear skip path.

### Mask Configurers

Several mask configuration strategies are available in `masks.py`:

| Configurer | Lateral | Sparsity | Notes |
|-----------|---------|----------|-------|
| `SmoothKWTA` | Uniform conv, 2 steps | KWTA | Default single-kernel approach |
| `SmoothKWTAFast` | Large kernel (17), 2 steps | KWTA | Fast convergence |
| `SmoothKWTASlow` | Small kernel (3), 15 steps | KWTA | Finer-grained patterns |
| `PolyakKWTAFast` | Polyak-averaged (alpha=0.95) | KWTA | Preserves gradient structure |
| `KWTA` | None | KWTA | Ablation: no spatial structure |
| `GridCell` | Dual exc/inh kernels | Emergent (softmax) | Hexagonal grid-like patterns |
| `MexicanHatCA` | Dual exc/inh kernels | Emergent (softmax) | Blob-like task regions |
| `Disabled` | None | Fixed | Ablation: static mask |

## Project Structure

```
func_net_2/
  run_experiments.py   # Main experiment script (phases 0-4)
  config.py            # Configuration classes (Classification, Regression)
  models.py            # Neural network architectures (MLP, MaskedMLP, MaskedMLPSpatial)
  masks.py             # Mask configurers (SmoothKWTA, DualKernel, etc.)
  tasks.py             # Synthetic data generators (DataGen, DataGenSine)
  utils.py             # Utilities (parameter counting, frame creation)
  images/              # Output directory for plots and GIFs (created at runtime)
```

## Usage

### Requirements

- Python 3
- PyTorch (with CUDA)
- NumPy, Pillow, Matplotlib

### Running

```bash
python run_experiments.py
```

The script runs on CUDA by default. It will:
1. Generate a task visualization image (`task_vis.png`) by invoking `tasks.py`
2. Run the configured experiment phases
3. Save loss plots, evaluation bar charts, and mask GIFs to `images/`

### Configuration

Edit the top of `run_experiments.py` to swap components:

```python
Model = models.MaskedMLPSpatial   # or models.MaskedMLP, models.MLP
DataGen = tasks.DataGenSine       # or tasks.DataGen
MaskConfigurer = masks.SmoothKWTAFast  # or any configurer from masks.py
Config = config.Classification()  # or config.Regression()
```

Key parameters in `config.py`:

| Parameter | Default (Classification) | Description |
|-----------|--------------------------|-------------|
| `PHASES` | `[1, 2, 4]` | Which experiment phases to run (subset of 0-4) |
| `N_TASK` | 3 | Number of tasks |
| `X_DIM` / `Y_DIM` | 4 / 2 | Input / output dimensions |
| `SQUARE_DIM` | 32 | Side length of 2D neuron grid (hidden_size = 32^2 = 1024) |
| `K` | 128 | Number of winners in KWTA (active neurons per task) |
| `LAYERS` | 5 | Number of hidden layers |
| `TRAIN_DURATION` | 100 | Training steps per task per epoch |
| `N_EPOCHS` | 20 | Number of epochs |
| `TRAIN_LR` | 3e-4 | Learning rate (AdamW) |
| `N_SAMPLE_PER_TASK` | 20 | Replay buffer samples stored per task |
| `N_REPLAY_SAMPLES` | 1 | Number of replay samples drawn per training step |

### Outputs

All outputs are saved to `images/`:

- `loss_<phase>.png` -- Per-task training loss curves
- `eval_<phase>.png` -- Per-task evaluation bar charts
- `eval_over_time_<phase>.png` -- Evaluation snapshots across training
- `bw_<phase>.gif` -- Grayscale mask activation over time (EMA)
- `rgb_<phase>.gif` -- RGB-coded mask activation (each color channel = one task)
- `rgb_<phase>.png` -- Final frame of the RGB mask
- `ng_mask_allocation.png` -- Neurogenesis mask analysis (phase 4 only)
- `task_vis.png` -- Visualization of the synthetic task data

## Experiment Walkthrough

Results from a full run with the default `Classification` config (`SmoothKWTAFast` mask configurer, `MaskedMLPSpatial` model, 3 tasks, 20 epochs, K=128 winners on a 32x32 grid).

### Task Data

The synthetic tasks are generated by `DataGenSine`: 2D inputs are projected through a fixed encoder, masked by task-specific binary masks, passed through a sinusoidal nonlinearity, and decoded into binary classification outputs. Each panel below shows one task's decision boundary over the input space (black = class 0, white = class 1).

![Task visualization](images/test_experiment/task_vis.png)

---

### Phase 1: Multi-task Learning

All 3 tasks are trained block-sequentially. The mask configurer discovers spatially separated subnetworks for each task on the 2D cortical grid.

**Training loss** -- Per-task cross-entropy over 2000 steps (20 epochs x 100 steps). All tasks converge well below the 0.693 chance level.

<img src="images/test_experiment/loss_p1_mtl.png" width="320">

**Mask evolution (grayscale)** -- EMA-smoothed mask activations over training. Bright pixels are frequently active neurons. The mask starts diffuse and consolidates into compact, spatially contiguous regions as the lateral dynamics and KWTA enforce structure.

<img src="images/test_experiment/bw_p1_mtl.gif" width="128">

**Mask evolution (RGB)** -- Same mask data but color-coded by task: red = task 0, green = task 1, blue = task 2. Each task claims a distinct spatial region with minimal overlap, demonstrating the spatial separation property.

<img src="images/test_experiment/rgb_p1_mtl.gif" width="128">

**Final mask snapshot** -- The converged mask allocation at the end of multi-task training. The three colored blobs show that each task has settled into its own contiguous region of the cortical grid.

<img src="images/test_experiment/rgb_p1_mtl.png" width="128">

---

### Phase 2a: Continual Learning -- OOD Neuron Combinations

Evaluates (without training) on out-of-distribution mask configurations -- novel neuron combinations that weren't seen during MTL. This tests whether the learned representations generalize beyond the specific masks found in Phase 1.

**Loss during OOD evaluation** -- Loss stays near chance level (~0.7) since the model is being evaluated with unfamiliar mask configurations it wasn't trained on, which is expected.

<img src="images/test_experiment/loss_p2a_ood_neurons.png" width="320">

**Mask snapshot (RGB)** -- The OOD masks used during evaluation. These differ from the MTL masks, testing generalization.

<img src="images/test_experiment/rgb_p2a_ood_neurons.png" width="128">

---

### Phase 2b: Continual Learning -- New Data Generator

Evaluates on data from a different generator seed (same task structure, different random projection). This tests robustness to distribution shift in the input data.

**Loss during OOD data evaluation** -- Similar to phase 2a, loss is near chance since the data distribution has shifted.

<img src="images/test_experiment/loss_p2b_ood_data.png" width="320">

**Mask snapshot (RGB)** 

<img src="images/test_experiment/rgb_p2b_ood_data.png" width="128">

---

### Phase 4: Neurogenesis

The hidden layer is doubled from 1024 to 2048 neurons (32x32 to 64x32 grid). Two sub-phases are run: training new-generation tasks on the expanded model, then retraining the original tasks on it. This tests whether the system can incorporate new capacity without disrupting existing knowledge.

#### New-generation tasks (on expanded model)

New tasks are trained from scratch on the expanded 2048-neuron model. They are free to use any neurons, including the newly added ones.

**Training loss** -- All three new-gen tasks converge from chance level, training successfully on the expanded architecture.

<img src="images/test_experiment/loss_p4_ng_new.png" width="320">

**Mask evolution (grayscale / RGB)** -- The new-gen tasks discover their own subnetwork regions within the expanded grid.

<img src="images/test_experiment/bw_p4_ng_new.gif" width="128"> <img src="images/test_experiment/rgb_p4_ng_new.gif" width="128">

**Final mask snapshot** -- Converged mask allocation for the new-generation tasks on the expanded grid.

<img src="images/test_experiment/rgb_p4_ng_new.png" width="128">

#### Original tasks (retrained on expanded model)

The original MTL tasks are retrained on the expanded model. Because their old weights are preserved in the first half of the grid, they start with low loss and the mask configurer tends to reuse the original neurons.

**Training loss** -- The original tasks start near their previous loss values (not from chance), confirming that expanding the model preserves prior learning.

<img src="images/test_experiment/loss_p4_ng_orig.png" width="320">

**Mask evolution (grayscale / RGB)** -- The original tasks largely reuse their pre-expansion neuron regions.

<img src="images/test_experiment/bw_p4_ng_orig.gif" width="128"> <img src="images/test_experiment/rgb_p4_ng_orig.gif" width="128">

**Final mask snapshot** -- Converged mask allocation for the original tasks on the expanded grid.

<img src="images/test_experiment/rgb_p4_ng_orig.png" width="128">

#### Mask allocation analysis

The neurogenesis analysis shows where each task's mask falls on the expanded grid: old neurons (left half, from pre-expansion) vs. new neurons (right half, added during expansion).

- **New-gen tasks** use 100% new neurons -- they settle entirely into the freshly added capacity.
- **Original tasks** use ~89-100% old neurons -- they stick to their established subnetworks, with only minor spillover into new neurons.

Each row shows one task; columns are old neurons, new neurons, and the full combined mask on the expanded 64x32 grid.

![Neurogenesis mask allocation](images/test_experiment/ng_mask_allocation.png)
