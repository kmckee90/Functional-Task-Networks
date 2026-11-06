# Functional Task Networks (FTN)

A research monorepo for **Functional Task Networks**: a continual / multi-task
learning approach built on a fixed **NeuroModel feed-forward backbone** whose
behaviour is specialised per task by a learned **binary mask with
reconfiguration** (top-k / KWTA over hidden units). The same two ideas — the
masked backbone and the reconfiguration search — are recycled across
classification, regression, reinforcement learning, and language modelling.

## Layout

```
functional_task_networks/
├── src/                  shared, recycled core
│   ├── model/            NeuroModel FFN, mask classes + reconfiguration,
│   │                     synthetic task/data generators, default config, utils
│   └── vis/              shared visualizations:
│                           • plotting.py  — eval-result matrices (forward
│                             transfer + backward recall, "both" variants)
│                           • vis_ftn.py / viz_latents*.py — binary-mask EMA
│                             rasters (purple↔blue) and latent demos
├── ftn/                  core FTN and continual-learning experiments
│   │                     (classification / regression / MNIST knowledge
│   │                     transfer; primary lineage = func_net_kt)
│   └── streams_project_c/  real-world concept-drift streams (ELEC2, energy)
├── m2e/                  original M2E explorers, environments, and experiments
├── Globo/                shared Craftium game, WarpCraft alternative, and experiments
│   ├── craftium/game/    authoritative Lua game source
│   ├── warpcraft/explorer/ inactive JAX explorer
│   └── experiments/     current globo_dreamer and original explorer
├── experiments/
│   ├── mnist/            restored manuscript MNIST suite, SupSup/Progressive, and controls
│   ├── rl/               FTN-DDPG robotics and Atari DQN (PyTorch/JAX)
│   ├── nlp/              FTN-LoRA, expert chat, and LoRA multiplexing
│   ├── nar/              neural algorithmic reasoning experiments
│   ├── options/          option learning and planning prototypes
│   ├── sit/              continual diffusion experiments + sit dependency
│   ├── doom/             FTN-M2E agent + vizdoom dependency
│   ├── clrs/             experiment notes + clrs dependency
│   ├── e2e/              independent E2E experiment repository
│   └── ftn_robotics/     independent robotics experiment repository
└── manuscripts/          papers (arxiv / neurips / icml formatted LaTeX)
    └── spatial_drafts/   earlier func_net_spatial drafts (v4/v5, ICML, CATS)
```

## Modularity

The `functional-task-networks` pod syncs this entire repository's source to
`/home/dev/repo_source`. Start/reconnect it with `bash connect.sh` locally
(defaults to a small CPU profile; pass `--profile` when compute is needed).
Generated data, checkpoints and media are excluded. The retained environment is
`/home/dev/workspace/.venv/bin/python`; the old checkout is preserved.
For example, on the pod:

```bash
PYTHON=/home/dev/workspace/.venv/bin/python bash /home/dev/repo_source/ftn/run.sh ftn/run_experiments.py --help
PYTHON=/home/dev/workspace/.venv/bin/python bash /home/dev/repo_source/m2e/run.sh experiment_dm_control.py --help
```

M2E uses this same repo-wide workspace. Its connect helper delegates to the root
helper. Project-specific pods can still sync their own folders independently.

`src/` holds the genuinely shared core. Each application folder is a
**self-contained subproject** that currently keeps its own working copy of the
model/mask code so it runs unchanged — these are intentionally faithful copies
pending a later dedup pass against `src/`. Project-specific pieces
(FTN-DDPG, FTN-DQN, FTN-LoRA, the chat app) live in their own folders.

The canonical shared modules were taken from `func_net_spatial` (the most
complete model/mask/vis lineage); `src/vis/plotting.py` is the `func_net_kt`
superset of the eval-matrix plots.

## Outputs

Anything an experiment produces is written into **organized, git-ignored**
folders: `output/`, `checkpoints/`, `results/`, `figures/`, `images/`,
`gifs/`. New generated files are ignored (see `.gitignore`); existing tracked
publication assets remain tracked. Source configuration directories are retained. Source,
small config/table assets, and LaTeX live in git; generated data does not.

> **Note (follow-up):** `experiments/rl/robotics` already routes its artifacts into
> `output/{results,images,configs}/`. Other subprojects still write to their
> historical `results/` and `figures/` folders (git-ignored); pointing them at a
> single `output/` root is a straightforward later cleanup.

## Environment

Python via `uv` (`uv run <script>`). Most experiments assume a CUDA device set
once at entry (`torch.set_default_device("cuda")`).
