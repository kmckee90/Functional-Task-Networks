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
├── base/                 base synthetic continual-learning experiments
│   │                     (classification / regression / MNIST knowledge
│   │                     transfer; primary lineage = func_net_kt)
│   └── streams_project_c/  real-world concept-drift streams (ELEC2, energy)
├── rl/                   reinforcement-learning applications
│   ├── robotics/         FTN-DDPG on MetaWorld MT10 + grid DQN/PPO
│   ├── atari/            FTN-DQN (PyTorch)
│   └── atari_jax/        FTN-DQN (JAX)
├── nlp/                  natural-language / transformer + LoRA applications
│   ├── ftn_lora/         FTN-LoRA: bank of LoRA atoms on a frozen LLM
│   ├── ftn_chat/         live perplexity-routed expert-chat demo app
│   └── lora_mux/         LoRA-multiplexing style-transfer experiments
└── manuscripts/          papers (arxiv / neurips / icml formatted LaTeX)
    └── spatial_drafts/   earlier func_net_spatial drafts (v4/v5, ICML, CATS)
```

## Modularity

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
`configs/`, `gifs/`. None of these are committed (see `.gitignore`). Source,
small config/table assets, and LaTeX live in git; generated data does not.

> **Note (follow-up):** `rl/robotics` already routes its artifacts into
> `output/{results,images,configs}/`. Other subprojects still write to their
> historical `results/` and `figures/` folders (git-ignored); pointing them at a
> single `output/` root is a straightforward later cleanup.

## Environment

Python via `uv` (`uv run <script>`). Most experiments assume a CUDA device set
once at entry (`torch.set_default_device("cuda")`).
