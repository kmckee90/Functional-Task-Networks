# FTN-Chat

Live, prompt-routed expert-knowledge chat demo. A frozen Qwen3-8B
backbone with a grid-arranged bank of LoRA atoms attached to the **last 25% of
decoder layers**; each user prompt is used as a self-supervised reconfiguration
batch (minimize prompt perplexity) to update a persistent BayesLoRA mask
posterior and select the expert atom subspace, which is
then visualized live as an animated square mask while the answer streams.

The same rule handles training batches and live prompts: a short persistent
posterior update (2 steps, temperature 3, KL weight 1) on stable inputs, and a
longer reset/refit (30 steps, temperature 3, no KL) when prompt NLL under the
previous mask crosses an EMA change threshold. Both use Gumbel top-k with an
exact-k forward mask. A small mask-filtered replay buffer helps retain earlier
knowledge. Bank size: H=32²=1024 rank-1 atoms; k is read from the checkpoint.
The GUI exposes the detector and mask search parameters.

Optional input-conditioned routing uses a separate `routing.pt` archive. At
training block boundaries, the trainer searches a mask for each learned task
using cached Q&A and accepts repairs only when answer-token cross-entropy
improves on separate validation Q&A. The app can then retrieve a repaired mask
from a question using discriminative identifiers and frozen-model token
embeddings; uncertain matches fall back to the ordinary Bayes search. The
app's route checkbox and threshold control this at inference. Matched routes
can be applied directly (the default) or refined for a configurable number of
prompt-NLL steps with KL to the retrieved logits. Neither mode needs an answer
or task label. On a routing miss, ordinary Bayes search runs instead.

Post-training is **context distillation** by default: the teacher
is the same frozen backbone with the source excerpt in context and the mask
zeroed; the student (masked bank, closed book) minimizes KL to the teacher's
answer-token distributions over synthesized Q&A. Optional source CE adds
next-token prediction on training-split source chunks under the same
question-selected mask. It never uses support or held-out chunks.

To mix source CE into an existing training run, add
`--source-ce-prob 0.1 --source-ce-weight 0.1` to `train.py` (and `--retrain`
for tasks already marked trained). This adds a source pass on 10% of Q&A steps;
it does not replace Q&A steps. Both settings can also be set in `config.py`.
The probability defaults to zero, so existing training is unchanged.

**Full spec / rationale / handoff:** see [`CONTINUATION.md`](./CONTINUATION.md).

## Domains

All sources are snapshotted into `corpus/` by `make_corpus.py` (the actl pod
syncs only this folder, so the surrounding repo is not available at runtime).
Synthesized Q&A is saved on the pod home PVC at
`~/ftn_chat_state/data_cache/qa/`, outside the synced workspace. Pretrained
model downloads use `~/.cache/huggingface/hub/`, also outside the workspace.

- `base_ftn` — `base/ftn.py` + `base/config.py` (FTN class + every experiment parameter)
- `explorer` — `explorer/agents/agent_recode.py` + RECODE + reservoir encoders
  (RDDPG_Dual architecture and its exploration method)
- `rl_mt10` — `rl/robotics/train_mt10_bc.py` + `train_mt10_ddpg.py` + their
  `ftn.py` (FTN adapted to behavior cloning and DDPG)

## Run

Everything runs on the actl pod `ftn-chat` (8×H200). Edit locally; sync
propagates. From the pod's `/home/dev/workspace`:

```bash
uv run python qa_synth.py rl_mt10            # once: this task's Q&A cache is missing
FTN_CKPT_DIR=~/ftn_chat_state/checkpoints_bayes uv run python train.py --fresh
FTN_CKPT_DIR=~/ftn_chat_state/checkpoints_bayes uv run python routing_check.py
uv run python app.py --ckpt-dir ~/ftn_chat_state/checkpoints_bayes
```

To extend an existing Q&A cache, set the desired **total accepted pairs** for
each task in `config.py`'s `qa_target_counts`, then run:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python qa_synth.py --resume base_ftn explorer
```

Resume preserves the existing JSON files, generates from train and support
source chunks, filters exact and close question/answer repeats against the full
cache, and saves atomically after each answer batch. Re-running skips tasks
already at their target. Synthesis stops with an explicit shortfall if the
source yields no new pairs for several rounds or reaches `qa_max_rounds`.
These lexical similarity checks reduce repetition but cannot prove semantic
uniqueness; inspect a sample before using a much larger cache for training.

To build the route archive for an existing bank without retraining its weights:

```bash
FTN_CKPT_DIR=~/ftn_chat_state/checkpoints_bayes uv run python train.py --build-routes
```

To maintain it during later training, pass `--input-routing`. Once an archive
exists in the checkpoint, subsequent training maintains it automatically.
Route repair reads synthesized Q&A from `data_cache/qa` and uses a separate
fit/validation split; `replay.pt` remains the mask-filtered weight-training
buffer. Only `routing.pt` is needed by the app. Building routes can take time
because repair differentiates through the frozen transformer several times.

The `base_ftn` and `explorer` Q&A caches already exist. To train just those
ready tasks, use `--tasks base_ftn explorer` with `train.py`. `--fresh` resets
only the selected checkpoint directory; omit it to resume an interrupted run.
The old base FTN bank lives in the separate default checkpoint directory.

## GPU placement

`Config.model_parallel=False` places the Qwen3-8B backbone and LoRA bank on
one H200 even when the pod exposes eight. The mask posterior remains one
persistent state in the same training process. An actual batch-16 training
benchmark (prompt update, teacher, student, optimizer) measured about 2.26 s
per steady step on one H200, versus 2.35–2.39 s with the model split across
2, 4, or 8. A longer 40-step comparison including replay measured 2.24 s
over the last five steps on one H200 versus 2.32 s on eight. One GPU used
about 55 GB on that run and 103 GB on a
synthetic batch near the configured token limits. Gradient checkpointing stays
enabled; disabling it ran out of memory. SDPA attention and a sparse hard-mask
bank path did not improve the full training step in these checks.

Set `Config.model_parallel=True` to restore balanced sharding across all
visible GPUs for a larger backbone or a memory-heavy dataset. The
[`bench_train_devices.py`](./bench_train_devices.py) reruns the comparison
without reading or writing training checkpoints.

## Adding a dataset (continual learning)

It's block-sequential CL: new tasks are learned on top of the existing bank.
The trainer occasionally stores one random Q&A example with its hard training
mask. Storage proposals are spread across the block by a low probability and
an eight-step minimum gap; a per-task reservoir holds at most 16 entries.
On replay steps, only stored masks with partial overlap with the current mask
are eligible. The replay example is distilled under its stored mask, then the
live mask is restored. Replay does not update the posterior or detector.

1. Add its source files to `make_corpus.py`, run `python3 make_corpus.py` locally.
2. Register it in `registry.py` (one line — a key, label, and corpus globs).
3. Synthesize + train just that block:
   ```bash
   uv run python qa_synth.py mydata
   FTN_CKPT_DIR=~/ftn_chat_state/checkpoints_bayes uv run python train.py --tasks mydata
   ```
4. Restart the app — the new task's color/label appear in the legend automatically.

Other knobs: `--epochs N --steps M` (depth), `--reconfig-steps S`, `--retrain`
(re-run an already-trained block), `--fresh` (wipe and start over). State lives
OUTSIDE the synced workspace in `~/ftn_chat_state/checkpoints_bayes/` for the
commands above. Without `FTN_CKPT_DIR`, the default is
`~/ftn_chat_state/checkpoints/`. The checkpoint contains
`{bank,refs,mask,posterior,replay}.pt` + `state.json`; the latter persists the
prompt NLL detector. The trainer checkpoints after every epoch, so it's
resumable. Older bank checkpoints remain loadable. Since they
lack posterior logits, their first new support batch starts a cold fit. The
app keeps its posterior in memory across prompts and can initialize it from
the latest training checkpoint; serving does not overwrite that checkpoint.
Use a separate `FTN_CKPT_DIR` for BayesLoRA training to preserve the existing
base FTN bank. That bank can be loaded for compatibility checks, but quality
under the new search should be judged after BayesLoRA post-training.

To view the app from your laptop, forward port 7860 through actl (the pod has
no tailscale; never use the public gradio tunnel — the data is in-house).
Either restart your sync session with the port:

```bash
actl pod up --port 7860        # then run app.py in its remote shell
```

or, without disturbing a running sync session, from a second terminal:

```bash
actl pod exec --port 7860      # opens a pod shell with the forward
```

and open http://localhost:7860 in your browser. The forward lives only while
that actl session stays open.

All knobs live in `config.py`.

### 27B training memory (2026-10-08)

On the existing actl pod, Qwen3.6-27B in BF16 with eager attention,
gradient checkpointing, H=2304, k=256, rank=1, and banks on the last quarter
of layers has 4.20B trainable atom parameters. The backbone is frozen before
installing the atoms; this preserves backpropagation into the adapters while
preventing unused gradients on 20.8B previously unfrozen backbone parameters.

Batch 4 passed three consecutive training updates on one H200, peaking at
116.1 GB of PyTorch allocated memory. The probe used maximum-length synthetic
answers (1024 student / 1680 teacher tokens), source CE on every update, and
replay on updates two and three. Batch 8 ran out of memory on the second
update in the full-vocabulary distillation KL. Batch 4 is the current default;
this is a short memory stress test, not a completed training run. Reducing
batch size also reduces examples seen per fixed number of updates.

Reproduce in the pod shell (does not load or save checkpoints):

```bash
CUDA_VISIBLE_DEVICES=0 FTN_BENCH_BATCH=4 FTN_BENCH_AUX=1 \
  FTN_BENCH_STEPS=3 FTN_BENCH_SYNTHETIC_MAX=1 \
  uv run python -u bench_train_devices.py
```

The hybrid model's TileLang backward path required `nvcc`; the missing
`cuda-nvcc-12-6` package was installed on this pod. That system installation
must be retained in its image or repeated after recreating the runtime.
Use a separate checkpoint directory when switching from an 8B backbone to
27B; their atom-bank shapes differ.

### 27B chat prefix and routing diagnosis

The app passes its thinking toggle to `apply_chat_template`. Qwen3.6-27B's
default assistant prefix already opens `<think>`; banning generated thinking
tags did not disable reasoning and also prevented its closing tag. Explicit
`enable_thinking=False` closes the thinking section before answer generation.
Existing training still uses the default template in `data._chat_prompt_ids`,
so checkpoints trained that way learned answers inside the thinking section.
The serving fix does not retrain those checkpoints or establish factual recall
under the changed prefix.

Input-route keyword anchors may be shared across tasks when their smoothed
document frequency is at least three times higher in the candidate task.
Previously, requiring complete exclusivity rejected even strongly favored FTN
questions because the robotics material also mentions FTN. The overall score
margin gate remains active. Verified on `checkpoints_bayes_qa`: the question
"What is the purpose of the `reconfigure` method in the FTN class?" now matches
`base_ftn`; RDDPG_dual and MT10 probes retain their intended routes, and lexical
probes for greetings and weather are rejected.

`diagnose_chat_modes.py` is a read-only GPU probe for the route scores and
generation with both template modes. Logs from this investigation are on the
pod under `~/ftn_chat_state/diagnostics/chat_modes_27b*.log`.

### Bayes lateral masks

Bayes search now applies SmoothKWTA lateral dynamics to every candidate and
final mask, including route refinement. Routing repair uses the same selection
rule and saves its validated hard mask as saturated logits; direct input
routing restores that exact mask without applying smoothing again.

`lateral_steps <= 0` or `lateral_polyak >= 1.0` bypasses smoothing entirely,
including its noise and convolution. Otherwise the initial top-k mask undergoes
the configured number of box-sum convolutions, mixed with its preceding
activity using `polyak * activity + (1-polyak) * smoothed`. Intermediate activity
stays continuous; final top-k preserves the active count. `torus` controls
circular versus zero padding. Search retains its sigmoid straight-through
gradient and KL objective. This changes the sampled mask distribution from
plain Gumbel top-k when enabled.

New checkpoints record the lateral settings in `state.json`; the app restores
them before loading the model. Earlier Bayes checkpoints ignored these knobs
and should be retrained to learn under spatial masks. Reference-map rendering
is unchanged. Run `uv run python test_lateral_bayes.py` on the pod for bypass,
legacy parity, gradient/rollback, and repaired-mask recall checks.
