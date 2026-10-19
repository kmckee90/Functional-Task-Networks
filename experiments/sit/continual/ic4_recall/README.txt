IC4 recall: recovered shared implementation
=========================================

This directory is a dependency of union11, memory_budget, and memory_refinement.
Keep its Python sources even when removing old experiment artifacts.

Recovery (2026-10-07)
---------------------
common.py, recognition.py, search.py, fixed_conditioner.py, and
atom_conditioner.py were recovered from original source-writing patches in
the local Codex session records, including subsequent unified-feature and
numerical-fit corrections. This was not a reconstruction from memory alone.

Primary record:
~/.codex/sessions/2026/10/06/
rollout-2026-10-06T09-42-49-01a11218-ac81-75b2-894a-e1f7919a2374.jsonl

Shared identity sampling/generation helpers were recovered using:
~/.codex/sessions/2026/09/29/
rollout-2026-09-29T16-32-26-01a0ef83-2e3b-7a43-becc-a5eaa34053c3.jsonl

Compatibility sources were also restored under identity_distinct and
mixed_anteater. The latter provides the CLIP wrappers and evaluator labels
used by union11.metrics. It is not the complete old mixed-object project.
identity_distinct/common.py binds directly to the surviving data root rather
than requiring the deleted preparation program.

universal.py is a new reusable API around the recovered implementation.
verify_recovery.py is a new read-only checkpoint compatibility check.
Historical training, dataset caches, checkpoints, and reports were not changed.
Deleted old experiment outputs and preparation scripts were not recreated.
Legacy experiment CLIs in recovered modules may require those old outputs;
use the universal API or the current union11/memory_budget entry points.

Unified direct4 architecture
----------------------------
Every input image uses the same three frozen branches:
  DINOv2 ViT-B/14: 768 dimensions.
  Fixed color/chroma/edge statistics and histograms: 104 dimensions.
  CASIA-WebFace InceptionResnetV1: 512 dimensions, MTCNN face crop with
  whole-image fallback. These are separate from VGGFace2 evaluation weights.

Normalize each branch, concatenate, divide by sqrt(3): 1384 dimensions.
Average the four support-image vectors.
Append ReLU(x @ P)/sqrt(128), with fixed Gaussian P of shape 1384 x 128,
seed 93811, and a constant bias: 1513 features total.
Linear readouts score the 1024 distributed rank-one LoRA atoms; select top 256.
There is one readout per atom, not a classifier selecting one stored mask.
Inference takes support images alone, without a supplied task identity.

Fit uses repaired teacher masks as +/-1 atom targets, 512 four-image episodes
per association, ridge penalty 0.01, and a float64 joint normal-equation solve.
In memory_budget the fit uses only retained images from completed blocks.
The encoders and projection are fixed. The readout weights are refitted;
fixed encoders alone do not guarantee retention of the learned readouts.

API (run on the SiT pod, from /home/dev/workspace)
------------------------------------------------
from continual.ic4_recall.universal import FrozenEncoder, load_readout
encoder = FrozenEncoder()
mapper = load_readout(path_to_saved_conditioner)
features = encoder(four_pil_images)
mask = mapper.mask(features.cuda()[None], 256)[0]

fit_readout(groups, teachers) exposes the checkpoint-compatible fitting step.
groups contains combined per-image features and teachers contains binary
atom masks with matching association keys. The caller controls which memory
is eligible; this API does not enforce an experiment's retention policy.

Validation
----------
Run only on the pod:
/home/dev/sit_state/env/bin/python -m continual.ic4_recall.verify_recovery

The recovery_verification.json file records the successful checks:
  Current training/evaluation/refinement modules import successfully.
  All 880 final direct4 masks match the historical records exactly:
    four union11 models and four memory-budget models, 110 masks each.
  The seeded projection matches all eight saved conditioners exactly.
  Refitting the final 5% model yields max absolute weight error 0.0.
  Features re-extracted for 20 images spanning all four data families agree
    with caches (minimum cosine > 0.999999); style features match exactly.
    Small neural-feature differences occur with different batch sizes.

This checks recovery compatibility, not new generation quality. No model
training or new quality comparison was performed during recovery.

Runtime assets remain under /home/dev/sit_state, outside source sync.
The original dependency environment is /home/dev/sit_state/env; frozen
encoder weights and feature caches remain under /home/dev/sit_state/ic4_recall.
