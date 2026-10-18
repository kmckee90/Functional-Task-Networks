# Continuous density + model exploration

Four fresh seed-7 single-stream runs on GPUs 0, 1, 2, 6. Each uses density config
24 and immediate model-only ratio 256 (one batch-8, length-32 model optimizer
update per observation after replay warmup). Dreamer actor/critic are disabled.
The exploration policy is the existing RECODE density DQN with a frozen recurrent
reservoir. There is no environment step cap or automatic stagnation stop.

Model reward weights: 0.25, 1, 4, 16. This is a weight comparison with model
feedback enabled in all four arms, not a feedback ablation or replicated study.

## Novelty and temporal alignment

The ensemble predicts coarse next-image/body features from Dreamer state and
candidate action. Raw novelty is the square root of mean feature variance across
heads, computed before executing the selected action. This is a custom
Plan2Explore-inspired estimator; it is not full rendered-video variance.

For the executed action at t, raw disagreement is transformed using statistics
from earlier selected actions. Its reward is stored with transition t -> t+1.
Its scalar feedback enters the MF reservoir at t+1, alongside the existing
action/density feedback. Reset clears previous feedback and the pending
transition, but retains lifetime normalization statistics. No gradients cross
between the Torch explorer and JAX world model.

Input: tanh(raw / (3 * max(0.001, running RMS))). Range [0, 1]. Running second
moment starts at 0.01, uses EMA coefficient 0.001, and clips its incoming sample
to three times the previous RMS. This limits contamination by outliers.

Objective: 0.1 * clip(raw / reward_scale, 0, 3). reward_scale starts at 0.1 and
can only increase to the running RMS. It never shrinks as the model improves;
near-zero disagreement remains near-zero reward. No mean subtraction or
variance standardization. Maximum unweighted model bonus is 0.3. The run weight
is applied separately from the density/frontier multiplier. Historical replay
bonuses decay with a 10k-transition half-life. This decay is a heuristic for stale
uncertainty, not recomputation using the latest model. Replay stores collection-
time normalization; normalization state is checkpointed.

Raw, input, objective and weighted objective values are logged separately.

## Monitoring and storage

Pod state: /home/dev/globo_dreamer_state/runs/continuous_fb_*
Supervisor: /home/dev/globo_dreamer_state/continuous_runs/
Gallery: http://127.0.0.1:8765/continuous/ (requires the local port forward).
Launcher: bash /home/dev/workspace/run_continuous.sh on the pod, detached.

Current-trajectory videos overwrite obs.mp4 and recent.mp4 every 1k observations
or five minutes. Each has 64 real frames at 6 fps; world-model movie shows actual,
posterior reconstruction, and 32 context + 32 action-conditioned open-loop frames.
Metadata reports current model step, actual frame range and age. These are recent
training trajectories, not held-out evaluations. Old-lifetime sequences may be
shown temporarily until a new life has 64 frames; frame age remains explicit.

Keep one committed model checkpoint, current MF replay chunks, active Dreamer
replay chunks, and bounded verbose logs. Buffers: 1M MF transitions, 100k frontier,
10k RECODE centroids, 100k Dreamer replay. Compact per-1k progress and stagnation
histories intentionally persist. All runtime data lives outside source sync.

Stagnation: no newly visited 4m spatial bin for 5k observations; log estimated
onset, confirmation, recovery, and longest stall. It is not a semantic discovery
metric. Density novelty, sky/low-detail fraction and action entropy provide
additional context. Crashed workers resume checkpoints after backoff; restarts
are visible and reset the physical environment, so uninterrupted-lifetime claims
must be checked against the process-start log.

## Scaled cohort (2026-10-09)

Four additional fresh agents run indefinitely on GPUs 3, 4, 5, 7, with reward
weights 0.25, 1, 4, 16 respectively. They share the original seed, spawn, density
hyperparameters, training ratio and feedback/reward normalization. Differences:

- Reservoir: 128 channels (4,608 recurrent activations).
- RSSM: deterministic 2,048; hidden 1,024; stochastic 128 variables × 32 classes.
- Image embedding: 2,048, plus the unchanged 256-dimensional vector branch.
- MF replay: 2M transitions; frontier: 500k; RECODE: 100k centroids.
- Dreamer replay remains 100k; batch 8 × sequence 32; model-only ratio 256.

The same continuous supervisor and gallery manage all eight runs. Entries label
Baseline versus Scaled and show buffer capacities. Existing baseline workers are
adopted without restarting when the supervisor is reloaded. Large runtime files
remain under /home/dev/globo_dreamer_state, outside source sync.

A 3k-step capacity trial completed 2,745 model updates and video/checkpoint writes
with 95.03 GiB highest sampled total GPU usage and 44.77 GiB free on GPU 3 (including
its idle watcher). GPUs 4, 5, 7 additionally retain ~23 GiB each in old NFS-blocked
processes, so their remaining headroom is lower. No trial checkpoint seeds the
new agents; all four start fresh.
