# Globo Dreamer: first integrated experiment

**Current experiment:** `density_bootstrap` restores the existing model-free
RECODE DQN from `globo_explorer/agent.py` as the primary collector while Dreamer
learns from that same real stream. The early objective is informative observation
collection and improving world-model predictions. Survival is logged, not used
as the primary early-stage success criterion.

One sequential environment stream. The agent starts without demonstrations or
pretrained weights. Real trajectories train DreamerV3; imagined trajectories train
the behavior. There are no parallel training worlds.

Source lives here. **Every command below runs on the actl pod**, from
`/home/dev/globo_dreamer_state`, with `PYTHONPATH=/home/dev/workspace` and
`PYTHONDONTWRITEBYTECODE=1`. Environments, virtualenv, vendor dependencies,
videos, replay, density memory, checkpoints and logs live below that state root,
outside automatic source sync. `GLOBO_STATE` can override the root; runtime paths
inside the source tree are rejected.

Upstream Dreamer: `danijar/dreamerv3`, commit
`e01491fad6434b2245a3b8ca201dd7faedcc458c`. Installed separately under
`/home/dev/globo_dreamer_state/vendor/dreamerv3`; no upstream source is modified.

## Current agent

- 64x64 first-person RGB plus normalized health, food, water, energy, temperature,
  breath and inventory quantities. World position is **diagnostic only**.
- 25 discrete motor/interaction primitives, including move+turn and move+jump,
  dig/use, place/use, sneak and slot selection. No scripted navigation policy.
- NumPy port of the existing RECODE weighted-kNN memory, using fixed 8x8 RGB
  block means. Online novelty is measured before insertion. Count/centroid memory
  and its RNG state survive episode resets and process restarts.
- Five independently initialized, bootstrap-trained, action-conditioned predictors
  of next image block means and bodily state. Disagreement is computed over means,
  not over stochastic samples from the RSSM. It is a Plan2Explore-style proxy,
  not a calibrated information gain measurement.
- Actual density feedback and per-action disagreement enter the policy. In
  imagination a density head supplies predicted feedback and intrinsic reward.
- Separate survival and intrinsic critics and return normalization. The survival
  reward is -1 on death and zero otherwise. Intrinsic reward combines density and
  current disagreement. Continuation handles death; time limits are not death.
- All model weights, replay and lifetime memory are checkpointed. On process
  restart the simulator starts a new life; exact physical-state resume is not
  implemented. This is one agent across sequential lives, not parallel universes.

Limitations of this first iteration: density-head targets are collection-time
novelty, so its predictions can become stale as memory grows; feedback in dreamed
rollouts is approximate. FTN options and high-level planning are not yet inserted.
Learned behavior must be evaluated from traces/videos and repeatable outcomes,
not assumed from a falling loss. `--no-feedback` removes the direct policy
feedback features only; density remains part of the RSSM observation.

`--replay-value` restores upstream Dreamer's survival-value objective on real
replay returns. The initial pilot omitted this term and trained both values only
in imagination. This flag changes the training objective without adding parameters,
so existing checkpoints remain compatible. It does not add health shaping or
change the death-only environmental reward.

## Run

```bash
cd /home/dev/globo_dreamer_state
export PYTHONPATH=/home/dev/workspace
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Engine/telemetry check. Video is scripted, not learned.
venv/bin/python -u -m globo_dreamer.train --smoke --name interface_check --steps 300 --episode-steps 150

# Small full learner check, including real transitions and gradient updates.
CUDA_VISIBLE_DEVICES=0 venv/bin/python -u -m globo_dreamer.train \
  --name learner_check --size size1m --batch 4 --length 16 --train-ratio 2 \
  --steps 350 --episode-steps 150

# First actual learning run, with no artificial episode timeout.
CUDA_VISIBLE_DEVICES=0 venv/bin/python -u -m globo_dreamer.train \
  --name pilot --steps 50000 --size size12m --train-ratio 64 --episode-steps 0

# Read-only behavioral summary while the agent runs.
venv/bin/python -m globo_dreamer.report runs/pilot

# Audit completed lives and compare stored novelty with current saved memory.
venv/bin/python -m globo_dreamer.audit runs/pilot
```

Reusing a run name restores its checkpoint and density memory. `--steps` is a
cumulative target. Keep the model configuration unchanged when resuming.
A file lock prevents duplicate writers. SIGTERM/Control-C saves at the next
collection-loop boundary. Latest checkpoints overwrite older generations through
upstream Elements (keep=1), rather than accumulating full models indefinitely.

Inspect `runs/<name>/status.json`, `metrics.jsonl`, `agent_feedback.jsonl`, and
`environment/{trace,episodes}.jsonl`. Finalized annotated MP4 clips live alongside
those environment files. Initial clips show blank-slate behavior; later clips
are sampled periodically. Video playback is fixed-rate; trace game time is the
authoritative simulation duration.

The pilot disables Globo's text HUD, which otherwise covers the small camera
image. Numeric bodily observations remain available. It also measures simulation
time from the engine's accumulated globalstep clock; the preliminary
`interface_check` and `learner_check` used a last-frame time that undercounted
elapsed simulation time, so their `game_seconds` fields are not comparable.

## Engine decision

The existing single-world Warpcraft adapter was probed on the pod. It reached
approximately 104 steps/s for 100 idle steps after warmup, but startup reported
CUDA compilation failures in fire, mob, survival and light kernels. This is not
evidence that all survival mechanics ran correctly. Craftium is the first-run
backend to avoid blocking the agent experiment on that engine investigation.
Warpcraft remains a candidate once relevant mechanics and runtime compatibility
are checked. A GPU is not reserved merely to keep all four H200s busy.

Next behavioral milestone: identify the first learned movement/interaction
pattern and its failure mode, then use those trajectories to decide whether the
next change belongs in control, exploration, FTN skill retention, or prediction.

## Iteration gates

1. Verify the camera, control, bodily observations, replay, finite updates and
   recovery on the pod. These checks establish operation, not learned survival.
2. Inspect single-stream behavior and compare against independent random and
   single-signal runs without sharing their experience with the pilot. Track
   survival time, spatial coverage, resource interactions and repeated failures.
3. Introduce FTN first in the policy/option repertoire once repeatable behaviors
   exist. Use trajectory support to recover overlapping masks; retain successful
   and exploratory behaviors with replay, and measure retention versus an
   unsegmented policy. Avoid assigning a new mask to every noisy transition.
4. Learn option termination, outcome, duration and survival cost. Plan over these
   learned outcomes, with short-horizon primitive control as fallback. Test
   recombination on new support before claiming abstract or OOD planning.
5. Add FTN to dynamics residuals only if observed context interference/forgetting
   warrants it. Keep ensemble uncertainty distinct from mask-selection
   uncertainty, and evaluate both prediction quality and behavioral benefit.

First-run simplifications include fixed clear midday lighting, a fixed spawn,
pixel-block density features and a small discrete motor interface. Climate still
affects survival. These are initial experimental conditions, not a demonstration
of broad survival competence or a finished continual-learning architecture.

## First observed failure and correction

The original pilot was stopped cleanly at 44,800 observations after 11,129
updates. It reached 108 diagnostic spatial cells, but later behavior favored
turning/lateral motion around the spawn and repeatedly died from terrain damage.
At the 27-life audit, the first five lives averaged 132 seconds and 24.8 cells;
the last five averaged 45 seconds and 6.8 cells. Falling prediction losses did
not imply better survival. These are descriptive comparisons from one run.

An audit of 104 replay images found mean collection-time novelty 0.094 versus
0.131 against the current saved memory. This does not support stale optimism as
the sole explanation; image-based novelty and memory drift remain limitations.

The pre-correction checkpoint, density memory and run metadata are preserved on
the pod at `runs/pilot/phase1_snapshot`. The same agent resumes toward step 70,000
with `--replay-value`, keeping the other learning settings unchanged. Phase
boundaries are recorded in `runs/pilot/phases.jsonl`. Simulator restart starts a
new physical life. Diagnose the new phase with `--after-step 44800`; end-of-life
filtering is intended for this boundary, which falls between complete lives.

```bash
CUDA_VISIBLE_DEVICES=0 venv/bin/python -u -m globo_dreamer.train \
  --name pilot --steps 70000 --size size12m --train-ratio 64 \
  --replay-value --episode-steps 0
venv/bin/python -m globo_dreamer.audit runs/pilot --after-step 44800
```

Run only one writer; if the pilot is already active, inspect it instead of
launching this command. `status.json` counts updates since the most recent
process start, whereas agent step is cumulative. The optimizer and model
counters are restored from the checkpoint.

`random_reference` is a separate 10,000-step random-action reference using the
same seed, interface and survival rules. Its experience is never shared with the
learner. Small samples and engine timing variability limit comparisons. The next
decision depends on survival and coverage after the correction, including the
reward model's prediction on actual terminal observations. FTN integration remains
gated on identifying behaviors worth retaining.

## Model-free density collection and model diagnostics

The original integration made a mistake in emphasis: density entered the policy
as feedback but its actor objective was learned through imagined density rewards.
It did not implement the original explorer's direct model-free density objective.
`--model-free` now reuses that DQN, including its frozen visual reservoir and
feedback, online novelty, refreshed offline novelty and frontier replay.

The DQN selects 90% of actions and Dreamer selects 10% in the initial experiment.
This is an explicit starting mixture, not a learned competence gate. Dreamer's
density and disagreement objectives remain enabled. Both learners receive the
actual executed action, including in Dreamer's recurrent state. The DQN does
not need forecasts or imagined rewards. Its survival reward coefficient and death
penalty are zero for this early collection phase.

The DQN's RECODE memory is the single authoritative novelty source for both
learners in this mode. It uses the existing Torch implementation with 16x32
grayscale image features (2,000 atoms in the initial experiment; configurable,
20,000 by default for new long runs). The older pilot used 8x8 RGB features
and a NumPy port, so reward magnitudes across these experiments are not directly
comparable. For compatibility, the observation key `density` stores the
**novelty distance**: larger values indicate less-dense memory support. Novelty
is computed before insertion. Weights, optimizer, density, main/frontier replay
and RNG state are checkpointed together. Recurrent rollout state resets with the
new physical life on process restart.

```bash
CUDA_VISIBLE_DEVICES=0 venv/bin/python -u -m globo_dreamer.train \
  --name density_large_gpu0 --model-free --dreamer-action-probability 0.1 \
  --spawn 11,5.1,121 --size size12m --train-ratio 64 \
  --mf-replay-capacity 4000000 --mf-goal-capacity 500000 \
  --recode-capacity 20000 --episode-steps 0 --video-every 4000
```

This starts both learners from scratch on one real experience stream. Dreamer
and the model-free explorer share logical GPU 0. Set `CUDA_VISIBLE_DEVICES` to
one distinct physical GPU per experiment, with a unique run name. `--steps 0`
(the default) runs until stopped. The scalable replay adapter handles ring
wraparound, truncates trajectories at resets, and ends frontier trajectories at
the selected frontier rather than the end of its episode. Random replay now
samples random transition endpoints instead of only episode tails. Goal chunks
are explicitly separated. These are sampling fixes relative to the old run.

Both buffers store full-precision reservoir features. Main capacity is 200 times
the initial 20,000 entries; goal capacity is 25 times larger. A transition costs
20,498 bytes (two 64x4x9 float32 states, a 512-float embedding and metadata).
The 4M+500K buffers occupy approximately 85.9 GiB. On the H200, an explorer-only
allocation/training probe with 6M+500K and 100K populated RECODE atoms reached
124.65 GiB peak; this is a tested explorer capacity, not a joint-training limit.
The 4M+500K default leaves more room for Dreamer and compilation workspaces.
8192-query offline novelty scans measured 0.019s at 20K atoms and 0.063s at 100K
atoms on synthetic data. Queries use chunks of 128, and both replay frontier
searches sample at most 8192 rows. A sampled frontier can miss the global maximum.

Dreamer's image replay remains host-side, default 100K observations, configurable
with `--dreamer-replay-capacity`; it does not consume these GPU replay budgets.
RECODE capacity is a separate algorithmic choice, not just an allocation limit:
increasing it delays reaching the full clustering/eviction regime.

Checkpoint interval defaults to 900 seconds. Immutable 2048-row replay chunks
under the run's `mf_replay` directory are rewritten only when dirty. Superseded
chunks are removed only after the enclosing checkpoint commits. CPU staging is
bounded by chunk size, weights are CPU-portable, and unused GPU slots are never
saved. Resume requires the same capacities and preserves all replay. At full
capacity each run needs about 86 GiB of replay storage, plus Dreamer replay,
models and videos; four full runs need over 344 GiB. All artifacts remain under
`/home/dev/globo_dreamer_state`, outside source sync. The current PVC's requested
size is 50 GiB, even though its shared NFS backing reports more free space.

The spawn audit found `nodes_nature:tiken` (50 damage/second) at the original
`8,8,115` starting coordinate, with adjacent damaging plants. A 600-step idle
probe lost 400 health there. The nearby sand clearing `11,5.1,121` had no health
loss during a 600-step idle verification. These diagnostics do not change world
mechanics or supply geometry to the policy.

`world_probe.py` evaluates frozen checkpoints on 32 freshly collected, withheld
32-frame sequences: 16 context frames followed by 16 open-loop predictions using
the actual future actions. These are fresh trajectories from the same seed and
region, not evidence of broad OOD generalization. It compares posterior
reconstruction, open-loop prediction, last-frame persistence and deliberately
wrong-action rollouts. Stochastic latent sampling remains enabled.

| Checkpoint | Posterior image MSE | 16-step forecast MSE |
|---|---:|---:|
| Random initialization | 0.03961 | 0.04284 |
| 44,800 steps | 0.00530 | 0.01739 |
| 70,000 steps | 0.00516 | 0.01275 |
| Last-frame persistence | — | 0.03081 |

The 70,000-step checkpoint beats persistence by 31%, 39%, 50% and 59% at horizons
1, 4, 8 and 16. The advantage remains on the textured-frame subset, so flat sky
does not explain the result. Wrong-action 16-step error is 0.03544. Visual review
shows coarse scene/camera-motion prediction but substantial blur and missing
fine geometry; low pixel error does not establish semantic prediction competence.

Pod artifacts: `diagnostics/world_probe/{scores.json,ckpt.mp4}` and
`diagnostics/{spawn_check,clear_spawn_check}/`. Evaluation files stay outside sync.
`test_model_free.py` checks repeated-view novelty suppression without a world
model and correct executed-action/episode-boundary replay bookkeeping. A 650-step
integration check and 100-step resume trained both learners and restored state.

New traces also record image texture, intensity entropy, a blue-sky fraction
heuristic, and camera pitch. These are diagnostic proxies, not rewards or model
inputs. `report.py` summarizes low-detail/sky-dominated frame fractions and
upward/downward fixation. Demonstrating that the learned collector reduces those
fractions is the next live behavioral check; passing the reward tests alone does
not establish that result.

## Immediate experiment objective

The first milestone is sustained informative exploration: successive windows
should keep discovering new territory/views without collapsing onto sky, ground,
or stationary action loops. Survival is not the early success criterion. Use
`report.py`'s nonoverlapping `progress_windows`: first-visit spatial cells are
counted against the entire trace, so repeated respawns do not create discoveries.
Read these alongside novelty, low-detail/sky fractions and world-model validation;
spatial coverage alone misses discovery through interaction and new viewpoints.

The existing collector uses epsilon-greedy actions: epsilon falls from 1.0 to
0.025 over 10,000 transitions and retains that nonzero floor. In the current
mixture Dreamer supplies 10% of executed actions; its density and ensemble
objectives remain active while the model-free collector supplies the rest.
There is currently no learned model-competence gate. Early training-loss reduction
alone is not evidence that prediction novelty helps exploration. Evaluate that
with withheld rollouts and matched behavioral comparisons as data accumulates.

Keep the current large buffers while they fit. The user-approved fallback for
this initial phase is 1M main transitions, 100K frontier transitions and 10K
RECODE centroids. Do not silently shrink a resumed run: replay and density
checkpoint shapes currently require matching capacities.

## Four-GPU hypothesis sweep

Keep seed 7, single sequential world per agent, 4M/500K replay, 20K RECODE,
train ratio 64 and 10% Dreamer actions fixed. Baseline is `density_large_gpu0`.
Variants are `epsilon10_gpu1` (`--mf-epsilon-end .10`), `offline025_gpu2`
(`--mf-online-coef .25`) and `horizon097_gpu3` (`--mf-gamma .97`). Each changes
one explorer setting. The reward reweighting leaves online RNN feedback enabled.
Use `python -m globo_dreamer.compare runs/<name> ...` for matched-age windows.

Do not rank by raw training reward across differently weighted objectives.
Compare newly visited cells, low-detail and sky fractions, and original density
novelty over multiple windows. Check beyond the 10K epsilon decay and 20K RECODE
capacity transition. Consider a run lagging only after repeated poor windows
at matched ages, with process/simulator health checked first. Keep checkpoints
of retired candidates. Winners need seed replication and world-model validation;
spatial discovery is a useful proxy, not proof of semantic repertoire growth.

First sweep review (2026-10-06): at matched 20K observations, unique 4m cells
were baseline 133, epsilon .10 174, online weight .25 133, gamma .97 209.
These are single-seed trajectory outcomes, not statistically established effects.
At about 34K the baseline discovered only one new cell across five complete 1K
windows, while mean novelty stayed around .10-.12. It continued moving within
old territory, with frequent left turns, upward gaze and deaths/respawns. This
motivated retiring it gracefully, retaining its final checkpoint, and replacing
GPU 0 with `frontier2_gpu0` (`--mf-frontier-scale 2`), all other baseline settings
unchanged. Hypothesis: a smaller frontier bonus reduces fixation on isolated
locally novel views. It is not yet established that bonus strength caused the stall.

Second sweep review: gamma .97 reached 326 spatial cells around 38K observations,
adding 17 in five complete 1K windows, with mean low-detail fraction under 2%.
Epsilon .10 reached 180 cells but added only 2; its energy reached zero, so
exhaustion is a confound (it was still moving locally). Online weight .25 reached
138 cells around 37K and added zero over five complete windows despite ongoing
movement and repeated deaths/respawns. Retire that run with its checkpoint intact.
GPU 2 replacement: `horizon097_eps10_gpu2`, gamma .97 plus epsilon floor .10,
otherwise baseline settings. Compare against the retained gamma .97 run to test
whether persistent randomness helps longer-horizon exploration. No reliable
self-sustaining cycle has been established yet. `frontier2_gpu0` remains younger
and continues while crossing the same milestones.

## Autonomous 8-GPU search (40K observations per trial)

The new pod is actl project **globo-dreamer**, with this directory synced directly
to `/home/dev/workspace`. `legacy/` contains the required explorer source and
13 MiB of fixed game assets. All generated worlds, packages, checkpoints, replay,
videos and search history live under `/home/dev/globo_dreamer_state` outside sync.
Run `bash /home/dev/workspace/setup_pod.sh` on a fresh pod. The isolated Python
3.11 venv uses tested CUDA 12 PyTorch/JAX wheels on the CUDA 13-capable driver;
the image's preinstalled Python/CUDA environment is not modified.

The screened seeds are 7 (sand), 19 (grassland), and 101 (woodland), each with a
recorded safe spawn in `seed_screen/selected.json`. Screening uses invincibility
only to locate terrain, then a **fresh, normally vulnerable** environment for
400 idle and 300 movement observations. All three idle tests were damage-free;
woodland lost 50 health during movement. Screening frames never enter training.

Launch on pod, then disconnect freely:

```bash
nohup bash /home/dev/workspace/run_search.sh \
  --steps 40000 --gpus 8 \
  > /home/dev/globo_dreamer_state/search40k-launcher.log 2>&1 < /dev/null &
```

Each worker receives exactly one physical GPU and one sequential environment.
Trials have 1M main replay, 100K frontier replay and 10K RECODE atoms. Every
configuration is evaluated independently on all three seeds. Initial hypotheses
cover gamma .90/.97, epsilon .025/.10, frontier bonus 5/2, and online coefficient
1/.25. Subsequent configurations mutate one or two settings of fully evaluated
elites, with 20% broad random proposals. Mutation scale expands/contracts with
three-seed improvement. Horizon, epsilon floor, online coefficient, frontier
bonus, Dreamer action fraction and DQN learning rate are searchable. There is no
finite generation limit. Empty GPU slots are filled independently.

Objective version 1 emphasizes discoveries during the **last 10K observations**:
new spatial cells plus new detailed visual views, productive-window frequency,
and low-detail/sky penalties. Visual views are sampled every 10 observations and
clustered for evaluation with a fixed RMS RGB threshold .06 on 8x8 RGB features.
These evaluation features and world positions never enter the policy or reward.
Raw intrinsic reward is reported but does not determine search ranking. Three-seed
scores are baseline-adjusted, averaged, and penalized by .25 times their standard
deviation. These are proxies, not semantic achievement or survival benchmarks.

`search40k/leaderboard.json` reports active GPUs and fully evaluated rankings.
`search40k/state.json` contains durable parameters, assignments, RNG and results;
`events.jsonl` records launches, completions, failures and pruning. The launcher
restarts a crashed supervisor; the supervisor adopts surviving workers, retries
failed trials up to three launches, and terminates workers that make no progress
for 20 minutes. Training resumes from checkpoints, with a new physical life.
The final checkpoint must be complete before a trial is accepted. A clean stop
of the supervisor stops/checkpoints workers and exits the launcher.

Completed heavy artifacts are retained for the top two fully evaluated
configurations and the six most recent terminal trials; incomplete configurations
and active trials are protected. Other completed runs retain small summaries,
metrics, parameters and logs while their worlds/replay/checkpoints are pruned.
The search never prunes unrelated or historical runs. This bounds the number of
heavy retained trials; storage grows temporarily while seed groups finish.
Full observations remain recorded for retained winners. Scores do not establish
robustness beyond these three seeds or beyond the 40K exploration horizon.

To stop gracefully, signal the PID in `search40k/launcher.pid`; to resume, run the
same launch command. Do not start a second launcher while the first is active.

Overnight recovery note (2026-10-07): 77 trials completed in roughly nine hours.
Two attempts remained in kernel D state (`nfs_lookup_revalidate`) even after
SIGKILL. The supervisor now quarantines such attempts only when uninterruptible
sleep AND a pending SIGKILL are verified, never reads/reuses their artifact
folders, and starts a fresh attempt in a new directory. At least 40,000 MiB GPU
headroom is required before further launches while quarantined processes exist.
The underlying kernel/NFS wait is not cured by this workaround; blocked tasks
can retain memory until the I/O returns or the pod/node is restarted. Quarantined
partial attempts are never scored or promoted, and their data is preserved.

## Stage 2: fixed config24 explorer, 60k Dreamer novelty search

The 40k search was stopped with 90 completed trials. Its state, rankings, journal,
config24 and source snapshot are preserved in
`/home/dev/globo_dreamer_state/archives/search40k_closed`. Retained `runs/s40_*`
artifacts remain in place and are never touched by the new search. Earlier
automatic pruning already removed some nonwinning checkpoints; all trial
parameters, metrics and summaries that existed at closure remain available.

`search60.py` starts fresh 60k lifetimes on the same seeds 7/19/101. It fixes all
config24 model-free settings, including gamma .9036665304, epsilon floor
.0153669002, online density coefficient .8728610896, frontier multiplier
4.5749983907, learning rate .0001899493, offline coefficient 1, and the
Dreamer action fraction .1337261838. Buffers, train ratio and model size match
stage 1. The model-free policy continues learning; its hyperparameters are fixed.

Search parameters concern the imagined intrinsic reward and its policy:
disagreement weight 0 or .1–100, disagreement exponent .5–2 (1 means ensemble
standard deviation, 2 means variance), expected prediction-error weight 0 or
.1–30, intrinsic actor weight .05–1, and intrinsic discount horizon 20–300.
Imagined trajectory length stays 15; discount horizon is not rollout length.
Dreamer's density reward coefficient stays 1. Initial candidates include the
unchanged config24 reward, density-only reward, stronger disagreement, variance,
expected error, combined error/disagreement, and actor/horizon variations.
After these, adaptive mutations and 20% global proposals continue indefinitely.

The error novelty is a learned **expected residual**, not access to a future real
frame during planning: a state/action head learns the detached one-step RMS error
of the ensemble mean predicting next-image 8x8 grayscale means and eight body
features. Training excludes resets. Features and residual targets are detached;
only the executed action's error estimate is supervised. It shares the existing
Dreamer RSSM features and can also reward unpredictable noise, so density-only
and disagreement-only controls are retained. No intrinsic signal is inserted
into the survival reward head. Weighted components and observed/predicted errors
are logged for diagnosis.

Objective version 2 uses the same fixed spatial/visual novelty diagnostics as
stage 1, evaluated over observations **40,001–60,000**, plus explicit final-10k
counts. Rankings compare three-seed results against config0 (config24 rerun for
60k), penalizing variability. Scores from the two stages are not directly ranked
together. These are discovery proxies, not proof of meaningful achievements.

On the pod:

```bash
nohup bash /home/dev/workspace/run_search60.sh \
  > /home/dev/globo_dreamer_state/search60k-launcher.log 2>&1 < /dev/null &
```

The independent launcher uses `search60k/launcher.pid`, state/leaderboard/journal
in `search60k`, and `runs/s60_*`. It inherits crash recovery and top-two/recent-six
heavy-artifact retention only for this new stage. Signal its launcher PID to stop.

Config24 world-model diagnostics are in `diagnostics/config24_seed{7,19,101}_canonical`
and linked from `reviews/config24/world_seed{7,19,101}`. Each uses 32 fresh,
scripted 32-observation sequences, withheld from training, at the correct seed
and screened spawn. Videos compare actual frames, posterior reconstruction, and
16-step open-loop forecasts following 16 context frames and the recorded actions.
Metrics include initialization, trained weights, persistence, wrong actions and
a textured subset. The gallery shows four fixed probes and a textured example
selected using only real-frame variation. All large files remain outside sync.
The canonical probes use the same explorer RECODE preprocessing as training,
with separate, freshly initialized evaluation memory. Earlier diagnostic outputs
are retained separately; gallery links point to the corrected canonical probes.

## Model-only replay-volume study

The 60k novelty evolution is stopped. Its metadata and config6 reference are in
`archives/search60k_closed`; retained trial artifacts stay in place. Config6 was
the provisional two-seed leader, not a validated three-seed winner.

`run_model_study.sh` launches eight fixed 60k-observation experiments, all seed7:
GPUs 0–3 use model replay ratios 256/512/1024/2048 from initial replay readiness;
GPUs 4–7 use the same ratios only AFTER collecting 20k observations. No catch-up
debt is accumulated. Batch 8 × length32 means 1/2/4/8 model optimizer updates per
eligible new observation. Initial replay needs 256 observations. The delayed
group therefore receives 40k/80k/160k/320k updates by observation60k; early runs
receive approximately 60k/120k/240k/480k. `model_updates` is checkpointed, so
restarts resume the cumulative budget without double counting completed updates.

Only encoder, RSSM, decoder, reward/continuation, density and disagreement heads
are optimized. The actor, critics, slow critics and imagined actor training are
excluded, not merely multiplied by zero. The same density explorer settings
from config6/config24 are used, but Dreamer's action probability is zero. Thus
the model-free agent continues collecting data independently of model quality;
model novelty is measured rather than fed into a new controller in this study.
No previous trained model is loaded: this measures learning from a blank model.

Every 2k observations, plus initialization and exit, read-only inference on the
fixed 32 held-out sequences writes actual/reconstruction/open-loop videos and
MSE at horizons1/4/8/16, persistence/wrong-action comparisons, and disagreement
versus actual auxiliary forecast error correlation. These data never enter
replay. Snapshots are in each `runs/wm_*/model_progress`; the page
`reviews/config24/models/index.html` shows current observation/update counts,
the precise snapshot step, and historical videos. The page refreshes every
minute only when videos are paused. Data, checkpoints and videos stay outside
source sync. Engine worlds/templates use per-run `/tmp/globo_model_engines`
scratch to avoid hot NFS metadata access; checkpoints and diagnostics remain
on the persistent home volume.

Launch on pod with `nohup bash /home/dev/workspace/run_model_study.sh` (redirect
output to the state directory). State and launcher/supervisor PIDs live in
`model_study`. Worker failures resume checkpoints up to three launches; this
fixed experiment does not mutate reward settings or prune previous searches.

The model-study gallery now also shows the two latest actual-observation clips
per agent (recorded every10k observations and on process restarts). These are
separate from the fixed held-out evaluation videos.

`trajectory_watch.py --run <name>` runs on the same pod with the run's GPU exposed.
It watches saved checkpoints and renders the newest complete within-life
32-frame sequences from that agent's saved replay. It never creates or resets a
training environment. A temporary hard link pins just the model checkpoint while
loading; replay is read-only. Videos and exact chunk/offset provenance are saved
under `runs/<name>/trajectory_progress` and automatically shown in the gallery.
These recent-trajectory renderings update approximately every10minutes with
checkpoints, and are NOT held-out quality tests: the frames may have been used
for training. The displayed model observation count belongs to the checkpoint;
frame provenance is separately recorded rather than inventing absolute frame
indices across replay recovery. Fixed held-out diagnostics remain every2k steps.

## Plan2Explore and achievement evaluation (opt-in, not yet launched)

`--plan2explore` selects the JAX implementation in `plan2explore.py`. It runs
only the exploration agent indefinitely when `--steps 0` is used. It does not
load TensorFlow or train a task-reward policy. The reference is the original
ramanans1/plan2explore commit `f8ba930bf7d1c6b2a6aa7685d26e5f8335c129da`,
default `modeltype_1`: five independently initialized two-layer ReLU predictors,
width equal to the image embedding size, action concatenated at each layer,
sequence bootstrap with replacement, detached inputs/targets, and mean ensemble
variance over image-embedding dimensions. `--p2e-reward-scale` defaults to the
original 10000; there is no square root, reward clipping, or density reward.
The predictor uses the RSSM deterministic state before incorporating the current
image and the preceding action. Imagined rewards are aligned to arriving states.

This is Plan2Explore on DreamerV3, not a numerical replication of the TF1 agent:
it retains our DreamerV3 categorical RSSM, discrete REINFORCE actor, return
normalization, continuation prediction, and combined optimizer. The encoder's
image features are the target; body inputs remain available to the world model.
The reward head is diagnostic, detached from the representation and unused by
the actor. Start a fresh run: old pixel-statistics ensemble checkpoints are not
compatible. Supervisor trial option: `plan2explore: true`; do not combine with
`ftn`, `model_free_only`, or `dreamer_baseline`. No trial is created automatically.

`--achievements` (supervisor: `achievements: true`) enables evaluation-only
instrumentation at launch. Six achievement types are tracked: direct player
killing blows on wolves/cows, successful dig/place callbacks, successful egg
laying, and active hydration from freshwater or water drops. Passive rain and
hydrating food do not count as drinking water. Blocks and animal variants have
separate detail counters. These diagnostics are excluded from observations,
replay inputs, and rewards. Repeated placements/digs count repeatedly; the
distinct-achievement chart measures breadth separately (maximum six types).

Counters persist in the run's environment directory outside sync, independently
of model checkpoints. Startup frames are excluded. Each game's monotonically
increasing counters are differenced to avoid counting repeated telemetry; new
lives accumulate into run totals. Histories include instrumentation onset,
environment step, first-success steps, and totals. A model checkpoint can lag
environment telemetry, so the monitor retains both training and environment
step provenance. Existing uninstrumented runs are shown as unavailable, never
backfilled with zeros. New achievement charts appear when the supervisor code
is next loaded; the current supervisor and workers have not been restarted.

Validation runs only on the pod, on CPU with GPUs hidden:
`PYTHONPATH=packages:vendor/dreamerv3 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" venv/bin/python -m unittest globo_dreamer.test_achievements globo_dreamer.test_plan2explore -v`
The Lua callback tests use the test dependency `lupa`.

## Phase 0 archive and fresh phase 1

Phase 0 is the preliminary work, archived on October 12, 2026. Run `./rehost.sh`
on the laptop to host both `/continuous/` (current phase) and `/phase0/`
(frozen preliminary monitor). Both URLs use the port printed by the script.
The archive keeps independent copies of the final videos, histories, and
charts, so resuming an old run cannot change its archived display.

All large files remain on the pod under `/home/dev/globo_dreamer_state`:

- `phases/phase0/`: source snapshots, package versions, configurations,
  final metadata, inventory, and links to retained experiment directories.
- `reviews/config24/phase0/`: frozen monitor, videos, logs, and inventory.
- `runs/<original-name>/`: original checkpoints, replay, density memory,
  achievement counters, and spatial history. Do not delete these directories;
  the archive references them for continuation instead of duplicating replay.
- Prior searches, model-training studies, diagnostics, and benchmarks remain
  in their original directories, linked from `phases/phase0/`.

To list resumable phase-0 configurations, run **on the pod**:

```bash
cd /home/dev/globo_dreamer_state
PYTHONPATH=packages venv/bin/python -m globo_dreamer.resume_phase0
```

To resume one on an unused GPU, append its original run name and `--gpu N`.
The helper checks scheduling conflicts and checkpoint presence, queues the
original configuration, and reloads only the supervisor. Existing training
workers continue. Check `runs/<name>/console.log` for checkpoint loading and
the current monitor for progress. Resume restores learned state and saved
replay; the environment starts a new game instance/life, not an exact saved
simulator state. Source snapshots preserve the implementation if future code
changes make an older checkpoint incompatible; the helper uses current code.

Phase 1 starts six fresh unlimited runs on seeds 19 and 101, at the same spawn
coordinates as phase 0. No phase-0 checkpoint or replay initializes them.
M2E uses the tuned 128-channel model-free explorer, 2M replay, 500k frontier,
100k RECODE centroids, and config 24. Dreamer+Novelty repeats the prior scaled
Dreamer-only disagreement objective and architecture; P2E repeats its size12m
embedding-disagreement setup. Thus Dreamer and P2E retain their previous,
different model sizes; this is not an architecture-matched ablation.
Both model-based methods use train ratio 256. All six log achievements from
the beginning and provide the same live curves, checklists, and video controls.
M2E has observation videos only; model-based runs additionally reconstruct
and predict their actual recent trajectory.

Initial GPU assignment: M2E 2/3, Dreamer+Novelty 4/5, P2E 6/7 (seed 19/101
respectively). GPUs 0 and 1 are reserved for experiments. The supervisor now
uses only its explicit state list and never automatically fills spare GPUs.

The supervisor also checks progress timestamps. After 30 minutes without a
status update (or since process startup), it requests a graceful stop. If the
worker remains blocked for another five minutes, it forces termination so the
normal retry can resume its checkpoint. Events are recorded in
`continuous_runs/watchdog.jsonl`. This is a process-health safeguard, not an
exploration-stagnation stopping rule; it does not change learning objectives.

Warm-start rendering experiment: `phase1_m2e_render256_s19` and
`phase1_m2e_render256_s101` use GPUs 0/1, same seeds/spawns as their Phase 1
parents. `--render-size 256` renders 256² RGB, BOX averages to 64², then uses
the unchanged M2E area resize to 32×64. Raw mouse actions are scaled by 64/256
to preserve turning increments. Default direct 64² rendering is unchanged.
`--mf-init-weights` loads only Q/target networks and fixed reservoir/feedback
parameters when no own checkpoint exists. Optimizer, replay, frontier, RECODE,
recurrent state, counters and environment are fresh. Epsilon starts and stays
at its configured minimum (config 24: 0.015366900171141888). Own checkpoints
subsequently resume normally. Immutable initialization weights and source-step
metadata live at `$GLOBO_STATE/warm_starts/m2e_render256/`; new checkpoints and
artifacts live under each new run, never in the source run or synced workspace.
The bottom monitor cards pair the original 256² render with a 64² display tile:
32×64 RGB above 32×64 projected feedback. The actual reservoir input is 4×32×64
(channel concatenation, not vertical); tanh feedback is shown on [-1,1].
Overwritten `live/inputs.npz` stores exact float reservoir inputs and feedback
vectors aligned with the latest video. No accumulating frame archive is made.
