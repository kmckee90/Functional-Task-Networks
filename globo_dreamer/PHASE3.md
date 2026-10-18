# Phase 3: paired exploration comparison

Six M2E runs and six convolutional FTN+M2E runs, all fresh and unlimited.
World seed 19; spawn `12.0,39.1,120.0`. Agent seeds 1901–1906, paired
between methods. `--seed` controls the learner; `--world-seed 19` controls
world generation independently. No previous weights, replay, or centroids.

Pairs 1–4 occupy adjacent GPU pairs on globo-dreamer; pair 5 runs on
globo1, pair 6 on globo2. Even GPU: M2E; odd GPU: convolutional FTN.
Existing Phase 2 M2E settings are retained, including 256px rendering,
downsampled inputs, tuned frozen reservoir, 2M replay, 500k frontier,
100k RECODE centroids, and config-24 exploration parameters. FTN uses
the same isolated convolutional decoder and reconfiguration as Phase 2.
There are no Dreamer components or survival/task rewards.

`phase3.py` defines assignments and starts the autonomous supervisors
and pod-specific publishers. `phase3_monitor.py` publishes the shared
monitor at `/phase3/`. Individual curves show all observations; thick
means and paired comparisons use the horizon shared by all 12 runs.
Shading is the six-run range, not a confidence interval. Do not interpret
repeated steps within a trajectory as independent statistical samples.
The six seed pairs are the units for comparison, on this one world.

Runtime/checkpoints: `/home/dev/globo_dreamer_state/runs/phase3_*` on
each owning pod. Published videos/stats: `/mnt/obelisk-shared/kmckee90/globo_phase3`.
The terrain cache/CPU reference mapper is shared with the archived Phase 2.
Checkpoints and videos overwrite/rotate using the existing bounded policy.
Stopping an old run was SIGTERM after disabling its supervisor, followed
by verification of fresh checkpoints and `running=false` status.
Old trials remain paused in `continuous_runs/state.json`; their pre-stop
metadata is retained in `continuous_runs/before_phase3.json` on each pod.

## Event video capture

Enabled only for FTN pairs 1–4 on globo-dreamer. Checkpoint/resume was
required to load the read-only game hooks. No reward/policy inputs changed.
Files stay in each owning run's `event_clips/`, outside source sync.
Each MP4 is 15 seconds at 12 fps, with the first observation containing
the event at 7.5 seconds. Frame timestamps use simulation time, with
sample-and-hold between available observations. A fixed spatial layout
from the trigger frame displays the recorded FTN mask EMA; the encoder
never consolidates or reconfigures the learner. Encoding runs in a CPU
thread, spilling queued work to pod disk if necessary.

Triggers: tree ascent (5 m from nearby actual ground within 30 game seconds,
also at least 5 m above ground beneath the tree, and 1 s on trunk/canopy
support); two connected ground removals near/below the feet
within 10 seconds (hole); placed block; player-attributed kill of any
species; inventory increase of fruit/tree products, items marked rare
or precious, or positive-health food (including explicit goldcap/coffee);
existing visible fruit harvest/consumption or bonus-object use events;
10 m and 20 m climb milestones; six consecutive entries into previously
unvisited 4 m cells. Remaining in a cell preserves the sequence, entering
an old cell or taking over 10 game seconds between entries resets it.
Long runs generate another event at each multiple of six new cells.

Simultaneous triggers share a clip; events at different observations
retain their own centered windows. Startup/death/stop can truncate the
available context: hold the first/last frame to preserve 15 seconds and
record padding explicitly in the JSON sidecar. `events.jsonl` indexes
saved clips and encoding errors. Automatic retention deletion is not
enabled. This is a recording layer, not new achievement/reward scoring.
