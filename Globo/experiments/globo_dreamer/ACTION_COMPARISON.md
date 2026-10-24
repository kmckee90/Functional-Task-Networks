# Reduced-action FTN-M2E

Started 2026-10-19 on `globo-actions`, four H200 GPUs, world 19.
Four fresh runs use agent seeds 1901–1904, one per GPU, without a step limit.
The reference is all six archived Phase 4 convolutional FTN-M2E runs.

`train.py --reduced-actions` keeps IDs 0–13: idle, movement/look, dig/attack,
and place/interact. It removes sneak, sneak+place, and inventory selection.
It changes both the environment and policy action dimensions to 14.
Run directories record the action mapping and reject incompatible resumes.

The comparison fixes `--m2e-ftn-k 296`, retaining Phase 4's 2,368-neuron bank,
depth/width and reservoir. Otherwise the FTN budget allocator would increase
the number of neurons when the action decoder became smaller. The output head
necessarily has fewer parameters. Remaining settings come from the same
continuous-run launcher/config24 as Phase 4; frontier rotation remains disabled.
Training has no extrinsic reward and uses 256px renders downsampled to 64px.

Runtime locations (on pod, outside source sync):

- `/home/dev/globo_dreamer_state/runs/actions14_conv_r{1..4}_s19`: checkpoints, replay, logs.
- `/home/dev/globo_dreamer_state/continuous_runs/state.json`: supervisor run list.
- `/mnt/obelisk-shared/kmckee90/globo_actions`: published results and monitor.
- `/mnt/obelisk-shared/kmckee90/globo_phase4`: read-only historical reference.

Monitor: `http://localhost:8014/` through the pod's port forward. It compares
spatial coverage and unique achievements, with means and min/max bands clipped
to the shortest trajectory across both conditions. Six historical vs four new
runs are explicitly identified; legend controls allow selecting matched seeds.
Current runs include observation/goal/FTN video, maps, achievement checklists,
survival diagnostics, and stagnation history. Videos overwrite themselves.
Phase 4 event-triggered clips are also enabled and retained in each run's
`event_clips/` folder, displayed in the monitor's saved-event gallery. These
include up to 80 observations before and after an event at 6 FPS. They were
enabled after initial launch; `event_capture_enabled.json` records the resume
step per run. Earlier events have no retrospective clip. Retained event clips,
unlike the rolling videos, accumulate on pod storage outside source sync.

Fresh deployment (pod only, after runtime setup):

```bash
export PYTHONPATH=/home/dev/globo_dreamer_state/packages:/home/dev/globo_dreamer_state/vendor/dreamerv3
/home/dev/globo_dreamer_state/venv/bin/python -m globo_dreamer.action_comparison launch
```

The launch command refuses an existing experiment list. The existing
`continuous_runs` supervisor resumes each run from its own checkpoint after
a worker failure. Publisher and monitor can be run separately with
`action_comparison publish` and `action_comparison monitor --port 8014`.
Do not start duplicate supervisors or monitor writers.

## Fresh corrected-engine comparison — 2026-10-20

The previous four runs were stopped and checkpointed at 406,728 / 423,970 /
422,160 / 447,004 observations. Their run directories and published monitor
remain under the original paths above. The supervisor list is archived at
`continuous_runs/actions14_before_engine_v2.json`.

Four new runs, `enginev2_14_conv_r{1..4}_s19`, start entirely untrained with empty
buffers on globo-actions. Seeds, architecture and training settings match the
previous four. Environment changes: native input protocol v2 and animal HP
capped at 20. No step limit. All animal kills trigger event clips.
Both globo1 continuation agents were also stopped with checkpoints preserved.

`engine_comparison monitor` serves the replacement monitor on port 8014.
Published data lives in `/mnt/obelisk-shared/kmckee90/globo_engine_v2`;
runtime checkpoints remain in `/home/dev/globo_dreamer_state/runs`.
Spatial and achievement charts compare four old vs four new runs using means
and min/max bands, clipped to the shortest shared duration. This tests both
environment changes together. Start the publisher using `engine_comparison publish`.

Single-click placement was enabled at checkpoint steps 10,086 / 13,680 /
12,290 / 10,624. Learned state and buffers were resumed. Each run records this
boundary in `single_place_restart.json`. The separate native binary is
`engines/input-v2-single-place/luanti`; PLACE lasts one frame per command.
Block stacks that previously held fewer than four now hold four; the engine's
stack capacity is per item type (so this also applies in slot 2). No starting
blocks are granted. Human exact-parity's four-block grant remains human-only.
