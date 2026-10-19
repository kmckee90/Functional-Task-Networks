# FTN-M2E in Doom

Current deployment: Doom 1, starting at E1M1, using the user-provided
`wad/Doom + Doom II Collection WAD files/doom.wad`. All four agents now start
from scratch in Doom 1: no Freedoom weights, optimizer state, replay, frontier,
RECODE memory, FTN masks or learning schedules are reused. The prior transferred
runs are archived separately on the pod. Frontier resampling remains enabled.
WAD transitions are recorded in each run's events.jsonl. Map names are read from the IWAD;
Doom episodes progress through their ordinary maps, skipping secret maps.

Run on the ViZDoom pod only. Source is synchronized through actl. Runtime is
`/home/dev/vizdoom_state`, outside the synchronized workspace. The engine is
the official ViZDoom 1.3.2 wheel; no C++ changes. The Python explorer core was
copied from `globo_dreamer` on 2026-10-16, preserving Phase 4's density frontier,
FTN convolutional atoms, masked replay, and spatial mask visualization.

Frontier resampling is enabled: choose among up to five distinct high-novelty
candidates, favor alternatives not recently used, and increase switching
probability with dwell time. A distinct alternative forces a switch by 1,000
interactions. This uses the existing FrontierRotation implementation.

Each process runs one independent, synchronous lifetime. No reward from the
engine reaches the learner. Death ends a transition and restarts the map;
successful exits advance maps (pistol starts). No episode time limit.
Originally Freedoom 2 MAP01; skill 3, seeds 19/101/202/303. WAD selectable with `--wad`.

The renderer produces 640x480 RGB including the HUD. Area downsampling produces
320x240 agent input. A frozen 128-channel reservoir uses 16x16 input patches,
15x20 recurrent grid, 20% recurrent connections, and exact linear spectral
radius .99 on this grid. This extends the Globo configuration to a larger
grid; it does not claim that HUD reading has already been learned. Health
and selected-weapon ammo are also bounded scalar observations in the feedback
projection. Health: clip(h/200,0,1). Ammo: clip(log1p(a)/log1p(600),0,1).
The monitor's input video shows actual RGB and projected feedback separately.

Actions enumerate compatible movement/strafe/turn/fire/run combinations,
plus use and weapon selection; 4 engine tics per decision (35 tics/second).
Turning uses Doom's digital turn buttons, retaining native turn dynamics;
movement inertia is also native. Action catalogue is saved in config.json.

Replay: 100k transitions; frontier: 20k; RECODE: 100k centroids. The larger
visual representation requires substantially more replay memory than Globo.
50 offline updates per 50 interactions after 500 transitions; batch 64;
two FTN mask reconfiguration steps per training batch and every 8 interactions.
The atom bank is rounded up to a square sheet near eight times the active K,
so the spatial mask display remains a genuine two-dimensional layout.
No step cap. Fresh initialization unless the run directory has a checkpoint.
Checkpoint includes weights, optimizer, density memory, replay, masks and RNG.
Resume starts a new episode on the saved map, retaining learning memory.

Videos overwrite obs.mp4 and input.mp4; checkpoints atomically replace the
prior checkpoint and remove superseded replay chunks. Metrics append to JSONL;
monitor uses a bounded recent history. Graceful stop: SIGTERM the learner.
Create runtime/STOP before stopping all supervisors. Never start a second
learner in the same run directory.

From the pod workspace:
```
nohup bash ftn_m2e/supervise.sh 0 19 > /home/dev/vizdoom_state/supervisor0.log 2>&1 &
/home/dev/vizdoom_state/venv/bin/python -m ftn_m2e.monitor --root /home/dev/vizdoom_state
```

Monitor port: 8766. Use a port-forward to view it locally. Full-resolution
video, FTN mask EMA, density-view frontier, novelty curves, throughput, memory,
deaths, and update freshness are visible. No spatial cells or achievement maps.
