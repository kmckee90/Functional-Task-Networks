# Initial deployment — 2026-10-16 UTC

Pod: `actl-ws-kmckee90-vizdoom-devspace-7844db76b7-qkdg8` (4 H200).
Runtime: `/home/dev/vizdoom_state`; all training and validation ran remotely.
Monitor: pod port 8766, forwarded to localhost:8766.

- Custom integration tests: 2 passed, checking simultaneous controls, bounded
  telemetry, direct MAP01 startup, rendered HUD, zero engine rewards, movement,
  death and reset. No menu interaction required.
- GPU smoke: 510 interactions, 50 FTN updates, mask-filtered replay active,
  finite metrics, video encoding, atomic checkpoint.
- Resume smoke: continued to 560 interactions and 100 updates with replay,
  optimizer and density state restored. New episode boundary applied.
- Final square-sheet smoke: 510 interactions and 50 updates, successfully saved.
- Four fresh production runs: seeds 19/101/202/303 on GPUs 0/1/2/3. All passed
  1,200 interactions and 700 updates with current video feeds. No smoke checkpoint
  was loaded into these runs. No maximum interaction count.
- At step 650: about 20.3 GiB allocated per learner, 113.6 GiB GPU memory free.
  Replay/goal feature arrays allocate capacity up front; optimizer/FTN replay
  allocations and CUDA caches can grow beyond that initial measurement.
- Monitor and video GETs returned HTTP 200. Checkpoint access returned 404.
  Manually inspected rendered gameplay, HUD, FTN spatial EMA and frontier panel.

Full cloned upstream test suite against installed ViZDoom 1.3.2 wheel:
427 passed, 272 skipped, 4 failed in 392.67 seconds. All failures concerned audio
buffer sizes (`test_get_state` plus three Gymnasium wrapper audio-size cases).
This runner directly uses DoomGame, with no audio observations or Gymnasium
wrapper. No claim that the full upstream suite passed. Full output remains at
`/home/dev/vizdoom_state/engine-tests.log`.

The larger reservoir uses the same spectral normalization procedure on its new
15x20 grid. Memory/decodability on this grid has not been separately benchmarked.
Likewise a visible HUD and explicit health/ammo feedback do not establish that
the agent has learned to interpret them. These are fresh exploration runs.
