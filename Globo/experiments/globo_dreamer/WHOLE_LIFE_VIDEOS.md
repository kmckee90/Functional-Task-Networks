# Entire-life recording

Enabled for all eight resumed `lifetime_v2` runs on 2026-10-23, on the
`globo-actions` and `globo1` four-GPU pods. Existing checkpoints and training
settings are retained.

`continuous_runs.py` enables the streaming recorder when a trial has
`life_capture: true`. Every observation is encoded, with the observation,
current frontier goal, and FTN activity. Playback is 6 FPS, not a measurement
of game-clock time. Encoding uses bounded memory rather than keeping a whole
life in RAM.

Videos and metadata live outside source sync at:

`/mnt/obelisk-shared/kmckee90/globo_lifetime_v2/runs/<run>/life_videos/`

An active recording has a `.partial.mp4` suffix. On death, the recorder closes
the file and writes JSON metadata. The monitor's **Entire-life videos** section
then displays it. Only videos with more than 600 seconds of game time are
retained. The monitor also prunes shorter finalized recordings from recorders
that were already running when this policy was introduced. Active partial
files are left untouched. Lives interrupted by shutdown are labeled
partial and follow the same duration threshold. Recording begins with this continuation; past lives cannot be
reconstructed from the old event clips. Files accumulate on shared storage,
not the local workstation.

The resume helper is `python -m globo_dreamer.resume_lifetime_recordings --pod
<pod>`, run with the existing pod venv and package path. It requires four
paused runs with checkpoints and refuses a live supervisor.

Fresh containers need the Luanti runtime libraries, including `libluajit-5.1-2`
and the X11/OpenGL libraries, plus FFmpeg for the existing video diagnostics.
The venv supplies PyAV for streaming recordings.
