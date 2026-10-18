# Phase 4: frontier exhaustion ablation

Resumed again October 16, 2026 across all five Globo pods (18 GPUs total,
including six active Phase 5 runs). Phase 4 ownership is now:

| Pod | GPU assignments |
| --- | --- |
| globo-dreamer | M2E r1/r2 on GPUs 6/7; Phase 5 r1–r6 remain on 0–5 |
| globo1 | M2E r3/r4 on GPUs 0/1 |
| globo2 | M2E r5/r6 on GPUs 0/1 |
| globo3 | FTN-M2E r1–r4 on GPUs 0–3 |
| globo4 | FTN-M2E r5/r6 on GPUs 0/1 |

Migrated runs retain complete checkpoint, replay, density memory and history
under `/mnt/obelisk-shared/kmckee90/phase4_18gpu_migration/`, linked from their
new owner's normal runtime run directory. Original source copies are retained
as inactive backups; their entries are removed from the old supervisor and
publisher to prevent duplicate workers or stale publication. Runtime dependencies
on the new pods use the existing shared `globo_runtime_v1` installation.
The new pod images also require the system software-rendering libraries:
`libegl1 libgl1 libegl-mesa0 libgl1-mesa-dri libglx-mesa0 libopengl0`
plus the X11/audio dependencies. Verify `ldd software_egl.so` resolves all
libraries and `/usr/share/glvnd/egl_vendor.d/50_mesa.json` exists before launch.

All twelve runs were gracefully paused October 16, 2026 to let Phase 5 catch up.
Final checkpoints and replay remain in their original run directories.
M2E stopped at approximately 358k–405k observations; FTN-M2E at 296k–337k.
The /phase4/ monitor remains available with the saved results.

Resumed all twelve original runs on October 15, 2026 (October 16 UTC), after
stopping Phase 5. These are checkpoint continuations, not fresh trials: learned
parameters, masks, replay and density memory are restored. Each starts a new
game instance/life, retaining cumulative evaluation history. Original settings,
including frontier_rotation=False and unlimited steps, remain unchanged.
The live monitor is again /phase4/. The pre-continuation archive remains under
/mnt/obelisk-shared/kmckee90/globo_archives/phase4/.

Fresh repeat of Phase 3: six M2E and six convolutional FTN+M2E agents,
world seed 19, paired learner seeds 1901–1906, same GPU assignments.
All settings and architectures match Phase 3, including unlimited duration,
empty initial buffers, 256px rendering, downsampling, and event capture on
FTN pairs 1–4. No checkpoints are imported.

The sole behavioral change is `frontier_rotation=False`, passed to workers as
`GLOBO_FRONTIER_ROTATION=0`. This bypasses the bounded-dwell top-five random
frontier scheduler in scalable replay and restores the original density-ranked
frontier selection and goal-buffer training. Ordinary M2E already used that
selection. FTN mask reconfiguration, learning losses and replay remain unchanged.

Phase 3 agents were gracefully stopped with final checkpoints preserved;
their frozen monitor remains at `/phase3/`. Phase 4 is at `/phase4/`, with
identical charts, selectable aggregates, achievement table, maps, videos,
mask diagnostics, survival metrics and pooled event clips.

Launch: `python -m globo_dreamer.phase4 launch --pod POD` on each owning pod.
Monitor: `python -m globo_dreamer.phase4_monitor` on globo-dreamer.
Runtime data: `/home/dev/globo_dreamer_state/runs/phase4_*` on each owning pod.
Published data: `/mnt/obelisk-shared/kmckee90/globo_phase4`.

Validation: compared all 12 generated configurations against Phase 3 (only
name/phase and the rotation flag differ). A GPU smoke test completed FTN
training and frontier display using the original selector while replacing the
rotation chooser with a function that raises if called.

Event capture now records 80 consecutive observations before the event, the
event observation, and 80 afterward: 161 frames at 6 FPS (26.83 seconds).
Missing context at a life boundary is truncated, never padded with frozen
frames. Existing clips keep their original simulation-time timing. Only the
four capture-enabled FTN workers were checkpointed/resumed for this update.
