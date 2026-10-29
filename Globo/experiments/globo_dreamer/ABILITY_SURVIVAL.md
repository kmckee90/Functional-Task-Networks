# Four-slot inventory, egg laying and sleep

Eight fresh runs on globo-actions (world 19) and globo1 (world 101), agent seeds
2901–2904 on each. No inherited weights, buffers, memories or checkpoints.
The previous natural-survival runs were checkpointed and stopped, and remain
available as references. Their artifact directories are not reused.

Changes from that batch:
- 18 actions: old reduced 14 plus slot selection 1–4; stationary jump, no sneak.
- Collected items stack to at most two. Single-item tools retain capacity one;
  the egg ability counter and sleep tool are exempt. No starting blocks.
- Egg and sleep support both primary and secondary use, including right-click
  into air. Sleep is idempotent and reports already-rested state. Existing
  resource, location and waking rules are retained.

Unchanged: gamma .90, 50 offline updates, full novelty scales plus death penalty
equal to the goal bonus, hard survival, original animal health, day/night/clouds,
speed 2.5, acceleration 2, memory carried across lives. Unlimited run duration.

Deploy: `python -m globo_dreamer.ability_survival launch --pod POD`
(globo-actions first, then globo1; refuses existing run directories).
Monitor: port 8019. Artifacts/checkpoints/videos stay outside source sync in
`/mnt/obelisk-shared/kmckee90/globo_ability_survival`.
Life-video threshold inherits the previous setting and only affects recordings
when they finish; existing saved videos are never deleted by the monitor.

Video HUD update: subsequent live, event and whole-life clips include a compact
42-pixel footer with raw HP, hunger, thirst, energy, temperature, air and four
inventory counts. Yellow indicates the selected slot. Values are captured at
the same observation as each frame; the footer is display-only, not policy input.
Workers were checkpointed and resumed to load this recording change; interrupted
lives are not counted as deaths. Historical life/event videos are unchanged.

## Higher-discount comparison

Four additional fresh runs: globo2 GPUs 0–1 on world 19; globo3 GPUs 0–1 on
world 101. Agent seeds 2901–2902 on each world match the first two original
runs. Gamma .995 in both Q streams; otherwise identical settings, including
50 offline updates, full novelty scales, death penalty, 18 actions, hard
survival and video HUD. Separate `abilities18_g995_*` directories; no warm start.
These appear as separate per-world groups on the same port-8019 monitor.
