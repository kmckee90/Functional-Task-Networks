# Fresh natural-cycle hard-survival experiment

Eight fresh FTN-M2E agents; no exploration or survival checkpoint initialization.
Agent seeds 2901–2904 are matched across world 19 (globo-actions GPUs 0–3) and
world 101 (globo1 GPUs 0–3). Original spawns: 12,39.1,120 and 16,58.1,118.
All preceding 12 agents were saved and stopped; their results remain archived.

- Gamma .90 for both Q streams; 50 offline updates per round.
- Original config24 explorer novelty coefficients and goal bonus, with ordinary
  novelty multiplier 1.0. Death incurs the unscaled batch goal bonus as a penalty.
- Hard survival: original species animal HP; 4x hunger/thirst depletion and
  starvation/dehydration health penalties. Energy rates/recovery gains unchanged.
- Clouds/weather visuals and day/night cycle restored. Initial time remains midday.
- Walk speed 2.5; ground and air acceleration 2. Climb/jump/gravity unchanged.
- Current 14 actions, nondirectional jump, single-click placement, minimum block
  stack capacity 4, no starting blocks, full numeric HUD feedback.
- Continuing-life memory and TD targets; tuned reservoir; FTN K296/H2368;
  2M replay, 500k frontier, 100k RECODE capacity. No step limit.

Monitor: http://localhost:8018/ (globo-actions).
Artifacts: /mnt/obelisk-shared/kmckee90/globo_natural_survival, outside sync.

Life retention is controlled live through `life_settings.json` and the monitor's
`/api/life-settings` endpoint: `minimum_minutes` is a nonnegative integer.
Default 0 keeps every finalized life. Positive values retain lives strictly
longer than that many game minutes. Recorder checks at life completion, including
lives in progress when the setting changes. Previously saved videos remain
visible and are never pruned by the monitor. Event clips and
lifespan/achievement logs are not filtered by this setting.
