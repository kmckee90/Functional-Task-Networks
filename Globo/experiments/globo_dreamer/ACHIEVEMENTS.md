# Achievement evaluation

Version 2 extends the six original outcomes with food harvesting/consumption,
reproductive score, recovery, sleep, other animal kills, food/resource/species
and biome discoveries, climbing, water crossings and predator survival.
These are read-only counters: no policy input, intrinsic reward or training loss changes.

The monitor places spatial and cumulative achievement charts side by side and
collapses the detailed checklist. Achievement curves count checked boxes exactly:
unlocked categories plus individual discovery entries. Repetitions do not raise
the curves; their counts remain available inside the checklist. Raw event totals
are preserved in logs separately.
Expanded instrumentation has no retroactive history. Older baselines retain their
original six-event records, so expanded totals are not directly comparable to them.

- Eggs: count successful `globo.lay_egg`; separately count owner score increases
  from egg timers. Diagnostic attempts/successes are in the engine JSON.
- Food: confirm a consumed stack or fruit-to-plant change; harvests require an
  inventory increase. Goldcap and coffee require actual HP/energy increases.
- Sleep: successful transition into sleep. Recovery only if measured; the current
  game's explicit sleep recovery block is commented out, so those may stay zero.
- Hunting: player killing blows only; unique species merge male variants.
- Discoveries: persisted identities avoid repeats across lives/restarts.
- Climbing: 5/10/20 m above life start, once each per life (not proof of tree climbing).
- Water: enter a water-group node, exit alive at least 5 m away horizontally.
- Biomes: first observed biome identity since instrumentation, including spawn
  only if observed after initialization; not a claim that a boundary was crossed.
- Predator survival: wolf/puma-attributed damage, alive 30 game seconds later
  without another such hit; this is an operational proxy, not proof of escape.

Tests run only on the pod: `python -m unittest globo_dreamer.test_achievements`.
`python -m globo_dreamer.verify_achievement_game` creates an isolated test world,
forces outcomes there, and checks the real engine's counters. Never uses live worlds.
`python -m globo_dreamer.deploy_achievement_v2` gracefully checkpoints/resumes
all current workers and the supervisor without changing run configurations.
