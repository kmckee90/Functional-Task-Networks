# Harder survival on the same world (corrected cohort)

All four agents continue on **world 19**, original spawn `12.0,39.1,120.0`.
Different seeds mean independently trained **agent seeds**, not different worlds.

| Pod/GPU | Agent seed | Exploration source | Checkpoint step |
|---|---|---|---|
| globo2/0 | 1901 | lifetime2_14_conv_r1_s19 | 1,189,474 |
| globo2/1 | 1902 | lifetime2_14_conv_r2_s19 | 1,215,681 |
| globo3/0 | 1903 | lifetime2_14_conv_r3_s19 | 1,181,824 |
| globo3/1 | 1904 | lifetime2_14_conv_r4_s19 | 1,173,580 |

These are the four original 14-action exploration agents, not current survival
checkpoints and not duplicated weights with new RNG seed labels. Their exploration
phase used gamma .9036665; all four now use gamma .99 (both Q streams), 50 offline
updates, LR 1.89949e-4 and .25 ordinary novelty reward. Goal reward is unchanged;
death cost matches the unscaled batch goal reward. Historical replay receives no
retroactive death labels.

Retain learned weights, optimizer moments, replay, frontier, density memory and
world-specific coverage/achievement accounting. Each fork has private checkpoints
and future replay writes. Run cards and video labels retain the training counter;
comparison plots use observations since the fork. Completed-life plots contain
only post-fork lives, with no smoothing or averaging.

`GLOBO_HARD_SURVIVAL=1` patches private engine game copies before startup:

- Restore original species HP by removing the 20-HP cap.
- Multiply negative hunger/thirst rates, movement/action losses and illness
  losses by four; rate feedback reports the effective rates.
- Multiply starvation/dehydration health penalties by four.
- Preserve food/water gains, energy rates and other damage.

The 14-action layout, HUD feedback, continuing-life memory, single-click placement
and four-block stack capacity are unchanged. Lifetime depends on behavior and is
not guaranteed to be exactly one quarter.

Data live outside sync under
`/mnt/obelisk-shared/kmckee90/globo_survival_hard_s19`.
Whole-life videos retain only lives over 10 game minutes. The monitor remains
`http://localhost:8017/` on globo2, with event clips, maps and lifespan charts.
Launch with `python -m globo_dreamer.survival_hard launch --pod POD`;
serve with `python -m globo_dreamer.survival_hard monitor`.

The mistaken new-world cohort (worlds 101, 2, 7, 8) was stopped and archived under
`globo_survival_hard`; none of its trained checkpoints initialize this cohort.
The earlier eight survival runs remain unchanged.
