# FTN-M2E discount and offline optimization comparison

The four `lifetime2_14_conv` runs on globo-actions continue unchanged:
gamma 0.9036665304295952, 50 offline optimization steps every 50 interactions.

Four fresh runs on globo1, world seed 19 and spawn 12.0,39.1,120.0:

| GPUs | Run prefix | Agent seeds | Gamma | Updates per round |
|---|---|---|---|---|
| 0–1 | lifetime2_g99_u50 | 1901, 1902 | 0.99 | 50 |
| 2–3 | lifetime2_g99_u200 | 1901, 1902 | 0.99 | 200 |

Rounds still occur every 50 interactions once 500 transitions are available.
All start with untrained weights and empty replay, frontier and RECODE memories.
No step limit. Retain 14 actions, nondirectional jump, normalized HUD feedback,
cross-life value propagation, persistent reservoir/feedback across respawns,
single-click placement, four-block capacity, empty starting block inventory,
20-HP animals, and no extrinsic reward. FTN K=296, bank=2368.
All other settings match the current lifetime-v2 cohort, including frontier
rotation disabled. The 200-update condition has four times the update budget
per collected transition; FTN's existing auxiliary replay behavior is unchanged.

The existing port-8016 monitor combines both pods. Each condition has its own
mean/min-max color for coverage, achievements and recent lifespan. The previous
engine experiment remains available through its archive link. Charts clip to the
shortest available trajectory; videos show all eight current runs.

Runtime files remain under /home/dev/globo_dreamer_state and the shared
globo_lifetime_v2 directory, outside source sync. launch.json records the actual
gamma and mf_offline_updates settings for each run.
