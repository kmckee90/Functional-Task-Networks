# HUD feedback and continuing lives

Fresh cohort on globo-actions: four FTN-M2E runs, world 19, agent seeds
1901–1904. Reference: the preceding four `enginev2_14_conv` runs, archived
with their weights, buffers, videos and results. Mean/min-max charts use a
shared endpoint. No checkpoint initializes the new cohort.

The reference runs stopped at 69,706 / 64,790 / 67,260 / 63,294 observations.
Their supervisor list is saved as `continuous_runs/engine_v2_before_lifetime_v2.json`.
The new cohort may resume its own checkpoints; it never loads reference weights.

## Changes

- Action 5: jump only; no forward+jump primitive. Other 13 actions unchanged.
- Numeric HUD enters the frozen feedback projection alongside previous action
  and novelty. This includes health, hunger, thirst, energy, breath, internal
  and external temperature, energy capacity, displayed score and four rates.
  Also includes inventory quantities normalized by actual stack capacity,
  one-hot selected slot, and indicators for the 11 named health effects plus
  an unknown-effect indicator. Item identities are not encoded.
- All added inputs are finite and clipped to [0,1]. HP/energy/breath divide by
  their maxima; hunger/thirst by 1000; temperature maps 20–50 C internally and
  -50–100 C externally; energy capacity divides by 10000; score uses s/(1000+s)
  for nonnegative s; signed rates use 0.5+0.5*tanh(rate/100).
- HUD values are observations only, not rewards. RECODE novelty remains
  image-based, so bodily changes do not themselves earn novelty.
- No reservoir, previous-action, novelty-feedback or FTN reset on respawn.
  The reservoir encodes the death observation and then the respawn observation.
  The last executed action before death is stored with the actual respawn
  reservoir state as its successor and respawn novelty as its reward signal.
  No fictional action is inserted for the death screen or reset operation.
- These transitions have terminal=false and boundary=false in main replay.
  Contiguous offline batches can cross deaths; TD bootstrap is enabled.
  Frontier copies still retain boundaries between separately copied snippets.
- Actual deaths/lives remain logged and rendered normally. Extrinsic reward
  and death penalty remain zero. Learning to avoid death is an experimental
  consequence to assess, not a guaranteed result of continuing TD targets.

## Retained settings

14 actions, single-click placement, four-block stack capacity, no starting
blocks, 20-HP animals; tuned 128-channel reservoir; FTN K296/H2368; 2M replay,
500k frontier and 100k RECODE centroids; config24 density learning settings.
Unlimited steps, event-triggered videos, same spawn and matched random seeds.

Monitor: port 8016. Runtime stays outside source sync under
`/home/dev/globo_dreamer_state/runs/lifetime2_14_conv_r*_s19` and
`/mnt/obelisk-shared/kmckee90/globo_lifetime_v2`.

Continuity is across automatic deaths/respawns within a running process.
Process restart does not reconstruct the old physical world or transient
reservoir stream. The fresh cohort starts with a newly initialized stream.
