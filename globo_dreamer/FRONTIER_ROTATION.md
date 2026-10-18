# Frontier rotation

The FTN variants use a bounded-dwell scheduler for their offline frontier.
The existing density estimator and novelty rewards remain in place. The goal
buffer's reward multiplier no longer decides whether a fresh candidate can be
selected in the basic flat/convolutional variants.

At each offline round, candidates are ranked by current density novelty.
Candidates with RMS visual-embedding distance below 0.03 are treated as the same
view. Five diverse candidates are retained from the bounded replay scan. Thus
"top five" means the scanned candidate set, not an exhaustive sort of 2M rows.
The current frontier is retained until a switch, eviction, or timeout.

Switch survival is exp(-(age / 600)^2), evaluated in environment observations.
Approximately 94% of selections switch before 1,000 observations; at 1,000 the
scheduler forces a switch if another distinct candidate exists. Current views
are excluded, and recent views are avoided where possible. If all candidates
represent the same view, the scheduler does not claim to have switched.

For basic FTN–M2E, prioritized goal batches end at the selected frontier and put
their sparse bonus there, preventing a different high-novelty frame within that
trajectory from silently taking over. Main/random replay remains unchanged.
Old archived mask replay retains its original reward-target convention.

For retained-frontier and waypoint agents, the same scheduler rotates offline
discovery targets. At execution boundaries, final destinations are sampled from
the top five accessible scored goals, excluding the last destination and
preferring those outside the recent-choice history. If none is available, the
agent continues density exploration. A composed route also has a 1,000-step
total cap, in addition to existing individual option timeouts.

Scheduler state and private RNG are checkpointed. Earlier checkpoints load with
a fresh scheduler; networks, optimizers, replay, density memory and counters are
preserved. Status metrics include frontier age, switch count, and probability.
