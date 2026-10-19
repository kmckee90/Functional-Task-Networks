# Retained frontier policies

Two experiments, each on seeds 19 and 101, supplement the flat and convolutional
FTN readout experiments. Both use the same frozen reservoir and convolutional
population budget (K=296, H=2368; eight-layer, width-eight private neurons).

**Retained frontiers:** each registered frontier owns a saved Bayesian mask
posterior. Its reaching objective remains fixed when memory density changes.
Density controls which frontier is desirable, rather than continually changing
the meaning of its low-level policy. Acting retrieves the selected goal's mask.

**Waypoint planning:** observed trajectories also supply intermediate goal
observations and directed graph edges. A shortest-path planner selects a series
of retained reaching policies. Edges are admitted from observed transitions;
reverse edges are not assumed.

Goals are detected using visual embeddings and frozen reservoir context, without
privileged coordinates. Reported goal hits are therefore detector matches, not
independent proof of successful navigation. Original goal images are retained
for inspection in the video and mask diagnostic.

Each goal has a stationary 0/1 reaching reward with terminal bootstrap at a hit,
gamma 0.98, a small conservative Q penalty (0.02), and behavior-cloning support
from observed approaches (0.05, temperature 0.2). Two Bayesian reconfiguration
steps precede each update. Search and weight updates use the same objective;
published mask changes require improved support loss. Overlapping policies are
replayed under their saved masks. Every fifth offline update trains the ordinary
density objective; periods of density exploration provide fresh experience.
The goal bank is bounded at 1,024 entries. Saved goal identities are not recycled.

Runs are unlimited and checkpoint into separate pod-only directories. Their
videos overwrite previous clips and show synchronized observation, mask EMA,
and goal observation. Waypoint runs distinguish immediate waypoint and final
frontier. A mask pop-out provides activity history and current goal thumbnails.

The monitor's focused spatial comparison includes these two variants, the flat
and convolutional FTN variants, historical base M2E, and Plan2Explore. Seeds are
charted separately. The observation axis follows the longest new FTN run;
baseline histories are clipped to this window. Warm-start runs are excluded.

Validation: `verify_frontier_ftn.py` runs on the pod and checks stationary goal
labels, terminating targets, independent saved posteriors, retrieval, checkpoint
recovery, directed routing, real scheduler updates, and both video layouts.
