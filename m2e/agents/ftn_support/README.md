These two modules are copied unchanged from
`Globo/experiments/globo_dreamer/legacy/{bayes_reconfig,overlap_regularizer}.py`,
whose Bayes inference originates in `ftn/projects/bayesian_adv`.
They keep the independently synced m2e project runnable without Globo or a
second pod source root. Algorithm changes should be reconciled with that source.

`spatial_permute.py` originated in `ftn/spatial_permute.py` (also used by
Globo). The m2e copy precomputes the mean neuron co-activation matrix instead
of carrying the full mask-history vectors through every swap. It preserves
the spatial objective, initialization, annealing and overlap invariants;
floating-point changes can produce a different, equivalent display layout.
