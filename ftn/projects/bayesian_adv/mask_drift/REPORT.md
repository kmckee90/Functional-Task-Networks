# Exact-mask prediction drift

Each row compares fixed query examples under exactly the same task mask
at task completion and after the last task block. The mask is passed
explicitly to the model; no retrospective search occurs. `replay_best`
is chosen on separate support from 16 evenly spaced masks saved in
the ordinary training replay buffer. `last_train` is the final
training mask, and `cold_oracle` is searched once at task completion.
All subsequent evaluations keep those masks fixed. The table pools
tasks 0–4; task 5 has no subsequent training block.

| Model | Shift | Mask | Origin acc | Final acc | Δ acc | Δ CE | Mean KL | Mean TV | Flipped classes | Origin overlap with task-end mask |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | domain | replay_best | 0.907 | 0.921 | +0.014 | -0.026 | 0.056 | 0.068 | 0.064 | 0.590 |
| Base | domain | last_train | 0.919 | 0.909 | -0.010 | +0.006 | 0.058 | 0.063 | 0.055 | 1.000 |
| Base | domain | cold_oracle | 0.918 | 0.913 | -0.005 | +0.002 | 0.060 | 0.064 | 0.059 | 0.682 |
| Base | concept | replay_best | 0.919 | 0.921 | +0.002 | +0.003 | 0.067 | 0.070 | 0.063 | 0.569 |
| Base | concept | last_train | 0.924 | 0.887 | -0.037 | +0.058 | 0.106 | 0.089 | 0.093 | 1.000 |
| Base | concept | cold_oracle | 0.918 | 0.897 | -0.021 | +0.044 | 0.093 | 0.082 | 0.081 | 0.599 |
| LoRA | domain | replay_best | 0.694 | 0.670 | -0.025 | +1.063 | 0.830 | 0.219 | 0.313 | 0.266 |
| LoRA | domain | last_train | 0.684 | 0.588 | -0.097 | +0.880 | 0.955 | 0.332 | 0.384 | 1.000 |
| LoRA | domain | cold_oracle | 0.644 | 0.568 | -0.076 | +0.976 | 1.231 | 0.360 | 0.595 | 0.561 |
| LoRA | concept | replay_best | 0.850 | 0.716 | -0.134 | +0.567 | 0.676 | 0.251 | 0.282 | 0.200 |
| LoRA | concept | last_train | 0.848 | 0.711 | -0.137 | +0.579 | 0.537 | 0.222 | 0.263 | 1.000 |
| LoRA | concept | cold_oracle | 0.845 | 0.715 | -0.130 | +0.587 | 0.616 | 0.237 | 0.269 | 0.273 |
| Bayes | domain | replay_best | 0.820 | 0.826 | +0.007 | -0.028 | 0.047 | 0.076 | 0.096 | 0.599 |
| Bayes | domain | last_train | 0.835 | 0.831 | -0.004 | -0.006 | 0.053 | 0.080 | 0.082 | 1.000 |
| Bayes | domain | cold_oracle | 0.821 | 0.815 | -0.006 | -0.003 | 0.062 | 0.082 | 0.075 | 0.688 |
| Bayes | concept | replay_best | 0.845 | 0.839 | -0.006 | +0.022 | 0.047 | 0.073 | 0.077 | 0.501 |
| Bayes | concept | last_train | 0.854 | 0.829 | -0.025 | +0.061 | 0.076 | 0.091 | 0.095 | 1.000 |
| Bayes | concept | cold_oracle | 0.833 | 0.814 | -0.020 | +0.102 | 0.109 | 0.098 | 0.087 | 0.580 |
| BayesLoRA | domain | replay_best | 0.747 | 0.740 | -0.007 | +0.250 | 0.377 | 0.192 | 0.212 | 0.312 |
| BayesLoRA | domain | last_train | 0.915 | 0.697 | -0.218 | +1.340 | 1.239 | 0.279 | 0.291 | 1.000 |
| BayesLoRA | domain | cold_oracle | 0.782 | 0.746 | -0.036 | +0.127 | 0.185 | 0.167 | 0.193 | 0.354 |
| BayesLoRA | concept | replay_best | 0.879 | 0.832 | -0.048 | +0.344 | 0.432 | 0.147 | 0.146 | 0.395 |
| BayesLoRA | concept | last_train | 0.939 | 0.836 | -0.103 | +0.493 | 0.559 | 0.151 | 0.155 | 1.000 |
| BayesLoRA | concept | cold_oracle | 0.882 | 0.839 | -0.043 | +0.204 | 0.259 | 0.131 | 0.131 | 0.406 |

## Stored replay mask by task

Accuracy and cross-entropy are evaluated on the identical 512
held-out examples at both times. KL and flip fraction compare
the two predictions on those same examples.

| Model | Shift | Task | Origin acc | Final acc | Origin CE | Final CE | KL | Flips |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base | domain | 0 | 0.906 | 0.928 | 0.226 | 0.173 | 0.097 | 0.049 |
| Base | domain | 1 | 0.900 | 0.910 | 0.237 | 0.197 | 0.061 | 0.064 |
| Base | domain | 2 | 0.893 | 0.928 | 0.237 | 0.215 | 0.066 | 0.082 |
| Base | domain | 3 | 0.910 | 0.900 | 0.230 | 0.227 | 0.037 | 0.084 |
| Base | domain | 4 | 0.928 | 0.939 | 0.175 | 0.162 | 0.019 | 0.039 |
| Base | concept | 0 | 0.900 | 0.947 | 0.263 | 0.157 | 0.147 | 0.074 |
| Base | concept | 1 | 0.936 | 0.936 | 0.152 | 0.136 | 0.045 | 0.051 |
| Base | concept | 2 | 0.906 | 0.914 | 0.203 | 0.237 | 0.064 | 0.066 |
| Base | concept | 3 | 0.951 | 0.936 | 0.109 | 0.135 | 0.038 | 0.055 |
| Base | concept | 4 | 0.902 | 0.871 | 0.200 | 0.274 | 0.043 | 0.070 |
| LoRA | domain | 0 | 0.469 | 0.531 | 0.695 | 3.280 | 2.938 | 1.000 |
| LoRA | domain | 1 | 0.551 | 0.551 | 0.963 | 2.864 | 0.515 | 0.000 |
| LoRA | domain | 2 | 0.828 | 0.795 | 0.342 | 0.410 | 0.134 | 0.174 |
| LoRA | domain | 3 | 0.773 | 0.646 | 0.405 | 1.107 | 0.466 | 0.225 |
| LoRA | domain | 4 | 0.850 | 0.824 | 0.302 | 0.361 | 0.097 | 0.166 |
| LoRA | concept | 0 | 0.711 | 0.492 | 0.604 | 1.828 | 1.544 | 0.559 |
| LoRA | concept | 1 | 0.891 | 0.748 | 0.294 | 0.579 | 0.370 | 0.268 |
| LoRA | concept | 2 | 0.855 | 0.650 | 0.295 | 1.442 | 1.199 | 0.342 |
| LoRA | concept | 3 | 0.920 | 0.854 | 0.179 | 0.298 | 0.188 | 0.137 |
| LoRA | concept | 4 | 0.871 | 0.836 | 0.289 | 0.351 | 0.077 | 0.105 |
| Bayes | domain | 0 | 0.811 | 0.846 | 0.427 | 0.324 | 0.106 | 0.121 |
| Bayes | domain | 1 | 0.838 | 0.877 | 0.381 | 0.280 | 0.081 | 0.113 |
| Bayes | domain | 2 | 0.811 | 0.799 | 0.387 | 0.390 | 0.017 | 0.078 |
| Bayes | domain | 3 | 0.811 | 0.783 | 0.383 | 0.424 | 0.020 | 0.062 |
| Bayes | domain | 4 | 0.828 | 0.826 | 0.343 | 0.363 | 0.011 | 0.107 |
| Bayes | concept | 0 | 0.809 | 0.834 | 0.415 | 0.372 | 0.080 | 0.088 |
| Bayes | concept | 1 | 0.828 | 0.807 | 0.353 | 0.427 | 0.084 | 0.076 |
| Bayes | concept | 2 | 0.869 | 0.854 | 0.341 | 0.390 | 0.033 | 0.082 |
| Bayes | concept | 3 | 0.902 | 0.904 | 0.204 | 0.193 | 0.019 | 0.041 |
| Bayes | concept | 4 | 0.816 | 0.797 | 0.356 | 0.399 | 0.022 | 0.098 |
| BayesLoRA | domain | 0 | 0.744 | 0.721 | 0.536 | 0.980 | 0.419 | 0.203 |
| BayesLoRA | domain | 1 | 0.707 | 0.758 | 0.593 | 0.696 | 0.269 | 0.223 |
| BayesLoRA | domain | 2 | 0.729 | 0.744 | 0.634 | 0.827 | 0.328 | 0.160 |
| BayesLoRA | domain | 3 | 0.756 | 0.738 | 0.504 | 0.690 | 0.547 | 0.279 |
| BayesLoRA | domain | 4 | 0.799 | 0.738 | 0.389 | 0.714 | 0.322 | 0.197 |
| BayesLoRA | concept | 0 | 0.846 | 0.799 | 0.428 | 0.872 | 0.512 | 0.172 |
| BayesLoRA | concept | 1 | 0.910 | 0.852 | 0.236 | 0.823 | 0.693 | 0.137 |
| BayesLoRA | concept | 2 | 0.840 | 0.830 | 0.391 | 0.732 | 0.399 | 0.115 |
| BayesLoRA | concept | 3 | 0.924 | 0.836 | 0.171 | 0.390 | 0.296 | 0.146 |
| BayesLoRA | concept | 4 | 0.877 | 0.842 | 0.322 | 0.453 | 0.259 | 0.160 |

## No replay gradient control

The replay buffer still stores its usual data and masks, but
`N_REPLAY_SAMPLES=0` disables replay gradient updates. These are
separate training trajectories at the same seed; compare each
run's own origin and final measurements for a fixed mask.

| Model | Shift | Replay batches | Origin acc | Final acc | Δ acc | Δ CE | Mean KL | Flipped classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base | domain | 1 | 0.907 | 0.921 | +0.014 | -0.026 | 0.056 | 0.064 |
| Base | domain | 0 | 0.877 | 0.748 | -0.129 | +0.296 | 0.195 | 0.193 |
| Base | concept | 1 | 0.919 | 0.921 | +0.002 | +0.003 | 0.067 | 0.063 |
| Base | concept | 0 | 0.910 | 0.705 | -0.205 | +0.659 | 0.642 | 0.284 |
| Bayes | domain | 1 | 0.820 | 0.826 | +0.007 | -0.028 | 0.047 | 0.096 |
| Bayes | domain | 0 | 0.807 | 0.771 | -0.036 | +0.026 | 0.038 | 0.091 |
| Bayes | concept | 1 | 0.845 | 0.839 | -0.006 | +0.022 | 0.047 | 0.077 |
| Bayes | concept | 0 | 0.812 | 0.771 | -0.041 | +0.131 | 0.119 | 0.131 |

## BayesLoRA DomainShift second seed

The task-end mask is a separate exact mask snapshot at the end
of each task block. The training loop only adds masks to the
ordinary replay buffer during the first 100 batches, so it
does not explicitly add this task-end snapshot.

| Seed | Mask | Origin acc | Final acc | Δ acc | Δ CE | Mean KL | Origin overlap with task-end mask |
|---:|---|---:|---:|---:|---:|---:|---:|
| 123 | last_train | 0.915 | 0.697 | -0.218 | +1.340 | 1.239 | 1.000 |
| 123 | replay_best | 0.747 | 0.740 | -0.007 | +0.250 | 0.377 | 0.312 |
| 124 | last_train | 0.930 | 0.727 | -0.203 | +1.200 | 1.168 | 1.000 |
| 124 | replay_best | 0.823 | 0.807 | -0.017 | +0.161 | 0.423 | 0.344 |

## Figures

- [BayesLoRA DomainShift task-end masks](artifacts_full/BayesLoRA_domain_123/drift_last_train.png)
- [BayesLoRA DomainShift stored replay masks](artifacts_full/BayesLoRA_domain_123/drift_replay_best.png)
- [Base concept shift with replay](artifacts_full/Base_concept_123/drift_replay_best.png)
- [Base concept shift without replay gradients](artifacts_full/Base_concept_123_r0/drift_replay_best.png)

## Interpretation

Nonzero KL and class flips with an unchanged hard mask directly
show that changes in predictor weights alter the function
associated with that task mask. Accuracy may improve or decline;
prediction drift alone is not evidence of harmful forgetting.
The task-wise rows identify where the stored mask became worse.

With standard replay, Base and Bayes stored masks exhibit
prediction drift, yet most old-task accuracy remains intact.
The no-gradient-replay Base controls lose 12.9 points of
DomainShift accuracy and 20.5 points of concept-shift accuracy
relative to their own task-end baselines. Bayes loses 3.6 and
4.1 points, respectively. These are separate model training
trajectories, not paired weight interventions.

BayesLoRA's task-end masks become markedly worse even with
standard replay: 21.8 and 20.3 accuracy points lost in two
DomainShift seeds. The masks sampled from the replay buffer
start weaker and remain more stable. They overlap the task-end
mask by only about one third at the end of training. This is
direct evidence that upkeep of stored early masks does not
guarantee upkeep of a later task-end solution.

LoRA and BayesLoRA have substantially larger fixed-mask
prediction drift than Base and Bayes in the seed-123 runs.
This is one seed with the parent's usual filtered replay and
training schedules, except for the second BayesLoRA DomainShift
seed. DomainShift disables concept shift; concept uses the
parent's default task generator settings. Fixed query examples
are generated from a separate RNG stream of the same task
generator; mask selection uses a disjoint support batch.
