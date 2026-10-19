# Changepoint storage comparison

Pure DomainShift, BayesLoRA, standard training schedule and one filtered
replay batch per step. Each accuracy is the mean over old tasks 0–4 on
512 fixed query examples per task. The same exact hard mask is applied at
task completion and after task 5; no evaluation mask search is used.
`replay_best` is selected at task completion from masks then in the
buffer; later changepoint storage may overwrite or evict that entry.

## Two-seed averages

| Storage | Mask | Task-end acc | Final acc | Δ acc |
|---|---|---:|---:|---:|
| standard | last_train | 0.923 | 0.712 | -0.210 |
| standard | replay_best | 0.785 | 0.773 | -0.012 |
| replace | last_train | 0.914 | 0.838 | -0.076 |
| replace | replay_best | 0.806 | 0.646 | -0.161 |
| append 8 | last_train | 0.904 | 0.735 | -0.170 |
| append 8 | replay_best | 0.801 | 0.784 | -0.017 |
| append 32 | last_train | 0.914 | 0.797 | -0.118 |
| append 32 | replay_best | 0.781 | 0.779 | -0.002 |

## Individual runs

| Seed | Storage | Mask | Task-end acc | Final acc | Δ acc | Δ CE | KL | Events | Final buffer by task |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 123 | standard | last_train | 0.915 | 0.697 | -0.218 | +1.340 | 1.239 | 0 | — |
| 123 | standard | replay_best | 0.747 | 0.740 | -0.007 | +0.250 | 0.377 | 0 | — |
| 123 | replace | last_train | 0.911 | 0.824 | -0.087 | +0.176 | 0.215 | 9 | — |
| 123 | replace | replay_best | 0.790 | 0.632 | -0.157 | +0.423 | 0.526 | 9 | — |
| 123 | append 8 | last_train | 0.889 | 0.722 | -0.168 | +0.440 | 0.460 | 11 | 0:36, 1:116, 2:124, 3:108, 4:108, 5:108, unlabeled:0 |
| 123 | append 8 | replay_best | 0.791 | 0.759 | -0.032 | +0.371 | 0.397 | 11 | 0:36, 1:116, 2:124, 3:108, 4:108, 5:108, unlabeled:0 |
| 123 | append 32 | last_train | 0.908 | 0.769 | -0.139 | +0.321 | 0.318 | 10 | 0:0, 1:72, 2:164, 3:132, 4:132, 5:100, unlabeled:0 |
| 123 | append 32 | replay_best | 0.769 | 0.771 | +0.002 | +0.108 | 0.306 | 10 | 0:0, 1:72, 2:164, 3:132, 4:132, 5:100, unlabeled:0 |
| 124 | standard | last_train | 0.930 | 0.727 | -0.203 | +1.200 | 1.168 | 0 | — |
| 124 | standard | replay_best | 0.823 | 0.807 | -0.017 | +0.161 | 0.423 | 0 | — |
| 124 | replace | last_train | 0.917 | 0.852 | -0.065 | +0.141 | 0.205 | 9 | — |
| 124 | replace | replay_best | 0.823 | 0.659 | -0.164 | +0.413 | 0.441 | 9 | — |
| 124 | append 8 | last_train | 0.920 | 0.748 | -0.172 | +0.524 | 0.567 | 11 | 0:36, 1:116, 2:108, 3:116, 4:124, 5:100, unlabeled:0 |
| 124 | append 8 | replay_best | 0.811 | 0.808 | -0.003 | +0.116 | 0.259 | 11 | 0:36, 1:116, 2:108, 3:116, 4:124, 5:100, unlabeled:0 |
| 124 | append 32 | last_train | 0.921 | 0.824 | -0.096 | +0.248 | 0.232 | 8 | 0:0, 1:72, 2:132, 3:132, 4:164, 5:100, unlabeled:0 |
| 124 | append 32 | replay_best | 0.793 | 0.787 | -0.006 | +0.147 | 0.347 | 8 | 0:0, 1:72, 2:132, 3:132, 4:164, 5:100, unlabeled:0 |

`replace` refreshes existing buffer slots with recent outgoing-mask data;
`append` adds recent outgoing-mask data to the same fixed-capacity deque,
evicting its oldest entries. Neither policy changes the replay overlap
filter or the number of replay gradient batches per step.

## Oldest task (task 0)

| Seed | Storage | Mask | Task-end acc | Final acc | Final buffer entries |
|---:|---|---|---:|---:|---:|
| 123 | standard | last_train | 0.910 | 0.695 | — |
| 123 | standard | replay_best | 0.744 | 0.721 | — |
| 123 | replace | last_train | 0.883 | 0.842 | — |
| 123 | replace | replay_best | 0.838 | 0.631 | — |
| 123 | append 8 | last_train | 0.906 | 0.723 | 36 |
| 123 | append 8 | replay_best | 0.775 | 0.725 | 36 |
| 123 | append 32 | last_train | 0.906 | 0.686 | 0 |
| 123 | append 32 | replay_best | 0.830 | 0.713 | 0 |
| 124 | standard | last_train | 0.949 | 0.672 | — |
| 124 | standard | replay_best | 0.836 | 0.812 | — |
| 124 | replace | last_train | 0.914 | 0.908 | — |
| 124 | replace | replay_best | 0.846 | 0.451 | — |
| 124 | append 8 | last_train | 0.951 | 0.822 | 36 |
| 124 | append 8 | replay_best | 0.801 | 0.814 | 36 |
| 124 | append 32 | last_train | 0.951 | 0.756 | 0 |
| 124 | append 32 | replay_best | 0.869 | 0.746 | 0 |
