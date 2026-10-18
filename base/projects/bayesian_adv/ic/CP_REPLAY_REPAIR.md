# BayesLoRA: does better replay storage reduce IC repair's gain?

Pure DomainShift, parent-scale six-task run, four input rows for direct
IC routing, one replay-gradient batch per training step.
The refresh policy replaces each outgoing task's existing replay slots
with recent completed batches and the outgoing hard mask at a detected
changepoint. Buffer capacity remains 600 batches; repair does not change
predictor weights or gradient replay. Priority uses the same refreshed
buffer and one replay-gradient batch per step, but samples an old task's
late-mask snapshot even when the ordinary overlap filter excludes it.
Priority-half does so every other step; full priority does so whenever
an old late-mask snapshot is available.
Repair uses 800 fit rows, separate replay validation rows, and three
100-step mask-search restarts after each task block.

| Seed | Buffer | IC repair | Current-block train acc | Cumulative recall | Final old-task recall | Final IC direct probe | Accepted repairs | Replaced slots | Eligible changes | Priority draws |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 123 | standard | off | 0.913 | 0.700 | 0.691 | 0.737 | 0 | — | — | — |
| 123 | standard | on | 0.913 | 0.895 | 0.903 | 0.900 | 15 | — | — | — |
| 123 | refresh | off | 0.907 | 0.714 | 0.720 | 0.767 | 0 | 900 | 6 | — |
| 123 | refresh | on | 0.907 | 0.865 | 0.853 | 0.862 | 14 | 900 | 6 | — |
| 123 | priority | off | 0.901 | 0.668 | 0.679 | 0.726 | 0 | 900 | 5 | 7500 |
| 123 | priority | on | 0.901 | 0.817 | 0.822 | 0.832 | 13 | 900 | 5 | 7500 |
| 123 | priority-half | off | 0.908 | 0.723 | 0.747 | 0.770 | 0 | 800 | 4 | 3750 |
| 123 | priority-half | on | 0.908 | 0.796 | 0.814 | 0.815 | 13 | 800 | 4 | 3750 |
| 124 | standard | off | 0.924 | 0.666 | 0.722 | 0.758 | 0 | — | — | — |
| 124 | standard | on | 0.924 | 0.915 | 0.918 | 0.921 | 15 | — | — | — |
| 124 | refresh | off | 0.920 | 0.721 | 0.724 | 0.746 | 0 | 700 | 5 | 0 |
| 124 | refresh | on | 0.920 | 0.918 | 0.926 | 0.917 | 15 | 700 | 5 | 0 |
| 124 | priority-half | off | 0.917 | 0.762 | 0.755 | 0.787 | 0 | 1000 | 8 | 3750 |
| 124 | priority-half | on | 0.917 | 0.887 | 0.927 | 0.920 | 13 | 1000 | 8 | 3750 |

## Repair gain in cumulative recall

| Seed | Standard buffer | Refreshed buffer | Full priority | Half priority |
|---:|---:|---:|---:|---:|
| 123 | +0.195 | +0.151 | +0.149 | +0.073 |
| 124 | +0.249 | +0.197 | — | +0.126 |

Cumulative recall is the experiment's backward evaluation averaged
over task blocks, including the current task. Final old-task recall
averages tasks 0–4 after task 5. Both use a fresh query batch per task
and direct IC mask retrieval from four unlabeled support inputs.
