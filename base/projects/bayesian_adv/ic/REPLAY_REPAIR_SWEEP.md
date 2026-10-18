# Gradient replay versus offline mask repair

This experiment keeps the DomainShift task stream, predictor size, replay
buffer capacity and sampling budget, four-input evaluation router, and repair search fixed. It changes
only the number of replay batches contributing gradients per training step and
whether old masks are repaired after each task block. The buffer holds 100
batches per task with 16 rows per batch even when gradient replay is zero.
`NoReplay` is deliberately not used because it also shrinks the stored data
and would leave too few rows for repair. All runs use `CONCEPT_SHIFT=False`.

`ICGradReplay0`, `ICGradReplay4`, and `ICGradReplay8` set
`N_REPLAY_SAMPLES`; one replay batch is the parent default. The existing
`replay_step` **sums** each sampled batch's loss into the gradient, so a larger
count means both broader sampling and greater total replay weight.
`ICGradReplay8Normalized` is an additional control that divides each of eight
replay losses by eight, keeping their total nominal weight equal to the
one-batch setting.

Repair uses the unchanged `IC_REPAIR_*` settings: 800 fitting rows, at least
800 separate validation rows, three independent 100-step cold mask searches,
and acceptance only on validation improvement. The no-repair condition routes
directly to the stored mask under the same input-domain key. Repair never
changes predictor weights. All conditions use seed 123 unless indicated.

## BayesLoRARNN, parent scale

| Replay batches per step | Current-block training accuracy | Recall without repair | Recall with repair | Repair gain | Accepted repairs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7474 | 0.5182 | 0.7337 | +0.2156 | 16 |
| 1 | 0.8171 | 0.6898 | 0.8214 | +0.1316 | 15 |
| 4 | 0.7933 | 0.7041 | 0.8021 | +0.0980 | 15 |
| 8 | 0.7690 | 0.7205 | 0.7960 | +0.0755 | 12 |
| 8, normalized loss | 0.8401 | 0.7025 | **0.8462** | +0.1436 | 15 |

With zero gradient replay, increasing repair from three 100-step restarts to
five 200-step restarts raised recall from 0.7337 to **0.7654** at seed 123.
The current-block training accuracy was unchanged at 0.7474. At seed 124,
heavy repair reached **0.7648** versus 0.7426 for standard repair and 0.5036
without repair. At seed 125, heavy repair reached **0.7823** versus 0.7436 for
standard repair and 0.5164 without repair. Current-block training accuracy
for seeds 124 and 125 was 0.8375 and 0.8536, respectively.
The matched one-batch-plus-standard-repair runs recalled 0.8214 at seed 123
and 0.8883 at seed 124, so larger repair alone did not fully replace one
gradient replay batch.

Recall averages retrospective held-out accuracy after successive task blocks.
Current-block training accuracy averages the training tail while that task is
current, so the two columns do not evaluate identical examples. At a fixed
replay count, repair and no-repair runs had identical training accuracy.
Input-domain retrieval matched on every final diagnostic trial in all runs.

Without gradient replay, repair brought recall near the 0.747 training score
of that same predictor. More gradient replay narrowed the repair gain but did
not remove it by eight batches per step. Among unnormalized conditions, both
the best absolute recall and the best current-block training performance were
at one replay batch **with** repair. The unnormalized eight-batch condition gave the closest
no-repair result to its own training score, but trained less well than the
one-batch condition. This is a single-seed result on favorable rank-two
domains; it does not establish a monotonic replay/repair tradeoff.
Normalizing the eight replay losses restored current-task training performance
and yielded the highest repaired recall, but its direct old-mask recall stayed
near the one-batch case. Extra replay-batch diversity without extra total
replay weight did not substitute for repair.

Run configs and metrics are in `artifacts/sweep_rnn_r*/`. Earlier one-batch
reference results are in `artifacts/rnn_full_direct_123/` and
`artifacts/rnn_full_repair_123/`.

## BayesLoRA, parent scale

| Replay batches per step | Current-block training accuracy | Recall without repair | Recall with repair | Repair gain | Accepted repairs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9033 | 0.5063 | 0.6576 | +0.1512 | 15 |
| 1 | 0.9133 | 0.7003 | 0.8948 | +0.1945 | 15 |
| 4 | 0.9073 | 0.7785 | 0.8880 | +0.1094 | 15 |
| 8 | 0.9045 | 0.7758 | 0.8932 | +0.1173 | 15 |
| 8, normalized loss | 0.9055 | 0.7040 | not run | — | — |

For LoRA, removing gradient replay barely changed current-block training
accuracy from the one-batch reference (0.9133), but old-mask recall fell near
chance. Standard offline repair helped substantially yet remained far below
the current-task training score. This differs from the recurrent model's
near-training recall with repair and no gradient replay.
Four replay batches improved direct old-mask recall substantially, but repair
still added 0.1094. The combined one-batch condition had slightly higher
absolute recall than the combined four-batch condition.
Eight unnormalized replay batches did not improve direct recall further:
0.7758 versus 0.7785 at four, despite comparable current-block training
accuracy. Repair still added 0.1173 at eight batches, reaching 0.8932.
Normalizing eight replay losses to the total one-batch weight
reduced direct recall to 0.7040, almost identical to the one-batch result
of 0.7003. Thus the higher direct recall at four and eight mostly tracks
extra replay gradient weight, rather than greater replay sample diversity.

Increasing zero-replay repair to five 200-step restarts raised seed-123 recall
from 0.6576 to **0.7536**, with the same 0.9033 current-block training
accuracy. This remains below the one-batch plus standard-repair recall of
0.8948. A second zero-replay heavy-repair run (seed 124) reached **0.8230**
recall versus 0.9171 current-block training accuracy. The two seeds therefore
show sizable repair-only recovery, but neither closes the gap to its own
current-block training performance.
At seed 124, one replay batch plus standard repair recalled **0.9150** with
0.9243 current-block training accuracy, versus 0.8230 recall and 0.9171
current-block training accuracy for zero replay plus heavy repair.

## What the substitution tests show

On seed 123, raising unnormalized gradient replay from one to eight batches
improved direct recall from 0.7003 to 0.7758 for LoRA and from 0.6898 to
0.7205 for LoRARNN. Neither reached the corresponding one-batch-plus-repair
recall (0.8948 and 0.8214). Increasing repair search from 300 to 1000
optimization steps across restarts improved zero-replay recall, but it did
not match one-batch-plus-standard-repair recall across matched seeds. The
results favor using both mechanisms at these budgets. They do not rule out a
much larger replay budget or a stronger repair algorithm.
