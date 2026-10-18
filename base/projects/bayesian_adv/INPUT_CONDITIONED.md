# Input-conditioned routing in the standard experiment

Add `InputConditioned` after `Bayes`, `BayesLoRA`, or `BayesLoRARNN` in
`--config`. The standard runner and its normal PNG/GIF output remain in use.
For the DomainShift setting studied in `ic/`, also add `DomainShift` and
`NoConceptShift`:

```sh
cd /home/dev/workspace/projects/bayesian_adv
CUDA_VISIBLE_DEVICES=0 uv run --project /home/dev/workspace python run_experiments.py \
  --config BayesLoRA DomainShift NoConceptShift InputConditioned \
  --name lora_ic --out-root artifacts
```

Replace `BayesLoRA` with `Bayes` or `BayesLoRARNN` to use the same router on
those models. `BayesLoRARNN_Ftn` can be used with the same subconfig to make the
input encoder and output decoder maskable. The effective parameters, scalar
metrics, loss plot, forgetting/transfer matrix, task plot, and mask GIFs are
written under `artifacts/<name>/`. Use `--metrics-only` only when plots are
unnecessary.

The router stores an optimal rank-`IC_AE_RANK` linear reconstruction model for
each segment, paired with its Bayesian mask logits. At evaluation, it chooses
the most recent near-best reconstruction match using the first
`IC_SUPPORT_SIZE` **unlabeled** input rows. The default `direct` strategy applies
that mask without evaluation-time optimization or labels. If there is no match,
the normal cold search runs on at most four labeled rows. `refine` instead
starts a posterior update from the retrieved logits; it uses evaluation labels
and can damage a good stored mask, so `direct` is the tested default.

After each training block, replay repair groups the existing labeled replay
examples by their input reconstruction model. It cold-searches a candidate
mask on `IC_REPAIR_FIT_ROWS`, checks it on separate
`IC_REPAIR_VALIDATION_ROWS`, and appends it only if validation improves. Old
router projectors and old mask entries remain immutable. This keeps routing
from forgetting through weight updates, while letting the mask memory track
changes to the predictor. It does require substantial labeled *training*
replay; the default BayesLoRA buffer supplies exactly 1600 rows per task block.

All routing and repair knobs are declared in the `InputConditioned` class in
`config.py`: support size, reconstruction rank and thresholds, optional input
novelty detection, optional training-time prior, evaluation strategy, replay
fit/validation sizes, optimization settings, repair interval, acceptance
margin, and stored-logit magnitude. Compose an additional subconfig after
`InputConditioned` to override them. `InputConditionedSmoke` reduces the replay
budget for a fast path check when also composed with `Smoke`.

The demonstrated high-recall result in `ic/RESULTS.md` used DomainShift with
concept shift disabled. Input-only routing cannot distinguish tasks whose
input distributions are the same, even when their labels differ. The standard
`Bayes` training schedules and prospective evaluations also differ from that
specific experiment, so its scores are a reference rather than a promised
result for every config.
