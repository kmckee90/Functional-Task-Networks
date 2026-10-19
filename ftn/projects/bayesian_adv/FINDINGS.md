# BayesLoRA findings (September 2026)

This note preserves the diagnostic results from the LoRA/BayesLoRA work before moving to the broader FTN repository. The current `BayesLoRA` config has since changed to `K=1024` and `TRAIN_DURATION=1500`; **the measured comparisons below used `K=256` and 500 training batches per task**. The newer setting reportedly works well, but no matching saved metrics were available here when this note was written.

## What the implementations do

- `LoRA` trains a bank of low-rank edits on each linear layer of one frozen MLP. Every layer has its **own** atom weights, and one mask selects the same atom indices across all layers. Its training mask search normally starts fresh each batch and retains the lowest-loss candidate seen during the search.
- `BayesLoRA` uses the same kind of backbone and layer-specific atom banks, but carries mask logits between batches. Stable batches get two stochastic posterior-update steps at temperature 3 with a KL penalty; detected changepoints get a 30-step reset/refit. Evaluation starts from zero logits and searches for 30 steps, regardless of the training mask state.
- `BayesLoRARNN` is a different architecture: a weight-tied recurrent cell has one atom bank shared across recurrent steps, with a separate mask at each step. It is not an option that gives the feedforward `BayesLoRA` a shared bank and independent per-layer masks.

The intended direction is compute-efficient FTN mask selection for deep architectures, including post-training pretrained LLMs and DiTs. The current LoRA experiment uses a frozen random MLP, so transfer to those architectures remains untested here.

## Matched five-seed comparison

Seeds 123–127; six tasks; 500 training batches per task; `K=256`; two routine mask-search steps; 30 evaluation search steps; same task-preview RNG sequence; 30 retrospective evaluation batches. `LoRA` searches cold on every training batch. `BayesLoRA` carries logits and uses a reactive 30-step reset. Recall is `avg_recall_eval_acc` from `MetricsCollector`.

| Seed | LoRA recall | BayesLoRA recall |
| --- | ---: | ---: |
| 123 | 85.50% | 85.51% |
| 124 | 70.64% | 83.41% |
| 125 | 72.06% | 83.75% |
| 126 | 79.37% | 84.19% |
| 127 | 81.34% | 83.43% |
| **Mean** | **77.78%** | **84.06%** |

LoRA_4 and BayesLoRA_14 are the matching seed-123 saved artifacts and genuinely tie on recall. Across five seeds, BayesLoRA had a 6.28-point higher mean and a narrower range (83.41–85.51% versus 70.64–85.50%). Mean measured training time per task block was 11.5 s for LoRA and 13.7 s for BayesLoRA. Thus BayesLoRA's advantage over a **two-step** LoRA baseline was reliability, with a modest training-time cost. It is not accurate to compare two-step BayesLoRA only against the much slower 30-step LoRA baseline.

The original 30-step cold-start LoRA was measured on seeds 123 and 124: recall 84.04% and 79.77%, with about 91 s per training block. These two runs show the cost of the longer search but are too few for a reliable mean-performance ranking.

An earlier two-step LoRA probe also used **two evaluation steps** and had poor recall. That is not comparable to LoRA_4 or BayesLoRA_14, which each used 30 evaluation steps. The matched comparison above corrected this budget error.

## Simple LoRA transfer ablations

All use the five-seed protocol above and 30-step evaluation. For reactive variants, the detector used `RECONFIG_REACTIVE_SD=5` and `RECONFIG_REACTIVE_EMA=0.9`; detected changepoints got 30 cold-search steps. `LoRAWarmReactive` keeps the selected search logits across ordinary batches and resets them on a detected changepoint. Its implementation and launchable config remain in this project.

| Training mask policy | Mean recall | Mean training accuracy | Time per task block |
| --- | ---: | ---: | ---: |
| Cold two-step search each batch (`LoRA`) | 77.78% | 78.66% | 11.5 s |
| Reactive cold two-step search on stable batches | 73.14% | 77.60% | 12.6 s |
| Warm two-step search without reactive resets | 76.07% | 92.86% | 12.1 s |
| **Warm two-step search with reactive resets** | **79.58%** | **91.78%** | **13.8 s** |
| BayesLoRA | 84.06% | 89.32% | 13.7 s |

Warm starts plus resets transferred a modest 1.8-point recall gain to LoRA, but did not match BayesLoRA. Warm starts alone dramatically improved training fit while reducing recall, evidence that training accuracy is a poor proxy for cold-search recoverability here. Reactive cold search often missed task changes because its loss signal was noisy. Multiplying carried logits by one-third before each warm search reduced the five-seed recall mean to 76.09%; that experimental parameter was removed.

## Other diagnostic results

- In an earlier five-seed BayesLoRA comparison at `K=256`, raising the stable-update temperature from 1 to 3 raised mean recall from about 78.2% to 84.5%, winning all five paired seeds. This was the clearest small hyperparameter effect. The changepoint and evaluation searches still used their own temperature of 1.
- Removing the stable-update KL penalty while keeping persistent logits and temperature 3 reduced mean recall from about 84.2% to 82.6% on three paired seeds. This suggests a modest benefit from KL, but it does not establish that KL accounts for the larger gap to LoRA.
- Gauge fixing (shifting every posterior logit by a common value to make the relaxed mask sum to `K`) did not improve five-seed BayesLoRA recall at temperature 3: about 84.1% versus 84.5% without it. It also added roughly 5–6% LoRA task-block time, so it was not adopted. On basic, non-LoRA BayesFTN, four-seed temperature changes were small (about 80.2% at temperature 1 versus 80.7% at 3); gauge fixing gave modest gains there.
- The warm/reactive LoRA ablations do not isolate the remaining BayesLoRA advantage. Bayesian mask sampling, temperature, KL, and the exact search update differ. High train accuracy under warm LoRA and lower cold evaluation recall are consistent with a mask bank that becomes too specialized, but this is an interpretation rather than a demonstrated unique cause.

## Reproduction pointers

- Current configs: `config.LoRA`, `config.LoRAWarmReactive`, and `config.BayesLoRA` in `config.py`. The default `BayesLoRA` now uses `K=1024` and 1,500 training batches, so reproducing the tables requires setting those back to `K=256` and 500 for the run.
- Model code: `lora_ftn.py`, `bayes_lora_ftn.py`, `bayes_reconfig.py`, and `bayes_lora_rnn_ftn.py`.
- Saved reference artifacts: `artifacts/LoRA_4/` and `artifacts/BayesLoRA_14/`; run logs are in `logs/`. The additional seed and ablation runs were launched on the existing actl pod and summarized above; their temporary probe scripts were removed after use.
