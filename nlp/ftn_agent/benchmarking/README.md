# FTN-Agent benchmarking — LoCoMo-10

Does FTN-LoRA parametric memory actually work? We ingest each LoCoMo
conversation session-by-session into the atom bank (weights, not RAG), then
answer the benchmark's questions **closed-book** — reconfiguring the mask on
the question to route to the knowledge-containing subspace.

Decisions locked with Kevin 2026-07-08 (see `../design.md` for the agent itself):

- **Benchmark:** LoCoMo-10 (10 conversations × ~26k tokens, ~1,540 questions
  after dropping the adversarial category, whose ground truth is unusable).
  Corrected answer key from the `dial481/locomo-audit` audit where available.
  **Contamination caveat:** LoCoMo has been public since Feb 2024 and may be in
  Qwen3-32B's pretraining data — the closed-book zero-ingestion floor doubles
  as the contamination probe and is reported prominently.
- **Judge:** cloud LLM judge (Claude via the authenticated `claude` CLI, or
  GPT-4o with an API key) + token-F1 cross-check. Judging runs laptop-side on
  the small synced prediction files; the pod only generates.
- **Training objective:** context distillation from the start — teacher = the
  same frozen Qwen3-32B *with the session transcript in context* (mask zeroed),
  student = the masked bank closed-book; KL on self-study synthetic data.
  Data recipe per the 2024–26 knowledge-injection literature: self-contained
  fact inventory → paraphrases + multi-angle/reversed QA + session-level
  templates, ≥30–100× source tokens, 1:1 QA:document mixture, ~10% generic
  replay.
- **Baselines:** closed-book floor, full-context ceiling, single monolithic
  LoRA trained sequentially on identical data (isolates FTN routing). Published
  memory-system numbers (Mem0 ~67 J, Zep, Locas ~35–42 F1) are cited, not
  re-run.

## Storage (design.md §7 applies)

Everything large lives on the PVC, outside the synced tree:

- `$FTN_AGENT_STATE/benchmarking/locomo/` — dataset + audit clone
- `$FTN_AGENT_STATE/benchmarking/banks/` — experiment bank checkpoints (rolling)
- `$FTN_AGENT_WORK/benchmarking/` — synthetic datasets (pruned after training)
- `benchmarking/results/` — small prediction/score JSONs only (synced, OK)

## Run

```bash
# pod, one-time: fetch data
bash benchmarking/setup_data.sh

# pod: baselines (floor = contamination probe, ceiling = full-context)
python benchmarking/run_baselines.py --mode floor
python benchmarking/run_baselines.py --mode ceiling

# laptop: score synced predictions
python benchmarking/score.py results/floor.jsonl --judge claude-cli
```
