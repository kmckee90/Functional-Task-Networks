# Long-Context FTN Sequence Benchmark

This folder contains a small PyTorch benchmark pipeline for comparing long-context
sequence prediction models on the same examples:

1. `hf_frozen`: a frozen pretrained causal LM reference, defaulting to
   `Qwen/Qwen2.5-7B`.
2. `ftn_lora`: the same frozen backbone with a mask-gated FTN-LoRA bank.
3. `gru`: an untrained vanilla `torch.nn.GRU` causal LM placeholder.

The benchmark is intentionally plain PyTorch and Hugging Face Transformers. It is
designed for quick model iteration rather than maximum throughput.

## Setup

On the ACTL NLP pod, install the Python dependencies in the active environment:

```bash
uv pip install -r requirements.txt
```

This package deliberately uses the narrow `requirements.txt` instead of the
repo-root `pyproject.toml`; the root project pulls in unrelated robotics/JAX
dependencies. The ACTL PyTorch image provides torch/CUDA.

## Artifact Storage

Do not write heavyweight generated data or checkpoints under the synced
workspace on the ACTL pod. Relative generated paths such as `runs` and
`data/processed/...` are rooted under either:

1. `run.artifact_root` in the config, or
2. `LONG_CONTEXT_ARTIFACT_ROOT` in the environment.

The Qwen pod configs use:

```yaml
run:
  artifact_root: /home/dev/artifacts/long_context
```

That keeps manifests, logs, and checkpoints outside
`/home/dev/workspace/long_context`, while preserving simple relative paths in
the config. Absolute paths, such as LoCoMo source data under
`/home/dev/datasets`, are left unchanged.

## Quick Smoke Run

Use the tiny config for local/pod sanity checks:

```bash
python -m long_context_bench.run prepare --config configs/gru_smoke.yaml
python -m long_context_bench.run train --model gru --config configs/gru_smoke.yaml
python -m long_context_bench.run eval --model gru --config configs/gru_smoke.yaml \
  --checkpoint runs/gru_smoke/gru/checkpoints/latest.pt
```

Use the full smoke config to exercise all three model paths with a tiny
Qwen/Llama-style backbone that supports FTN-LoRA target modules:

```bash
python -m long_context_bench.run all --config configs/full_smoke.yaml \
  --models hf_frozen ftn_lora gru
```

## H100 Reference Run

The default long-context config uses `Qwen/Qwen2.5-7B` as the pretrained sequence
model and synthetic needle-in-a-haystack examples. It also includes a LoCoMo
loader stanza; point it at a local LoCoMo JSON/JSONL export or Hugging Face
dataset id before enabling it.

```bash
python -m long_context_bench.run prepare --config configs/qwen25_7b_niah_locombo.yaml
python -m long_context_bench.run eval --model hf_frozen --config configs/qwen25_7b_niah_locombo.yaml
python -m long_context_bench.run train --model ftn_lora --config configs/qwen25_7b_niah_locombo.yaml
python -m long_context_bench.run train --model gru --config configs/qwen25_7b_niah_locombo.yaml
```

For full-sequence long-context comparisons, keep `eval.max_length` at or above
the generated context length. The default overflow behavior skips examples that
do not fit rather than silently truncating away the evidence.

## Data Format

`prepare` writes inspectable JSONL manifests:

```json
{"id":"...","dataset":"niah","split":"eval","prompt":"...","target":" answer","answers":["answer"],"metadata":{}}
```

Training/evaluation tokenizes these records at runtime into causal LM examples
with labels masked to the answer/prediction span.

## FTN-LoRA As Long-Context Memory

The LoCoMo memory path tests a different claim than ordinary long-context QA:
FTN-LoRA first writes a conversation into a masked LoRA subspace, then answers
later questions without receiving the conversation again.

Smoke test all memory stages on the ACTL pod:

```bash
python -m long_context_bench.run memory-all \
  --config configs/locomo_memory_smoke.yaml
```

For a real LoCoMo run, copy `configs/qwen25_7b_locomo_memory.yaml`, set the
LoCoMo `path` or `dataset_name`, and enable the dataset. The protocol is:

1. `memory-prepare` normalizes LoCoMo rows into episodes with memory blocks and
   held-out QA pairs.
2. `memory-write` trains only active FTN-LoRA atoms on conversation text from
   `memory.write.split` while never using QA answers.
3. `memory-eval` asks question-only prompts and reports recall under:
   `zero`, `oracle`, `query`, `wrong`, and `full_context` modes.

Interpretation:

- `oracle` high, `query` low: the bank can store memory, but routing needs work.
- `oracle` and `query` both low: the write/capacity setup is weak.
- `query` close to `oracle`, and both above `zero`/`wrong`: evidence for FTN-LoRA
  acting as a recallable long-context memory.
