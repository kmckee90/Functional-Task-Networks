# FTN-Chat

Live, prompt-routed expert-knowledge chat demo. A frozen Qwen2.5-1.5B-Instruct
backbone with a grid-arranged bank of LoRA atoms; each user prompt is used as a
self-supervised reconfiguration batch (minimize prompt perplexity) to select the
expert atom subspace, which is then visualized live as an animated square mask
while the answer streams.

**Full spec / rationale / handoff:** see [`CONTINUATION.md`](./CONTINUATION.md).

## Domains
- `sandbox` — `astera-org/sandbox` repo (code)
- `ftn_paper` — `../paper/arxiv/*.tex`
- `robotics` — `../../robotics/*.py`

## Run

```bash
cd experiments/func_net_kt/ftn_chat
uv run python data.py                        # ingest + per-task token report
uv run python train.py --fresh               # train all registered tasks -> checkpoints/
uv run python routing_check.py               # routing accuracy + overlap matrix
uv run python app.py                         # chat UI on 127.0.0.1:7860
```

## Adding a dataset (continual learning)

It's block-sequential CL: new tasks are learned ON TOP of the existing bank, with
replay protecting earlier tasks. To add a dataset:

1. Register it in `registry.py` (one line — a key, label, and file globs):
   ```python
   register("mydata", "My Dataset", ["/abs/path/**/*.py", "/abs/path/*.md"])
   ```
2. Train just that block (the bank, refs, and replay buffers are loaded from the
   existing checkpoint; old tasks are skipped and replayed, not retrained):
   ```bash
   uv run python train.py --tasks mydata
   ```
3. Restart the app — the new task's color/label appear in the legend automatically.

Other knobs: `--epochs N --steps M` (depth), `--reconfig-steps S`, `--retrain`
(re-run an already-trained block), `--fresh` (wipe and start over). State lives in
`checkpoints/{bank,refs,replay}.pt` + `state.json`; training checkpoints after
every block, so it's resumable.

Then on the server expose it privately over Tailscale (not the public gradio
tunnel — the data is in-house):

```bash
tailscale serve 7860
```

and open the printed MagicDNS `https://…ts.net` URL from your laptop.

All knobs live in `config.py`.
