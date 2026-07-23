# FTN-Chat

Live, prompt-routed expert-knowledge chat demo. A frozen Qwen2.5-7B-Instruct
backbone with a grid-arranged bank of LoRA atoms attached to the **last 25% of
decoder layers**; each user prompt is used as a self-supervised reconfiguration
batch (minimize prompt perplexity) to select the expert atom subspace, which is
then visualized live as an animated square mask while the answer streams.

The training design mirrors the base/ experiment
`run_experiments.py --config Large NoReplay Reactive_Low`:
annealSTE mask search (30 steps), plain KWTA (SmoothKWTA options available but
off), **no replay**, and **reactive (loss-triggered) reconfiguration** during
training (sd=5.0, ema=0.9). Bank size: H=32²=1024 atoms, k=128, rank-1 atoms.
At inference the mask still reconfigures on every user prompt.

Post-training is **context distillation** (no raw-text training): the teacher
is the same frozen backbone with the source excerpt in context and the mask
zeroed; the student (masked bank, closed book) minimizes KL to the teacher's
answer-token distributions over synthesized Q&A.

**Full spec / rationale / handoff:** see [`CONTINUATION.md`](./CONTINUATION.md).

## Domains

All sources are snapshotted into `corpus/` by `make_corpus.py` (the actl pod
syncs only this folder, so the surrounding repo is not available at runtime).

- `base_ftn` — `base/ftn.py` + `base/config.py` (FTN class + every experiment parameter)
- `explorer` — `explorer/agents/agent_recode.py` + RECODE + reservoir encoders
  (RDDPG_Dual architecture and its exploration method)
- `globo` — `globo_explorer/globo_game` docs + Globo-authored mods (what Globo
  adds/changes on top of the Luanti/Minetest engine)
- `rl_mt10` — `rl/robotics/train_mt10_bc.py` + `train_mt10_ddpg.py` + their
  `ftn.py` (FTN adapted to behavior cloning and DDPG)

## Run

Everything runs on the actl pod `ftn-chat` (8×H100). Edit locally; sync
propagates. From the pod's `/home/dev/workspace`:

```bash
python3 make_corpus.py                       # LOCAL: refresh corpus/ snapshots
uv run python data.py                        # ingest + per-task token report
uv run python smoke_shard.py                 # sharding + distill-step smoke test
uv run python qa_synth.py                    # synthesize distillation triples (once)
uv run python train.py --fresh               # train all registered tasks -> checkpoints/
uv run python routing_check.py               # routing accuracy + overlap matrix
uv run python app.py                         # chat UI on 127.0.0.1:7860
```

## Adding a dataset (continual learning)

It's block-sequential CL: new tasks are learned ON TOP of the existing bank
(no replay — retention rests on mask separation + the frozen backbone).

1. Add its source files to `make_corpus.py`, run `python3 make_corpus.py` locally.
2. Register it in `registry.py` (one line — a key, label, and corpus globs).
3. Synthesize + train just that block:
   ```bash
   uv run python qa_synth.py mydata
   uv run python train.py --tasks mydata
   ```
4. Restart the app — the new task's color/label appear in the legend automatically.

Other knobs: `--epochs N --steps M` (depth), `--reconfig-steps S`, `--retrain`
(re-run an already-trained block), `--fresh` (wipe and start over). State lives
OUTSIDE the synced workspace in `~/ftn_chat_state/checkpoints/` on the pod
(`{bank,refs,mask}.pt` + `state.json`, which also persists the reactive-reconfig
EMA state; override with `FTN_CKPT_DIR`); training checkpoints after every
block, so it's resumable.

To view the app from your laptop, forward port 7860 through actl (the pod has
no tailscale; never use the public gradio tunnel — the data is in-house).
Either restart your sync session with the port:

```bash
actl pod up --port 7860        # then run app.py in its remote shell
```

or, without disturbing a running sync session, from a second terminal:

```bash
actl pod exec --port 7860      # opens a pod shell with the forward
```

and open http://localhost:7860 in your browser. The forward lives only while
that actl session stays open.

All knobs live in `config.py`.
