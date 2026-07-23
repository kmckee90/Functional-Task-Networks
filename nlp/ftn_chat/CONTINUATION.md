# FTN-Chat: live prompt-routed expert-knowledge chat demo — continuation doc

**Status:** v2 REDESIGN (2026-07-19) — code complete, retraining required
(`--fresh`; the old bank is shape-incompatible). Older sections below describe
v1 and are historical.

### v2 redesign (2026-07-19): Large/NoReplay/Reactive_Low + context distillation
Project moved repos; new home is `nlp/ftn_chat` with its own pyproject, running
on actl pod **ftn-chat** (8×H100 80GB, 250Gi PVC, namespace obelisk). The pod
syncs ONLY this folder, so all task sources are snapshotted into `corpus/` by
`make_corpus.py` (run locally after upstream edits).

- **Training matches `base/run_experiments.py --config Large NoReplay
  Reactive_Low`** (reactive + no-replay + plain-KWTA parts): annealSTE reconfig
  30 steps @ lr 0.3 (init logits ~N(0,1e-5), matching base), plain KWTA —
  SmoothKWTA lateral options (`lateral_steps/kernel/polyak/noise_sd/torus`)
  kept in config + GUI but OFF by default. Bank size set independently of the
  Large class (Kevin, rev 2 of this redesign): grid_dim=32 (H=1024 atoms),
  k=128, rank-1 atoms, alpha=16.
- **Banks on the LAST 25% of decoder layers only** (`lora_layer_frac=0.25` →
  layers 21–27 of Qwen2.5-7B): the LoCoMo layer sweep showed extractable
  knowledge lives in the last ~25% of blocks; mid-stack deltas were dead
  weight. ~0.65B atom params total (~1.3GB bank.pt); model.py's explicit
  device_map packs the 21 un-banked layers on GPU 0 and spreads the 7 banked
  layers one per GPU 1–7.
- **Checkpoints live OUTSIDE the synced workspace**: default ckpt_dir is
  `~/ftn_chat_state/checkpoints` on the pod's home PVC (FTN_CKPT_DIR
  overrides). DevSpace sync is bidirectional and once started pulling a bank.pt
  to the laptop; large artifacts must never sit in /home/dev/workspace.
  `.actlignore` additionally excludes checkpoints*/, data_cache/, *.pt, logs
  (takes effect on sync-session restart).
- **Reactive (loss-triggered) reconfiguration during training** (Reactive_Low):
  reconfig once on the first batch ever, then only when batch loss > EMA mean
  + 5.0·sd (ema=0.9, warmup ≈10 batches). Boundary-free: EMA state persists
  across blocks AND across train.py invocations (state.json). Reconfig target
  = one random batch member's bare question (the chat-time signal). Adam state
  resets per block (as base does per task). Inference is unchanged: the app
  reconfigures on EVERY user prompt.
- **NO replay** (NoReplay): replay buffers/`replay.pt`, mid-band overlap
  filtering, and rehearsal-by-default are gone from train.py.
- **Context distillation, no raw-text training** (Cartridges-style self-study,
  the recipe locked in the LoCoMo work): `qa_synth.py` generates diverse
  self-contained questions per source chunk, then the teacher answers WITH the
  chunk in context (batched, greedy); triples {question, answer, context} are
  cached. train.py recomputes teacher logits on the fly — teacher = the SAME
  backbone with the mask zeroed (bank inert) + context in the prompt; student
  = masked bank, closed book; loss = KL over the answer tokens (kl_temp,
  optional ce_weight). Teacher logits are mask-independent, so a reactive
  re-reconfig only re-runs the student pass.
- **Four new domains** (old sandbox/cscg/sudoku/genetic/robotics tasks dropped):
  `base_ftn` (base/ftn.py + config.py — parameter-level Q&A), `explorer`
  (agent_recode.py + recode.py + reservoir encoders — RDDPG_Dual architecture
  + RECODE exploration), `globo` (globo_game docs + Globo-authored mods only;
  vendored mods excluded except mod.confs — "what does Globo add vs the
  engine"), `rl_mt10` (train_mt10_bc.py + train_mt10_ddpg.py + rl/robotics/
  ftn.py — FTN adapted to BC and DDPG).
- Misc: chunk_tokens 256→512 (richer distillation contexts), app UPSCALE now
  adaptive (~1024px map at any grid_dim), refs built from QA-question reconfigs,
  state.json guards H/k/rank/n_wrappers. `_probe_memorization.py` is a stale v1
  throwaway (hardcoded cscg).

**Status (older, v1):** v1 COMPLETE and running (2026-06-10). All modules implemented; bank
trained (`checkpoints/bank.pt`, `refs.pt`); app serving on `127.0.0.1:7860`.

### Current state / results (2026-06-10, rev 2)
- Bank: H=256 (16×16 grid), k=48, rank 4 → 1.18B atom params on frozen Qwen2.5-1.5B.
- **Reconfig steps = 30 for BOTH train and inference** (paper-documented AnnealSTE
  budget; `reconfig_steps=train_reconfig_steps=30`). Replay uses stored masks.
- **Domains (rev 2):** swapped the diffuse whole-sandbox domain for just the CSCG
  model (`packages/algorithms/src/algorithms/cscg/*.py` + quickstart notebook,
  ~49k tok). Now all three are peer-sized: cscg 49k, ftn_paper 41k, robotics 30k.
- Trained 1 epoch × 200 steps/domain @ 30 reconfig steps (~6.1s/step, ~70 min on
  one H100). NLL → 0.27 (cscg) / 0.47 (ftn_paper) / 0.01 (robotics).
- **Overlap matrix is now clean & on-thesis:** cscg vs {ftn_paper, robotics} =
  0.12 / 0.15 (≈random floor → genuinely distinct), ftn_paper vs robotics = 0.46
  (correlated FTN pair shares an atom core). This is the result we want — the
  CSCG swap fixed the all-high overlap of the sandbox run.
- App verified live (gradio_client) with a CSCG question: coherent answer, routes
  to CSCG, no handler errors. Serving on the unchanged Tailscale URL.

### Process-management gotcha (cost a debugging cycle — DO NOT repeat)
`pkill -f "python app.py"` does NOT match the child `python3 app.py` (the "3"),
so old app instances survived restarts, kept port 7860 bound, and new launches
silently died — leaving a STALE app serving without new fixes. Always
`pkill -9 -f "app.py"` (match on the script name only) and confirm port 7860 is
free + the right PID owns it before concluding a fix "didn't work".

### Incremental CL refactor (2026-06-10, rev 3)
The trainer is now true block-sequential continual learning with persisted state,
so you can keep adding datasets:
- **`registry.py`** — the extension point. `register(key, label, globs, color=...)`
  defines a task from file globs. New tasks append to `DEFAULT_ORDER` and flow
  through config/data/app/train automatically (labels + mask colors included).
- **`data.py`** — `build_task(cfg, tok, key)` builds ONE task from its globs
  (`.ipynb` aware); `build_tasks` builds several; `build_domains` is a back-compat
  alias. No more per-domain hardcoded gatherers.
- **`train.py`** — load-or-init from `checkpoints/`, train each not-yet-trained
  task as a block (`--tasks`, `--fresh`, `--retrain`, `--epochs`, `--steps`,
  `--reconfig-steps`). Per step: reconfigure on the chunk → train selected atoms;
  replay over PRIOR tasks' persisted (chunk, mask) buffers (mid-band overlap
  filter) for retention. **Checkpoints after every block** (bank.pt, refs.pt,
  replay.pt, state.json) → resumable and incremental across runs.
- Deeper default schedule: `epochs=1, steps_per_domain=400` (was 200).
- Add a dataset: register it, then `uv run python train.py --tasks <key>` — it
  trains the new block on top, replays old tasks, appends its ref mask, restarts
  the app showing the new color. No retraining of old tasks.

### Q&A SFT recipe (2026-06-10, rev 4)
Raw-text-only training over-memorized (deep robotics block, 12 chunks @ 400 steps
→ NLL 0.001, regurgitated raw LaTeX instead of answering). Fix = the standard
domain-QA recipe: **mix raw continued-LM with answer-masked Q&A SFT.**
- **`qa_synth.py`** — generates grounded Q&A per task with a LOCAL model
  (`CFG.qa_gen_model`, default Qwen2.5-7B-Instruct — no data egress). For each
  train+support chunk it asks for `qa_per_chunk` self-contained Q&A pairs, caches
  to `data_cache/qa/<task>.json`. Run once before training: `uv run python qa_synth.py`.
- **`data.py`** — `format_qa` (answer-masked chat: -100 on prompt, answer ids on
  the answer span; reconfig sees the BARE question to match chat-time inference);
  `build_examples` returns {raw, qa, support} example pools.
- **`train.py`** — each step draws a Q&A example with prob `CFG.qa_frac` (0.6)
  else a raw chunk; reconfigure on `recfg` → train on (ids, lbl). Replay buffer
  now stores (ids, lbl, mask) so prior-task Q&A is replayed correctly. Mixing Q&A
  also raises the example count, lowering the oversampling that caused the
  over-fit.
- Config knobs: `qa_gen_model`, `qa_per_chunk`, `qa_frac`, `qa_max_len`, `qa_sys`.
- Workflow now: `qa_synth.py` → `train.py --fresh` → restart app.
- Still NOT doing general-instruction rehearsal (a further anti-forgetting lever);
  answer-masked Q&A alone should largely preserve chat behavior.

### GUI: live k + mid-rollout reconfiguration (2026-06-12, rev 8)
Three new Controls-accordion sliders in `app.py`, all plumbed through
`PerplexityConfigurer.iterate(... k, init_mask, init_a, init_b)`:
- **`k` — active atoms in the mask.** Overrides the active-atom count per request
  and is written back to `MaskController.k`, so the answer also generates at that
  k. `BankedLinear.scaling` is now a **property** (`alpha / controller.k`, was a
  baked `alpha/cfg.k`) so changing k at inference stays magnitude-normalized
  (more atoms don't blow up the delta, fewer don't fade it). Bank trained at
  k=128; this is a live experiment knob.
- **Mid-rollout reconfig every N tokens (0 = off).** When >0, generation decodes
  in chunks of N tokens; between chunks it re-runs `mid_steps` of reconfiguration
  on the FULL sequence so far (prompt + rollout) and animates it, so the expert
  subspace tracks the answer as it grows. Implemented as a manual chunked-decode
  loop (`return_dict_in_generate` in a thread to recover the token ids, re-fed
  each chunk — no KV-cache reuse across chunks, fine at demo lengths). `0` keeps
  the old single-shot decode. Companion slider **Mid-rollout reconfig steps**.
- **Mask init `a·current_mask + b·noise`.** `iterate` now seeds the mask logits
  from `init_a * current_mask + init_b * noise` (falls back to `init_b*noise`
  when a=0). So a=1,b=0 = warm-start from current mask; a=0,b=1 = explore from
  pure noise; small a,b = near-zero. Defaults a=1.0, b=0.3 (inert on the first
  prompt search since the mask is all-zeros then). Verified in isolation:
  a=1,b=0 reproduces the current mask exactly; a=0,b=1 does not.

**Viz:** dropped the cumulative-EMA "experts used this conversation" raster — the
heatmap now shows ONLY the current mask. `render()` lost its `ema` arg; `_animate()`
lost its `ema` raster. Atom coloring: colored (domain-claimed) atoms render at full
domain color when active, `FAINT=0.18` when inactive; UNcolored atoms render 50%
gray when active, black when inactive. A per-network **prevalence bar panel**
(`_bar_panel`) is composited to the right of the heatmap inside `render()` (one
image out, so no new Gradio output wiring): one horizontal bar per domain, in the
network's color, length = fraction of the active atoms claimed by that domain's
reference mask — so you can read off whether the search configured on-topic.
App renamed **ChatFTN** (window title + header).

Pure code change — no retraining. Restart the app to pick it up (see the
process-management gotcha above: `pkill -9 -f "app.py"`, confirm port 7860 owner
before concluding).

### Known follow-ups (deferred by Kevin — do not block on these)
- Revisit data mixture + mask separation (the high overlap). Possible levers: more
  distinct domains, load-balancing / diversity pressure on atom selection, lower
  token ceiling, more training.
- QA-pair synthesis per domain for sharper answers (v1 is plain causal-LM).
- Scale to 7B. Longer training (epochs=2, steps_per_domain=300 are the original
  defaults, trimmed for fast first validation).

### To connect (server is `g093`, tailnet `g093.taila1eba.ts.net`)
App binds 127.0.0.1:7860. On the server run `tailscale serve 7860`, then open
`https://g093.taila1eba.ts.net` from the laptop. For persistence run the app under
tmux/systemd (not tied to an SSH shell).

---

**Original plan below (started 2026-06-10)**
**Owner notes for Kevin + future-Claude:** this is the working spec/handoff. If the
session is lost, read this top-to-bottom and you can resume without re-deriving the
plan. Code lives in `experiments/func_net_kt/ftn_chat/`.

---

## 1. What we are building (the one-paragraph version)

An interactive chat agent built on a **frozen Qwen2.5-1.5B-Instruct** backbone with a
**bank of LoRA atoms** arranged on a 2-D square grid. When the user types a question,
we use the **question text itself as a self-supervised reconfiguration batch**: we
optimize a binary top-k mask over the atom bank to **minimize the language-model
perplexity of the prompt** (backbone + bank frozen, only the mask logits move). The
selected atoms = the "expert knowledge subspace" for that question. We then generate
the answer under that mask. The whole thing is visualized live: a square heatmap of
the mask `m` animates as it "searches" for the right subspace, then the answer streams
in. Served from the GPU server, viewed from Kevin's laptop over Tailscale.

This is the **scaled, label-free, interactive** instantiation of FTN: same
support-conditioned reconfiguration idea as the paper, but (a) the support signal is
unsupervised next-token loss on the prompt instead of a labeled batch, and (b) it runs
live in a chat loop.

## 2. Why this design (decisions already made with Kevin)

- **Prompt-perplexity reconfiguration, not retrieval.** The configurer minimizes the
  NLL of the user's prompt under the selected atoms. Whatever subspace best *models*
  the question's domain language is selected. This is label-free (we don't have the
  answer at inference), and it's what makes the live "search" animation meaningful.
- **Correlated domains are fine / on-point.** FTN's thesis is that correlated tasks
  *share* computation rather than interfere. Two of our domains (the FTN paper and the
  robotics/RL adaptation of FTN) overlap heavily in vocabulary. If they end up sharing
  a mask region, that is the method correctly reporting correlation — a feature, not a
  bug. No task labels means there is no "wrong" partition to violate.
- **`B ~ N(0, 0.01)` init is mandatory (not standard B=0).** In the classification
  FTN-LoRA work we found standard LoRA `B=0` makes the configurer gradient-blind to
  untrained atoms — it can only *deselect*, never *select*. For a selection-driven demo
  that is fatal. So `init_b_zero=False` is the default here.
- **2-D grid topology over the atom bank + lateral smoothing (PolyakKWTA style).** The
  atom bank is laid out as a `grid_dim x grid_dim` square so the mask renders as a
  clean image, and we apply a lateral conv-smoothing step during reconfiguration
  (ported from `func_net_spatial/masks.py`) so active atoms form spatially coherent
  blobs instead of salt-and-pepper. This is purely for legibility of the demo; the
  mechanism is unchanged.
- **Data balancing by competence, not proportion.** `astera-org/sandbox` (~21 MB)
  dwarfs the FTN paper (~13k words) and the robotics folder (~12k words). We cap each
  domain's contribution to the training mixture at a per-domain **token ceiling** and
  oversample the small domains up toward it, so sandbox doesn't swamp the bank. The
  ceiling is a config knob. Goal: a credible "general Astera code" expert + detailed
  FTN/robotics experts.
- **1.5B first.** Get it running at 1.5B on a single H100; 7B is a later scale-up
  (architecture is identical, just heavier).
- **Serving: host on server, view over Tailscale.** The model lives where the GPU is;
  the laptop only runs a browser. Bind the web app to localhost and expose via
  `tailscale serve <port>` (private, encrypted, TLS, MagicDNS URL) — NOT Gradio's
  public `share=True` tunnel (proprietary in-house data). SSH `-L` forward is the
  fallback. Run the app as a persistent process (tmux/systemd) so it survives SSH
  disconnects.

## 3. The three data domains

| Domain key | Source | Notes |
|---|---|---|
| `cscg` | The CSCG model inside `astera-org/sandbox`: `packages/algorithms/src/algorithms/cscg/*.py` + `packages/algorithms/notebooks/cscg/cscg_quickstart.ipynb` | Focused, peer-sized (~49k tok). Still clones the repo to `data_cache/sandbox` to get the files. (rev 2: replaced the whole-sandbox domain, which gave diffuse high overlap.) |
| `ftn_paper` | `experiments/func_net_kt/paper/arxiv/*.tex` (main + `sections/` + `tables/`), optionally `refs.bib` | ~13k words of LaTeX/ML prose. Ingest fully. Exclude `.aux/.log/.bbl/.pdf`. |
| `robotics` | `experiments/robotics/*.py` (`ftn.py`, `train*.py`, `config.py`) + `nohup.out` | FTN adapted to RL. ~12k words, code-heavy. Ingest fully. Exclude the 100+ `.npz`/`.png` result files. |

Paths are resolved relative to the `ftn_chat/` package dir via `__file__` (paper is
`../paper/arxiv`, robotics is `../../robotics`).

## 4. Architecture / file map (`ftn_chat/`)

- `config.py` — all knobs: model name, `H` (bank size, perfect square), `grid_dim`,
  `k` (active atoms), rank, alpha, dropout, `init_b_zero=False`, per-domain token
  ceiling, reconfig steps/lr, lateral kernel size/torus, paths, port.
- `spatial.py` — `kwta`, `mask_to_grid`, `mask_to_vec`, `LateralSmoother` (circular/zero
  conv for blob coherence). Ported from `func_net_spatial/masks.py`.
- `lora_bank.py` — tailored copy of `ftn_lora/lora_bank.py`: `LoRABankConfig`,
  `MaskController`, `BankedLinear` (memory-safe einsum `...kr,kro->...o`),
  `install_lora_bank`, state-dict save/load. Default `init_b_zero=False`.
- `model.py` — load Qwen + tokenizer, install bank, helpers to count params, set/get
  mask, save/load bank.
- `data.py` — clone sandbox; walk + chunk each domain into token windows; build a
  **balanced** mixture with per-domain token ceiling + oversampling. Returns per-domain
  lists of tokenized chunks (train/support/heldout splits).
- `perplexity_configurer.py` — THE core piece. AnnealSTE inner loop over H mask logits
  minimizing prompt NLL (backbone+bank frozen), + lateral smoothing + final kwta.
  Records the mask at every step for animation. Self-supervised analogue of
  `ftn_lora/configurer.py`.
- `train.py` — sequential-over-domains training: per-batch perplexity reconfig on
  domain text, train the selected atoms on causal-LM of domain chunks, light
  mask-overlap-filtered replay. Saves bank + per-domain reference masks + overlap
  matrix. Entry: `cd ftn_chat && uv run python train.py`.
- `routing_check.py` — headless de-risk. For held-out prompts per domain, run the
  configurer and print (a) which domain each prompt's mask overlaps most, (b) the full
  cross-domain mean mask-overlap matrix. Run BEFORE trusting the demo. Works on an
  untrained bank too (baseline) and after training.
- `app.py` — Gradio chat UI: prompt box, streamed response, live square mask heatmap +
  cumulative-EMA "experts used this conversation" raster. On each message: reconfigure
  (record frames) → animate frames at fixed cadence → generate under final mask via
  `TextIteratorStreamer`. Binds `127.0.0.1:<port>`; prints the `tailscale serve` hint.
- `README.md` — short pointer to this doc + run commands.
- `data_cache/`, `checkpoints/` — gitignored artifacts (cloned repo, bank weights,
  saved masks).

## 5. Implementation order (de-risk routing before UI)

1. `config.py`, `spatial.py`, `lora_bank.py`, `model.py` — scaffolding.
2. `data.py` — ingest + balance; smoke-test chunk counts per domain.
3. `perplexity_configurer.py` — the self-supervised configurer.
4. `routing_check.py` — **run on an UNTRAINED bank first** to confirm the loss/grad
   plumbing works and see baseline overlap; then we have the harness ready.
5. `train.py` — train the bank on the 3 balanced domains. Save bank + ref masks +
   overlap matrix.
6. Re-run `routing_check.py` post-training — **this is the crux gate.** We want
   held-out prompts to route to their own domain, and a sensible overlap matrix
   (sandbox distinct; paper/robotics may correlate — fine).
7. `app.py` — wire the Gradio UI once routing is confirmed.
8. Serve via `tailscale serve`; document in README.

## 6. Key technical gotchas (carried from the classification FTN-LoRA work)

- **Configurer memory:** freeze backbone + bank (`requires_grad=False`) and DISABLE
  gradient checkpointing during the inner loop — only the H-dim mask logits need grad.
  Re-enable after. (See `ftn_lora/configurer.py` for the exact toggle pattern.)
- **BankedLinear memory:** never materialize the `(B, L, k, out)` intermediate; contract
  atom+rank dims in one einsum (`proj = einsum("...i,kir->...kr"); delta =
  einsum("...kr,kro->...o")`). Already in the tailored `lora_bank.py`.
- **B-init:** `init_b_zero=False` (`B ~ N(0,0.01)`), else the configurer can't select
  untrained atoms.
- **Default device:** set `torch.set_default_device("cuda")` at entry points; avoid
  per-tensor device plumbing (Kevin's standing preference). Use `uv run`, not bare
  python.
- **Reconfig loss is NLL of the PROMPT tokens** (all of them), not answer tokens — this
  is the self-supervised change vs the classification version.

## 7. Open questions / v2 ideas (not blocking v1)

- QA-pair synthesis per domain (use a strong model to generate Q&A from docs) to make
  the experts sharper at *answering* vs just modeling domain text. v1 uses plain
  causal-LM on domain chunks + the frozen instruct backbone's chat ability.
- Scale to 7B once routing + demo are solid.
- Smoother animation via FastAPI + WebSocket + `<canvas>` if Gradio's frame cadence
  feels chunky for a showcase.
- Encourage/measure mask separation explicitly if domains don't separate enough.

## 8. How to run (once built)

```bash
cd experiments/func_net_kt/ftn_chat
uv run python data.py            # ingest + report per-domain chunk/token counts
uv run python routing_check.py --untrained   # baseline plumbing check
uv run python train.py           # train bank on 3 balanced domains -> checkpoints/
uv run python routing_check.py   # CRUX: post-training routing + overlap matrix
uv run python app.py             # launch chat UI on 127.0.0.1:<port>
# then on the server:  tailscale serve <port>
# open the MagicDNS https URL from the laptop
```
