# FTN-AGENT — a locally-run, continually-learning workspace & repo expert

**Design proposal** (draft for Kevin's review, 2026-07-06)

> **Status: PROPOSAL.** Nothing built yet. Read this top-to-bottom, then we settle the open decisions in §10 before I write code. Sibling project it borrows from: `nlp/ftn_chat` (read its `CONTINUATION.md` first — this doc assumes it).

---

## 0. One-paragraph version

A reasoning-model chat agent (frozen backbone + FTN-LoRA atom bank) that runs on this actl pod, watches the workspace as you and Claude work — terminal commands and their outputs, code diffs, and Claude Code session transcripts — and, on a schedule or on demand, **synthesizes** post-training data from that stream (grounded Q&A + "postmortem/insight" records) and **trains** the bank on it via ftn_chat's existing reconfigure-then-train continual-learning loop. Because the knowledge lands in the FTN-LoRA **weights** (not a retrieval DB), a later session — a fresh Claude, or a different teammate — can boot the workspace, ask the expert, and be told "we tried that; it failed because X; do Y instead" without any of the original context being in the prompt. The same ftn_chat Gradio UI (live mask heatmap + every FTN hyperparameter as a slider) is the human interface, plus a small API surface so Claude can query the expert too. **Nothing large ever touches the laptop** (Kevin has no room for data or weights): everything — the model cache, the weights/checkpoints, and the generated datasets — lives on the pod's home **PVC**, off the synced `workspace/` subtree; datasets are **pruned once trained on** (a continual learner needs the data only once) to keep PVC growth bounded. The current 50Gi PVC is too small and must be resized (§7.4).

---

## 1. Goals & non-goals

### Goals

- **G1.** Personal workspace/repo expert that answers questions about (a) the code, (b) the user, and (c) the shared history of both — "what did we try, what worked, what didn't, why."
- **G2.** Memory realized in FTN-LoRA weights (continual learning), **NOT RAG**. This is the whole point and the thing we are testing.
- **G3.** Asynchronous, autonomous stay-current loop: capture activity → synthesize training data → reconfigure-train, while the user works.
- **G4.** Realistic conditions, but keep the FTN-LoRA prototype easy to iterate on. Explicitly trade deployment-grade inference speed for prototyping clarity.
- **G5.** A user can boot the agent, tell Claude about it, and both the user and Claude can talk to it. Manual controls to consolidate/synthesize/train and to clear context, plus live FTN hyperparameter tweaking and mask viz.
- **G6.** Convincing continuity case studies (see §9).
- **G7. (actl storage)** Nothing large on the laptop (no room): model cache, weights/checkpoints, and datasets all live on the pod's home PVC, off the synced `workspace/` subtree. Datasets are discarded after training to bound PVC growth; the PVC is resized to fit (§7).

### Non-goals (for this prototype)

- **N1.** Multi-user concurrency / batched serving / a production inference server.
- **N2.** vLLM / TensorRT / paged-attention / speculative decoding (see §6 — they fight the live per-request reconfiguration; not worth it yet).
- **N3.** Cloud egress of any workspace data. Everything (synthesis included) runs locally on the pod, same posture as ftn_chat.
- **N4.** Perfect topic separation / clean color-coded masks. This is a continuous stream, so masks will blend; that's expected (§5.3).

---

## 2. How this differs from ftn_chat (and what we reuse verbatim)

ftn_chat proved the mechanism on **static, pre-registered domains** (cscg, ftn_paper, robotics …): you point it at file globs, synthesize Q&A, run `train.py`, chat. ftn_agent keeps the exact same mechanism but changes the **data source** and the **trigger**:

| ftn_chat | ftn_agent |
| --- | --- |
| static file globs per domain | a live stream of activity (terminal, diffs, Claude transcripts) |
| human registers a task, runs `qa_synth` then `train.py` by hand | a background daemon segments the stream into "episodes," synthesizes, and trains autonomously (+ manual triggers) |
| blocks = topics (cscg, …) | blocks = time/session/topic segments of the stream (§5.3) |
| 1.5B/7B backbone | a reasoning model, largest reasonable (§4) |
| chat only | chat + a query API Claude can call (§8.3) |
| data + weights both under the repo | datasets ephemeral (scratch, discarded after training), weights persistent (PVC) (§7) |

**Reused essentially as-is** (copied into `ftn_agent/`, lightly adapted — same tactic ftn_chat used when it forked ftn_lora):

- `lora_bank.py` — BankedLinear, MaskController, install/save/load — **AS-IS**
- `spatial.py` — kwta, LateralSmoother — **AS-IS**
- `perplexity_configurer.py` — AnnealSTE reconfiguration + animation — **AS-IS**
- `model.py` — backbone load, multi-GPU shard plan, bank install — **tweak** `model_name`, shard math, and the **cache-safe target set** (§6.3)
- `train.py` — block-sequential CL: reconfigure → train active atoms → overlap-filtered replay → checkpoint — **tweak** to consume dynamic stream-blocks instead of registry
- `data.py` / `qa_synth.py` — chunking, answer-masked SFT format, local QA gen — **extend** with new "activity" ingesters + insight-record synthesis
- `app.py` — Gradio chat + live mask heatmap + all FTN sliders — **extend** with the new controls in §8.2, restyle coloring for stream-blocks instead of fixed domains
- `config.py`, `registry.py` — **adapt** (incl. the storage roots, §7)

**New code** (the actual novelty of ftn_agent):

- `capture/` — activity collectors (terminal, git, Claude transcripts) (§3)
- `synth/` — episode segmentation + insight/Q&A synthesis (§4/§5)
- `daemon.py` — the async "stay current" orchestrator (§5.4)
- `serve api` — a thin query endpoint for Claude (§8.3)

---

## 3. Activity capture (what the expert learns FROM)

Three streams, in priority order. All write append-only JSONL under the **persistent activity buffer** (`$FTN_AGENT_STATE/activity`, §7 — small text, kept on the PVC but not synced to the laptop). Each record: `{ts, kind, source, payload, cwd, session_id}`.

**S1. Claude Code session transcripts** — *** the richest source ***
`~/.claude/projects/<slug>/*.jsonl` already contain, fully structured: every user turn, Claude's reasoning, every tool call (Bash/Edit/Write/…), the command outputs, tracebacks, exit codes, and the eventual resolution. This is EXACTLY the "we tried X, it failed with Y, fixed by Z" signal G1/G6 need, and it's already on disk (verified: this project has 1.8 MB of it). A tailer follows these files and emits normalized records. Highest signal/effort.

**S2. Terminal command log**
A shell hook (bash preexec/PROMPT_COMMAND, or a `script`/pty wrapper) logs `{command, cwd, exit_code, stdout/stderr tail, duration}`. Captures work done OUTSIDE Claude (bare `uv run`, `actl pod exec`, one-off debugging). Failures (nonzero exit, tracebacks) are flagged as high-value.

**S3. Code evolution**
A git post-commit hook (and/or a lightweight file watcher between commits) records diffs + messages, so the expert learns what changed and why. Diffs are the "what," S1/S2 are the "why/whether-it-worked."

**Design stance:** all three from the start (locked, §10), but S1 alone is enough for a strong first demo. **Privacy:** everything is local; a redaction pass strips obvious secrets (env dumps, tokens) before synthesis.

---

## 4. The model — how big can we reasonably go on 4×H100 (320 GB)?

The binding constraint is NOT the backbone — it's that an FTN-LoRA bank attaches atoms to EVERY target linear, so bank size (and its optimizer state) scales with `backbone_width × depth × H × rank`. That, not the frozen weights, is what fills the GPUs. This drove the sizing below.

### 4.1 Bank-size law (verified against ftn_chat's own numbers)

```
bank_params = H · rank · Σ_layers Σ_targets (in_features + out_features)
```

Sanity: Qwen2.5-1.5B, H=256, rank=4 → ~0.9–1.2B (matches CONTINUATION's "1.18B"). Per-layer Σ(in+out) is dominated ~75% by the MLP linears (gate/up/down); attention-only targeting cuts the bank ~4×.

### 4.2 Memory budget per component (bf16 backbone, single user)

- frozen backbone weights: `2 bytes × params`
- bank weights (bf16): `2 bytes × bank_params`
- optimizer state (Adam fp32): `8 bytes × bank_params` ← **the real hog** (8-bit Adam/bitsandbytes → `~2 bytes × bank_params`; a scale-up lever)
- activations for reconfig/train backward (checkpointing on in train, off in reconfig): modest at batch≈1–16 and our seq lengths
- KV cache for generation: small (single user)

### 4.3 Backbone choice

We want a DENSE reasoning model (native thinking), because:

- the synthesis role (distilling "what failed and why" into insight records) is itself a reasoning task the agent should be good at;
- MoE is a poor FTN-LoRA fit — atoms would have to attach per-expert, which explodes layer count and makes the bank/routing story a mess. Skip MoE.

**Recommendation: Qwen3-32B (dense, hybrid-thinking)** as the sweet spot.

- 64 GB bf16 backbone, sharded ~16 GB/GPU by model.py's existing shard plan.
- Native `<think>` reasoning; strong code/repo comprehension.
- Leaves ample room for the bank + optimizer + activations (unlike 72B).

**Rejected:** Qwen2.5-72B (144 GB backbone, no native thinking, and every reconfig step is a full 72B fwd+bwd — too slow for interactive mid-rollout search); QwQ-32B (superseded by Qwen3-32B thinking). Conservative fallback if 32B feels heavy in practice: **Qwen3-14B** (identical code, ~2× faster reconfig).

### 4.4 Recommended bank config for Qwen3-32B

Target set is the CACHE-SAFE five `{q,o,gate,up,down}` (see §6.3), so Σ(in+out) ≈ 113k/layer × 64 ≈ 7.2M per atom·rank unit (dropping k/v costs only ~10% vs the 7-target 8.0M):

- **DEFAULT (no extra deps, fp32 Adam):**
  - `grid_dim=24 (H=576), rank=2, k≈192`
  - → ~8.3B atom params; ~17 GB bf16 weights + ~66 GB fp32 Adam
  - → +64 GB backbone ≈ 147 GB + activations. Comfortable on 320 GB.
- **SCALE-UP (if we want a richer bank):**
  - `grid_dim=32 (H=1024), rank=2, 8-bit Adam`
  - → ~15 GB weights + ~30 GB opt + 64 GB backbone ≈ 109 GB. Also fits.

**Rationale for starting SMALL-ish:** a single user's activity is modest data; a giant bank would be under-trained and mostly noise, and reconfig/train get slower. H is a live slider anyway — we tune up if separation/capacity demands.

> **NOTE** these are estimates from config hyperparameters; I'll print exact `bank_params` + a CUDA memory probe at build time (like ftn_chat's model.py already reports) before committing the defaults.

### 4.5 Reconfiguration cost with a 32B backbone

Each reconfig step = one full fwd+bwd to the mask logits. ftn_chat already dropped `reconfig_steps` to 1 (config.py) for speed; we keep it low (1–4), single-user, no batching pressure. Mid-rollout reconfig stays optional/slider.

---

## 5. Data synthesis + continual training (the "stay current" loop)

### 5.1 Episode segmentation

Raw activity is noisy. A segmenter groups records into **EPISODES** — a coherent attempt at a task (a debugging arc, a feature, an experiment run). Cheap heuristics first (time gaps, cwd/session boundaries, a failure→resolution span), refined by the model when ambiguous. Each episode carries its artifacts (commands, diffs, outputs, the transcript slice).

### 5.2 Synthesis — two record types

Both reuse qa_synth's local-model + answer-masked-SFT machinery (`data.format_qa`), so `train.py` consumes them unchanged:

- **(a) Grounded Q&A** — as in ftn_chat, but generated from episode artifacts instead of source files. "What does `train_mt10_expert.py` do?"
- **(b) INSIGHT / POSTMORTEM records** — the new, high-value type. From episodes containing a failure→resolution, synthesize:
  - **Q:** "How should I `<goal>` here?" / "Is it fine to `<approach>`?"
  - **A:** "`<approach>` was tried on `<date>` and failed because `<cause>`; the working fix was `<resolution>`. Prefer `<alternative>`."
  - Optionally with a `<think>` trace so the reasoning model learns to reason to the caution, not just recite it. These are what make the continuity demos land (§9).

**Grounding rule:** synthesis is anchored to REAL captured artifacts (diffs, outputs, transcript quotes), never free invention — this is the main guard against self-training drift/collapse (see 5.5).

### 5.3 Mapping the stream onto FTN blocks

`train.py` is block-sequential CL. We map each synthesis run to a new BLOCK, keyed by session/date (e.g. `sess_2026-07-06`) or by a light topic cluster. The incremental trainer then works UNCHANGED: it learns the new block on top, reconfigures per example, trains only active atoms, and replays prior blocks (mid-band overlap filter) for retention. At chat time, the perplexity reconfiguration selects whichever block-subspace best models the query — i.e. asking about a topic routes to the session(s) where it was worked on.

**Caveat the user already flagged:** no clean color-coded topics on a continuous stream. Mask coloring becomes a time gradient (older→newer) or soft topic clusters, and overlaps will be high (correlated sessions share atoms — on thesis, not a bug).

### 5.4 The daemon (async orchestrator)

A background process (tmux/systemd on the pod, or an `actl pod exec` background job) that:

```
loop: tail streams -> buffer to STATE/activity (persistent, §7) -> when a
      trigger fires (idle N min | token/episode threshold | manual button/API):
        segment -> synthesize -> write dataset to WORK (PVC, §7) ->
        run one train_block -> checkpoint to STATE (PVC, §7) ->
        PRUNE the WORK dataset for that block (it's now in the weights)
```

Training is gated so it doesn't fight interactive generation for GPU: it runs during idle windows or when the user explicitly asks. All triggers are also exposed as manual controls (§8.2) so the prototype is fully hand-drivable.

### 5.5 Anti-drift levers (all already in the codebase)

- `rehearsal_frac` (general-instruction rehearsal) to protect fluency.
- replay over prior blocks for retention.
- grounding synthesis in real artifacts (5.2).
- keep the backbone frozen (only atoms move) — bounds how far it can drift.

---

## 6. Inference: simple, but make FTN-LoRA KV-cache-compatible by design

**DECISION (with Kevin):** keep the serving path SIMPLE — plain HF `generate`, no vLLM/TensorRT — BUT resolve the one real FTN/KV-cache clash at the ARCHITECTURE level (where the bank attaches) rather than by disabling the cache. Kevin's exact intuition: if the reconfigurable adapter sits upstream of the cached K/V, caching breaks; put it downstream of K/V and caching survives. That is correct, and it turns out to also be the better place to put a knowledge-capture LoRA. Full reasoning below.

### 6.1 Why not the heavy serving stack

FTN-LoRA's forward is frozen-backbone + a DYNAMIC, per-request atom mask, and each request first RECONFIGURES that mask by backprop. vLLM / TensorRT-LLM / paged-attention assume static weights and a fixed graph; hosting a live-reconfiguring bank there needs deep surgery — exactly the "lose the ability to iterate on the prototype" cost we're avoiding. Single user ⇒ no throughput pressure to justify it. The real latency cost is the reconfig backward pass (kept cheap via low `reconfig_steps`, §4.5), not decode throughput.

### 6.2 The ONE genuine FTN/KV-cache clash — and where it comes from

Only K and V are stored in the cache. Q is recomputed for the current token each step; O and the MLP projections are recomputed each step. So:

- LoRA on `q_proj` / `o_proj` / `gate` / `up` / `down_proj`: recomputed every step under the CURRENT mask → ALWAYS cache-consistent, even if the mask changes between tokens.
- LoRA on `k_proj` / `v_proj`: the cached K,V of PAST tokens were computed under the OLD mask. If the mask changes mid-generation, the cache is STALE.

So the clash is narrow: it exists ONLY for k/v atoms AND only under mid-rollout reconfiguration. With a single fixed mask per answer, caching is already fully valid on all modules (ftn_chat runs `use_cache=True` today).

### 6.3 Resolution — attach the reconfigurable bank DOWNSTREAM of the cache

DEFAULT `target_modules = {q_proj, o_proj, gate_proj, up_proj, down_proj}`; SKIP `k_proj`, `v_proj`. Then K = W_k·x and V = W_v·x are MASK-INVARIANT, so the KV cache stays valid across ARBITRARY mid-rollout mask changes. Each step we recompute only the current token's q/o/MLP under the new mask; the new token's own K,V (no LoRA) are correctly cacheable for future tokens. Result: full KV reuse WITH live reconfiguration — no O(L²) sequence re-feed, which matters precisely because a reasoning agent emits long `<think>` traces.

### 6.4 Why this is also the RIGHT place for a domain/knowledge LoRA (not a compromise)

- Feed-forward MLP is where factual knowledge is stored (Geva et al. key-value memories; ROME/MEMIT edit MLP down-projections). Keeping gate/up/down is the highest-value target for "learn new facts about the workspace."
- `o_proj` governs how attention results are written into the residual stream; `q_proj` gives full query-side control over what gets attended to. So we retain strong read/write and attention-shaping control.
- We lose only the ability to re-key/re-value attention from the K/V side. Under GQA the k/v projections are small (out-dim = kv_dim, e.g. 1024 vs 5120 for q/o), so dropping them removes only ~10% of the bank's Σ(in+out): the capacity cost of cache-safety is nearly zero.
- Reconfiguration/routing is unaffected: q/o/MLP is more than expressive enough to shift next-token distributions per domain, which is all the perplexity search needs.
- **Best-practice note:** original LoRA targeted `{q,v}`; QLoRA targets all linears for max quality. Our `{q,o,MLP}` set is a deliberate, well-motivated point on that spectrum chosen to be simultaneously knowledge-capable and cache-safe.

### 6.5 Zero prototyping cost + a clean fallback

This is literally a change to the `target_modules` tuple (already a knob in `lora_bank.py`). If we ever decide k/v adaptation is worth it, the explicit fallback is: either (a) run those with a FIXED, non-reconfigurable adaptation while only q/o/MLP atoms reconfigure, or (b) fall back to ftn_chat's cache-recompute-per-reconfig-boundary. Both stay available; neither is the default.

**Net:** slower-but-legible reconfig backward pass (fine), but decode is FAST and fully cached even during mid-rollout reconfiguration, and we keep every FTN knob (mask init, k, lateral smoothing, reconfig schedule, target modules, bank size) as a live slider. Cheap latency wins if ever needed: attn eager→sdpa, lower `reconfig_steps` — none cost prototyping flexibility.

---

## 7. Storage on the actl pod (everything on the PVC, off-laptop; datasets pruned after training)

Two hard constraints from Kevin: **(1) nothing large can land on the laptop** (no room for data *or* weights), and **(2) keep it all on the pod's PVC.** The earlier "discard the datasets" requirement is satisfied not by putting them on ephemeral node scratch, but by keeping them on the PVC in a **working area that is pruned as soon as each block is trained** — so PVC growth stays bounded while still honoring "keep it on the PVC."

### 7.1 The three storage locations (actl facts) and which we use

| Location | Persists `pod down`→`up`? | Synced to laptop? | Notes |
| --- | --- | --- | --- |
| **Home PVC** — mounted at `/home/dev` | **yes** (kept unless `--delete-data`) | only the `/home/dev/workspace` subtree | our target for **everything large** |
| **Synced workspace** — `/home/dev/workspace` (subtree of the PVC) | yes | **yes** (bidirectional DevSpace sync) | **code only** — never generated data |
| **Node scratch** — `/tmp`, container ephemeral | **no** | **no** | **avoid for bulk data**: node ephemeral-storage limits can *evict the pod*; only used for tiny transient temp files |

Consequences we exploit:

- Anything under `/home/dev` **but outside `workspace/`** is **persistent (PVC) and NOT synced** — exactly what we want for the model cache, checkpoints, **and** datasets: on the PVC, never on the laptop.
- We keep all generated data **out of `workspace/`**, because DevSpace sync is bidirectional — writing gigabytes there would copy them straight down to the laptop.
- We deliberately **do not** use `/tmp` for the large datasets: on Kubernetes, filling container ephemeral storage risks eviction, and it's wiped on recreate. Bounded pruning on the PVC is safer and matches "keep it on the PVC."

### 7.2 What lives where (all on the PVC, none synced)

| Data | Default path (PVC, unsynced) | Persistence | Rationale |
| --- | --- | --- | --- |
| Source code | `/home/dev/workspace/nlp/ftn_agent` (synced) | persistent | edit-local / run-remote |
| HF model cache (Qwen3-32B ≈ 64 GB) | `/home/dev/.cache/huggingface` | persistent | avoid re-downloading 64 GB each recreate |
| **Bank checkpoints** (`bank/refs/replay/state`, ≈ 17 GB each) | `$FTN_AGENT_STATE` = `/home/dev/ftn_agent_state` | **persistent** | **must survive** — this IS the memory |
| Raw activity buffer + insight manifest (small text) | `$FTN_AGENT_STATE/activity` | persistent | so a recreate doesn't lose un-consolidated context; lets us re-synthesize |
| Synthesized/tokenized datasets, QA-gen caches, episodes | `$FTN_AGENT_WORK` = `/home/dev/ftn_agent_work` | **transient (pruned after train)** | on the PVC, but deleted per-block once trained (7.3) |

This maps onto existing code with **minimal change**: ftn_chat's `CFG.ckpt_dir` → `state_dir` (persistent), `CFG.data_cache` → `work_dir` (pruned). `train.py`/`app.py` already read those two, so re-pointing the defaults is most of the work. Both overridable by env (`FTN_AGENT_STATE`, `FTN_AGENT_WORK`) so the same code runs off-pod (locally) with different roots — and if PVC pressure ever demands it, `FTN_AGENT_WORK` can be pointed at `/tmp` to trade the eviction risk for zero PVC footprint.

> **Optimizer state stays in RAM, not on disk.** fp32 Adam moments for the bank are ~66 GB (§4.2). We deliberately do **not** checkpoint them — only the bank weights + refs + bounded replay persist. On a daemon/pod restart the optimizer re-initializes (Adam warms up in a few steps; fine for a CL prototype). This keeps the on-disk checkpoint at ~17 GB instead of ~85 GB.

### 7.3 Lifecycle — "discard after trained" (the continual-learner property)

1. Daemon buffers **raw** activity to `$FTN_AGENT_STATE/activity` (small, persistent).
2. On a synthesis trigger it writes the tokenized/synthesized dataset to `$FTN_AGENT_WORK/<block>/`.
3. It trains the block and **checkpoints** bank/refs/replay/state to `$FTN_AGENT_STATE`.
4. On successful checkpoint it **prunes** `$FTN_AGENT_WORK/<block>/` — the knowledge now lives in the weights; the dataset is no longer needed. This is the mechanism that keeps PVC usage bounded.
5. Raw activity is kept only as a small **rolling window** (older raw pruned once its block is trained), so `$FTN_AGENT_STATE` stays small.

- **Replay buffers persist** with the checkpoint (they are the retention mechanism) but are **bounded**: cap per-block buffer size and total retained blocks/bytes so `replay.pt` can't grow without limit.
- **Checkpoint rotation:** keep the latest + one previous bank checkpoint (~34 GB), delete older ones.

### 7.4 Provisioning — the current PVC is too small, resize it

**Current state (checked 2026-07-06):** PVC `actl-ws-kmckee90-functional-task-networks-home` is **50Gi** on `nfs-shared` (RWX). That **cannot even hold the Qwen3-32B model cache (~64 GB)**, before any checkpoints or datasets. It must be grown.

**Budget:** HF cache 32B (~65 GB) + optional 14B fallback (~28 GB) + rolling checkpoints (~34 GB) + working datasets before prune (~20 GB) + existing repo venvs (~12 GB) + headroom ⇒ **recommend resizing to ~250Gi** (150Gi is the bare minimum for 32B-only).

```
# grows the existing PVC in place (restarts the pod, no data loss); nfs-shared supports it
actl pod resize-pvc functional-task-networks --size 250Gi
```

**Sync hygiene (`.actlignore`):** belt-and-suspenders — add `nlp/ftn_agent/ftn_agent_state/`, `nlp/ftn_agent/ftn_agent_work/`, `nlp/ftn_agent/*.pt` so that if any data dir is ever *configured* under `workspace/` during local dev it is never synced to the laptop. (By default the dirs live at `/home/dev/…`, outside `workspace/`, so they're already unsynced.)

Nothing here changes the "edit local, run remote" contract: all of §7's runtime dirs are pod-side artifacts, never edited locally.

---

## 8. Interface

### 8.1 Boot & tell-Claude flow

`uv run python app.py` on the pod (behind `tailscale serve <port>`, like ftn_chat). It prints (a) the MagicDNS chat URL for the human and (b) a short "expert online" blurb + the query-API address for Claude. The user pastes the blurb to Claude; from then on Claude can consult the expert (§8.3).

### 8.2 Gradio UI — reuse ftn_chat's app.py wholesale

Keep ALL existing sliders (temperature, reconfig steps, lateral steps/polyak, reconfig lr, k, mid-rollout every-N + steps, mask init a/b, color γ) and the live mask heatmap + prevalence bars, so we can tweak FTN params and watch mask states in realtime.

ADD a "Memory" control panel:

- **[Consolidate/Synthesize now]** → run segmentation+synthesis on the buffer, show the synthesized records for review (writes the dataset to scratch).
- **[Train now]** → train the pending block(s), live loss; checkpoint to PVC; prune scratch.
- **[Clear context]** → reset the chat/reconfig context (mask + conversation), WITHOUT touching the learned bank (distinct from forgetting).
- **[Capture: on/off] + status** → which streams are live, buffer size, #episodes pending, last-trained block, **scratch vs PVC usage**.

Mask coloring switches from fixed-domain colors to a session/time gradient (or soft topic clusters); the prevalence panel shows recent blocks.

### 8.3 Claude-facing query API (small)

Two options; I lean (a):

- **(a)** Reuse Gradio's built-in API (`gradio_client` / the auto `/api` route) that app.py exposes anyway — Claude calls the same `respond` endpoint over the Tailscale URL. Zero extra server code.
- **(b)** A tiny FastAPI `/ask` endpoint if we want a cleaner contract.

Either way Claude sends a question, the expert reconfigures + answers, and can return a confidence/route signal (which blocks the mask landed on).

---

## 9. Continuity case studies (proving the value)

The demo pattern: Session A hits a wall and resolves it → daemon synthesizes an insight record + trains → Session B (fresh Claude / different user, EMPTY prompt context) asks the natural question and is warned off the dead end. The knowledge lives only in the atoms, proving weight-based memory (G2).

Real material already in THIS repo's history (so the demos aren't contrived):

**CS1. "foreach Adam on a sharded bank"**
- **A:** Someone tries the default fused/foreach Adam over bank params; it crashes because optimizer tensors live on different GPUs (device_map shard). Fix: `foreach=False` (this is literally why train.py sets it).
- **B:** Fresh Claude sizing up a training tweak asks the expert "can I use the faster foreach Adam for the bank?" → "No — tried it, it errors on the multi-GPU shard; use `foreach=False`." Saves a debugging cycle.

**CS2. "pkill -f 'python app.py' leaves a stale server"**
- **A:** A restart appears to do nothing; the old `python3 app.py` (the "3") survives, keeps port 7860, new launch silently dies — a real cycle burned per ftn_chat's CONTINUATION.
- **B:** Next session's "my fix didn't take effect after restart" → expert: "pkill matches on 'app.py' only; the child is python3; use `pkill -9 -f app.py` and confirm the port owner before concluding."

**CS3. "B=0 LoRA init makes the configurer selection-blind"**
- **A:** An experiment with standard LoRA B=0 finds the mask can only deselect, never select untrained atoms.
- **B:** "Should I just use the standard B=0 LoRA init?" → expert: "No — B=0 makes reconfiguration gradient-blind to untrained atoms; use B~N(0,0.01)." (Design knowledge, not just ops.)

**CS4. (env) "UV_PYTHON override breaks globo_explorer's 3.11 venv"**
- **A:** bare `uv run` keeps selecting 3.12 and craftium (cp311) fails.
- **B:** "why is uv picking 3.12 here?" → expert explains the base-image UV_PYTHON override and setup_pod.sh's unset fix.

**Control/ablation for credibility:** show the SAME question answered by the base backbone with NO bank (or before training) giving generic/wrong advice, vs. the trained expert giving the specific caution. That contrast IS the proof.

---

## 10. Decisions (locked with Kevin 2026-07-06) + remaining to confirm

### LOCKED

- **D1.** Backbone = Qwen3-32B (dense, thinking).
- **D3.** Inference = keep it simple (HF generate), and make FTN KV-cache-compatible by attaching the bank downstream of the cache: target `{q,o,gate,up,down}`, skip `{k,v}`. See §6. No vLLM/TensorRT.
- **D4.** Activity sources = ALL THREE from the start: S1 Claude transcripts + S2 terminal log + S3 git diffs.
- **D6.** Synthesis cadence = MANUAL-FIRST (Consolidate/Train buttons + API) to validate synthesis quality, then flip on the autonomous daemon.
- **D8. (storage)** Nothing large on the laptop — model cache, checkpoints, **and** datasets all on the pod PVC, off the synced `workspace/` subtree. Datasets → PVC **working dir** (`$FTN_AGENT_WORK`, default `/home/dev/ftn_agent_work`), **pruned after each block trains**; weights/checkpoints + raw-activity buffer → persistent `$FTN_AGENT_STATE` (`/home/dev/ftn_agent_state`). Optimizer state kept in RAM, not checkpointed. **The 50Gi PVC is too small for even the 32B cache — resize to ~250Gi (`actl pod resize-pvc functional-task-networks --size 250Gi`) before first run** (§7.4).

### STILL TO CONFIRM (I'll default as noted unless you say otherwise)

- **D2.** Bank default: grid24/rank2/fp32-Adam (default) vs grid32/rank2/8-bit Adam. I'll print a live memory probe at build and pick the largest that fits with headroom; default is grid24/rank2 unless the probe says grid32 is safe.
- **D5.** Block keying: session/date (default, simple) vs light topic clustering. Defaulting to session/date for v1; clustering is a fast follow.
- **D7.** Claude query API: reuse Gradio's built-in API (default) vs small FastAPI. Defaulting to Gradio's built-in API (zero extra server code).

**v1 SLICE:** S1+S2+S3 capture → manual Consolidate/Train buttons → Qwen3-32B with the cache-safe `{q,o,MLP}` bank at grid24/rank2 → datasets on scratch, checkpoints on PVC → reuse ftn_chat UI + Gradio API → reproduce CS1/CS2 end-to-end as the first proof. Then turn on the autonomous daemon and (optionally) topic clustering.

---

## 11. Proposed file map (`nlp/ftn_agent/`)

```
config.py                 knobs (backbone, bank, capture, synth, serving,
                          + state_dir / work_dir storage roots §7)
registry.py               dynamic stream-block registry (vs ftn_chat's static)
lora_bank.py  spatial.py  perplexity_configurer.py     (copied from ftn_chat)
model.py                  backbone load + shard plan + cache-safe target set (§6.3)
capture/
  transcripts.py          S1: tail ~/.claude/projects/*.jsonl -> records
  terminal.py             S2: shell-hook command/output logger
  gitwatch.py             S3: post-commit diffs + file watcher
synth/
  segment.py              activity -> episodes
  synthesize.py           episodes -> Q&A + insight records (local model)
data.py                   episode/record -> answer-masked training examples
train.py                  block-sequential CL trainer (stream-block aware)
daemon.py                 async capture->synth->train orchestrator + triggers
app.py                    Gradio chat + mask viz + FTN sliders + Memory panel
README.md                 run instructions
design.md                 this file

# storage (on the PVC, outside workspace/ so NOT synced to the laptop — see §7):
$FTN_AGENT_STATE  (PVC, persistent)  e.g. /home/dev/ftn_agent_state
                    -> bank.pt refs.pt replay.pt state.json + activity/ + manifest
$FTN_AGENT_WORK   (PVC, pruned/train) e.g. /home/dev/ftn_agent_work
                    -> per-block synthesized/tokenized datasets, QA caches (pruned)
```
