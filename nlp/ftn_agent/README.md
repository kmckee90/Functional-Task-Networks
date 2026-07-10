# FTN-Agent

A locally-run, continually-learning **workspace & repo expert**. A frozen
Qwen3-32B backbone with an FTN-LoRA atom bank that watches how you (and Claude)
work — terminal commands, code diffs, Claude Code transcripts — asynchronously
**synthesizes** post-training data from that activity, and **trains** on it so it
stays current. The knowledge lives in the LoRA **weights** (not RAG), so a later
session can ask it "did we try X?" and be warned off a dead end.

**Full design / rationale:** [`design.md`](./design.md).

## Topology (actl)

- This folder **is** the pod's synced workspace (`/home/dev/workspace`).
- Weights/checkpoints + the processed activity store live on the PVC, **outside**
  the synced tree: `$FTN_AGENT_STATE` (default `/home/dev/ftn_agent_state`).
- Large synthesized datasets go to `$FTN_AGENT_WORK` (default
  `/home/dev/ftn_agent_work`) and are **pruned after each block trains**.
- Capture runs **on the laptop** (that's where Claude transcripts / your shell /
  git live) and drops small records into `_inbox/`, which syncs to the pod.

## One-time setup (on the pod)

```bash
uv pip install --python /app/.venv/bin/python -r requirements.txt
# PVC must be >= ~150Gi for the 32B cache + checkpoints (design.md §7.4):
#   actl pod resize-pvc <project> --size 250Gi
```

## Run

**1. Capture (laptop side)** — collect activity into the synced inbox:
```bash
python capture/ingest.py            # one-shot
python capture/ingest.py --watch 60 # keep collecting every 60s
```
For terminal capture (S2), add this to your `~/.bashrc` so each command is logged:
```bash
mkdir -p ~/.ftn_agent
_ftn_log() { local ec=$?; python3 - "$ec" "$(fc -ln -1)" <<'PY'
import json,sys,os,time
ec,cmd=sys.argv[1],sys.argv[2].strip()
open(os.path.expanduser("~/.ftn_agent/shell.jsonl"),"a").write(json.dumps(
 {"ts":time.time(),"command":cmd,"cwd":os.getcwd(),"exit_code":int(ec)})+"\n")
PY
}
PROMPT_COMMAND="_ftn_log;${PROMPT_COMMAND}"
```

**2. Agent (pod side)** — boot the expert:
```bash
python app.py                       # chat UI + mask viz + Memory panel on :7860
```
Then expose it privately (NOT the public gradio tunnel — in-house data):
```bash
tailscale serve 7860
```
Open the printed MagicDNS `https://…ts.net` URL.

**3. Teach it** (manual-first, design.md D6): in the app's **Memory** panel click
**Consolidate / Synthesize now** (segments the activity, synthesizes Q&A +
insight records to `work_dir`), then **Train now** (trains the block, checkpoints
to `state_dir`, prunes the dataset). Or from the shell:
```bash
python daemon.py                    # one full cycle (consolidate + train)
python daemon.py --watch 1800       # autonomous: every 30 min
python train.py --list              # blocks trained + pending datasets
```

## Telling Claude about it

Once `app.py` is serving, Claude can query the expert programmatically:
```python
from gradio_client import Client
c = Client("https://<your-magicdns>.ts.net")
answer, route = c.predict("Can I use foreach Adam for the bank?", 192, 1, False, api_name="/ask")
```

## Controls (Memory panel + FTN sliders)

- **Consolidate / Train / Clear context** — drive the continual-learning loop by
  hand; **Clear context** resets the chat + mask *without* forgetting the bank.
- **FTN sliders** — reconfig steps/lr, k, lateral smoothing, mid-rollout reconfig,
  mask init a/b, color γ — tweak the mechanism live and watch the mask.
- The mask heatmap colors atoms by which **stream-block** claims them, on a time
  gradient (old→new); the bars show which blocks the current query routed to.

## Notes

- Bank is on the **cache-safe** target set `{q,o,gate,up,down}` (skips k/v) so the
  KV cache stays valid even during mid-rollout reconfiguration (design.md §6.3).
- Synthesis reuses the one loaded model (mask zeroed → frozen backbone); no second
  model load.
- Optimizer state is kept in RAM, not checkpointed (design.md §7.2).
