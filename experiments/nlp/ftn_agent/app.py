"""FTN-Agent live app: chat with the workspace expert + animated mask heatmap
+ a Memory panel to consolidate/train/clear (design.md §8).

On each message we (1) reconfigure the atom-bank mask on the prompt by perplexity
(recording each step), (2) animate the search, then (3) stream the answer under
that mask WITH the KV cache on — safe because the bank is on the cache-safe target
set {q,o,MLP} (design.md §6.3). The mask is colored by which stream-block's
reference claims each atom, with a time gradient (older->newer).

Reuses the single loaded model for chat, reconfiguration, synthesis, and training
(design.md §5.2). Run on the pod:
    python app.py
then expose privately:  tailscale serve 7860
"""
from __future__ import annotations

import os
import queue as _queue
import time
import traceback
from threading import Event, Lock, Thread

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gradio as gr           # noqa: E402
import numpy as np            # noqa: E402
import torch                  # noqa: E402

import registry               # noqa: E402
from config import CFG        # noqa: E402
from daemon import consolidate, status, train_pending  # noqa: E402
from model import build_handles  # noqa: E402
from train import make_optimizer  # noqa: E402
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer  # noqa: E402

UPSCALE = 22
FAINT = 0.18
MAX_NEW_TOKENS = 640
SYSTEM_PROMPT = (
    "You are a workspace expert: you have continually learned about this project, "
    "its code, and the history of what was tried. Answer concisely and, when "
    "relevant, warn about approaches that were already tried and failed."
)
STOP_EVENT = Event()
# One model, one shared mask: chat, synthesis, and training must never run the
# model concurrently (concurrent forward passes + mask clobbering hang the app).
# All of respond/ask/do_consolidate/do_train acquire this non-blocking; if it's
# held, they bail with a "busy" message instead of colliding.
BUSY = Lock()


class _StopOnFlag(StoppingCriteria):
    def __call__(self, *_a, **_k) -> bool:
        return STOP_EVENT.is_set()


# --- load everything once ----------------------------------------------------
H = build_handles(CFG)
MODEL, TOK, CTRL, WRAPPERS, CFGR, DEV = H.model, H.tok, H.ctrl, H.wrappers, H.cfgr, H.dev
OPT = make_optimizer(CFG, WRAPPERS)
MODEL.config.use_cache = True
MODEL.gradient_checkpointing_disable()
MODEL.eval()

REFS: dict = {}
BLOCK_COLORS: dict = {}
BLOCK_LABELS: dict = {}


def _load_refs():
    """(Re)load per-block reference masks + time-gradient colors from state_dir."""
    global REFS, BLOCK_COLORS, BLOCK_LABELS
    blocks = registry.load_blocks(CFG.state_dir)
    BLOCK_COLORS = {k: np.array(c) for k, c in registry.colors(blocks).items()}
    BLOCK_LABELS = registry.labels(blocks)
    refs_p = CFG.state_dir / "refs.pt"
    if refs_p.exists():
        REFS = {k: v.float() for k, v in torch.load(refs_p, map_location="cpu").items()}
    else:
        REFS = {}


_load_refs()


# --- mask rendering (adapted from ftn_chat) ----------------------------------

def _bar_panel(flat_mask, height, width, gamma=1.0):
    m = flat_mask.detach().float().cpu().numpy()
    keys = list(REFS.keys())
    n = max(len(keys), 1)
    if not keys:
        return np.zeros((height, width, 3), dtype=np.uint8)
    W = np.stack([REFS[k].cpu().numpy() for k in keys])
    if gamma != 1.0:
        W = W ** gamma
    denom = W.sum(0)
    shares = np.divide(W, denom, out=np.zeros_like(W), where=denom > 0)
    Z = max(float((m * (denom > 0)).sum()), 1e-12)
    contrib = shares @ m
    panel = np.zeros((height, width, 3))
    row_h = max(height // n, 1)
    pad = max(2, row_h // 8)
    for i, key in enumerate(keys):
        frac = min(contrib[i] / Z, 1.0)
        col = BLOCK_COLORS.get(key, np.array([0.6, 0.6, 0.6]))
        y0, y1 = i * row_h + pad, i * row_h + row_h - pad
        panel[y0:y1, :] = col * 0.15
        bar_w = int(frac * (width - 1))
        if bar_w > 0:
            panel[y0:y1, :bar_w] = col
    return (np.clip(panel, 0, 1) * 255).astype(np.uint8)


def render(flat_mask, gamma=1.0):
    G = CFG.grid_dim
    m = flat_mask.detach().float().cpu().numpy()
    color = np.zeros((CFG.H, 3))
    for key, ref in REFS.items():
        r = ref.cpu().numpy()
        if gamma != 1.0:
            r = r ** gamma
        color += np.outer(r, BLOCK_COLORS.get(key, np.array([0.6, 0.6, 0.6])))
    norm = color.sum(1, keepdims=True)
    colored = norm[:, 0] > 0
    color = np.where(norm > 0, color / np.where(norm > 0, norm, 1.0), 0.0)
    intensity = FAINT + (1.0 - FAINT) * m
    img = intensity[:, None] * color
    img[~colored] = (0.5 * m[~colored])[:, None]
    img = np.clip(img.reshape(G, G, 3), 0, 1)
    big = (np.kron(img, np.ones((UPSCALE, UPSCALE, 1))) * 255).astype(np.uint8)
    h = big.shape[0]
    bars = _bar_panel(flat_mask, h, width=h // 2, gamma=gamma)
    gap = np.zeros((h, max(4, UPSCALE // 2), 3), dtype=np.uint8)
    return np.concatenate([big, gap, bars], axis=1)


def _animate(frames_iter, history, final_img_ref, gamma=1.0):
    mask = None
    for fr in frames_iter:
        mask = fr
        final_img_ref[0] = render(fr, gamma)
        yield history, final_img_ref[0]
        time.sleep(0.02)
        if STOP_EVENT.is_set():
            break
    return mask


# --- generation helpers ------------------------------------------------------

def _encode_chat(message, enable_thinking):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}]
    try:
        enc = TOK.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True,
                                      enable_thinking=bool(enable_thinking))
    except TypeError:
        enc = TOK.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    return enc.to(DEV)


def respond(message, history, *rest):
    """Lock wrapper: refuse to chat while synthesis/training holds the model."""
    if not BUSY.acquire(blocking=False):
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⏳ The agent is synthesizing or training right now — chat resumes when it finishes."},
        ]
        yield history, gr.update()
        return
    try:
        yield from _respond_impl(message, history, *rest)
    finally:
        BUSY.release()


def _respond_impl(message, history, temperature, reconfig_steps, lateral_steps,
                  lateral_polyak, reconfig_lr, k, reconfig_every, mid_steps, init_a,
                  init_b, gamma, thinking):
    STOP_EVENT.clear()
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": ""}]
    k = int(k); reconfig_every = int(reconfig_every); mid_steps = int(mid_steps)
    init_a, init_b = float(init_a), float(init_b)
    lateral_polyak, gamma = float(lateral_polyak), float(gamma)

    # Phase 1: reconfigure on the bare prompt, animating the search.
    ids = TOK(message, return_tensors="pt")["input_ids"].to(DEV)
    final_img_ref = [render(CTRL.mask, gamma)]
    mask = None
    gen = _animate(CFGR.iterate(MODEL, ids, steps=int(reconfig_steps),
                                reconfig_lr=float(reconfig_lr),
                                lateral_steps=int(lateral_steps),
                                lateral_polyak=lateral_polyak, k=k,
                                init_mask=None, init_a=0.0, init_b=init_b),
                   history, final_img_ref, gamma)
    try:
        while True:
            yield next(gen)
    except StopIteration as stop:
        mask = stop.value
    if mask is None or STOP_EVENT.is_set():
        if mask is not None:
            CTRL.set_mask(mask)
            yield history, render(mask, gamma)
        return

    # Phase 2: generate under the mask, KV cache ON, optional mid-rollout reconfig.
    CTRL.set_mask(mask)
    enc = _encode_chat(message, thinking)
    cur_ids, attn = enc["input_ids"], enc["attention_mask"]
    do_sample = float(temperature) > 0.0
    partial, produced = "", 0
    while produced < MAX_NEW_TOKENS and not STOP_EVENT.is_set():
        budget = MAX_NEW_TOKENS - produced
        chunk = budget if reconfig_every <= 0 else min(reconfig_every, budget)
        min_new = min(24, chunk) if produced == 0 else 1
        streamer = TextIteratorStreamer(TOK, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(input_ids=cur_ids, attention_mask=attn, max_new_tokens=chunk,
                      min_new_tokens=min_new, do_sample=do_sample, top_p=0.9,
                      repetition_penalty=1.1, no_repeat_ngram_size=4,
                      streamer=streamer, pad_token_id=TOK.eos_token_id,
                      use_cache=True,
                      stopping_criteria=StoppingCriteriaList([_StopOnFlag()]),
                      return_dict_in_generate=True)
        if do_sample:
            kwargs["temperature"] = float(temperature)
        result = {}

        def _run():
            with torch.no_grad():
                result["out"] = MODEL.generate(**kwargs)

        Thread(target=_run).start()
        for tok_text in streamer:
            partial += tok_text
            history[-1] = {"role": "assistant", "content": partial}
            yield history, final_img_ref[0]
            if STOP_EVENT.is_set():
                break
        out = result.get("out")
        if out is None:
            break
        new_ids = out.sequences
        gained = new_ids.shape[1] - cur_ids.shape[1]
        produced += gained
        cur_ids = new_ids
        attn = torch.ones_like(cur_ids)
        eos_hit = cur_ids[0, -1].item() == TOK.eos_token_id
        if STOP_EVENT.is_set() or eos_hit or gained < chunk or reconfig_every <= 0:
            break
        # Mid-rollout reconfig on prompt+rollout; KV cache stays valid (§6.3).
        gen = _animate(CFGR.iterate(MODEL, cur_ids, steps=mid_steps,
                                    reconfig_lr=float(reconfig_lr),
                                    lateral_steps=int(lateral_steps),
                                    lateral_polyak=lateral_polyak, k=k,
                                    init_mask=CTRL.mask, init_a=init_a, init_b=init_b),
                       history, final_img_ref, gamma)
        try:
            while True:
                yield next(gen)
        except StopIteration as stop:
            if stop.value is not None:
                mask = stop.value
        CTRL.set_mask(mask)


def ask(question: str, k: int = 192, reconfig_steps: int = 1,
        thinking: bool = False) -> tuple[str, dict]:
    """Non-streaming programmatic endpoint for Claude (design.md §8.3).
    Returns (answer_text, route_summary)."""
    if not BUSY.acquire(blocking=False):
        return "The agent is busy synthesizing or training; try again shortly.", {"busy": True}
    try:
        STOP_EVENT.clear()
        ids = TOK(question, return_tensors="pt")["input_ids"].to(DEV)
        mask, _, _ = CFGR(MODEL, ids, steps=int(reconfig_steps))
        CTRL.set_mask(mask)
        # which blocks did the mask land on?
        route = {}
        for key, ref in REFS.items():
            ov = float((mask * ref.to(mask.device)).sum() / max(1, int(k)))
            if ov > 0.05:
                route[BLOCK_LABELS.get(key, key)] = round(ov, 2)
        enc = _encode_chat(question, thinking)
        with torch.no_grad():
            out = MODEL.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                 repetition_penalty=1.1, no_repeat_ngram_size=4,
                                 pad_token_id=TOK.eos_token_id)
        ans = TOK.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return ans, {"routed_to": route}
    finally:
        BUSY.release()


# --- Memory panel actions ----------------------------------------------------

def _run_streaming(fn):
    """Run fn(log) in a thread; yield the growing log text as lines arrive.
    Returns the accumulated lines list + a result dict via the last yield's
    caller state (callers read `res` after the loop)."""
    q = _queue.Queue()
    res = {}

    def log(m):
        print(m, flush=True)   # mirror to the pod terminal (persistent record)
        q.put(str(m))

    def run():
        try:
            # set_default_device is thread-local; re-assert it in this worker so
            # bare tensor creations (e.g. torch.zeros) land on cuda, not cpu.
            if torch.cuda.is_available():
                torch.set_default_device("cuda")
            res["value"] = fn(log)
        except Exception as e:
            q.put("ERROR: " + repr(e))
            q.put(traceback.format_exc())
        finally:
            q.put(None)

    Thread(target=run, daemon=True).start()
    lines = []
    while True:
        m = q.get()
        if m is None:
            break
        lines.append(m)
        yield "\n".join(lines), res, lines


def do_consolidate():
    if not BUSY.acquire(blocking=False):
        yield "⏳ Busy — a chat or another memory op is running. Try again in a moment."
        return
    try:
        for text, res, _lines in _run_streaming(lambda log: consolidate(CFG, H, log=log)):
            summ = res.get("value")
            yield text + (f"\n\nSUMMARY: {summ}" if summ else "")
    finally:
        BUSY.release()


def do_train():
    if not BUSY.acquire(blocking=False):
        yield "⏳ Busy — a chat or another memory op is running. Try again in a moment.", gr.update()
        return
    try:
        last = ""
        for text, _res, _lines in _run_streaming(lambda log: train_pending(CFG, H, opt=OPT, log=log)):
            last = text
            yield text, gr.update()   # stream log; leave legend unchanged
        _load_refs()                  # refresh coloring/legend after the bank changed
        MODEL.eval()
        yield last + "\n\nTraining done. Legend/colors refreshed.", _legend_html()
    finally:
        BUSY.release()


def clear_context():
    """Reset chat + reconfig mask WITHOUT touching the learned bank (§8.2)."""
    CTRL.set_mask(torch.zeros(CFG.H, device=CTRL.mask.device))
    return [], render(CTRL.mask, 8.0), "context cleared (bank untouched)"


def capture_status():
    return str(status(CFG))


def reset_busy():
    """Force-release the BUSY lock if a prior op leaked it (e.g. a hung/cancelled
    synthesis). A plain Lock can be released from any thread. Use only when the
    agent is wrongly stuck 'busy' and GPU is idle."""
    try:
        BUSY.release()
        return "🔓 busy lock released."
    except RuntimeError:
        return "already free — nothing to reset."


def _legend_html():
    items = []
    for key in REFS:
        c = BLOCK_COLORS.get(key, np.array([0.6, 0.6, 0.6]))
        rgb = f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
        label = BLOCK_LABELS.get(key, key)
        items.append(f"<span style='display:inline-block;width:12px;height:12px;"
                     f"background:{rgb};margin-right:4px;border-radius:2px'></span>{label}")
    return "Blocks (old→new): " + (" &nbsp; ".join(items) if items else "(none trained yet)")


def main():
    with gr.Blocks(title="FTN-Agent") as demo:
        gr.Markdown("### FTN-Agent — continually-learning workspace expert")
        legend = gr.Markdown(_legend_html())
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=520)  # gradio 6: messages format only, no type=
                with gr.Row():
                    box = gr.Textbox(scale=8, show_label=False,
                                     placeholder="Ask about the workspace, the code, or what's been tried…")
                    stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
            with gr.Column(scale=2):
                mask_img = gr.Image(label="atom mask m (expert subspace) + per-block prevalence",
                                    height=360)
                thinking = gr.Checkbox(value=False, label="reasoning (<think>) — slower")
                with gr.Accordion("FTN controls", open=False):
                    temperature = gr.Slider(0.0, 1.5, value=0.6, step=0.05, label="temperature")
                    reconfig_steps = gr.Slider(1, 100, value=CFG.reconfig_steps, step=1, label="reconfig steps")
                    lateral_steps = gr.Slider(0, 10, value=CFG.lateral_steps, step=1, label="lateral smoothing steps")
                    lateral_polyak = gr.Slider(0.0, 1.0, value=CFG.lateral_polyak, step=0.05, label="lateral polyak")
                    reconfig_lr = gr.Slider(0.02, 1.0, value=CFG.reconfig_lr, step=0.02, label="reconfig lr")
                    k_atoms = gr.Slider(8, min(1024, CFG.H), value=CFG.k, step=8, label="k — active atoms")
                    reconfig_every = gr.Slider(0, 64, value=0, step=1, label="mid-rollout reconfig every N tok (0=off)")
                    mid_steps = gr.Slider(1, 60, value=3, step=1, label="mid-rollout reconfig steps")
                    init_a = gr.Slider(0.0, 1.0, value=0.01, step=0.05, label="mask init a·mask (mid-rollout)")
                    init_b = gr.Slider(0.0, 1.0, value=0.0035, step=0.05, label="mask init b·noise")
                    gamma = gr.Slider(0.2, 100.0, value=8.0, step=0.2, label="color sharpness γ")
                with gr.Accordion("Memory (consolidate / train / clear)", open=True):
                    mem_log = gr.Textbox(label="memory log", lines=8, interactive=False)
                    with gr.Row():
                        consolidate_btn = gr.Button("🧠 Consolidate / Synthesize now")
                        train_btn = gr.Button("🎓 Train now", variant="primary")
                    with gr.Row():
                        clear_btn = gr.Button("🧹 Clear context")
                        status_btn = gr.Button("📊 Capture status")
                        reset_busy_btn = gr.Button("🔓 Reset busy")

        inputs = [box, chat, temperature, reconfig_steps, lateral_steps, lateral_polyak,
                  reconfig_lr, k_atoms, reconfig_every, mid_steps, init_a, init_b, gamma, thinking]
        submit_event = box.submit(respond, inputs, [chat, mask_img], show_progress="hidden")
        submit_event.then(lambda: "", None, box)
        stop_btn.click(lambda: (STOP_EVENT.set(), "")[1], None, box, cancels=[submit_event])

        consolidate_btn.click(do_consolidate, None, mem_log)
        train_btn.click(do_train, None, [mem_log, legend])
        clear_btn.click(clear_context, None, [chat, mask_img, mem_log])
        status_btn.click(capture_status, None, mem_log)
        reset_busy_btn.click(reset_busy, None, mem_log)

        # Programmatic endpoint for Claude (gradio_client): api_name="ask".
        gr.api(ask, api_name="ask")

    print(f"\nServing on 127.0.0.1:{CFG.port} — expose with:  tailscale serve {CFG.port}")
    print(f"Claude can query it via gradio_client predict(api_name='/ask').\n")
    demo.queue().launch(server_name="127.0.0.1", server_port=CFG.port)


if __name__ == "__main__":
    main()
