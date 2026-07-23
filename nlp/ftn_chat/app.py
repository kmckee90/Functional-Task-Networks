"""ChatFTN live demo: chat + animated mask heatmap.

On each user message we (1) reconfigure the atom-bank mask on the prompt by
perplexity (recording each step), (2) animate the mask as it locates the expert
subspace, then (3) stream the answer generated under that mask. The mask is
colored by which domain reference claims each active atom, so you can watch the
"expert" light up.

Run on the server:
    cd ftn_chat && uv run python app.py
Then expose it privately over Tailscale (NOT the public gradio share tunnel):
    tailscale serve 7860
and open the printed MagicDNS https URL from your laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from threading import Event, Thread

# Must be set BEFORE the first CUDA op so cuBLAS picks a deterministic workspace;
# without it torch.use_deterministic_algorithms() raises on cuBLAS GEMMs. Safe to
# set unconditionally at import (no CUDA context exists yet).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gradio as gr
import numpy as np
import registry
import torch
from config import CFG, DOMAIN_LABELS
from model import build, load_bank
from perplexity_configurer import PerplexityConfigurer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

# Colors come from the registry, so newly-added tasks are colored automatically.
DOMAIN_COLORS = {k: np.array(c) for k, c in registry.colors().items()}
# Pixel upscale per atom, sized so the rendered map stays ~1000px regardless of
# grid_dim (22 was tuned for the old 48x48 grid; 96x96 needs ~10).
UPSCALE = max(4, 1024 // CFG.grid_dim)

# Whether the model reasons ("thinks") before answering. Defaults to on (the
# Qwen chat-template default); flipped off by the --no-thinking CLI flag in
# _parse_cli(), which runs before the first generate() call.
ENABLE_THINKING = True

# Set by the Stop button; checked both in the reconfig/stream loops and inside
# generate() so the worker thread actually halts (and frees the GPU) rather than
# running to completion in the background after the UI stops updating.
STOP_EVENT = Event()


class _StopOnFlag(StoppingCriteria):
    """Halt generation as soon as the Stop button sets STOP_EVENT."""

    def __call__(self, *_args, **_kwargs) -> bool:
        return STOP_EVENT.is_set()


def _seed_all():
    """Reset EVERY RNG generation draws on back to the boot seed (CFG.seed):
    Python's `random`, NumPy, and torch (CPU + all CUDA devices). Pair this with
    the deterministic-kernel flags set in _load so re-seeding actually reproduces
    bit-identical output rather than just re-seeding torch's CPU generator."""
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)  # also seeds the current CUDA device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.seed)  # every CUDA device


def _load():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    # Force deterministic GPU kernels — by default cuDNN/cuBLAS pick fast but
    # non-reproducible algorithms, so the same seed still yields different output.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    _seed_all()
    model, tok, ctrl, wrappers, _ = build(CFG)
    try:
        load_bank(wrappers, CFG.ckpt_dir / "bank.pt")
        print(f"  loaded bank from {CFG.ckpt_dir / 'bank.pt'}")
    except FileNotFoundError:
        print("  WARNING: no trained bank found; running on untrained atoms.")
    # Prefer usage-weighted refs (refs_usage.pt) so a neuron shared by several
    # tasks blends their colors by relative usage; fall back to the binary
    # top-k refs.pt. See rebuild_refs_usage.py.
    refs_p = CFG.ckpt_dir / "refs_usage.pt"
    if not refs_p.exists():
        refs_p = CFG.ckpt_dir / "refs.pt"
    try:
        refs = torch.load(refs_p, map_location="cpu")
        refs = {k: v.float() for k, v in refs.items()}
        print(f"  loaded refs from {refs_p.name}")
    except FileNotFoundError:
        refs = {k: torch.zeros(CFG.H) for k in CFG.domains}
    # Inference: no gradient checkpointing (reconfig keeps a small graph itself)
    # and re-enable the KV cache for fast generation.
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    return model, tok, ctrl, wrappers, refs


def _parse_cli():
    """Honor a CLI checkpoint-dir override. Must run BEFORE _load(), which reads
    CFG.ckpt_dir at import time. Precedence: --ckpt-dir > FTN_CKPT_DIR > config
    default (the latter two are already baked into CFG.ckpt_dir). Uses
    parse_known_args so it ignores any extra args (e.g. gradio's)."""
    p = argparse.ArgumentParser(description="ChatFTN live demo")
    p.add_argument(
        "--ckpt-dir",
        "--checkpoints",
        dest="ckpt_dir",
        default=None,
        help="checkpoint folder to serve from (reads bank.pt, refs_usage.pt/"
        "refs.pt). Overrides FTN_CKPT_DIR and the config default.",
    )
    p.add_argument(
        "--no-thinking",
        dest="no_thinking",
        action="store_true",
        help="disable the model's thinking/reasoning mode (enabled by default). "
        "Passes enable_thinking=False to the chat template.",
    )
    args, _ = p.parse_known_args()
    if args.ckpt_dir:
        CFG.ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    global ENABLE_THINKING
    ENABLE_THINKING = not args.no_thinking
    print(f"  serving checkpoints from: {CFG.ckpt_dir}")
    print(f"  thinking mode: {'enabled' if ENABLE_THINKING else 'disabled'}")


def _sync_cfg_to_ckpt():
    """Make the served bank authoritative over the config: read k (and H) from
    state.json and the LoRA rank from bank.pt's tensor shapes (A is H x in x
    rank), overriding CFG before build(). The bank's H/rank MUST match or
    load_bank's shape-copy fails; k must match or the MaskController activates
    the wrong number of atoms. Runs for whatever ckpt_dir is in effect (CLI,
    FTN_CKPT_DIR, or default)."""
    state_p = CFG.ckpt_dir / "state.json"
    bank_p = CFG.ckpt_dir / "bank.pt"
    if state_p.exists():
        state = json.loads(state_p.read_text())
        if state.get("k") is not None:
            CFG.k = int(state["k"])
        if state.get("H") is not None:
            H = int(state["H"])
            g = int(round(H**0.5))
            if g * g != H:
                raise SystemExit(f"checkpoint H={H} is not a perfect square.")
            CFG.grid_dim = g
    if bank_p.exists():
        sd = torch.load(bank_p, map_location="cpu", mmap=True, weights_only=True)
        a_keys = [key for key in sd if key.split(".")[-1] == "A"]
        if a_keys:
            CFG.rank = int(sd[a_keys[0]].shape[-1])
    print(f"  synced config to checkpoint: H={CFG.H} k={CFG.k} rank={CFG.rank}")


_parse_cli()
_sync_cfg_to_ckpt()
MODEL, TOK, CTRL, WRAPPERS, REFS = _load()
CFGR = PerplexityConfigurer(CFG, CTRL)
DEV = next(
    MODEL.parameters()
).device  # event handler runs off-thread; place inputs explicitly


# Deactivated atoms aren't black — they glow faintly in their domain color so the
# whole expert map stays visible; active atoms light up to full color.
FAINT = 0.18


def _bar_panel(
    flat_mask: torch.Tensor, height: int, width: int, gamma: float = 1.0
) -> np.ndarray:
    """Horizontal colored bar per domain = its SHARE of the currently-active
    atoms' color. Each active atom contributes a per-domain color share
    p[d][n] = REFS[d][n]**g / sum_t REFS[t][n]**g (how the pixel's color splits
    across domains, sharpened by gamma g); the bar is the mean of those shares
    over the active atoms, so bars SUM TO 1 for any mask AND match the aggregate
    color of the lit-up map exactly. Bars use the same colors as the networks,
    so a tall on-topic bar = the mask configured onto the right expert."""
    m = flat_mask.detach().float().cpu().numpy()
    keys = list(REFS.keys())
    n = max(len(keys), 1)
    W = np.stack([REFS[key].cpu().numpy() for key in keys])  # (T, H) usage weights
    if gamma != 1.0:
        W = W**gamma  # sharpen/flatten contrast
    denom = W.sum(0)  # per-atom total usage
    shares = np.divide(W, denom, out=np.zeros_like(W), where=denom > 0)  # color split
    Z = max(float((m * (denom > 0)).sum()), 1e-12)  # active weight on colored atoms
    contrib = shares @ m  # per domain; sums to Z
    panel = np.zeros((height, width, 3))
    row_h = height // n
    pad = max(2, row_h // 8)
    for i, key in enumerate(keys):
        frac = min(contrib[i] / Z, 1.0)
        col = DOMAIN_COLORS.get(key, np.array([0.6, 0.6, 0.6]))
        y0, y1 = i * row_h + pad, i * row_h + row_h - pad
        panel[y0:y1, :] = col * 0.15  # faint full-length track
        bar_w = int(frac * (width - 1))
        if bar_w > 0:
            panel[y0:y1, :bar_w] = col
    return (np.clip(panel, 0, 1) * 255).astype(np.uint8)


def render(flat_mask: torch.Tensor, gamma: float = 1.0) -> np.ndarray:
    """Color the CURRENT flat mask by domain membership -> upscaled RGB uint8,
    with a per-domain prevalence bar panel composited to the right.
    Colored (domain-claimed) atoms: full domain color when active, FAINT when
    inactive. Uncolored atoms: 50% gray when active, black when inactive.
    gamma sharpens (>1) or flattens (<1) the per-atom color split across domains;
    the bar panel uses the same gamma so map and bars stay consistent."""
    G = CFG.grid_dim
    m = flat_mask.detach().float().cpu().numpy()
    color = np.zeros((CFG.H, 3))
    for key, ref in REFS.items():
        r = ref.cpu().numpy()
        if gamma != 1.0:
            r = r**gamma
        color += np.outer(r, DOMAIN_COLORS.get(key, np.array([0.6, 0.6, 0.6])))
    # Renormalize each colored atom's hue to unit strength. Divide by the TRUE
    # per-atom weight sum (not a fixed floor): gamma shrinks the weights toward
    # 0, so a fixed clip floor would collapse the colors toward black at high
    # gamma. Atoms with no usage (norm == 0) stay black.
    norm = color.sum(1, keepdims=True)
    colored = norm[:, 0] > 0
    color = np.where(norm > 0, color / np.where(norm > 0, norm, 1.0), 0.0)
    intensity = FAINT + (1.0 - FAINT) * m  # colored: inactive FAINT, active full
    img = intensity[:, None] * color
    img[~colored] = (0.5 * m[~colored])[
        :, None
    ]  # uncolored: inactive black, active 50% gray
    img = np.clip(img.reshape(G, G, 3), 0, 1)
    big = (np.kron(img, np.ones((UPSCALE, UPSCALE, 1))) * 255).astype(np.uint8)
    h = big.shape[0]
    bars = _bar_panel(flat_mask, h, width=h // 2, gamma=gamma)
    gap = np.zeros((h, max(4, UPSCALE // 2), 3), dtype=np.uint8)
    return np.concatenate([big, gap, bars], axis=1)


# Deliberately generic and domain-free: naming the domains here would inject
# shared tokens into every mid-rollout reconfiguration (which runs on the
# chat-templated prompt), inflating cross-domain mask overlap. Keep it neutral.
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely and accurately."
MAX_NEW_TOKENS = 512


def _animate(frames_iter, history, final_img_ref, gamma=1.0):
    """Consume a configurer iterate() generator, yielding (history, image) per
    frame so the search animates. The image is always just the CURRENT mask.
    Returns the final mask. final_img_ref is a 1-element list so the caller sees
    the latest image after the generator finishes."""
    mask = None
    for fr in frames_iter:
        mask = fr
        final_img_ref[0] = render(fr, gamma)
        yield history, final_img_ref[0]
        time.sleep(0.02)
        if STOP_EVENT.is_set():
            break
    return mask


def respond(
    message,
    history,
    temperature,
    reconfig_steps,
    lateral_steps,
    lateral_polyak,
    reconfig_lr,
    k,
    reconfig_every,
    mid_steps,
    init_a,
    init_b,
    gamma,
    enable_thinking,
):
    STOP_EVENT.clear()  # fresh request; ignore any prior Stop
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    k = int(k)
    reconfig_every = int(reconfig_every)  # 0 => reconfigure once, never mid-rollout
    mid_steps = int(mid_steps)
    init_a, init_b = float(init_a), float(init_b)
    lateral_polyak = float(lateral_polyak)
    gamma = float(gamma)

    # Phase 1: reconfigure on the bare prompt, animating the search LIVE — each
    # frame is one real forward pass. Search depth, smoothing, LR, and active-atom
    # count k come from the GUI. This initial search must NOT warm-start from the
    # previous prompt's mask, so init_a is forced to 0 (pure b*noise init); the
    # `a` slider only warm-starts the mid-rollout reconfigs in Phase 2.
    ids = TOK(message, return_tensors="pt")["input_ids"].to(DEV)
    final_img_ref = [render(CTRL.mask, gamma)]
    mask = None
    gen = _animate(
        CFGR.iterate(
            MODEL,
            ids,
            steps=int(reconfig_steps),
            reconfig_lr=float(reconfig_lr),
            lateral_steps=int(lateral_steps),
            lateral_polyak=lateral_polyak,
            k=k,
            init_mask=None,
            init_a=0.0,
            init_b=init_b,
        ),
        history,
        final_img_ref,
        gamma,
    )
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

    # Phase 2: generate under the located mask. If reconfig_every > 0 we decode
    # in chunks of that many tokens and, between chunks, re-run mid_steps of
    # reconfiguration on the FULL sequence so far (prompt + rollout) — the expert
    # subspace tracks the answer as it grows. reconfig_every == 0 keeps the old
    # single-shot behaviour (decode straight through under the initial mask).
    CTRL.set_mask(mask)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    enc = TOK.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=bool(enable_thinking),
    ).to(DEV)
    cur_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    do_sample = float(temperature) > 0.0
    partial = ""
    produced = 0
    while produced < MAX_NEW_TOKENS and not STOP_EVENT.is_set():
        budget = MAX_NEW_TOKENS - produced
        chunk = budget if reconfig_every <= 0 else min(reconfig_every, budget)
        # Force a few tokens on the first chunk so a snap EOS can't yield an
        # empty answer; afterwards a single token per chunk is fine.
        min_new = min(24, chunk) if produced == 0 else 1
        streamer = TextIteratorStreamer(TOK, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            input_ids=cur_ids,
            attention_mask=attn,
            max_new_tokens=chunk,
            min_new_tokens=min_new,
            do_sample=do_sample,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            streamer=streamer,
            pad_token_id=TOK.eos_token_id,
            stopping_criteria=StoppingCriteriaList([_StopOnFlag()]),
            return_dict_in_generate=True,
        )
        if do_sample:
            kwargs["temperature"] = float(temperature)
        result = {}

        def _run():
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
        # Stop the rollout if the model emitted EOS, the Stop button fired, the
        # chunk ended early (a stop reason other than the length cap), or
        # mid-rollout reconfig is disabled.
        eos_hit = cur_ids[0, -1].item() == TOK.eos_token_id
        if STOP_EVENT.is_set() or eos_hit or gained < chunk or reconfig_every <= 0:
            break

        # Mid-rollout reconfiguration on prompt + rollout-so-far, warm-started
        # from the current mask per the a/b sliders. Animate it, then resume.
        gen = _animate(
            CFGR.iterate(
                MODEL,
                cur_ids,
                steps=mid_steps,
                reconfig_lr=float(reconfig_lr),
                lateral_steps=int(lateral_steps),
                lateral_polyak=lateral_polyak,
                k=k,
                init_mask=CTRL.mask,
                init_a=init_a,
                init_b=init_b,
            ),
            history,
            final_img_ref,
            gamma,
        )
        try:
            while True:
                yield next(gen)
        except StopIteration as stop:
            if stop.value is not None:
                mask = stop.value
        CTRL.set_mask(mask)


def _reset_seed():
    """Re-seed every RNG to the value the app booted from (see _seed_all),
    restoring the deterministic generation state it had at load time."""
    _seed_all()


def _legend_html():
    # One colored swatch per TRAINED task (whatever is in the loaded refs).
    items = []
    for key in REFS:
        c = DOMAIN_COLORS.get(key, np.array([0.6, 0.6, 0.6]))
        rgb = f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})"
        label = DOMAIN_LABELS.get(key, key)
        items.append(
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{rgb};margin-right:4px;border-radius:2px'></span>{label}"
        )
    return " &nbsp; ".join(items)


def main():
    with gr.Blocks(title="ChatFTN") as demo:
        gr.Markdown(
            "### ChatFTN — prompt-routed expert knowledge\n"
            "Mask colors: " + _legend_html()
        )
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=520)  # gradio 6: messages format only
                with gr.Row():
                    box = gr.Textbox(
                        scale=8,
                        show_label=False,
                        placeholder="Ask about the base FTN code, the explorer agent, Globo, or the MT10 robotics scripts…",
                    )
                    stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
                    # Hidden for now: re-seeding doesn't reliably reproduce
                    # output yet. Keep the wiring; flip visible=True to restore.
                    seed_btn = gr.Button("🌱 Reset seed", scale=1, visible=False)
            with gr.Column(scale=2):
                mask_img = gr.Image(
                    label="atom mask  m  (expert subspace)  +  per-network prevalence",
                    height=380,
                )
                with gr.Accordion("Controls", open=False):
                    # Per-message thinking toggle. Initial state honors the
                    # --no-thinking CLI flag (ENABLE_THINKING); flipping it here
                    # takes effect on the next message with no restart.
                    thinking = gr.Checkbox(
                        value=ENABLE_THINKING,
                        label="Enable thinking (model reasons before answering)",
                    )
                    temperature = gr.Slider(
                        0.0,
                        1.5,
                        value=0.7,
                        step=0.05,
                        label="LLM temperature (answer sampling)",
                    )
                    reconfig_steps = gr.Slider(
                        1,
                        100,
                        value=CFG.reconfig_steps,
                        step=1,
                        label="Reconfig steps (mask search depth)",
                    )
                    # Default = CFG.lateral_steps (0 = plain KWTA, matching the
                    # base Large/NoReplay/Reactive_Low config); raise the slider
                    # to turn SmoothKWTA blob coherence on live.
                    lateral_steps = gr.Slider(
                        0,
                        10,
                        value=CFG.lateral_steps,
                        step=1,
                        label="Lateral smoothing steps (0 = plain KWTA)",
                    )
                    lateral_polyak = gr.Slider(
                        0.0,
                        1.0,
                        value=CFG.lateral_polyak,
                        step=0.05,
                        label="Lateral polyak (lower = stronger smoothing pull)",
                    )
                    reconfig_lr = gr.Slider(
                        0.02,
                        1.0,
                        value=CFG.reconfig_lr,
                        step=0.02,
                        label="Reconfig learning rate (mask logits)",
                    )
                    k_atoms = gr.Slider(
                        8,
                        min(1024, CFG.H),
                        value=CFG.k,
                        step=8,
                        label="k — active atoms in the mask",
                    )
                    reconfig_every = gr.Slider(
                        0,
                        64,
                        value=16,
                        step=1,
                        label="Mid-rollout reconfig every N tokens (0 = off)",
                    )
                    mid_steps = gr.Slider(
                        1,
                        60,
                        value=3,
                        step=1,
                        label="Mid-rollout reconfig steps (per re-search)",
                    )
                    init_a = gr.Slider(
                        0.0,
                        1.0,
                        value=0.01,
                        step=0.05,
                        label="Mask init: a · current mask (mid-rollout only)",
                    )
                    init_b = gr.Slider(
                        0.0,
                        1.0,
                        value=0.0035,
                        step=0.05,
                        label="Mask init: b · noise",
                    )
                    gamma = gr.Slider(
                        0.2,
                        100.0,
                        value=8.0,
                        step=0.2,
                        label="Color sharpness γ (domain contrast in map + bars)",
                    )
        # show_progress="hidden": otherwise Gradio overlays a processing spinner
        # on the outputs during the heavy phase-1 steps, masking the live mask
        # frames so only the first/last show through. The animation IS the
        # progress indicator.
        submit_event = box.submit(
            respond,
            [
                box,
                chat,
                temperature,
                reconfig_steps,
                lateral_steps,
                lateral_polyak,
                reconfig_lr,
                k_atoms,
                reconfig_every,
                mid_steps,
                init_a,
                init_b,
                gamma,
                thinking,
            ],
            [chat, mask_img],
            show_progress="hidden",
        )
        submit_event.then(lambda: "", None, box)
        # Stop: signal the worker thread to halt generation AND cancel the Gradio
        # event so the UI stops streaming immediately; clears the textbox too.
        stop_btn.click(
            lambda: (STOP_EVENT.set(), "")[1], None, box, cancels=[submit_event]
        )
        # Reset seed: re-apply the exact seed the app booted from, so generation
        # returns to the deterministic state it had at load time.
        seed_btn.click(_reset_seed, None, None)
    print(
        f"\nServing on 127.0.0.1:{CFG.port} — forward with e.g. "
        f"`actl pod exec --port {CFG.port}` and open http://localhost:{CFG.port}\n"
    )
    demo.queue().launch(server_name="127.0.0.1", server_port=CFG.port)


if __name__ == "__main__":
    main()
