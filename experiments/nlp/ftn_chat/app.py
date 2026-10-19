"""ChatFTN live demo: chat + animated mask heatmap.

On each user message we (1) update persistent BayesLoRA mask logits on prompt
perplexity using the same reactive rule as training, (2) animate the sampled
mask, then (3) stream the answer generated under that mask. The mask is
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
from bayes_configurer import BayesianPerplexityConfigurer
from input_routing import InputRoutingArchive
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


def _load_refs(cfg):
    """Prefer usage refs, falling back when an older bank's map is incompatible."""
    for name in ("refs_usage.pt", "refs.pt"):
        path = cfg.ckpt_dir / name
        if not path.exists():
            continue
        refs = torch.load(path, map_location="cpu", weights_only=True)
        shapes = {key: tuple(value.shape) for key, value in refs.items()}
        if any(shape != (cfg.H,) for shape in shapes.values()):
            print(f"  WARNING: ignoring {path}: reference shapes {shapes} "
                  f"do not match checkpoint H={cfg.H}.")
            continue
        print(f"  loaded refs from {name}")
        return {key: value.float() for key, value in refs.items()}
    return {key: torch.zeros(cfg.H) for key in cfg.domains}


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
    refs = _load_refs(CFG)
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
        for name in ("lateral_steps", "lateral_kernel", "lateral_polyak",
                     "lateral_noise_sd", "torus"):
            if name in state.get("lateral", {}):
                setattr(CFG, name, state["lateral"][name])
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
detector_state = None
state_p = CFG.ckpt_dir / "state.json"
if state_p.exists():
    saved_state = json.loads(state_p.read_text())
    if saved_state.get("detector_objective") == "prompt_nll":
        detector_state = saved_state.get("reactive")
CFGR = BayesianPerplexityConfigurer(CFG, CTRL, detector_state)
posterior_p = CFG.ckpt_dir / "posterior.pt"
if posterior_p.exists():
    CFGR.load_state_dict(torch.load(posterior_p, map_location="cpu", weights_only=True))
    print(f"  loaded mask posterior from {posterior_p}")
ROUTER = InputRoutingArchive(CFG, TOK, MODEL)
routing_p = CFG.ckpt_dir / "routing.pt"
if routing_p.exists():
    ROUTER.load_state_dict(torch.load(routing_p, map_location="cpu", weights_only=True))
    print(f"  loaded {len(ROUTER.routes)} input routes from {routing_p}")
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


def _animate(frames_iter, history, final_img_ref, gamma=1.0, route_status="Bayes search"):
    """Consume a configurer iterate() generator, yielding (history, image, status) per
    frame so the search animates. The image is always just the CURRENT mask.
    Returns the final mask. final_img_ref is a 1-element list so the caller sees
    the latest image after the generator finishes."""
    mask = None
    try:
        for fr in frames_iter:
            mask = fr
            final_img_ref[0] = render(fr, gamma)
            yield history, final_img_ref[0], route_status
            time.sleep(0.02)
            if STOP_EVENT.is_set():
                mask = None
                break
    finally:
        frames_iter.close()  # rollback an interrupted posterior update
    return mask


def respond(
    message,
    history,
    temperature,
    stable_steps,
    cp_steps,
    reconfig_lr,
    cp_lr,
    mask_temperature,
    cp_temperature,
    kl_weight,
    prior_decay,
    reactive_sd,
    reactive_ema,
    k,
    reconfig_every,
    gamma,
    enable_thinking,
    input_routing,
    route_margin,
    route_semantic_weight,
    route_mode,
    route_refine_steps,
    route_refine_kl,
):
    STOP_EVENT.clear()  # fresh request; ignore any prior Stop
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    k = int(k)
    reconfig_every = int(reconfig_every)  # 0 => reconfigure once, never mid-rollout
    gamma = float(gamma)

    search = dict(k=k, steps=int(stable_steps), changepoint_steps=int(cp_steps),
                  lr=float(reconfig_lr), changepoint_lr=float(cp_lr),
                  temperature=float(mask_temperature),
                  changepoint_temperature=float(cp_temperature),
                  kl_weight=float(kl_weight), prior_decay=float(prior_decay),
                  reactive_sd=float(reactive_sd), reactive_ema=float(reactive_ema))

    ids = TOK(message, return_tensors="pt")["input_ids"].to(DEV)
    final_img_ref = [render(CTRL.mask, gamma)]
    mask = None
    routed = False
    route_status = "Bayes search"
    if input_routing and ROUTER.routes and k == CFG.k:
        match, info = ROUTER.match(message, k=k, margin=route_margin,
                                   semantic_weight=route_semantic_weight)
        if match is not None:
            routed = True
            route_logits = ROUTER.routes[info["task"]]["logits"]
            if route_mode == "Refine":
                route_status = (f"Refining **{info['task']}** mask "
                                f"({int(route_refine_steps)} prompt steps)")
                mask = yield from _animate(
                    CFGR.iterate(MODEL, ids, **search,
                                 routed_prior=route_logits,
                                 routed_steps=int(route_refine_steps),
                                 routed_kl_weight=float(route_refine_kl)),
                    history, final_img_ref, gamma, route_status,
                )
                if mask is not None:
                    route_status = (f"Refined **{info['task']}** mask "
                                    f"({CFGR.last_stats['steps']} steps; "
                                    f"prompt NLL {CFGR.last_stats['nll']:.3f})")
                    yield history, final_img_ref[0], route_status
            else:
                mask = match.to(CTRL.mask.device)
                CTRL.k = k
                CTRL.set_mask(mask)
                CFGR.mask_logits.copy_(route_logits.to(CFGR.mask_logits.device))
                # A retrieved route starts a fresh dwell regime. Give a
                # mid-rollout update a baseline under this exact mask.
                with torch.no_grad():
                    fitted = float(CFGR._nll(MODEL, ids, torch.ones_like(ids)))
                CFGR.detector.mean, CFGR.detector.var, CFGR.detector.since = fitted, 0.0, 0
                route_status = (f"Retrieved **{info['task']}** mask "
                                f"(margin {info['margin']:.2f}; prompt NLL {fitted:.3f})")
                final_img_ref[0] = render(mask, gamma)
                yield history, final_img_ref[0], route_status
        else:
            route_status = f"Bayes search: route {info['reason']}"
    elif input_routing:
        route_status = ("Bayes search: routing masks were repaired for k=" + str(CFG.k)
                        if ROUTER.routes else
                        "Bayes search: no routing archive in checkpoint")
    if not routed:
        mask = yield from _animate(
            CFGR.iterate(MODEL, ids, **search),
            history,
            final_img_ref,
            gamma,
            route_status,
        )

    if mask is None or STOP_EVENT.is_set():
        if mask is not None:
            CTRL.set_mask(mask)
            yield history, render(mask, gamma), route_status
        return

    # Phase 2: generate under the located mask. If reconfig_every > 0 we decode
    # in chunks of that many tokens and update the posterior on the full
    # sequence so far (prompt + rollout) — the expert
    # subspace tracks the answer as it grows. reconfig_every == 0 keeps the old
    # single-shot behaviour (decode straight through under the initial mask).
    CTRL.set_mask(mask)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    # The 27B default prefix already opens <think>. Token bans cannot disable
    # that mode; let the model's template open/close it according to the toggle.
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
            yield history, final_img_ref[0], route_status
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

        # Mid-rollout uses the same detector and posterior as a new prompt.
        mask = yield from _animate(
            CFGR.iterate(MODEL, cur_ids, **search),
            history,
            final_img_ref,
            gamma,
            route_status,
        )
        if mask is None or STOP_EVENT.is_set():
            break
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
    website = os.environ.get('FTN_WEBSITE_MODE') == '1'
    with gr.Blocks(title="ChatFTN") as demo:
        gr.Markdown(
            "### ChatFTN — prompt-routed expert knowledge\n"
            "Mask colors: " + _legend_html()
        )
        with gr.Row(elem_id='chat-layout'):
            with gr.Column(scale=3,min_width=280):
                chat = gr.Chatbot(height=430 if website else 520,show_label=not website)
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
            with gr.Column(scale=2,min_width=260):
                mask_img = gr.Image(
                    value=render(CTRL.mask) if website else None,
                    label="Neural assemblies · domain activity" if website else "atom mask  m  (expert subspace)  +  per-network prevalence",
                    height=330 if website else 380,interactive=False,
                )
                route_status = gr.Markdown("Bayes search")
                with gr.Accordion("Controls", open=False,visible=not website):
                    input_routing = gr.Checkbox(
                        value=website,
                        label="Use repaired input routes when confident",
                    )
                    route_margin = gr.Slider(
                        0.0, 4.0, value=CFG.input_route_margin, step=0.05,
                        label="Minimum route score margin",
                    )
                    route_semantic_weight = gr.Slider(
                        0.0, 8.0, value=CFG.input_route_semantic_weight, step=0.1,
                        label="Question semantic similarity weight",
                    )
                    route_mode = gr.Radio(
                        ["Direct", "Refine"], value="Direct",
                        label="Matched route: apply directly or refine on prompt",
                    )
                    route_refine_steps = gr.Slider(
                        1, 120, value=CFG.input_route_refine_steps, step=1,
                        label="Route refinement steps",
                    )
                    route_refine_kl = gr.Slider(
                        0.0, 10.0, value=CFG.input_route_refine_kl, step=0.1,
                        label="Route refinement KL to retrieved logits",
                    )
                    # Per-message thinking toggle. Initial state honors the
                    # --no-thinking CLI flag (ENABLE_THINKING); flipping it here
                    # takes effect on the next message with no restart.
                    thinking = gr.Checkbox(
                        value=False if website else ENABLE_THINKING,
                        label="Enable thinking (model reasons before answering)",
                    )
                    temperature = gr.Slider(
                        0.0,
                        1.5,
                        value=0.7,
                        step=0.05,
                        label="LLM temperature (answer sampling)",
                    )
                    stable_steps = gr.Slider(
                        1, 20, value=CFG.bayes_reconfig_steps, step=1,
                        label="Stable prompt steps",
                    )
                    cp_steps = gr.Slider(
                        1, 120, value=CFG.bayes_changepoint_reconfig_steps, step=1,
                        label="Changepoint steps",
                    )
                    reconfig_lr = gr.Slider(
                        0.01, 1.0, value=CFG.bayes_reconfig_lr, step=0.01,
                        label="Stable mask learning rate",
                    )
                    cp_lr = gr.Slider(
                        0.01, 1.0, value=CFG.bayes_changepoint_reconfig_lr,
                        step=0.01, label="Changepoint mask learning rate",
                    )
                    mask_temperature = gr.Slider(
                        0.1, 10.0, value=CFG.bayes_temperature, step=0.1,
                        label="Stable mask temperature",
                    )
                    cp_temperature = gr.Slider(
                        0.1, 10.0, value=CFG.bayes_changepoint_temperature,
                        step=0.1, label="Changepoint mask temperature",
                    )
                    kl_weight = gr.Slider(
                        0.0, 10.0, value=CFG.bayes_kl_weight, step=0.1,
                        label="Stable KL weight",
                    )
                    prior_decay = gr.Slider(
                        0.0, 1.0, value=CFG.bayes_prior_decay, step=0.05,
                        label="Stable prior retention",
                    )
                    reactive_sd = gr.Slider(
                        0.0, 10.0, value=CFG.reactive_sd, step=0.1,
                        label="Change threshold (EMA standard deviations)",
                    )
                    reactive_ema = gr.Slider(
                        0.0, 0.99, value=CFG.reactive_ema, step=0.01,
                        label="Change detector EMA",
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
                        value=0,
                        step=1,
                        label="Mid-rollout update every N tokens (0 = off)",
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
                stable_steps,
                cp_steps,
                reconfig_lr,
                cp_lr,
                mask_temperature,
                cp_temperature,
                kl_weight,
                prior_decay,
                reactive_sd,
                reactive_ema,
                k_atoms,
                reconfig_every,
                gamma,
                thinking,
                input_routing,
                route_margin,
                route_semantic_weight,
                route_mode,
                route_refine_steps,
                route_refine_kl,
            ],
            [chat, mask_img, route_status],
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
    demo.queue().launch(server_name=os.environ.get('FTN_SERVER_NAME','127.0.0.1'),
        server_port=CFG.port,root_path=os.environ.get('FTN_ROOT_PATH') or None,
        theme=gr.themes.Base(primary_hue='emerald',neutral_hue='slate',font=['system-ui','sans-serif']) if website else None,
        css=Path(__file__).with_name('site.css').read_text() if website else None,
        footer_links=[] if website else None)


if __name__ == "__main__":
    main()
