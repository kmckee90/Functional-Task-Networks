#!/usr/bin/env python
"""Human-playable globo in your browser, over a headless SSH server.

Same world as the agent (seed / spawn / weather / daylight from default.yaml), at
higher resolution. No torch needed.

Three modes:
  * free-roam (default): real-time; the env steps at a fixed rate using your held
    keys, so you can hold/​combine controls. Smooth, but a key tap spans several
    steps and you can press combos the discrete agent can't.
  * --parity: agent-faithful. The env WAITS for each action and steps exactly once
    per action, using the agent's own action wrapper (discrete or factored, from
    default.yaml) with the same mouse magnitude (0.5). Each action has exactly the
    same effect as one of the agent's steps.
  * --agent CKPT: load a training checkpoint and demo the agent acting autonomously
    in real time (no human input). World + agent config come from the checkpoint; the
    agent's network sees its trained (config) resolution while you watch the full
    render resolution. Epsilon is pinned at its final low value (no annealing); the
    per-step flow + recurrence reset mirror train.py exactly.

Usage (remote):
    uv run python play.py            # free-roam
    uv run python play.py --parity   # one step per action, agent's action space
    uv run python play.py --agent artifacts/checkpoints/<name>ckpt_s<step>.pt --device cuda

Laptop:
    ssh -L 8000:localhost:8000 <user>@<server>   ;  open http://localhost:8000

Controls: W/S forward/back, A/D strafe, arrows look, Space jump.
In --parity factored mode: hold the controls for the step, then press Enter to take it.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

from env import make_globo_env

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- shared state ----
_held: set[str] = set()
_latest_jpeg: bytes = b""
_running = True
_lock = threading.Lock()        # guards _held
_step_lock = threading.Lock()   # guards env.step (parity: handler threads)
ENV = None
SEED = 0
JPEG_Q = 85
PARITY = False
MODE = "raw"
AGENT = False        # True when driving a loaded agent checkpoint (no human input)
TRAIN = False        # True for the live actor-learner training runtime (play.py --train)
PAGE_NOTE = ""       # status line shown on the agent/training page


def load_env_cfg(config_path: str) -> dict:
    base_path = os.path.join(HERE, "default.yaml")
    cfg = yaml.safe_load(open(base_path)) or {}
    if config_path and os.path.abspath(config_path) != os.path.abspath(base_path):
        cfg.update(yaml.safe_load(open(config_path)) or {})
    return {str(k).replace("-", "_"): v for k, v in cfg.items()}


def overlay_cfg(base: dict, config_path: str) -> dict:
    """Overlay a --config YAML on top of a base config (e.g. a checkpoint's embedded config).
    Lets --agent / --train demos override world keys (seed, spawn, weather_seed, clear_sky, …)
    without editing the checkpoint. NOTE: the demo reads the config embedded in the .pt, NOT the
    sidecar .yaml — pass overrides via --config. Architectural keys must still match the weights."""
    cfg = dict(base)
    base_path = os.path.join(HERE, "default.yaml")
    if config_path and os.path.abspath(config_path) != os.path.abspath(base_path):
        over = yaml.safe_load(open(config_path)) or {}
        cfg.update({str(k).replace("-", "_"): v for k, v in over.items()})
    return cfg


def encode_jpeg(obs) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(obs)[:, :, :3]).save(buf, format="JPEG", quality=JPEG_Q)
    return buf.getvalue()


def raw_action(held: set[str], look: float) -> dict:
    """Free-roam: held keys -> craftium native Dict action (combos allowed)."""
    keys = {k: 0 for k in ENV.action_space.spaces if k != "mouse"}
    if "w" in held: keys["forward"] = 1
    if "s" in held: keys["backward"] = 1
    if "a" in held: keys["left"] = 1
    if "d" in held: keys["right"] = 1
    if " " in held: keys["jump"] = 1
    mx = (look if "ArrowRight" in held else 0.0) - (look if "ArrowLeft" in held else 0.0)
    my = (look if "ArrowUp" in held else 0.0) - (look if "ArrowDown" in held else 0.0)
    keys["mouse"] = np.array([mx, my], dtype=np.float32)
    return keys


def game_loop(look, fps):
    """Free-roam only: step the env at fixed rate using held keys."""
    global _latest_jpeg
    dt = 1.0 / fps
    while _running:
        t0 = time.time()
        with _lock:
            held = set(_held)
        obs, _r, term, trunc, _i = ENV.step(raw_action(held, look))
        _latest_jpeg = encode_jpeg(obs)
        if term or trunc:
            obs, _ = ENV.reset(seed=SEED)
            _latest_jpeg = encode_jpeg(obs)
        nap = dt - (time.time() - t0)
        if nap > 0:
            time.sleep(nap)


def parity_step(action):
    """Parity: one env step per action, using the agent's wrapper. action is an int
    (discrete) or list[int] (factored MultiDiscrete)."""
    global _latest_jpeg
    a = np.array(action) if isinstance(action, list) else int(action)
    with _step_lock:
        obs, _r, term, trunc, _i = ENV.step(a)
        _latest_jpeg = encode_jpeg(obs)
        if term or trunc:
            obs, _ = ENV.reset(seed=SEED)
            _latest_jpeg = encode_jpeg(obs)


def frame_sink(obs):
    """Stream a rendered frame to the browser (used by the live training runtime)."""
    global _latest_jpeg
    _latest_jpeg = encode_jpeg(obs)


def _downscale_for_agent(obs, out_w: int, out_h: int) -> np.ndarray:
    """The human watches the full render resolution; the agent's network sees its trained
    (config) resolution. Uses the shared env.downsample_obs so the demo, train.py, and the
    live trainer all feed the agent through one identical pipeline (render==obs is a no-op)."""
    from env import downsample_obs
    return downsample_obs(obs, out_w, out_h)


def agent_loop(agent, factored, obs_w, obs_h, online_coef, env_reward_coef, fps):
    """Drive the env with a loaded agent, mirroring train.py's per-step inference flow
    (compute novelty -> encode/advance reservoir -> act -> step -> note_action), with
    epsilon pinned at its final low value (no annealing) and recurrence reset on each
    episode end exactly as in training. Streams the full-res frame to the browser."""
    global _latest_jpeg
    import torch
    from env import obs_to_chw
    dev = agent.device
    dt = 1.0 / max(1, fps)

    def encode_state(obs):
        # mirrors the per-episode reset block in train.py (reset_hidden + first encode)
        agent.reset_hidden()
        img = obs_to_chw(_downscale_for_agent(obs, obs_w, obs_h), device=dev)
        z = agent.encode(img, agent.intrinsic_reward(img))
        agent.update_density(img)
        return z

    with torch.no_grad():
        obs, _ = ENV.reset(seed=SEED)
        _latest_jpeg = encode_jpeg(obs)
        z = encode_state(obs)
        while _running:
            t0 = time.time()
            action = agent.act(z)                       # epsilon fixed at final (no anneal)
            env_action = action if factored else int(action[0])
            next_obs, env_r, term, trunc, _i = ENV.step(env_action)
            _latest_jpeg = encode_jpeg(next_obs)
            nimg = obs_to_chw(_downscale_for_agent(next_obs, obs_w, obs_h), device=dev)
            intr = agent.intrinsic_reward(nimg)
            agent.note_action(action, online_coef * float(intr) + env_reward_coef * float(env_r))
            z = agent.encode(nimg, intr)
            agent.update_density(nimg)
            if term or trunc:
                obs, _ = ENV.reset(seed=SEED)
                _latest_jpeg = encode_jpeg(obs)
                z = encode_state(obs)
            nap = dt - (time.time() - t0)
            if nap > 0:
                time.sleep(nap)


_JS_FREEROAM = """
 const held=new Set();
 const KEYS=new Set(["w","s","a","d"," ","ArrowLeft","ArrowRight","ArrowUp","ArrowDown"]);
 const post=()=>fetch("/input",{method:"POST",body:JSON.stringify([...held])});
 const norm=e=>e.key.length===1?e.key.toLowerCase():e.key;
 addEventListener("keydown",e=>{const k=norm(e); if(KEYS.has(k)){e.preventDefault(); if(!held.has(k)){held.add(k);post();}}});
 addEventListener("keyup",  e=>{const k=norm(e); if(KEYS.has(k)){e.preventDefault(); held.delete(k);post();}});
 addEventListener("blur",   ()=>{held.clear();post();});
 setInterval(post,150);
"""

_JS_PARITY_DISCRETE = """
 const MAP={"w":1,"s":2,"a":3,"d":4," ":5,"ArrowRight":6,"ArrowLeft":7,"ArrowUp":8,"ArrowDown":9,"n":0};
 const norm=e=>e.key.length===1?e.key.toLowerCase():e.key;
 const act=a=>fetch("/action",{method:"POST",body:JSON.stringify({action:a})});
 addEventListener("keydown",e=>{const k=norm(e); if(k in MAP){e.preventDefault(); act(MAP[k]);}});
"""

_JS_PARITY_FACTORED = """
 const held=new Set();
 const KEYS=new Set(["w","s","a","d"," ","ArrowLeft","ArrowRight","ArrowUp","ArrowDown"]);
 const norm=e=>e.key.length===1?e.key.toLowerCase():e.key;
 const act=a=>fetch("/action",{method:"POST",body:JSON.stringify({action:a})});
 function step(){ // [look_h, look_v, jump, move, strafe] matching FACTORED_GROUPS
   const lh=held.has("ArrowLeft")?0:(held.has("ArrowRight")?2:1);
   const lv=held.has("ArrowUp")?0:(held.has("ArrowDown")?2:1);
   const j =held.has(" ")?1:0;
   const mv=held.has("w")?0:(held.has("s")?2:1);
   const st=held.has("a")?0:(held.has("d")?2:1);
   act([lh,lv,j,mv,st]);
 }
 addEventListener("keydown",e=>{const k=norm(e); if(KEYS.has(k)){e.preventDefault(); held.add(k);} if(e.key==="Enter"){e.preventDefault(); step();}});
 addEventListener("keyup",  e=>{const k=norm(e); if(KEYS.has(k)){e.preventDefault(); held.delete(k);}});
 addEventListener("blur",   ()=>held.clear());
"""


def render_page(title_note, controls_note, js):
    return ("""<!doctype html><html><head><meta charset=utf-8><title>globo (human)</title>
<style>
 body{background:#0e1726;color:#e6edf3;font-family:system-ui,sans-serif;margin:0;
      display:flex;flex-direction:column;align-items:center;gap:14px;padding:18px}
 img{image-rendering:pixelated;width:min(90vw,960px);border:1px solid #3a4761;border-radius:8px;background:#000}
 .k{font-family:ui-monospace,monospace;background:#16223a;border:1px solid #3a4761;border-radius:5px;padding:2px 7px}
 #s{font-size:13px;color:#9fb0c8;max-width:90ch;text-align:center} h1{font-size:16px;font-weight:650;margin:0}
</style></head><body tabindex=0 autofocus>
 <h1>globo &mdash; human play <span style='color:#9fb0c8;font-weight:400'>(""" + title_note + """)</span></h1>
 <img id=v src="/stream">
 <div id=s>Click here first to capture the keyboard.<br>
   <span class=k>W/S</span> move &nbsp; <span class=k>A/D</span> strafe &nbsp;
   <span class=k>&larr;&uarr;&darr;&rarr;</span> look &nbsp; <span class=k>Space</span> jump<br>""" + controls_note + """</div>
<script>document.body.focus();""" + js + """</script></body></html>""")


def render_demo_page(note):
    """Minimal page for the agent demo: just the live stream + a status line (no controls)."""
    return ("""<!doctype html><html><head><meta charset=utf-8><title>globo (agent)</title>
<style>
 body{background:#0e1726;color:#e6edf3;font-family:system-ui,sans-serif;margin:0;
      display:flex;flex-direction:column;align-items:center;gap:14px;padding:18px}
 img{image-rendering:pixelated;width:min(90vw,960px);border:1px solid #3a4761;border-radius:8px;background:#000}
 #s{font-size:13px;color:#9fb0c8;max-width:90ch;text-align:center} h1{font-size:16px;font-weight:650;margin:0}
</style></head><body>
 <h1>globo &mdash; agent demo <span style='color:#9fb0c8;font-weight:400'>(real-time)</span></h1>
 <img id=v src="/stream">
 <div id=s>""" + note + """</div></body></html>""")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _page(self):
        if AGENT or TRAIN:
            return render_demo_page(PAGE_NOTE)
        if PARITY and MODE == "discrete":
            return render_page("parity: one step per keypress",
                               "Each keypress = one agent step (discrete).", _JS_PARITY_DISCRETE)
        if PARITY and MODE == "factored":
            return render_page("parity: hold combo, press Enter to step",
                               "Hold your controls, then <span class=k>Enter</span> takes one step (factored).",
                               _JS_PARITY_FACTORED)
        return render_page("free-roam (real-time)", "Hold keys to move continuously.", _JS_FREEROAM)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = self._page().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while _running:
                    f = _latest_jpeg
                    if f:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(f)).encode() + b"\r\n\r\n" + f + b"\r\n")
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        if self.path == "/input" and not PARITY:
            try:
                keys = set(json.loads(raw or b"[]"))
            except Exception:
                keys = set()
            with _lock:
                _held.clear()
                _held.update(keys)
            self.send_response(204); self.end_headers()
        elif self.path == "/action" and PARITY:
            try:
                parity_step(json.loads(raw)["action"])
            except Exception:
                pass
            self.send_response(204); self.end_headers()
        else:
            self.send_error(404)


def main():
    global ENV, SEED, JPEG_Q, PARITY, MODE, AGENT, _running
    p = argparse.ArgumentParser(description="Play globo in a browser over SSH (same world as the agent).")
    p.add_argument("--config", default=os.path.join(HERE, "default.yaml"),
                   help="config YAML. For --agent/--train it OVERLAYS the checkpoint's embedded "
                        "config, so use it to override world keys for a demo (seed, spawn, "
                        "weather_seed, clear_sky). The sidecar .yaml next to the .pt is NOT read.")
    p.add_argument("--parity", action="store_true",
                   help="agent-faithful: env waits for each action, steps once, using the agent's "
                        "action wrapper (discrete/factored from default.yaml) with mouse_mov=0.5")
    p.add_argument("--agent", default=None, metavar="CKPT",
                   help="path to a training checkpoint (.pt); demos the loaded agent acting "
                        "autonomously in real time (config/world taken from the checkpoint). "
                        "With --train, warm-starts the live run from this checkpoint instead.")
    p.add_argument("--train", action="store_true",
                   help="LIVE actor-learner training: the full train.py machinery (RECODE "
                        "centroid spawning, prioritized replay, eps anneal) runs while you watch "
                        "the FPV. Learner trains async on --learn-device; weights push to the actor.")
    p.add_argument("--infer-device", default="cuda:0", help="(--train) actor / inference device")
    p.add_argument("--learn-device", default="cuda:1", help="(--train) learner / optimization device")
    p.add_argument("--name", default="", help="(--train) output filename prefix for centroids/checkpoints")
    p.add_argument("--device", default=None, help="torch device for --agent (default: checkpoint's device)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--width", type=int, default=480, help="render width (keep 2:1 to match the agent's view)")
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=int, default=20, help="free-roam / agent-demo step rate (ignored in --parity)")
    p.add_argument("--look", type=float, default=0.6, help="free-roam look speed (parity forces agent's 0.5)")
    p.add_argument("--jpeg-quality", type=int, default=85)
    a = p.parse_args()
    global TRAIN, PAGE_NOTE

    # ---- LIVE actor-learner training (runs its own env + threads; serves the same stream) ----
    if a.train:
        import torch
        import live
        ckpt = None
        if a.agent:
            ckpt = torch.load(a.agent, map_location="cpu", weights_only=False)
            cfg = overlay_cfg(ckpt["config"], a.config)   # --config overlays the checkpoint config
        else:
            cfg = load_env_cfg(a.config)
        TRAIN = True
        SEED = int(cfg.get("seed", 0)); JPEG_Q = a.jpeg_quality
        PAGE_NOTE = (f"LIVE training (actor @ {a.infer_device}, learner @ {a.learn_device}). "
                     "Watch behaviour, RECODE centroids, and the policy co-evolve in real time.")
        args = SimpleNamespace(**cfg)
        args.name = f"{a.name}_" if a.name else ""
        args.config = a.config
        print(f"[play] mode=LIVE-TRAIN  seed={SEED} actor={a.infer_device} learner={a.learn_device} "
              f"warm_start={a.agent or 'none'}")
        print(f"[play] watching the agent's {cfg.get('obs_width')}x{cfg.get('obs_height')} FPV "
              "(upscaled in the browser)")
        threading.Thread(target=live.run_live, kwargs=dict(
            args=args, infer_device=a.infer_device, learn_device=a.learn_device,
            frame_sink=frame_sink, should_run=lambda: _running, ckpt=ckpt),
            daemon=True, name="live-actor").start()
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
        print(f"\n  On your laptop:  ssh -L {a.port}:localhost:{a.port} <user>@<server>")
        print(f"  then open:       http://localhost:{a.port}\n  (Ctrl-C to stop)\n")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            _running = False
            time.sleep(0.5)   # let the actor loop close its env
        return

    ckpt = None
    if a.agent:
        import torch
        # load on CPU to read the embedded config; module weights copy onto the agent's
        # device at build time, and density tensors are moved in load_state_dict.
        ckpt = torch.load(a.agent, map_location="cpu", weights_only=False)
        cfg = overlay_cfg(ckpt["config"], a.config)   # --config overlays the checkpoint's embedded config
        AGENT = True
        PARITY = False
        MODE = str(cfg.get("action_mode", "discrete"))   # use the agent's own action wrapper
        PAGE_NOTE = ("Autonomous agent (greedy at final epsilon) acting in the same world it "
                     "trained on. No controls.")
    else:
        cfg = load_env_cfg(a.config)
        PARITY = a.parity
        MODE = str(cfg.get("action_mode", "discrete")) if PARITY else "raw"
    SEED = int(cfg.get("seed", 0))
    JPEG_Q = a.jpeg_quality

    mode_str = f"AGENT/{MODE}" if AGENT else ("PARITY/" + MODE if PARITY else "free-roam")
    print(f"[play] mode={mode_str}  world: seed={SEED} "
          f"spawn={cfg.get('spawn')} weather_seed={cfg.get('weather_seed')} "
          f"freeze_daylight={cfg.get('freeze_daylight')} invincible={cfg.get('invincible')}")
    print(f"[play] render {a.width}x{a.height}"
          + (f"  agent sees {cfg.get('obs_width')}x{cfg.get('obs_height')}" if AGENT else ""))

    ENV = make_globo_env(
        obs_width=a.width, obs_height=a.height, max_timesteps=None,
        init_frames=int(cfg.get("init_frames", 200)), seed=SEED,
        action_mode=MODE,                         # raw (free-roam) or the agent's wrapper (parity)
        invincible=bool(cfg.get("invincible", False)),
        freeze_daylight=bool(cfg.get("freeze_daylight", False)),
        clear_sky=bool(cfg.get("clear_sky", False)),
        spawn=cfg.get("spawn"), weather_seed=cfg.get("weather_seed"),
        frameskip=int(cfg.get("frameskip", 1)),
        move_speed=float(cfg.get("move_speed", 8.0)), move_accel=float(cfg.get("move_accel", 500.0)),
        climb_speed=float(cfg.get("climb_speed", 3.0)), jump_speed=float(cfg.get("jump_speed", 6.5)),
        gravity=float(cfg.get("gravity", 9.81)), mouse_mov=float(cfg.get("mouse_mov", 0.5)),
        env_dir=os.path.join(HERE, "craftium_play_env"),
        run_dir_prefix=os.path.join(HERE, "runs", "play"),
    )

    global _latest_jpeg
    obs, _ = ENV.reset(seed=SEED)
    _latest_jpeg = encode_jpeg(obs)

    if AGENT:
        from train import build_agent
        dev = a.device or str(cfg.get("device", "cpu"))
        aargs = SimpleNamespace(**cfg); aargs.device = dev
        agent = build_agent(aargs, ENV)
        agent.load_state_dict(ckpt["agent"])
        agent.epsilon = float(cfg.get("epsilon_end", 0.0))   # final low epsilon, no annealing
        print(f"[play] loaded agent <- {os.path.relpath(a.agent, HERE)} on {dev} "
              f"(step={ckpt.get('global_step')}, eps={agent.epsilon}, atoms={agent.density.size})")
        threading.Thread(target=agent_loop, args=(
            agent, MODE == "factored",
            int(cfg.get("obs_width", 64)), int(cfg.get("obs_height", 32)),
            float(cfg.get("online_coef", 1.0)), float(cfg.get("env_reward_coef", 0.0)),
            a.fps), daemon=True).start()
    elif not PARITY:
        # normalize free-roam turn speed to the 64px reference so high-res play
        # doesn't turn many times faster per step than the agent
        look_eff = a.look * (64.0 / a.width)
        threading.Thread(target=game_loop, args=(look_eff, a.fps), daemon=True).start()

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"\n  On your laptop:  ssh -L {a.port}:localhost:{a.port} <user>@<server>")
    print(f"  then open:       http://localhost:{a.port}\n  (Ctrl-C to stop)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _running = False
        time.sleep(0.2)
        ENV.close()


if __name__ == "__main__":
    main()
