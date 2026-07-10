"""Human/agent play for the GPU globo explorer — browser MJPEG over a forwarded port.

GPU analog of ../globo_explorer/play.py. Drives our warpcraft GloboGPUEnv
(num_worlds=1) and streams the rendered frames as MJPEG to a browser, so you can
test the warpcraft version interactively.

  # on the pod (forward the port with:  actl pod exec --port 8000  or  pod up --port 8000)
  uv run python play.py                 # free-roam: WASD move, arrows look, space jump
  uv run python play.py --agent artifacts/run_ckpt.pkl   # watch a trained checkpoint play

Then open http://localhost:8000 on your laptop.

Controls (free-roam) map browser keys onto warpcraft's action channels:
  LOCOMOTION: w=forward s=back a=strafe-left d=strafe-right space=jump (w+space=jump-forward)
  LOOK:       ArrowUp/Down = pitch, ArrowLeft/Right = yaw
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# LocomotionAction / LookAction codes (from warpcraft.engine.contracts, verified).
LOCO = {"noop": 0, "forward": 1, "back": 2, "left": 3, "right": 4, "jump": 5,
        "jump_forward": 6, "jump_back": 7}
LOOK = {"noop": 0, "up": 1, "down": 2, "left": 3, "right": 4}

_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>globo (warpcraft)</title>
<style>body{background:#111;color:#ddd;font-family:monospace;text-align:center}
img{image-rendering:pixelated;width:70vmin;height:70vmin;border:1px solid #333}</style></head>
<body tabindex=0 autofocus><h3>globo — warpcraft GPU</h3>
<img id=v src="/stream"><p id=s>WASD move · arrows look · space jump</p>
<script>
const held={};
function send(){fetch('/input',{method:'POST',body:JSON.stringify(Object.keys(held).filter(k=>held[k]))})}
addEventListener('keydown',e=>{held[e.key.toLowerCase()]=1;send();e.preventDefault()});
addEventListener('keyup',e=>{delete held[e.key.toLowerCase()];send();e.preventDefault()});
</script></body></html>"""


def keys_to_action(held: set) -> jnp.ndarray:
    """Map held browser keys to [locomotion, look] head indices (active_channels=(0,1))."""
    jump = " " in held or "spacebar" in held
    if "w" in held:
        loco = LOCO["jump_forward"] if jump else LOCO["forward"]
    elif "s" in held:
        loco = LOCO["back"]
    elif "a" in held:
        loco = LOCO["left"]
    elif "d" in held:
        loco = LOCO["right"]
    elif jump:
        loco = LOCO["jump"]
    else:
        loco = LOCO["noop"]
    if "arrowup" in held:
        look = LOOK["up"]
    elif "arrowdown" in held:
        look = LOOK["down"]
    elif "arrowleft" in held:
        look = LOOK["left"]
    elif "arrowright" in held:
        look = LOOK["right"]
    else:
        look = LOOK["noop"]
    return jnp.array([loco, look], jnp.int32)


class Viewer:
    def __init__(self, env, jpeg_quality=80):
        self.env = env
        self.jpeg_quality = jpeg_quality
        self.latest = self._encode(env.reset())
        self.held: set = set()
        self.lock = threading.Lock()

    def _encode(self, obs) -> bytes:
        arr = (np.asarray(obs["render_rgb"][0]) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()

    def set_frame(self, obs):
        with self.lock:
            self.latest = self._encode(obs)

    def get_frame(self) -> bytes:
        with self.lock:
            return self.latest


def make_handler(viewer: Viewer):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_PAGE.encode())
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = viewer.get_frame()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                        time.sleep(1 / 30)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode() if n else "[]"
            if self.path == "/input":
                viewer.held = set(json.loads(body))
            self.send_response(204)
            self.end_headers()
    return H


def run_free_roam(env, viewer, fps):
    dt = 1.0 / fps
    while True:
        t = time.time()
        action = keys_to_action(viewer.held)
        obs, _r, _term, _to = env.step(action)
        viewer.set_frame(obs)
        time.sleep(max(0, dt - (time.time() - t)))


def run_agent(env, viewer, ckpt_path, fps):
    from agent import GloboExplorerBranchingDQN
    with open(ckpt_path, "rb") as f:
        blob = pickle.load(f)
    cfg = blob["config"]
    key = jax.random.PRNGKey(cfg["seed"])
    agent = GloboExplorerBranchingDQN(cfg, env.head_sizes, key)
    agent.params = jax.tree_util.tree_map(jnp.asarray, blob["params"])
    agent.epsilon = cfg["epsilon_end"]     # exploit
    from recode import RecodeState
    agent.density.set_state({k: jnp.asarray(v) for k, v in blob["density"].items()})

    dt = 1.0 / fps
    hidden = agent.init_hidden()
    obs = env.reset()
    zero = jnp.float32(0.0)
    z, action, emb, nov = agent.perceive(hidden, obs["image"], jnp.zeros((agent.n_heads,), jnp.int32),
                                         zero, zero, key)
    while True:
        t = time.time()
        obs2, env_r, terminal, timeout = env.step(action)
        viewer.set_frame(obs2)
        boundary = jnp.maximum(terminal, timeout)
        r_fb = jnp.float32(cfg["online_coef"]) * nov + jnp.float32(cfg["env_reward_coef"]) * env_r
        hidden_carry = jnp.where(boundary > 0.5, jnp.zeros_like(z), z)
        key, pk = jax.random.split(key)
        z, action, emb, nov = agent.perceive(hidden_carry, obs2["image"], action, r_fb, nov, pk)
        time.sleep(max(0, dt - (time.time() - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "default.yaml"))
    ap.add_argument("--agent", default=None, help="checkpoint .pkl to watch")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--render", type=int, default=256, help="human render resolution")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    import train as T
    cfg = T.load_config(args.config)
    if args.agent:  # use the checkpoint's config for env parity
        with open(args.agent, "rb") as f:
            cfg = pickle.load(f)["config"]

    from env import GloboGPUEnv
    env = GloboGPUEnv(
        seed=cfg["seed"], obs_width=cfg["obs_width"], obs_height=cfg["obs_height"],
        render_width=args.render, render_height=args.render,
        recode_down_h=cfg["recode_down_h"], recode_down_w=cfg["recode_down_w"],
        recode_grayscale=not cfg["recode_rgb"], active_channels=tuple(cfg["active_channels"]),
        max_episode_steps=cfg["max_episode_steps"], advance_weather=cfg["advance_weather"])

    viewer = Viewer(env)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(viewer))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"\n  Serving on http://localhost:{args.port}  (forward the pod port first)\n"
          f"  mode: {'agent ' + args.agent if args.agent else 'free-roam'}\n", flush=True)
    if args.agent:
        run_agent(env, viewer, args.agent, args.fps)
    else:
        run_free_roam(env, viewer, args.fps)


if __name__ == "__main__":
    main()
