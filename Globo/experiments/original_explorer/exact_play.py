"""Manual driver of the actual Globo-Dreamer Env. Run only on the pod."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image


PAGE = """<!doctype html><meta charset="utf-8"><title>Globo exact parity</title>
<style>
body{background:#0e1726;color:#e6edf3;font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px}
h1{font-size:22px}.views{display:flex;gap:20px;align-items:start;flex-wrap:wrap}
img{image-rendering:pixelated;width:min(70vw,512px)}#input{width:256px}
button,select{font:inherit;padding:8px;margin:6px 4px 6px 0;background:#233249;color:white;border:1px solid #64748b}
pre{white-space:pre-wrap}#error{color:#ff9090}
</style>
<h1>Globo-Dreamer: exact action parity</h1>
<p>Select an action, then press <b>Enter</b> or click Step. One request = one agent
interaction (8 engine frames). No automatic steps or mouse-look.
Action IDs and 64×64 input match the agent environment. This human test uses
single-click placement (no hold-to-repeat) and starts with four wet-silt blocks
in slot 1. Running agents do not yet use this placement change.</p>
<p><b>Manual action stepping, not a world pause:</b> the agent's asynchronous
server simulation continues while you wait. Images and telemetry below are
snapshots from the last step; the telemetry can lag a newly selected slot.</p>
<div class="views"><div>Render<br><img id="view" src="/frame.jpg"></div>
<div>Agent input (64×64)<br><img id="input" src="/input.jpg"></div></div>
<select id="action"></select><button id="step">Step — Enter</button>
<button id="reset">Reset world</button>
<p>Shortcuts select only: WASD movement; arrows look; Space jump-forward;
Q/R forward-left/right; F dig/attack; E place/interact; C sneak; V sneak-interact;
1–9 inventory; X idle. The selected action remains selected.</p>
<p id="error"></p><pre id="status"></pre><pre id="controls"></pre>
<script>
const action=document.querySelector('#action'),step=document.querySelector('#step');
let busy=false,stamp=0;
const shortcuts={w:1,s:2,a:3,d:4,' ':5,ArrowLeft:6,ArrowRight:7,ArrowUp:8,ArrowDown:9,
q:10,r:11,f:12,e:13,c:14,v:15,x:0};
for(let i=1;i<=9;i++)shortcuts[String(i)]=15+i;
function show(s){
 document.querySelector('#status').textContent=`Interactions: ${s.steps}; life: ${s.life}; last action: ${s.last}
HP: ${s.hp}; selected inventory slot: ${s.wield}
Slot quantities: ${JSON.stringify(s.slots)}; inventory: ${JSON.stringify(s.items)}
${s.done?'Episode ended. Use Reset world to continue.':''}`;
 document.querySelector('#view').src='/frame.jpg?'+(++stamp);
 document.querySelector('#input').src='/input.jpg?'+stamp;
}
async function send(reset=false){
 if(busy)return;
 busy=true;step.disabled=true;document.querySelector('#reset').disabled=true;
 try{
 const res=await fetch('/step',{method:'POST',body:JSON.stringify({action:Number(action.value),reset})});
 const s=await res.json();if(!res.ok)throw Error(s.error);show(s);
 document.querySelector('#error').textContent='';
 }catch(e){document.querySelector('#error').textContent=String(e);}
 finally{busy=false;step.disabled=false;document.querySelector('#reset').disabled=false;}
}
step.onclick=()=>send();document.querySelector('#reset').onclick=()=>send(true);
addEventListener('keydown',e=>{
 if(e.key==='Enter'){e.preventDefault();if(!e.repeat)send();return;}
 const key=e.key.length===1?e.key.toLowerCase():e.key, id=shortcuts[key];
 if(id!==undefined){e.preventDefault();if(id<action.options.length)action.value=String(id);}
});
fetch('/state').then(r=>r.json()).then(s=>{
 s.actions.forEach((a,i)=>action.add(new Option(`${i}: ${a[0]}  ${JSON.stringify(a[1])}`,i)));
 document.querySelector('#controls').textContent=`World seed ${s.seed}; spawn ${s.spawn}; render ${s.render_size}×${s.render_size}.
--fps, --look, --width, --height and --config do not change exact-parity settings.`;
 show(s);
});
</script>"""


def run(args):
    state = Path(os.environ.get("GLOBO_STATE", "/home/dev/globo_dreamer_state"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(state / "vendor/dreamerv3"))
    sys.path.insert(0, str(state / "packages"))
    from globo_dreamer.env import Globo

    root = state / "human_exact"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="play-", dir=root) as temp:
        # Never share a training engine directory or generate event media.
        previous = {k: os.environ.get(k) for k in ("GLOBO_ENGINE_ROOT", "GLOBO_EVENT_CAPTURE")}
        os.environ["GLOBO_ENGINE_ROOT"] = str(Path(temp) / "engine")
        os.environ["GLOBO_EVENT_CAPTURE"] = "0"
        env = Globo(Path(temp) / "environment", seed=args.world_seed,
                    episode_steps=0, video_every=0, memory_size=32,
                    render_size=args.render_size, spawn=args.spawn,
                    reduced_actions=args.reduced_actions, bounded_artifacts=True,
                    human_place_test=True)
        lock = threading.Lock()
        frames = {}
        steps = 0
        last = "initial reset"

        def capture(obs):
            for name, image in (("/frame.jpg", env.human_frame if env.human_frame is not None else obs["image"]),
                                ("/input.jpg", obs["image"])):
                buf = io.BytesIO()
                Image.fromarray(image).save(buf, format="JPEG", quality=args.jpeg_quality)
                frames[name] = buf.getvalue()

        def status():
            rec = env.record or {}
            return dict(steps=steps, life=env.life, last=last, done=env.done,
                        hp=rec.get("hp"), wield=rec.get("wield"), slots=rec.get("slots"),
                        items=rec.get("items"), actions=env.actions, seed=args.world_seed,
                        spawn=args.spawn, render_size=args.render_size)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value, code=200, mime="application/json"):
                body = value if isinstance(value, bytes) else json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self.reply(PAGE.encode(), mime="text/html; charset=utf-8")
                elif path == "/state":
                    with lock:
                        self.reply(status())
                elif path in frames:
                    self.reply(frames[path], mime="image/jpeg")
                else:
                    self.reply({"error": "Not found"}, 404)

            def do_POST(self):
                nonlocal steps, last
                if self.path != "/step":
                    self.reply({"error": "Not found"}, 404)
                    return
                try:
                    data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    aid, reset = int(data["action"]), bool(data.get("reset", False))
                    if not 0 <= aid < len(env.actions):
                        raise ValueError("Action outside this agent's action space")
                    with lock:
                        if env.done and not reset:
                            raise ValueError("Episode ended; use Reset world")
                        capture(env.step({"action": aid, "reset": reset}))
                        steps += int(not reset)
                        last = "reset" if reset else f"{aid}: {env.actions[aid][0]}"
                        self.reply(status())
                except (ValueError, KeyError, TypeError) as e:
                    self.reply({"error": str(e)}, 400)
                except Exception as e:
                    self.reply({"error": str(e)}, 500)

        server = None
        try:
            capture(env.step({"action": 0, "reset": True}))
            server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
            print(f"[exact-parity] http://localhost:{args.port} — seed {args.world_seed}, "
                  f"{len(env.actions)} actions, 8 engine frames per action; Enter to step.", flush=True)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if server:
                server.server_close()
            env.close()
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
