"""Two checkpoint continuations under the corrected native input protocol."""
import functools
import html
import json
import os
from pathlib import Path
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .phase2 import read, atomic
from .achievement_charts import achievement_checklist, run_charts
from .monitor_settings import read_settings, CLIP_CHOICES

STATE = Path(os.environ["GLOBO_STATE"])
OUT = STATE / "monitor"
SETTINGS = STATE / "clip_settings.json"


def update():
    from .consolidate_clip import process
    from .survival import summary, panel
    trials = read(STATE / "continuous_runs/state.json", [])
    OUT.mkdir(parents=True, exist_ok=True)
    parts = ['''<!doctype html><meta charset="utf-8"><title>Globo — corrected input continuations</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{background:#10151d;color:#e5eaf0;font:16px system-ui;max-width:1350px;margin:auto;padding:24px}
section{margin:24px 0;padding:24px;background:#19212c;border-radius:12px}
a{color:#9bd0ff}video{width:100%;max-width:1100px}svg{max-width:100%}
.run-charts{display:grid;grid-template-columns:1fr 1fr;gap:18px}
pre{white-space:pre-wrap}summary{cursor:pointer}button,select{padding:8px}
</style><h1>FTN-M2E — corrected input continuations</h1>
<p>Two independent checkpoint forks from the four-agent reduced-action experiment.
World 19; 14 actions; one GPU each. Weights, FTN, replay, frontier and density
memory are retained. Each continuation begins a new life at the original spawn.
Native eight-frame actions now deliver dig/place press events and hold place/sneak.
The original four runs are untouched.</p>
<p>After the latest checkpoint restart, animal health is capped at 20.
Every animal kill triggers a clip with observations before and after the event,
including the goal and FTN state. Clips appear under “Captured behavior events”.</p>
<p>The reduced action set still excludes inventory selection. These continuations
test the input correction without changing the learned action space.</p>
<label>Video length <select id="frames">''' +
             "".join(f'<option value="{n}">{n/6:.1f} seconds</option>' for n in CLIP_CHOICES) +
             '''</select></label> <button id="apply">Apply</button> <span id="message"></span>''']
    parts.append(f'<p>Updated {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}</p>')
    for trial in trials:
        name = trial["name"]
        root = STATE / "runs" / name
        live = root / "live"
        status = read(root / "status.json")
        provenance = read(root / "fork_provenance.json")
        env = status.get("environment") or {}
        try:
            if (live / "video.json").exists():
                process(live)
        except Exception:
            traceback.print_exc()
        video = read(live / "video.json")
        link = OUT / name
        if not link.exists():
            link.symlink_to(root, target_is_directory=True)
        step = status.get("step", provenance["step"])
        parts.append(f'<section><h2>Agent seed {trial["agent_seed"]} · GPU {trial["gpu"]}</h2>'
                     f'<p>Checkpoint: {provenance["source_run"]} at {provenance["step"]:,} interactions. '
                     f'Current: {step:,} (+{step-provenance["step"]:,} since fork). '
                     f'Status age: {max(0,time.time()-status.get("time",time.time())):.0f}s.</p>')
        counts = env.get("achievement_counts") or {}
        parts.append(f'<p><b>Blocks placed: {counts.get("place_block",0):,}</b> · '
                     f'Blocks dug: {counts.get("dig_block",0):,} (retained lifetime totals).</p>')
        if (live / "obs.mp4").exists():
            parts.append(f'<video controls playsinline preload="metadata" src="{name}/live/obs.mp4?v={video.get("step",0)}"></video>'
                         f'<p>Current observations, frontier goal and FTN activity; video through step {video.get("step",0):,}.</p>')
        else:
            parts.append("<p>Loading checkpoint / collecting the first video.</p>")
        parts.append(achievement_checklist(env))
        try:
            parts.append(panel(summary(root)))
            rows = [json.loads(line) for line in (live / "history.jsonl").read_text().splitlines()]
            parts.append(run_charts([r for r in rows if r["step"] >= provenance["step"]]))
        except (OSError, ValueError, KeyError):
            pass
        parts.append(f'<p><a href="{name}/live/history.jsonl">Progress history</a> · '
                     f'<a href="{name}/live/events.jsonl">Stagnation log</a> · '
                     f'<a href="{name}/environment/input_protocol.json">Input protocol</a></p>'
                     '<details><summary>Captured behavior events</summary>')
        events = []
        for path in (root / "event_clips").glob("*.json"):
            event = read(path)
            if event.get("file"):
                events.append(event)
        for event in sorted(events, key=lambda e:e.get("event_step",0), reverse=True)[:30]:
            labels = ", ".join(e["kind"].replace("_", " ") for e in event.get("events", []))
            parts.append(f'<details><summary>Step {event["event_step"]:,}: {html.escape(labels)}</summary>'
                         f'<video controls preload="none" src="{name}/event_clips/{html.escape(event["file"],quote=True)}"></video></details>')
        parts.append('</details><details><summary>Status</summary><pre>' +
                     html.escape(json.dumps(status, indent=2)) + '</pre></details></section>')
    parts.append('''<script>
fetch('/api/clip-settings').then(r=>r.json()).then(s=>document.querySelector('#frames').value=s.frames);
document.querySelector('#apply').onclick=async()=>{
let r=await fetch('/api/clip-settings',{method:'POST',body:JSON.stringify({frames:Number(document.querySelector('#frames').value)})});
document.querySelector('#message').textContent=r.ok?'Applied to future clips':'Invalid setting';};
setInterval(()=>{if(![...document.querySelectorAll('video')].some(v=>!v.paused&&!v.ended))location.reload()},30000);
</script>''')
    path = OUT / "index.tmp"
    path.write_text("".join(parts))
    path.replace(OUT / "index.html")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8015)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/clip-settings":
                body = json.dumps(read_settings(SETTINGS)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()
        def do_POST(self):
            if self.path != "/api/clip-settings":
                self.send_error(404); return
            try:
                size = int(self.headers.get("Content-Length",0))
                if not 0 < size <= 1024: raise ValueError()
                frames = json.loads(self.rfile.read(size))["frames"]
                if type(frames) is not int or frames not in CLIP_CHOICES: raise ValueError()
                atomic(SETTINGS, dict(frames=frames))
            except (ValueError, KeyError, TypeError):
                self.send_error(400); return
            self.send_response(204); self.end_headers()
    server = ThreadingHTTPServer(("0.0.0.0", args.port),
                                functools.partial(Handler, directory=str(OUT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    while True:
        try:
            update()
        except Exception:
            traceback.print_exc()
        time.sleep(20)


if __name__ == "__main__":
    main()
