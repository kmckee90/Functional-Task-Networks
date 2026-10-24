"""Bounded, in-memory MJPEG view of the actual sequential agent trajectory."""
import io
import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw


PAGE = b'''<!doctype html><html><head><meta charset="utf-8">
<title>FTN-M2E live</title><style>
body{background:#10151e;color:#e8edf5;font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}
img{width:100%;image-rendering:pixelated;border:1px solid #45536b;border-radius:8px}
p{color:#b8c5d8}#status{font-variant-numeric:tabular-nums}
</style></head><body><h1>FTN-M2E: live exploration</h1>
<p>Actual running agent: observation, active neural population, and frontier goal.</p>
<img src="/stream" alt="Live observation, FTN mask, and frontier goal">
<p id="status">Waiting for the first observation...</p>
<p>White: active neurons. Purple: recent activity. The goal is a compressed observation
from memory. During learning or checkpointing the last frame stays visible; its age is shown above.</p>
<script>async function poll(){try{let s=await(await fetch('/status',{cache:'no-store'})).json();
document.getElementById('status').textContent=s.step===undefined?'Starting the agent...':
`Step ${s.step.toLocaleString()} | ${s.action} | DQN updates ${s.updates} | Cells ${s.cells} | Latest frame ${s.age_seconds.toFixed(1)}s ago`;
}catch(e){document.getElementById('status').textContent='Disconnected; retrying...'}}
setInterval(poll,1000);poll();</script></body></html>'''


class LiveApplet:
    def __init__(self, port, fps=10, location_path=None):
        # Diagnostic provenance only; never supplied to the exploration policy.
        self.locations = sqlite3.connect(location_path) if location_path else None
        if self.locations:
            self.locations.execute('CREATE TABLE IF NOT EXISTS sightings '
                '(id INTEGER PRIMARY KEY, x REAL, y REAL, z REAL, step INTEGER, life INTEGER)')
        self.location_writes = 0
        self.condition = threading.Condition()
        self.jpeg = b''
        self.status = {}
        self.sequence = 0
        self.last_frame = 0.
        self.period = 1 / fps
        self.closed = False
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == '/':
                    body, content_type = PAGE, 'text/html; charset=utf-8'
                elif self.path == '/status':
                    with app.condition:
                        state = dict(app.status)
                    if state:
                        state['age_seconds'] = max(0., time.time() - state['time'])
                    body, content_type = json.dumps(state).encode(), 'application/json'
                elif self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    sequence = -1
                    try:
                        while not app.closed:
                            with app.condition:
                                app.condition.wait_for(lambda: app.closed or
                                    (app.jpeg and app.sequence != sequence), timeout=5)
                                if app.closed:
                                    break
                                if not app.jpeg or sequence == app.sequence:
                                    continue
                                frame, sequence = app.jpeg, app.sequence
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        pass
                    return
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f'Live applet: http://127.0.0.1:{port}', flush=True)

    def observe(self, row, record, step, behavior):
        # policy() just stored this observation with model_step=cur_step-1.
        # Reset frames are not replay transitions. This ID is NOT the env step.
        if self.locations and not bool(row['is_first']) and record.get('position') is not None:
            self.locations.execute('INSERT OR REPLACE INTO sightings VALUES (?,?,?,?,?,?)',
                (int(behavior.agent.cur_step)-1, *map(float, record['position']),
                 int(step), int(record.get('life', 0))))
            self.location_writes += 1
            if self.location_writes % 100 == 0:
                self.locations.commit()
        if time.monotonic() - self.last_frame < self.period:
            return
        self.last_frame = time.monotonic()
        canvas = Image.new('RGB', (768, 304), '#10151e')
        draw = ImageDraw.Draw(canvas)
        obs = row.get('log/human', row['image'])
        canvas.paste(Image.fromarray(np.uint8(obs)).resize((256, 256)), (0, 24))
        ftn = behavior.ftn
        mask = row['log/m2e_ftn_ema'][ftn.layout].reshape(*ftn.grid, 3)
        canvas.paste(Image.fromarray(mask).resize((256, 256), Image.Resampling.NEAREST), (256, 24))
        goal_step = int(row.get('log/offline_goal_step', -1))
        sighting = (self.locations.execute('SELECT x,y,z,step,life FROM sightings WHERE id=?',
            (goal_step,)).fetchone() if self.locations and goal_step >= 0 else None)
        if goal_step >= 0:
            goal = Image.fromarray(row['log/offline_goal']).convert('RGB')
            canvas.paste(goal.resize((256, 128), Image.Resampling.NEAREST), (512, 88))
        for x, label in ((4, f'Observation | step {step:,}'), (260, 'FTN active mask / EMA'),
                         (516, 'Frontier goal')):
            draw.text((x, 5), label, fill='white')
        draw.text((516, 260), f'Observed at step {goal_step:,}' if goal_step >= 0 else 'Collecting frontier memory', fill='white')
        draw.text((4, 286), str(record.get('action', '')), fill='white')
        output = io.BytesIO()
        canvas.save(output, 'JPEG', quality=88)
        with self.condition:
            self.jpeg = output.getvalue()
            self.status = dict(step=step, time=time.time(), action=record.get('action', ''),
                updates=behavior.updates, cells=record.get('cells_lifetime', 0),
                goal_memory_id=goal_step, goal_position=list(sighting[:3]) if sighting else None,
                goal_observed_step=sighting[3] if sighting else None,
                goal_observed_life=sighting[4] if sighting else None)
            self.sequence += 1
            self.condition.notify_all()

    def close(self):
        if self.locations:
            self.locations.commit()
            self.locations.close()
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
