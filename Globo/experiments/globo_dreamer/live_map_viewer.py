"""Read-only viewer sidecar: existing live stream plus cached terrain/visitation.

No training restart, environment stepping, or changes to agent inputs/rewards.
"""
import argparse
import io
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import terrain_maps


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class MapCache:
    def __init__(self, run, terrain, seed):
        self.run, self.seed = run, seed
        terrain_maps.SHARED = terrain.parent.parent
        self.lock = threading.Lock()
        self.png, self.meta = b'', {}
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def work(self):
        previous = None
        while not self.stop.is_set():
            try:
                visits = read(self.run / 'live/visitation.json')
                latest = read(self.run / 'environment/latest.json')
                bounds = terrain_maps.bounds([visits, latest], previous)
                if bounds:
                    # A square viewport keeps X and Z at the same scale.
                    x0, x1, z0, z1 = bounds
                    side = max(x1-x0, z1-z0)
                    bounds = [x0, x0+side, z0, z0+side]
                    t = time.perf_counter()
                    terrain, tiles, ready, _ = terrain_maps.raster(self.seed, bounds)
                    output = io.BytesIO()
                    terrain.save(output, 'PNG')
                    with self.lock:
                        self.png = output.getvalue()
                        self.meta = dict(bounds=bounds, terrain_ready=ready,
                            terrain_requested=len(tiles), generated=time.time(),
                            render_ms=(time.perf_counter()-t)*1000)
                        self.error = None
                    previous = bounds
            except Exception as exc:
                self.error = str(exc)
            self.stop.wait(10)

    def state(self):
        visits = read(self.run / 'live/visitation.json')
        latest = read(self.run / 'environment/latest.json')
        with self.lock:
            meta = dict(self.meta)
        # Existing engine diagnostics are saved every 50 observations. Reading
        # them never causes a simulation step or a write to the training world.
        current = latest if latest.get('step', -1) >= visits.get('last_step', -1) else visits
        return dict(**meta, counts=visits.get('counts', {}), recent=visits.get('recent', []),
            spawn=visits.get('spawn'), position=current.get('position'),
            yaw=current.get('look_yaw', current.get('yaw', 0)), life=current.get('life'),
            step=current.get('step', current.get('last_step', 0)),
            coverage_step=visits.get('last_step', 0), error=self.error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--upstream', default='http://127.0.0.1:8000')
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--terrain', type=Path,
        default=Path('/mnt/obelisk-shared/kmckee90/globo_phase2/terrain/19'))
    args = parser.parse_args()
    cache = MapCache(args.run, args.terrain, args.seed)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                body = Path(__file__).with_suffix('.html').read_bytes()
                kind = 'text/html; charset=utf-8'
            elif path == '/map':
                body, kind = json.dumps(cache.state()).encode(), 'application/json'
            elif path == '/terrain.png':
                with cache.lock:
                    body = cache.png
                if not body:
                    self.send_error(503, 'Terrain initializing')
                    return
                kind = 'image/png'
            elif path in ('/stream', '/status'):
                try:
                    with urllib.request.urlopen(args.upstream + path, timeout=30) as upstream:
                        self.send_response(200)
                        self.send_header('Content-Type', upstream.headers['Content-Type'])
                        self.send_header('Cache-Control', 'no-store')
                        self.end_headers()
                        while True:
                            block = upstream.read1(65536)
                            if not block:
                                break
                            self.wfile.write(block)
                            self.wfile.flush()
                except (OSError, TimeoutError):
                    self.close_connection = True
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Live map viewer on {args.port}; run={args.run}', flush=True)
    try:
        server.serve_forever()
    finally:
        cache.stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
