"""Read-only monitor, serving bounded live artifacts from the runtime directory."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(args.root), **kw)

        def do_GET(self):
            if self.path.split('?')[0] == '/':
                data = Path(__file__).with_name('monitor.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            # Only lightweight public monitor files, never checkpoints/replay.
            name = self.path.split('?')[0].split('/')[-1]
            if name not in ('status.json', 'history.json', 'config.json', 'video.json',
                            'obs.mp4', 'input.mp4', 'latest.jpg'):
                self.send_error(404)
                return
            super().do_GET()

        def end_headers(self):
            self.send_header('Cache-Control', 'no-cache')
            super().end_headers()
    ThreadingHTTPServer(('0.0.0.0', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
