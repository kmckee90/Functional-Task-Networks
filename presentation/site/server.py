#!/usr/bin/env python3
"""Local presentation server. No dependencies; repository artifacts stay in place."""
import argparse
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

SITE = Path(__file__).resolve().parent
REPO = SITE.parent.parent
ROOTS = {
    '/assets/preview/': Path(os.environ.get('PROPOSAL_MEDIA', '/home/dev/proposal_state/media')),
    '/assets/site/': SITE / 'assets',
    '/assets/proposal/': REPO / 'presentation/proposal assets',
    '/assets/sit/': REPO / 'SiT/continual/memory_budget/artifacts',
    '/assets/robotics/': REPO / 'ftn_robotics',
}
SOURCES = {'/source/theory': REPO / 'presentation/proposal.md',
           '/source/ftn': REPO / 'base/README.md',
           '/source/bayes': REPO / 'base/projects/bayesian_adv/FINDINGS.md',
           '/source/explorer': REPO / 'explorer/agents/agent_recode.py',
           '/source/robotics': REPO / 'ftn_robotics/train_eval_mt10_plan.py',
           '/source/chat': REPO / 'nlp/ftn_chat/README.md',
           '/source/globo': REPO / 'globo_dreamer/README.md'}
REPORT_STYLE = '<link rel="stylesheet" href="/report.css"><script defer src="/report.js"></script>'

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE), **kwargs)

    def reply(self, data, mime='application/json', status=200):
        if isinstance(data, str): data = data.encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        if self.command != 'HEAD': self.wfile.write(data)

    def proxy(self, upstream, prefix, styled=False):
        suffix = self.path[len(prefix):]
        headers = {'X-Forwarded-Host':self.headers.get('Host','localhost:8080'),
                   'X-Forwarded-Proto':'http'}
        if self.headers.get('Range'): headers['Range'] = self.headers['Range']
        body = None
        if self.command == 'POST':
            size = int(self.headers.get('Content-Length', 0))
            if size > 1_000_000: return self.reply('Request too large', 'text/plain', 413)
            body = self.rfile.read(size)
            headers['Content-Type'] = 'application/json'
        try:
            request = urllib.request.Request(upstream + '/' + suffix, data=body, headers=headers, method=self.command)
            with urllib.request.urlopen(request, timeout=180) as response:
                mime = response.headers.get('Content-Type', 'application/octet-stream')
                if styled and 'text/html' in mime:
                    html = response.read().decode()
                    html = re.sub(r'<meta[^>]+http-equiv=["\']refresh["\'][^>]*>', '', html, flags=re.I)
                    html = html.replace('</style>', '</style>' + REPORT_STYLE, 1)
                    return self.reply(html, mime)
                self.send_response(response.status)
                self.send_header('Content-Type', mime)
                for name in ('Content-Length', 'Content-Range', 'Accept-Ranges'):
                    if response.headers.get(name): self.send_header(name, response.headers[name])
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                if self.command != 'HEAD':
                    while True:
                        chunk = response.read1(65536)
                        if not chunk: break
                        self.wfile.write(chunk)
                        self.wfile.flush()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            self.reply(json.dumps({'error': 'Service is unavailable. Check the local tunnel and remote service.'}), status=503)

    def do_POST(self):
        if self.path.startswith('/chat/'):
            return self.proxy(self.server.chat, '/chat/')
        self.reply('Not found', 'text/plain', 404)

    def do_HEAD(self): self.do_GET()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path.startswith('/chat/'):
            return self.proxy(self.server.chat, '/chat/')
        if path.startswith('/globo/'):
            return self.proxy(self.server.globo, '/globo/', styled=True)
        if self.server.redirect_proposal and path not in {'/report.css', '/report.js'}:
            self.send_response(302)
            self.send_header('Location', self.server.redirect_proposal)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if path == '/reports/sit/':
            html = (ROOTS['/assets/sit/'] / 'report.html').read_text()
            html = html.replace('<style>', '<base href="/assets/sit/"><style>', 1)
            html = html.replace('</style>', '</style>' + REPORT_STYLE, 1)
            return self.reply(html, 'text/html; charset=utf-8')
        if path in SOURCES:
            return self.reply(SOURCES[path].read_bytes(), 'text/plain; charset=utf-8')
        for prefix, root in ROOTS.items():
            if path.startswith(prefix):
                target = (root / path[len(prefix):]).resolve()
                if not target.is_relative_to(root.resolve()) or not target.is_file():
                    return self.reply('Not found', 'text/plain', 404)
                if target.suffix.lower() not in {'.png', '.gif', '.jpg', '.jpeg', '.webp', '.svg', '.json', '.csv', '.txt', '.mp4', '.webm', '.pdf'}:
                    return self.reply('Not found', 'text/plain', 404)
                size = target.stat().st_size
                start, end = 0, size - 1
                range_header = self.headers.get('Range')
                if range_header:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header)
                    if not match or not any(match.groups()):
                        return self.reply('Invalid range', 'text/plain', 416)
                    left, right = match.groups()
                    if left:
                        start = int(left)
                        end = min(int(right), end) if right else end
                    else:
                        start = max(0, size - int(right))
                    if start > end or start >= size:
                        self.send_response(416)
                        self.send_header('Content-Range', f'bytes */{size}')
                        self.send_header('Content-Length', '0')
                        self.end_headers()
                        return
                with target.open('rb') as f:
                    self.send_response(206 if range_header else 200)
                    self.send_header('Content-Type', mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
                    self.send_header('Content-Length', str(end - start + 1))
                    self.send_header('Accept-Ranges', 'bytes')
                    if range_header:
                        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                    self.send_header('Cache-Control', 'public, max-age=600')
                    self.end_headers()
                    if self.command != 'HEAD':
                        f.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = f.read(min(65536, remaining))
                            if not chunk: break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                return
        # Serve only website files; never expose directory listings or server source.
        if path not in {'/', '/index.html', '/style.css', '/app.js', '/innovations.js', '/proposal.js', '/proposal.css', '/projects.js', '/projects.css', '/report.css', '/report.js', '/favicon.svg', '/marked.js', '/marked-LICENSE.txt'}:
            return self.reply('Not found', 'text/plain', 404)
        return super().do_GET()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--chat', default='http://127.0.0.1:17860')
    p.add_argument('--globo', default='http://127.0.0.1:17865')
    p.add_argument('--redirect-proposal', default='')
    args = p.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.chat, server.globo = args.chat, args.globo
    server.redirect_proposal = args.redirect_proposal
    print(f'TBD AGI → http://localhost:{args.port}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
