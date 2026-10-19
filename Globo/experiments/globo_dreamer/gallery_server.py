"""Loopback gallery server with a bounded runtime clip-length control."""
import argparse
import json
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from .monitor_settings import SETTINGS,CLIP_CHOICES,read_settings


class Handler(SimpleHTTPRequestHandler):
    settings_lock=Lock()

    def reply(self,status,data):
        body=json.dumps(data).encode()
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        if urlsplit(self.path).path=='/api/clip-settings':return self.reply(200,read_settings())
        return super().do_GET()

    def do_POST(self):
        if self.path!='/api/clip-settings':return self.reply(404,dict(error='Unknown control'))
        origin=self.headers.get('Origin')
        if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
            return self.reply(403,dict(error='Origin mismatch'))
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=512:raise ValueError('Invalid request size')
            if self.headers.get_content_type()!='application/json':raise ValueError('Expected JSON')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict):raise ValueError('Expected an object')
            frames=data.get('frames')
            if type(frames) is not int or frames not in CLIP_CHOICES:raise ValueError('Unsupported clip length')
        except (ValueError,TypeError) as exc:return self.reply(400,dict(error=str(exc)))
        with self.settings_lock:
            SETTINGS.parent.mkdir(parents=True,exist_ok=True)
            temp=SETTINGS.with_suffix('.tmp');temp.write_text(json.dumps(dict(frames=frames)))
            temp.replace(SETTINGS)
        self.reply(200,read_settings())


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765)
    p.add_argument('--directory',default='/home/dev/globo_dreamer_state/reviews/config24')
    args=p.parse_args()
    from functools import partial
    ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=args.directory)).serve_forever()


if __name__=='__main__':main()
