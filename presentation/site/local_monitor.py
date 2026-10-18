#!/usr/bin/env python3
"""Laptop HTTP server reading live monitor artifacts directly from the Globo pod.

No local training, checkpoint copies, video archive, or remote HTTP server needed.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

REMOTE = r'''
import json,mimetypes,sys
from pathlib import Path
root=Path('/home/dev/globo_dreamer_state/reviews/config24')
path,range_header,method,payload=sys.argv[1:]
try:
 if path in ('/api/clip-settings','/phase2/api/clip-settings','/phase3/api/clip-settings','/phase4/api/clip-settings','/phase5/api/clip-settings'):
  phase=path.split('/')[1]
  p=Path(f'/mnt/obelisk-shared/kmckee90/globo_{phase}/clip_settings.json' if phase.startswith('phase') else '/home/dev/globo_dreamer_state/continuous_runs/clip_settings.json')
  if method=='POST':
   frames=json.loads(payload)['frames']
   if type(frames) is not int or frames not in (64,128,192,256):raise ValueError('Unsupported clip length')
   temp=p.with_suffix('.local.tmp');temp.write_text(json.dumps(dict(frames=frames)));temp.replace(p)
  try:frames=json.loads(p.read_text())['frames']
  except (OSError,ValueError,KeyError):frames=192
  body=json.dumps(dict(frames=frames,seconds=frames/6,choices=[64,128,192,256])).encode()
  print(json.dumps(dict(status=200,mime='application/json',length=len(body))),flush=True)
  sys.stdout.buffer.write(body);sys.exit(0)
 if method=='POST':raise ValueError('Unknown control')
 rel=Path(path.lstrip('/'))
 if '..' in rel.parts:raise ValueError('Invalid path')
 target=root/rel
 if target.is_dir():target=target/'index.html'
 if target.suffix.lower() not in ('.html','.mp4','.webm','.png','.jpg','.jpeg','.svg','.gif','.json','.jsonl','.txt','.md','.css','.js') or not target.is_file():
  print(json.dumps(dict(status=404,length=0)),flush=True);sys.exit(0)
 # Hold an open descriptor so atomic clip replacement cannot mix generations.
 with target.open('rb') as f:
  import os,re
  size=os.fstat(f.fileno()).st_size;start,end=0,size-1
  if range_header:
   m=re.fullmatch(r'bytes=(\d*)-(\d*)',range_header)
   if not m or not any(m.groups()):raise ValueError('Invalid range')
   a,b=m.groups()
   if a:start=int(a);end=min(int(b),end) if b else end
   else:start=max(0,size-int(b))
   if start>end or start>=size:
    print(json.dumps(dict(status=416,length=0,content_range=f'bytes */{size}')),flush=True);sys.exit(0)
  header=dict(status=206 if range_header else 200,length=max(0,end-start+1),mime=mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
  if range_header:header['content_range']=f'bytes {start}-{end}/{size}'
  print(json.dumps(header),flush=True)
  if method!='HEAD':
   f.seek(start);remaining=end-start+1
   while remaining>0:
    data=f.read(min(65536,remaining))
    if not data:break
    sys.stdout.buffer.write(data);remaining-=len(data)
except (ValueError,KeyError) as e:
 print(json.dumps(dict(status=400,length=0)),flush=True)
'''

class Pod:
    def __init__(self,namespace,owner):
        self.prefix=['kubectl','-n',namespace];self.owner=owner
        self.name=None;self.updated=0;self.lock=threading.Lock()
    def get(self):
        with self.lock:
            if self.name and time.monotonic()-self.updated<60:return self.name
            raw=subprocess.check_output(self.prefix+['get','pods','--request-timeout=15s','-l',
                f'astera.org/owner={self.owner},astera.org/project=globo-dreamer','-o','json'],timeout=25)
            pods=json.loads(raw)['items']
            ready=[p for p in pods if p['status']['phase']=='Running' and not p['metadata'].get('deletionTimestamp')]
            if not ready:raise RuntimeError('Globo pod is not running')
            self.name=max(ready,key=lambda p:p['metadata']['creationTimestamp'])['metadata']['name']
            self.updated=time.monotonic();return self.name

class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):self.serve()
    def do_GET(self):self.serve()
    def do_POST(self):self.serve()
    def serve(self):
        process=None;timer=None;sent=False
        try:
            payload=''
            if self.command=='POST':
                origin=self.headers.get('Origin')
                if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
                    self.send_error(403);return
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=512:self.send_error(400);return
                payload=self.rfile.read(size).decode()
                if self.headers.get_content_type()!='application/json':self.send_error(400);return
            pod=self.server.pod
            process=subprocess.Popen(pod.prefix+['exec',pod.get(),'--',
                '/home/dev/globo_dreamer_state/venv/bin/python','-u','-c',REMOTE,
                unquote(urlsplit(self.path).path),self.headers.get('Range',''),self.command,payload],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            timer=threading.Timer(180,process.kill);timer.daemon=True;timer.start()
            header=json.loads(process.stdout.readline())
            self.send_response(header['status']);sent=True
            self.send_header('Content-Type',header.get('mime','text/plain'))
            self.send_header('Content-Length',str(header['length']))
            self.send_header('Cache-Control','no-store');self.send_header('Accept-Ranges','bytes')
            if 'content_range' in header:self.send_header('Content-Range',header['content_range'])
            self.end_headers()
            if self.command!='HEAD':
                while True:
                    chunk=process.stdout.read(65536)
                    if not chunk:break
                    self.wfile.write(chunk)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as exc:
            self.log_error('Pod read failed: %s',exc)
            if not sent:self.send_error(503,'Cannot reach Globo pod; check network and Kubernetes login')
        finally:
            if timer:timer.cancel()
            if process:
                if process.poll() is None:process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                process.stdout.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--bind',default='127.0.0.1')
    parser.add_argument('--namespace',default='obelisk')
    parser.add_argument('--owner',default='kmckee90')
    args=parser.parse_args()
    server=ThreadingHTTPServer((args.bind,args.port),Handler)
    server.pod=Pod(args.namespace,args.owner)
    print(f'Local monitor: http://{args.bind}:{args.port}/continuous/',flush=True)
    server.serve_forever()
