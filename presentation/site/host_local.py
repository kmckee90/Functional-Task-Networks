#!/usr/bin/env python3
"""Start/restart laptop hosting; user explicitly authorized local web processes.
Run with --fetch-media once while base is available. No training is launched.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request

SITE=Path(__file__).resolve().parent
CACHE=Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'globo-local-hosting'
KUBE=['kubectl','-n','obelisk']

def pod(project):
    result=subprocess.check_output(KUBE+['get','pods','--request-timeout=15s','-l',
        f'astera.org/owner=kmckee90,astera.org/project={project}','-o','json'],timeout=25)
    ready=[p for p in json.loads(result)['items'] if p['status']['phase']=='Running' and not p['metadata'].get('deletionTimestamp')]
    return max(ready,key=lambda p:p['metadata']['creationTimestamp'])['metadata']['name'] if ready else None

def stop_owned(role,marker):
    path=CACHE/(role+'.pid')
    if not path.exists():return
    pid=int(path.read_text())
    proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists() and marker.encode() in proc.read_bytes():
        os.kill(pid,signal.SIGTERM)
        for _ in range(30):
            stat=Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().split()[2]=='Z':break
            time.sleep(.1)

def release_old_forward(port):
    # Replace only old kubectl TCP forwards for this exact local port.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if not args or Path(os.fsdecode(args[0])).name!='kubectl' or b'port-forward' not in args:continue
        if any(a.startswith(f'{port}:'.encode()) for a in args):os.kill(int(proc.name),signal.SIGTERM)
    time.sleep(.2)

def start(role,args,env=None):
    with (CACHE/(role+'.log')).open('w') as log:
        proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    (CACHE/(role+'.pid')).write_text(str(proc.pid))
    return proc

def ready(url):
    for _ in range(50):
        try:
            with urllib.request.urlopen(url,timeout=3) as r:
                if r.status==200:return
        except OSError:time.sleep(.2)
    raise RuntimeError(f'Not responding: {url}; see logs in {CACHE}')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fetch-media',action='store_true');args=parser.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    with (CACHE/'launcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        media=CACHE/'media';media.mkdir(exist_ok=True)
        if args.fetch_media:
            name=pod('base')
            if not name:raise RuntimeError('Base must be running for the one-time media copy')
            proc=subprocess.Popen(KUBE+['exec',name,'--','tar','-C','/home/dev/proposal_state/media','-cf','-','.'],stdout=subprocess.PIPE)
            total=0
            with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
                for member in archive:
                    if not member.isfile():continue
                    path=Path(member.name)
                    if len(path.parts)!=1 or path.name in ('.','..'):raise RuntimeError('Unexpected media path')
                    target=media/path.name;temp=target.with_suffix(target.suffix+'.tmp')
                    with archive.extractfile(member) as src,temp.open('wb') as dst:
                        while chunk:=src.read(65536):dst.write(chunk);total+=len(chunk)
                    temp.replace(target)
            if proc.wait():raise RuntimeError('Media transfer failed')
            print(f'Copied {total/1024**2:.1f} MiB of proposal media',flush=True)
        # The proposal is published from proposal_githubio; only monitors run here.
        stop_owned('site',str(SITE/'server.py'))
        stop_owned('gateway',str(SITE/'server.py'))
        stop_owned('chat-forward','port-forward')
        for role,filename,port in [('monitor','local_monitor.py',8765)]:
            stop_owned(role,str(SITE/filename));release_old_forward(port)
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                sock.bind(('127.0.0.1',port))
            env=dict(os.environ,PROPOSAL_MEDIA=str(media),PYTHONDONTWRITEBYTECODE='1')
            cmd=[sys.executable,'-u',str(SITE/filename),'--port',str(port)]
            start(role,cmd,env)
        ready('http://127.0.0.1:8765/api/clip-settings')
        print('Proposal: https://kmckee90.github.io/nascent/\nMonitor: http://127.0.0.1:8765/continuous/\n'+
              f'Local media and logs: {CACHE}\nNo training workers or pods were stopped.',flush=True)

if __name__=='__main__':main()
