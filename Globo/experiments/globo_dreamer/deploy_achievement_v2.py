"""Pod-only controlled checkpoint/resume of the live supervisor and workers."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .continuous_runs import ROOT,STATE,read
from .search import alive

def main():
    trials=read(ROOT/'state.json')
    pid=int((ROOT/'supervisor.pid').read_text())
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'globo_dreamer.continuous_runs' in cmd,cmd
    record=dict(time=time.time(),runs=[dict(name=t['name'],pid=t['pid'],
        step=read(STATE/'runs'/t['name']/'status.json').get('step'))
        for t in trials if t['status']=='running' and alive(t)])
    (ROOT/'achievement_v2_restart.json').write_text(json.dumps(record,indent=2))
    # Supervisor's finally sends graceful SIGTERM to every worker. Each saves its
    # current learner/replay state before exiting; the new supervisor resumes it.
    os.kill(pid,signal.SIGTERM)
    for _ in range(60):
        stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split()[2]=='Z':break
        time.sleep(1)
    else:raise RuntimeError('Old supervisor still alive; refusing duplicate launcher')
    with (ROOT/'supervisor.log').open('a') as f:
        child=subprocess.Popen([sys.executable,'-u','-m','globo_dreamer.continuous_runs'],
            cwd=STATE,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
    print(json.dumps(dict(supervisor=child.pid,restarting=record['runs'])),flush=True)

if __name__=='__main__':main()
