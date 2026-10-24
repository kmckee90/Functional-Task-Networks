"""Resume the explicitly paused lifetime cohort, enabling whole-life recordings."""
import argparse
import os
from pathlib import Path
import time
from .phase2 import STATE,read,atomic,detach

def main():
    p=argparse.ArgumentParser();p.add_argument('--pod',choices=['globo-actions','globo1'],required=True);a=p.parse_args()
    path=STATE/'continuous_runs/state.json';trials=read(path)
    assert len(trials)==4 and all(t['status']=='paused' for t in trials)
    pidfile=STATE/'continuous_runs/supervisor.pid'
    if pidfile.exists():
        cmd=Path('/proc')/pidfile.read_text().strip()/'cmdline'
        assert not cmd.exists() or b'globo_dreamer.continuous_runs' not in cmd.read_bytes()
    atomic(STATE/f'continuous_runs/before_life_video_resume_{int(time.time())}.json',trials)
    for t in trials:
        assert (STATE/'runs'/t['name']/'ckpt/latest').exists(),t['name']
        t.update(status='pending',life_capture=True,retry_after=0)
        for key in ('pid','watchdog_term_at','watchdog_kill_at','pause_reason','paused_at'):t.pop(key,None)
    atomic(path,trials)
    print('supervisor',detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log'))
    print('publisher',detach('globo_dreamer.lifetime_comparison','lifetime_publish.log','publish','--pod',a.pod))
    if a.pod=='globo-actions':print('monitor',detach('globo_dreamer.lifetime_comparison','lifetime_monitor.log','monitor'))

if __name__=='__main__':main()
