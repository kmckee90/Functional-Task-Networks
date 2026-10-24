"""Save and pause all currently assigned learners without deleting results."""
import argparse,os,signal,time
from pathlib import Path
from .phase2 import STATE,read,atomic,stop_supervisor
p=argparse.ArgumentParser();p.add_argument('--pod',required=True);a=p.parse_args()
stop_supervisor();path=STATE/'continuous_runs/state.json';ts=read(path,[])
atomic(path.with_name(f'before_natural_survival_{int(time.time())}.json'),ts)
active=[]
for t in ts:
 proc=Path(f'/proc/{t.get("pid")}/cmdline')
 if proc.exists() and t['name'].encode() in proc.read_bytes():
  os.kill(t['pid'],signal.SIGTERM);active.append(t)
 t.update(status='paused',pause_reason='Replaced by fresh natural-cycle survival experiment')
atomic(path,ts)
# Freeze publishers so archived monitors keep their existing cohorts.
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 try:cmd=proc.read_bytes()
 except OSError:continue
 if b'publish' in cmd and (b'globo_dreamer.lifetime_comparison' in cmd or b'globo_dreamer.survival_hard' in cmd):
  os.kill(int(proc.parent.name),signal.SIGTERM)
for i in range(120):
 pending=[t for t in active if Path(f'/proc/{t["pid"]}/cmdline').exists() and Path(f'/proc/{t["pid"]}/cmdline').read_bytes()]
 if not pending:break
 time.sleep(5)
else:raise RuntimeError('Learners still saving')
for t in active:
 run=STATE/'runs'/t['name'];s=read(run/'status.json')
 assert s.get('running') is False,(t['name'],s)
 latest=(run/'ckpt/latest').read_text().strip();assert (run/'ckpt'/latest/'done').exists()
 print('SAVED',t['name'],s['step'],latest,flush=True)
for phase,folder in [('lifetime_v2','globo_lifetime_v2'),('survival_hard','globo_survival_hard_s19')]:
 subset=[t for t in ts if t.get('phase')==phase]
 if subset:atomic(Path('/mnt/obelisk-shared/kmckee90')/folder/'pods'/f'{a.pod}.json',dict(time=time.time(),pod=a.pod,trials=subset))
