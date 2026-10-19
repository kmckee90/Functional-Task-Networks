"""Render recent actual replay with saved model weights, without resetting agents."""
import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import pickle
import sys
import time

STATE=Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))
sys.path.insert(0,str(STATE/'vendor/dreamerv3'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import elements
import numpy as np
import ruamel.yaml
from .env import Globo,atomic_json
from .world_probe import ProbeAgent,video


def recent_batch(run):
    # Filename timestamp is chunk creation time, not checkpoint write order.
    windows=[];provenance=[]
    for path in sorted((run/'replay').glob('*.npz'),reverse=True)[:16]:
        try:
            with np.load(path) as archive:data={k:archive[k] for k in archive.files}
        except (OSError,ValueError,EOFError):continue
        end=len(data['image'])
        while end>=32 and len(windows)<8:
            start=end-32
            if not data['is_first'][start+1:end].any() and not data['is_last'][start:end-1].any():
                row={k:v[start:end].copy() for k,v in data.items()}
                row['consec']=np.zeros(32,np.int32)
                windows.append(row);provenance.append(dict(chunk=path.name,start=start,end=end))
                end=start
            else:end-=1
        if len(windows)>=8:break
    if not windows:raise ValueError('No saved 32-frame within-life sequence yet')
    # First sequence is the newest suitable window. Repeat only to meet batch shape.
    count=len(windows)
    while len(windows)<8:windows.append(windows[-1])
    return {k:np.stack([w[k] for w in windows]) for k in windows[0]},provenance,count


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=STATE/'runs'/args.run;out=run/'trajectory_progress';out.mkdir(exist_ok=True)
    lock=(out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'pid').write_text(str(os.getpid()))
    launch=json.loads((run/'launch.json').read_text())
    config=elements.Config(ruamel.yaml.YAML(typ='safe').load((run/'dreamer.yaml').read_text()))
    cfg=dict(config.agent)
    cfg['explore']=dict(ensemble=5,density=launch['density'],disagreement=launch['disagreement'],
        prediction_error=launch.get('prediction_error',0),disagreement_power=launch.get('disagreement_power',1),
        actor_weight=launch['actor_weight'],feedback=launch['feedback'],
        horizon=launch.get('intrinsic_horizon',100),model_only=launch.get('world_model_only',False))
    ac=elements.Config(**cfg,logdir=str(out),seed=config.seed,jax=config.jax,batch_size=8,
        batch_length=32,replay_context=0,report_length=32,replica=0,replicas=1)
    spaces=Globo(out/'spaces',video_every=0)
    agent=ProbeAgent(spaces.obs_space,{k:v for k,v in spaces.act_space.items() if k!='reset'},ac)
    previous=''
    if (out/'latest.json').exists():previous=json.loads((out/'latest.json').read_text()).get('checkpoint','')
    while True:
        try:
            checkpoint=(run/'ckpt/latest').read_text().strip()
            source=run/'ckpt'/checkpoint
            if checkpoint==previous or not (source/'done').exists():time.sleep(60);continue
            step=pickle.loads((source/'step.pkl').read_bytes())
            updates=pickle.loads((source/'model_updates.pkl').read_bytes())
            batch,provenance,count=recent_batch(run)
            # A hard link pins just the agent weights while checkpoint retention runs.
            weights=out/'weights';version=weights/'snapshot';version.mkdir(parents=True,exist_ok=True)
            weightfile=version/'agent.pkl'
            if weightfile.exists():weightfile.unlink()
            os.link(source/'agent.pkl',weightfile)
            (version/'done').touch();(weights/'latest').write_text('snapshot')
            cp=elements.Checkpoint(elements.Path(weights),write=False);cp.agent=agent;cp.load(keys=['agent'])
            weightfile.unlink()
            prepared=next(iter(agent.stream(itertools.cycle([batch]))))
            _,result=agent.report(agent.init_report(8),prepared)
            result={k:np.asarray(v) for k,v in result.items() if k!='params/summary'}
            dest=out/f'step{int(step):07d}_{checkpoint}';dest.mkdir(exist_ok=True)
            video(dest/'recent.mp4',result,index=0)
            video(dest/'textured.mp4',result)
            truth=result['truth'][0].astype(np.float32)/255
            metadata=dict(step=int(step),model_updates=int(updates),checkpoint=checkpoint,
                folder=dest.name,time=time.time(),source=provenance[0],distinct_sequences=count,
                heldout=False,note='Recent actual saved replay; may have been trained on. Not a generalization test. '
                    'No environment is created or reset. Model step is the checkpoint step, not the absolute index of these frames.',
                posterior_mse=float(np.mean((result['posterior'][0]-truth)**2)),
                forecast16_mse=float(np.mean((result['forecast'][0,-1]-truth[-1])**2)))
            atomic_json(dest/'metadata.json',metadata);atomic_json(out/'latest.json',metadata)
            previous=checkpoint;print(json.dumps(metadata),flush=True)
        except Exception as exc:print('Trajectory rendering retry:',repr(exc),flush=True)
        time.sleep(60)


if __name__=='__main__':main()
