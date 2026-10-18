"""Pod-only idempotent setup of the four explicitly requested decoder experiments."""
import json
import os
from pathlib import Path
import signal
import time
from .continuous_runs import STATE,ROOT,gallery,read
from .search import alive, atomic as atomic_json


def main():
    path=ROOT/'state.json';trials=read(path)
    if any(alive(t) for t in trials):
        raise RuntimeError('Previous workers must finish graceful shutdown before preparing new runs')
    archive=STATE/'archives/pre_m2e_ftn';archive.mkdir(parents=True,exist_ok=True)
    for t in trials:
        if t.get('m2e_ftn'):raise RuntimeError('FTN run set already installed; resume supervisor instead')
        t.update(status='paused',compare=True)
    atomic_json(archive/'stopped_state.json',trials)
    summary={t['name']:read(STATE/'runs'/t['name']/'status.json') for t in trials}
    atomic_json(archive/'final_status.json',summary)
    for gpu,(variant,seed) in enumerate((('flat',19),('flat',101),('conv',19),('conv',101))):
        trials.append(dict(name=f'm2eftn_{variant}_s{seed}',gpu=gpu,seed=seed,
            spawn='12.0,39.1,120.0' if seed==19 else '16.0,58.1,118.0',
            scaled=True,model_free_only=True,m2e_ftn=variant,reservoir_variant='tuned_v1',
            mf_replay_dtype='float16',software_renderer=True,weight=0.,status='pending',
            attempts=0,achievements=True,phase='m2e_ftn',render_size=256))
    atomic_json(path,trials)
    atomic_json(ROOT/'phase.json',dict(title='FTN–M2E · isolated policy decoder experiments',time=time.time()))
    gallery(trials)
    print('Prepared four fresh unlimited FTN–M2E runs; previous checkpoints and results retained.',flush=True)


if __name__=='__main__':main()
