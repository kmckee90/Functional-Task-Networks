"""Pod supervisor for the source ReactiveLow settings on the SiT protocol."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .bayesian_adv_config_snapshot import BayesLoRAReactiveLow


def main():
    root=Path('/home/dev/sit_state')
    source=BayesLoRAReactiveLow()
    snapshot=Path(__file__).with_name('bayesian_adv_config_snapshot.py')
    protocol={'name':'BayesLoRAReactiveLow on the existing SiT experiment',
        'source_snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        'source_bayes_settings':source.BAYES_FTN_PARS,
        'source_detector':{'sd':source.RECONFIG_REACTIVE_SD,'ema':source.RECONFIG_REACTIVE_EMA,
                           'warmup':'ceil(1/max(1-ema,1e-8))'},
        'training_lr':source.TRAIN_LR,'steps_per_domain':source.TRAIN_DURATION,
        'source_bank':{'H':source.HIDDEN_SIZE,'k':source.K,'rank':source.LORA_RANK,'alpha':source.LORA_ALPHA},
        'sit_bank':{'H':64,'k':16,'rank':1,'alpha':4},
        'architecture_mapping':'Preserve pretrained SiT-XL/2 and the previous 112 adapter locations/bank; source reactive and posterior parameters are read from a byte-identical config snapshot.',
        'data_mapping':'Same Imagenette photo/monochrome/poster/sketch caches, 16-image training batches, 1500 updates/domain, seeds123/124/125.',
        'replay':'No replay data stored, no gradient replay, no IC repair.',
        'ic':'Same clean-latent PCA router as previous SiT experiment: rank8, threshold0.5, 4 reference images, archive at known block boundaries. No training prior or novelty trigger. Direct/refine evaluation on the same weights as cold inference.',
        'evaluation':'Initial and after every block; cold4/cold40, IC direct/refine4, saved masks; 200 held-out queries with paired noise/times/search seeds.',
        'mask_analysis':'Change-point locations, boundary-mask overlap, unique masks and active-atom unions per block. No forced disjointness or boundary resets.',
        'differences_from_prior_run':['Stable logits lr0/1step/deterministic instead of lr.3/2steps/stochastic','Source training lr3e-4 instead of1e-4','Exact source warmup ceil expression instead of literal10','No replay storage, including no repair'],
        'interpretation':'This tests source ReactiveLow behavior with the existing SiT architecture, not the literal 4096-atom toy backbone.'}
    (root/'reactive_protocol.json').write_text(json.dumps(protocol,indent=2))
    jobs=[]
    for gpu,seed in enumerate([123,124,125]):
        name=f'bayes_reactive_low_s{seed}'
        env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',HF_HOME=str(root/'hf'))
        log=(root/f'{name}.log').open('a',buffering=1)
        cmd=[sys.executable,'-u','-m','continual.run','--name',name,'--seed',str(seed),
             '--steps',str(source.TRAIN_DURATION),'--bayes-preset','reactive_low','--ic','--resume']
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        jobs.append({'name':name,'proc':proc,'log':log,'env':env,'phase':'train'})
    while True:
        for job in jobs:
            if job['proc'].poll()==0 and job['phase']=='train':
                job['phase']='quality'
                job['proc']=subprocess.Popen([sys.executable,'-u','-m','continual.reactive_quality','--name',job['name']],
                    env=job['env'],stdout=job['log'],stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        status={j['name']:{'pid':j['proc'].pid,'phase':j['phase'],'exit_code':j['proc'].poll()} for j in jobs}
        temp=root/'reactive_status.tmp'; temp.write_text(json.dumps(status,indent=2)); temp.replace(root/'reactive_status.json')
        if all(j['proc'].poll() is not None for j in jobs):
            break
        time.sleep(20)
    if any(j['proc'].returncode for j in jobs):
        raise RuntimeError('ReactiveLow worker failed; inspect status/logs')


if __name__=='__main__':
    main()
