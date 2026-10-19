"""Replay-only mask maintenance over frozen checkpoints; no adapter retraining."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from .model import BayesSiT,InputPrior,transport_batch
from .run import balanced_indices,query_loss,atomic_json
from .prepare import DOMAINS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--root',default='/home/dev/sit_state')
    args=p.parse_args()
    torch.set_num_threads(4)
    root=Path(args.root); run=root/'runs'/args.name
    out=run/'repair'; out.mkdir(exist_ok=True)
    cfg=json.loads((run/'config.json').read_text())
    protocol={'source':'ftn/projects/bayesian_adv/input_conditioned.py:ic_repair_from_replay',
              'fit_rows':32,'validation_rows':32,'steps':100,'restarts':3,'take_best':True,
              'lr':.3,'temperature':1.,'logit_magnitude':4.,'threshold':.5,
              'same_domain_distance':.05,'weights_frozen':True,'evaluation_support':[4,40],
              'duplicate_policy':'deduplicate exact clean latent before fit/validation split',
              'schedule':'after each saved block; exclude live projector as in source',
              'quality_count':200,'quality_seed':99233}
    atomic_json(out/'protocol.json',protocol)
    torch.manual_seed(cfg['seed'])
    model=BayesSiT(root/'SiT-XL-2-256.pt',cfg['H'],cfg['k'],cfg['rank'],cfg['alpha']).cuda().eval()
    router=InputPrior(); history=[]
    for block,domain in enumerate(cfg['order'].split(',')):
        stage=f'after_{block+1}_{domain}'
        ck=torch.load(run/f'{stage}.pt',map_location='cuda',weights_only=False)
        model.load_adapter(ck['adapter'])
        # Carry repaired entries forward; append only the newly archived key.
        router.entries.append(copy.deepcopy(ck['router'][-1]))
        raw=InputPrior(); raw.entries=copy.deepcopy(ck['router'])
        before_state=model.get_reconfig_state()
        versions=[p._version for p in model.parameters()]
        cpu_rng=torch.get_rng_state().clone(); gpu_rng=torch.cuda.get_rng_state().clone()
        tic=time.time()
        repairs=router.repair_from_replay(model,ck['replay'],
                 live_projector=router.entries[-1]['projector'],seed=cfg['seed']+7401+block)
        assert versions==[p._version for p in model.parameters()]
        assert all(p.grad is None for p in model.parameters())
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        for key in ['m','mask_logits']:
            assert torch.equal(before_state[key],model.get_reconfig_state()[key])
        for r in repairs:
            if 'fit_hashes' in r:
                assert not set(r['fit_hashes']) & set(r['validation_hashes'])
            # Only diagnostics use the known replay insertion order.
            counts={}
            for idx in r['routed_batches']:
                label=cfg['order'].split(',')[idx//32]
                counts[label]=counts.get(label,0)+1
            r['batch_domain_counts_for_audit_only']=counts
        row={'stage':stage,'repairs':repairs,'repair_seconds':time.time()-tic,'domains':{},
             'checks':{'weights_unchanged':True,'posterior_restored':True,'rng_restored':True}}
        for d in DOMAINS:
            support={k:v.cuda() for k,v in torch.load(root/f'{d}_support.pt',weights_only=True).items()}
            test={k:v.cuda() for k,v in torch.load(root/f'{d}_test.pt',weights_only=True).items()}
            si=balanced_indices(support['y'],80); qi=balanced_indices(test['y'],200)
            sx,sy=transport_batch(support['z'][si],support['y'][si],torch.Generator(device='cuda').manual_seed(189))
            qx,qy=transport_batch(test['z'][qi],test['y'][qi],torch.Generator(device='cuda').manual_seed(901))
            candidates={}; metadata={}
            for budget in [4,40]:
                torch.manual_seed(7801+budget+DOMAINS.index(d)); start=time.time()
                model.cold_reconfigure(sx[:budget],sy[:budget],steps=30,take_best=False)
                key=f'cold{budget}'; candidates[key]=model.m.clone()
                metadata[key]={'seconds':time.time()-start}
            for prefix,r in [('raw',raw),('repaired',router)]:
                for refine in [False,True]:
                    key=prefix+('_refine4' if refine else '_direct4')
                    # A router miss must receive exactly the same fallback
                    # search noise as cold4 and the paired other router.
                    torch.manual_seed(7801+4+DOMAINS.index(d))
                    start=time.time()
                    metadata[key]=r.infer(model,support['z'][si[:4]],sx[:4],sy[:4],refine)
                    metadata[key]['seconds']=time.time()-start
                    candidates[key]=model.m.clone()
            row['domains'][d]={}
            for key,mask in candidates.items():
                result=query_loss(model,qx,qy,mask)
                result.update(mask=mask.cpu().tolist(),metadata=metadata[key])
                row['domains'][d][key]=result
            print(json.dumps({'stage':stage,'domain':d,'loss':{k:v['loss'] for k,v in row['domains'][d].items()}}),flush=True)
        history.append(row)
        atomic_json(out/'lifecycle.json',history)
        torch.save(router.entries,out/f'{stage}_router.pt')
        print('REPAIR_STAGE_COMPLETE',stage,json.dumps([{k:v for k,v in r.items()
              if k not in ('fit_hashes','validation_hashes','old_mask','candidate_mask')}
              for r in repairs]),flush=True)
    atomic_json(out/'completed.json',{'complete':True,'stages':len(history)})


if __name__=='__main__':
    main()
