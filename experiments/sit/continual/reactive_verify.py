"""Audit source preset, detector decisions, masks, replay absence and outputs."""
import hashlib
import argparse
import json
import math
from pathlib import Path
import torch
from .bayesian_adv_config_snapshot import BayesLoRAReactiveLow


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--sweep',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=Path('/home/dev/sit_state'); source=BayesLoRAReactiveLow()
    status=json.loads((root/('reactive_sweep_status.json' if args.sweep else 'reactive_status.json')).read_text())
    assert len(status)==(6 if args.sweep else 3) and all(v['phase']=='quality' and v['exit_code']==0 for v in status.values())
    protocol=json.loads((root/'reactive_protocol.json').read_text())
    assert protocol['source_bayes_settings']==source.BAYES_FTN_PARS
    assert protocol['source_snapshot_sha256']==hashlib.sha256(Path(__file__).with_name('bayesian_adv_config_snapshot.py').read_bytes()).hexdigest()
    results=[]
    for name in status:
        folder=root/'runs'/name; cfg=json.loads((folder/'config.json').read_text())
        assert cfg['bayes_preset']=='reactive_low' and cfg['lr']==source.TRAIN_LR
        assert cfg['replay']==0 and not cfg['ic_repair']
        r_sd=cfg.get('reactive_sd') or source.RECONFIG_REACTIVE_SD
        assert r_sd in ([1.,3.] if args.sweep else [5.])
        records=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
        assert len(records)==4500
        mu,var,since=None,0.,0
        previous=None; changes=0; boundary_scores=[]
        for r in records:
            assert r['replay_loss'] is None and math.isfinite(r['loss'])
            threshold=None if mu is None else mu+r_sd*math.sqrt(max(var,0.))
            if threshold is not None:
                assert abs(threshold-r['detector_threshold'])<1e-10
            expected=mu is None or (since>=math.ceil(1/max(1-source.RECONFIG_REACTIVE_EMA,1e-8)) and r['detector_loss']>threshold)
            if r['step']==0 and mu is not None:
                boundary_scores.append({'domain':r['domain'],'detector_loss':r['detector_loss'],
                    'previous_ema_mean':mu,'previous_ema_sd':math.sqrt(max(var,0.)),
                    'z_score':(r['detector_loss']-mu)/max(math.sqrt(max(var,0.)),1e-12),
                    'threshold':threshold,'triggered':r['changepoint']})
            assert r['changepoint']==expected,(name,r['block'],r['step'])
            assert sum(r['mask'])==cfg['k']
            if previous is not None and not r['changepoint'] and not previous['changepoint']:
                assert r['mask']==previous['mask'],'Stable mask changed'
            if r['changepoint']:
                changes+=1; mu,var,since=r['loss'],0.,0
            else:
                diff=r['loss']-mu; a=source.RECONFIG_REACTIVE_EMA
                mu=a*mu+(1-a)*r['loss']; var=a*var+(1-a)*diff*diff
            since+=1; previous=r
        history=json.loads((folder/'lifecycle.json').read_text()); assert len(history)==4
        previous_adapter=None; drift=[]
        for row in history:
            for d,methods in row['domains'].items():
                for mode,v in methods.items():
                    if mode=='inference':
                        continue
                    assert len(v['per_example'])==200 and math.isfinite(v['loss'])
                if row['stage']=='initial':
                    for mode in ['ic_direct','ic_refine']:
                        assert methods[mode]['mask']==methods['cold4']['mask']
                        assert methods[mode]['loss']==methods['cold4']['loss']
            if row['stage']!='initial':
                ck=torch.load(folder/f"{row['stage']}.pt",map_location='cpu',weights_only=False)
                assert ck['replay']==[]
                if previous_adapter is not None:
                    squared=torch.zeros(cfg['H'],dtype=torch.float64)
                    maximum=torch.zeros(cfg['H'],dtype=torch.float64)
                    for key,value in ck['adapter'].items():
                        if key.endswith(('.A','.B','.bias_bank')):
                            difference=(value-previous_adapter[key]).double().reshape(cfg['H'],-1)
                            squared+=difference.square().sum(1)
                            maximum=torch.maximum(maximum,difference.abs().max(1).values)
                    block=ck['completed_blocks']-1
                    used=torch.tensor([r['mask'] for r in records if r['block']==block]).bool().any(0)
                    assert torch.all(maximum[~used]==0),'Inactive atoms changed despite reset Adam and no replay'
                    drift.append({'stage':row['stage'],'per_atom_l2_change':squared.sqrt().tolist(),
                                  'atoms_changed':int((maximum>0).sum()),'training_atom_union':int(used.sum()),
                                  'inactive_atoms_exactly_unchanged':True})
                previous_adapter=ck['adapter']
        q=json.loads((folder/'reactive_quality'/'quality.json').read_text())
        assert len(q)==19 and all(v['generated_count']==200 and v['reference_count']==400 for v in q)
        results.append({'name':name,'threshold_sd':r_sd,'updates':len(records),'changepoints':changes,'no_replay_storage':True,
            'source_detector_reproduced':True,'stable_masks_unchanged':True,'quality_records':len(q),
            'unique_generation_sets':sum('identical_mask_to' not in v for v in q),'atom_drift':drift,
            'all_archived_logits_identical':all(torch.equal(e['logits'],ck['router'][0]['logits']) for e in ck['router']),
            'boundary_detector_scores':boundary_scores})
    out=Path('/home/dev/workspace/continual/artifacts')/('reactive_sweep' if args.sweep else 'reactive_low'); out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps({'passed':True,'runs':results},indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
