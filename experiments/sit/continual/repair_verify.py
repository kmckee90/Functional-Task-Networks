"""Independent saved-result checks; no inference, run on pod."""
import hashlib
import json
from pathlib import Path
import torch


def digest(z):
    return hashlib.sha256(z.contiguous().numpy().tobytes()).hexdigest()


def main():
    root=Path('/home/dev/sit_state'); all_rows=[]
    status=json.loads((root/'repair_status.json').read_text())
    assert len(status)==3 and all(v['phase']=='quality' and v['exit_code']==0 for v in status.values())
    forbidden=set()
    for domain in ['photo','monochrome','poster','sketch']:
        for split in ['support','test']:
            data=torch.load(root/f'{domain}_{split}.pt',map_location='cpu',weights_only=True)
            forbidden.update(digest(z) for z in data['z'])
    for name in status:
        run=root/'runs'/name; folder=run/'repair'
        rows=json.loads((folder/'lifecycle.json').read_text())
        assert len(rows)==3
        final_diag=json.loads((run/'diagnostics.json').read_text())
        for row in rows:
            ck=torch.load(run/f"{row['stage']}.pt",map_location='cpu',weights_only=False)
            allowed={digest(z) for batch,_,_ in ck['replay'] for z in batch}
            for r in row['repairs']:
                if 'fit_hashes' not in r:
                    continue
                fit=set(r['fit_hashes']); validation=set(r['validation_hashes'])
                assert len(fit)==32 and len(validation)==32 and not fit & validation
                assert (fit | validation) <= allowed
                assert not (fit | validation) & forbidden
                assert r['accepted']==(r['candidate_loss']<r['before'])
            for domain,methods in row['domains'].items():
                for method,result in methods.items():
                    assert len(result['per_example'])==200
                    assert sum(result['mask'])==16
                if row==rows[0]:
                    assert methods['raw_direct4']['mask']==methods['repaired_direct4']['mask']
                if row==rows[-1]:
                    for new,old in [('cold4','cold_4_30'),('cold40','cold_40_30'),('raw_direct4','ic_direct4')]:
                        assert methods[new]['mask']==final_diag[domain][old]['mask'],(name,domain,new)
                        assert abs(methods[new]['loss']-final_diag[domain][old]['loss'])<1e-6
        q=json.loads((folder/'quality.json').read_text())
        assert len(q)==16
        assert all(v['generated_count']==200 and v['reference_count']==400 for v in q)
        all_rows.append({'run':name,'stages':3,'quality_records':16,'baseline_reproduced':True,
                         'repair_fit_validation_training_only':True,'unique_generation_sets':sum('identical_mask_to' not in v for v in q)})
    result={'passed':True,'runs':all_rows}
    out=Path('/home/dev/workspace/continual/artifacts/repair'); out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
