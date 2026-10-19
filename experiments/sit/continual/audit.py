"""Audit logged mask reuse and reconstruct replay selection from its seeded RNG."""
import json
from pathlib import Path
import numpy as np


def main():
    root = Path('/home/dev/sit_state/runs')
    audit = {}
    for run in sorted(root.iterdir()):
        if run.name == 'smoke' or not (run/'completed.json').exists():
            continue
        cfg = json.loads((run/'config.json').read_text())
        records = [json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
        masks = np.array([r['mask'] for r in records])
        blocks = np.array([r['block'] for r in records])
        boundaries = [masks[np.where(blocks==b)[0][-1]] for b in range(3)]
        result = {'boundary_mask_overlap': (np.array(boundaries) @ np.array(boundaries).T / masks[0].sum()).tolist(),
                  'blocks': {}}
        rng = np.random.default_rng(cfg['seed']+1000)
        memory = []
        choices = []
        for r,m in zip(records,masks):
            if r['step']<32:
                memory.append((r['block'],m))
            if not cfg['replay']:
                choices.append(None)
                continue
            overlap = [oldmask @ m / m.sum() for _,oldmask in memory]
            eligible = [i for i,v in enumerate(overlap) if .2<v<=.8]
            if eligible:
                pick = int(rng.choice(eligible))
                choices.append(memory[pick][0])
                assert r['replay_loss'] is not None
            else:
                choices.append(None)
                assert r['replay_loss'] is None
        for b,d in enumerate(cfg['order'].split(',')):
            idx = np.where(blocks==b)[0]
            mm = masks[idx]
            old = sum(choices[i] is not None and choices[i]<b for i in idx)
            result['blocks'][d] = {'change_points': sum(records[i]['changepoint'] for i in idx),
                'distinct_hard_masks': len(np.unique(mm,axis=0)),
                'consecutive_overlap_mean': float(np.mean(np.sum(mm[:-1]*mm[1:],axis=1)/mm[0].sum())),
                'atom_usage': mm.mean(0).tolist(),
                'reconstructed_replay_from_earlier_domain_fraction': old/len(idx)}
        audit[run.name] = result
    out = Path('/home/dev/workspace/continual/artifacts/mask_audit.json')
    out.write_text(json.dumps(audit,indent=2))
    print(json.dumps({n:{'boundary_overlap':r['boundary_mask_overlap'],
        'blocks':{d:{k:v for k,v in b.items() if k!='atom_usage'} for d,b in r['blocks'].items()}}
        for n,r in audit.items()},indent=2))


if __name__ == '__main__':
    main()
