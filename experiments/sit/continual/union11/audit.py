import json
import hashlib
from pathlib import Path
from .common import ART,NAMES,atomic

def main():
    protocol=json.loads((ART/'protocol.json').read_text())
    for name in ['training.py','evaluate.py','metrics.py','common.py']:
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==protocol['source_sha256'][name],name
    assert protocol['class_dropout']==dict(filters=.1,faces_and_objects=0.)
    assert json.loads((ART/'evaluator_validation.json').read_text())['complete']
    checks={}
    for seed in [123,125]:
        for arm in ['noreplay','replay']:
            tag=f'{arm}_s{seed}';folder=ART/tag
            assert json.loads((folder/'complete.json').read_text())['complete']
            rows=[json.loads(x) for x in (folder/'train.jsonl').read_text().splitlines()]
            assert len(rows)==16500
            for j,n in enumerate(NAMES):
                rr=[r for r in rows if r['block']==j]
                assert len(rr)==1500 and all(r['domain']==n and sum(r['mask'])==256 for r in rr)
                if arm=='noreplay':assert all(r['replay_loss'] is None for r in rr)
            summaries=list(folder.glob('*/summary.json'));assert len(summaries)==12
            for path in summaries:
                s=json.loads(path.read_text());assert s['complete'] and s['training_tasks']==NAMES[:s['completed']]
                rr=json.loads((path.parent/'results.json').read_text());assert len(rr)==110
                assert all(len(set(r['support_indices']))==4 and sum(r['mask'])==256 for r in rr)
                teachers=json.loads((path.parent/'teachers.json').read_text())
                assert list(teachers)==NAMES[:s['completed']]
                for teacher in teachers.values():
                    assert not set(teacher['fit_indices']) & set(teacher['validation_indices'])
                for r in rr:
                    j=NAMES.index(r['domain'])
                    expected='forward' if j>=s['completed'] else 'current' if j==s['completed']-1 else 'backward'
                    assert r['relation']==expected
                marker=json.loads((folder/f'{s["stage"]}_complete.json').read_text())
                assert all(marker[k] for k in ['weights_unchanged','rng_restored','posterior_restored'])
            checks[tag]=dict(passed=True,updates=len(rows),checkpoints=len(summaries))
    atomic(ART/'audit.json',dict(passed=True,runs=checks));print('AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
