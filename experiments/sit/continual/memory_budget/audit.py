"""Final access, retention-budget, and evaluation audit."""
import hashlib,json
import torch
from pathlib import Path
from .common import ROOT,ART,NAMES,BUDGETS,atomic

def main():
    protocol=json.loads((ART/'protocol.json').read_text())
    for name in ['common.py','selection.py','replay.py','training.py','evaluate.py']:
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==protocol['source_sha256'][name]
    records={};selected={}
    for b in BUDGETS:
        tag=f'replay_p{b:02d}_s125';folder=ART/tag
        ck=torch.load(ROOT/'runs'/tag/'latest.pt',map_location='cpu',weights_only=False)
        assert ck['completed']==11 and set(ck['memory'])==set(NAMES)
        selected[b]={}
        for n,e in ck['memory'].items():
            count=e['source_count']*b//100;assert len(e['ids'])==len(set(e['ids']))==count
            assert len(e['z'])==len(e['y'])==count
            assert all(len(f)==count for f in e['features'].values())
            assert all(0<=i<count for entry in e['entries'] for i in entry['indices'])
            selected[b][n]=set(e['ids'])
        rows=0;replay_counts={};high=0
        for line in (folder/'train.jsonl').open():
            r=json.loads(line);rows+=1;assert NAMES[r['block']]==r['domain']
            if r['replay']:
                rep=r['replay'];assert .2<rep['overlap']<=.85
                replay_counts[r['domain']]=replay_counts.get(r['domain'],0)+1
                high+=rep['overlap']>.8
        assert rows==16500
        summaries=list(folder.glob('*/summary.json'));assert len(summaries)==12
        for p in summaries:
            s=json.loads(p.read_text());stage=p.parent
            marker=json.loads((folder/f'{s["stage"]}_complete.json').read_text())
            assert marker['weights_unchanged'] and marker['rng_restored'] and marker['posterior_restored']
            teachers=json.loads((stage/'teachers.json').read_text());assert list(teachers)==NAMES[:s['completed']]
            for n,t in teachers.items():
                assert not set(t['fit_indices'])&set(t['validation_indices'])
                assert all(i<len(ck['memory'][n]['ids']) for i in t['fit_indices']+t['validation_indices'])
            results=json.loads((stage/'results.json').read_text());assert len(results)==110
            assert all(len(set(r['support_indices']))==4 and sum(r['mask'])==256 for r in results)
        records[tag]=dict(passed=True,updates=rows,retained=sum(len(e['ids']) for e in ck['memory'].values()),
                          replay_counts=replay_counts,above80_replay=high,
                          replay_image_coverage=sum(c>0 for c in ck['replay']['image_counts'].values()))
        del ck
    for small,large in zip([5,10,25],[10,25,50]):
        for n in NAMES:assert selected[small][n]<=selected[large][n]
    atomic(ART/'audit.json',dict(passed=True,nested_subsets=True,runs=records))
    print('AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
