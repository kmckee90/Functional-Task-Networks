"""Compact transfer matrices, memory coverage, and first-draw sample grids."""
import csv
import html
import json
import shutil
from datetime import datetime,timezone
import numpy as np
from .common import ART,NAMES,BUDGETS,atomic
from ..union11.report import SPECS,matrix,plot,LABELS
import matplotlib.pyplot as plt

def main():
    ART.mkdir(exist_ok=True)
    allrecords={f'replay_p{b:02d}_s125':sorted([json.loads(p.read_text()) for p in (ART/f'replay_p{b:02d}_s125').glob('*/summary.json')],key=lambda r:r['completed']) for b in BUDGETS}
    bounds={}
    for key,targets,lower,title in SPECS:
        arrays=[matrix(rows,key,targets) for rows in allrecords.values()]
        vals=np.concatenate([a[np.isfinite(a)] for a in arrays])
        bounds[key]=(float(vals.min()),float(vals.max())) if lower and len(vals) else (0,1)
    body=['<!doctype html><meta charset="utf-8"><title>Retained-memory budgets</title>',
          '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px}img{max-width:100%}td,th{padding:7px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style>',
          '<h1>Replay with bounded retained memory</h1>',
          '<p>Seed 125; four independent replay runs, 50%, 25%, 10%, and 5% of each task training set retained. Full data accessible only during the active task block. After it ends, replay, mask repair, scaling, and rebuilt direct4 readouts access retained examples only. H=1024, K=256, rank-one atoms; 1500 updates per task, training batch16 plus one replay batch4. Replay receives anonymous masks and examples, including temporary within-block entries; eligibility is 20% &lt; overlap &le; 85%.</p>',
          '<p>Nested frozen-feature facility-location selection balances filter classes. Near-best trajectory masks provide secondary shared-atom coverage. Replay prioritizes overlap, neglected shared atoms, recency and observed loss increases, with 20% exploration over distinct eligible masks. Dataset identity is not passed to replay. This changes replay policy as well as memory size relative to the earlier union study; comparisons between these four runs isolate memory budget.</p>',
          '<p>Repair uses a retained-only fit/validation split (3/1 at four images). The scalar readouts then use all retained images; 512 four-image episodes per task, with just one distinct quartet at four retained images. Four held-out support images at inference; ten support draws; no refinement. Frozen evaluation references and held-out data are scoring-only. Source datasets remain on disk for earlier experiments, but are not reloaded after their block by these workers.</p>',
          '<p>Gray cells are pending. Outlined cells are current tasks; below is backward retention, above is forward behavior. Lower KID/MSE is better; higher acceptance is better. Independent metric families are not combined.</p>',
          '<p>Updated '+datetime.now(timezone.utc).isoformat()+'. <a href="protocol.json">Protocol</a> · <a href="status.json">Status</a> · <a href="transfer.csv">Numbers</a> · <a href="audit.json">Final audit (when complete)</a></p>']
    csvrows=[]
    for tag,rows in allrecords.items():
        body.append(f'<h2>{tag}: {len(rows)}/12 checkpoints evaluated</h2>')
        for key,targets,lower,title in SPECS:
            a=matrix(rows,key,targets);fname=f'xfer_{tag}_{key}.png'
            plot(a,targets,key,tag+': '+title,ART/fname,limits=bounds[key])
            body.append(f'<details{" open" if key in ["kid","acceptance"] else ""}><summary>{title}</summary><img src="{fname}"></details>')
            for i,j in zip(*np.where(np.isfinite(a))):
                k=NAMES.index(targets[j])+1
                csvrows.append(dict(run=tag,completed=int(i),target=targets[j],metric=key,value=float(a[i,j]),relation='forward' if i<k else 'current' if i==k else 'backward'))
        if not rows:continue
        for row in rows:
            stage=ART/tag/row['stage'];preview=stage/'previews';preview.mkdir(exist_ok=True)
            for r in json.loads((stage/'results.json').read_text()):
                if r['draw']==0:
                    source=ART/tag/r['grid'];dest=preview/f'{r["domain"]}.png'
                    if not dest.exists() and source.exists():shutil.copyfile(source,dest)
        latest=rows[-1];stage=ART/tag/latest['stage']
        teachers=json.loads((stage/'teachers.json').read_text())
        if teachers:
            names=list(teachers);m=np.array([teachers[n]['mask'] for n in names]);o=m@m.T/256
            fig,ax=plt.subplots(figsize=(9,8),layout='constrained');im=ax.imshow(o,vmin=0,vmax=1,cmap='Blues')
            labels=[LABELS[NAMES.index(n)] for n in names];ax.set_xticks(range(len(names)),labels,rotation=60,ha='right');ax.set_yticks(range(len(names)),labels)
            for i in range(len(names)):
                for j in range(len(names)):ax.text(j,i,f'{o[i,j]:.0%}',ha='center',va='center',fontsize=8,color='white' if o[i,j]>.55 else 'black')
            ax.set_title(tag+': repaired mask overlap / 256');fig.colorbar(im,ax=ax);fname=f'mask_overlap_{tag}.png';fig.savefig(ART/fname,dpi=130);plt.close(fig)
            body.append(f'<details><summary>Mask overlap</summary><img src="{fname}"></details>')
        body.append('<details><summary>Memory selection</summary><table><tr><th>Task</th><th>Retained</th><th>Full set</th><th>Feature coverage</th><th>Mask associations</th></tr>')
        for p in sorted((ART/tag).glob('memory_*.json')):
            r=json.loads(p.read_text());body.append(f'<tr><td>{r["task"]}</td><td>{r["retained"]}</td><td>{r["source_count"]}</td><td>{r["coverage"]["mean_similarity"]:.3f}</td><td>{r["mask_coverage"]["associations"]}</td></tr>')
        body.append('</table></details><details><summary>Sample grids by checkpoint</summary><table>')
        for row in rows:
            body.append('<tr><th>'+row['stage']+'</th>'+''.join(f'<td><a href="{tag}/{row["stage"]}/previews/{n}.png">{LABELS[i]}</a></td>' for i,n in enumerate(NAMES))+'</tr>')
        body.append('</table></details>')
    if csvrows:
        with (ART/'transfer.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    atomic(ART/'summaries.json',allrecords)
    (ART/'report.html').write_text('\n'.join(body))
    print('REPORT',{k:len(v) for k,v in allrecords.items()},flush=True)

if __name__=='__main__':main()
