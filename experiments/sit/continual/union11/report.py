"""Transfer matrices from genuinely sequential checkpoints, including forward cells."""
import json
import csv
import shutil
from datetime import datetime,timezone
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .common import ART,NAMES,FILTERS,FACES,atomic

LABELS=['Mono','Poster','Sketch','Bush','Powell','Blair','Whoopi','Willis','Kidman','Anteater','BMW']
SPECS=[('kid',FILTERS,True,'Filter KID ↓'),('acceptance',NAMES[3:],False,'Target acceptance ↑ (six-way face / closed-label object)'),
       ('legacy_acceptance',FACES,False,'Legacy three-way face acceptance ↑'),
       ('broad_acceptance',NAMES[-2:],False,'Object acceptance, 1,019 labels ↑'),
       ('query_mse',NAMES,True,'Direct4 fixed-noise velocity MSE ↓'),
       ('stored_query_mse',NAMES,True,'Archived-mask fixed-noise velocity MSE ↓'),
       ('exact_teacher',NAMES,False,'Exact repaired-teacher recovery ↑')]

def matrix(records,key,targets):
    x=np.full((12,len(targets)),np.nan)
    for r in records:
        for j,n in enumerate(targets):
            t=r['targets'][n];v=t.get(key,t['metrics'].get(key))
            if v is not None:x[r['completed'],j]=v
    return x

def plot(x,targets,key,title,path,delta=False,limits=None):
    fig,ax=plt.subplots(figsize=(max(6,len(targets)*1.05+3),8.3),layout='constrained')
    lower=key in ['kid','query_mse','stored_query_mse']
    cmap=plt.get_cmap('RdYlGn_r' if lower else 'RdYlGn').copy();cmap.set_bad('#e8edf1')
    finite=x[np.isfinite(x)]
    if delta:
        bound=max(float(np.max(np.abs(finite))) if len(finite) else .01,.001);lo,hi=-bound,bound
    elif not lower:lo,hi=0,1
    else:lo,hi=(float(finite.min()),float(finite.max())) if len(finite) else (0,1)
    if limits is not None:lo,hi=limits
    im=ax.imshow(x,cmap=cmap,vmin=lo,vmax=hi,aspect='auto')
    ax.set_xticks(range(len(targets)),[LABELS[NAMES.index(n)] for n in targets],rotation=25,ha='right')
    ax.set_yticks(range(12),['Initial']+[f'After {i+1}: {n}' for i,n in enumerate(LABELS)])
    for boundary in [3.5,6.5,9.5]:ax.axhline(boundary,color='#596d7a',lw=.8,alpha=.6)
    for i in range(12):
        for j,n in enumerate(targets):
            v=x[i,j]
            if np.isfinite(v):
                text=f'{v:+.3f}' if delta and lower else f'{v:+.0%}' if delta else f'{v:.3f}' if lower else f'{v:.0%}'
                ax.text(j,i,text,ha='center',va='center',fontsize=9)
            if i==NAMES.index(n)+1:ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor='#172a35',lw=2.2))
    ax.set_title(title+'\nOutlined cells = current task; rows below = backward, above = forward',fontsize=11)
    fig.colorbar(im,ax=ax,shrink=.7);fig.savefig(path,dpi=155);plt.close(fig)

def main():
    ART.mkdir(exist_ok=True)
    parts=['<!doctype html><meta charset="utf-8"><title>Union of four datasets: sequential transfer</title>',
      '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccc}</style>',
      '<h1>Eleven-target union: sequential transfer</h1>',
      '<p>One shared 1,024-rank-one-atom bank, 256 active; 1,500 updates per target. Replay and no-replay, seeds 123 and 125. Exactly four support images; unified frozen DINO/style/CASIA feature pipeline; scalar atom readouts; no refinement.</p>',
      '<p>Each checkpoint is evaluated before the next block starts. Offline training-only repair and readout fitting use only tasks already encountered. Both arms can revisit earlier training images during repair. Future-task cells use the existing direct4 readout without future-task fitting; initial cells use a fixed seeded top-k prior.</p>',
      '<p>Ten support draws per cell, overlapping pools; identical masks share the same paired sample batch. 200 generated images per unique filter mask, 64 otherwise. Class labels/CFG remain the original filter protocol; faces/objects use null class. Frozen evaluator galleries may contain future targets for scoring only.</p>',
      '<p>Gray = pending or not applicable, not zero. Outlined cells are immediately after learning that column target. Values below them show retention; values above them show forward behavior. Companion delta matrices subtract the same target’s diagonal value (backward) or initial value (forward). Different metric families are never averaged together.</p>']
    parts.append('<p><a href="protocol.json">Protocol</a> · <a href="data_check.json">Source-data audit</a> · <a href="evaluator_validation.json">Real-reference evaluator calibration</a> · <a href="retention.csv">Retention CSV</a>. The separate-suite filter and LFW studies used 64 atoms; this union uses 1,024 throughout. Filter class dropout is 10%; null-conditioned face/object dropout is zero. All earlier experiments remain unchanged.</p>')
    parts.append('<p>Report built '+datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')+'. <a href="status.json">Worker status</a> · <a href="transfer.csv">All evaluated transfer cells (CSV)</a>. Two training seeds are descriptive replication, not a confidence interval.</p>')
    parts.append('<p>Exact teacher recovery measures association to the selected repaired mask, not visual quality or optimality of that mask. A lower generation score does not prove that every useful mask has disappeared. Archived-mask MSE is a separate diagnostic, not a generation-quality certificate.</p>')
    allrecords={tag:[json.loads(p.read_text()) for p in sorted((ART/tag).glob('*/summary.json'))]
                for tag in [f'{a}_s{s}' for a in ['noreplay','replay'] for s in [123,125]]}
    bounds={};delta_bounds={}
    for key,targets,lower,title in SPECS:
        vals=[];deltas=[]
        for records in allrecords.values():
            x=matrix(records,key,targets);vals.extend(x[np.isfinite(x)].tolist())
            for j,n in enumerate(targets):
                diag=NAMES.index(n)+1
                for i in range(12):
                    d=x[i,j]-x[0 if i<diag else diag,j]
                    if np.isfinite(d):deltas.append(abs(d))
        bounds[key]=(min(vals),max(vals)) if lower and vals else (0,1)
        extent=max(deltas+[.001]);delta_bounds[key]=(-extent,extent)
    transfer={};retention=[]
    for arm in ['noreplay','replay']:
        arm_records=[]
        for seed in [123,125]:
            tag=f'{arm}_s{seed}';folder=ART/tag
            records=allrecords[tag];arm_records.append(records)
            parts.append(f'<h2>{tag}: {len(records)}/12 checkpoints evaluated</h2>')
            for key,targets,lower,title in SPECS:
                x=matrix(records,key,targets);fname=f'xfer_{tag}_{key}.png'
                plot(x,targets,key,f'{tag}: {title}',ART/fname,limits=bounds[key])
                parts.append(f'<details{" open" if key in ["kid","acceptance","broad_acceptance"] else ""}><summary>{title}</summary><img src="{fname}"></details>')
                transfer.setdefault(tag,{})[key]=dict(targets=targets,values=[[float(v) if np.isfinite(v) else None for v in row] for row in x])
                for j,n in enumerate(targets):
                    diag=NAMES.index(n)+1;baseline=x[diag,j]
                    later=x[diag:,j];finite=later[np.isfinite(later)]
                    if not np.isfinite(baseline):continue
                    final=x[-1,j]
                    worst=(max(finite) if lower else min(finite))
                    retention.append(dict(run=tag,target=n,metric=key,lower_is_better=lower,
                        initial=float(x[0,j]) if np.isfinite(x[0,j]) else None,
                        after_learning=float(baseline),final=float(final) if np.isfinite(final) else None,
                        final_drop=float(final-baseline if lower else baseline-final) if np.isfinite(final) else None,
                        worst_drop_so_far=float(worst-baseline if lower else baseline-worst),
                        final_complete=bool(np.isfinite(final))))
            if records:
                # Only these compact first-draw previews sync to the laptop.
                for stage_record in records:
                    stage_folder=folder/stage_record['stage']
                    preview=stage_folder/'previews';preview.mkdir(exist_ok=True)
                    for rr in json.loads((stage_folder/'results.json').read_text()):
                        if rr['draw']==0:
                            dest=preview/f'{rr["domain"]}.png'
                            if not dest.exists():shutil.copyfile(folder/rr['grid'],dest)
                latest=max(records,key=lambda r:r['completed']);stage=latest['stage']
                rr=json.loads((folder/stage/'results.json').read_text())
                parts.append('<details><summary>Latest evaluated stage: support and unselected generated samples</summary>')
                for r in rr:
                    if r['draw']:continue
                    n=r['domain'];parts.append(f'<h3>{html.escape(n)} — {r["relation"]}</h3><p>Four supports:</p><img width="520" src="{tag}/{stage}/{n}_support4.png"><p>Generated:</p><img src="{tag}/{stage}/previews/{n}.png">')
                parts.append('</details>')
                parts.append('<details><summary>Sample grids at every evaluated checkpoint</summary><p>Each link opens the unselected first generated batch for that stage and target, using the same paired noise. Supports are the first four-image draw.</p><table><tr><th>Checkpoint</th>'+''.join(f'<th>{n}</th>' for n in LABELS)+'</tr>')
                for stage_record in sorted(records,key=lambda r:r['completed']):
                    stage_name=stage_record['stage']
                    episodes=json.loads((folder/stage_name/'results.json').read_text())
                    first={r['domain']:r for r in episodes if r['draw']==0}
                    parts.append(f'<tr><td>{stage_name}</td>'+''.join(f'<td><a href="{tag}/{stage_name}/previews/{n}.png">Samples</a></td>' for n in NAMES)+'</tr>')
                parts.append('</table></details>')
                teachers=json.loads((folder/stage/'teachers.json').read_text())
                if teachers:
                    tn=list(teachers);masks=np.array([teachers[n]['mask'] for n in tn])
                    overlap=masks@masks.T/256
                    fig,ax=plt.subplots(figsize=(max(6,len(tn)*.65+2),max(5,len(tn)*.6+1)),layout='constrained')
                    im=ax.imshow(overlap,vmin=0,vmax=1,cmap='Blues')
                    short=[LABELS[NAMES.index(n)] for n in tn]
                    ax.set_xticks(range(len(tn)),short,rotation=35,ha='right');ax.set_yticks(range(len(tn)),short)
                    for i in range(len(tn)):
                        for j in range(len(tn)):
                            ax.text(j,i,f'{overlap[i,j]:.0%}',ha='center',va='center',color='white' if overlap[i,j]>.55 else 'black',fontsize=9)
                    ax.set_title(f'{tag}, after {latest["completed"]} tasks: repaired-mask overlap\nH=1,024; K=256; shared atoms / 256')
                    fig.colorbar(im,ax=ax,shrink=.7);name=f'mask_overlap_{tag}.png';fig.savefig(ART/name,dpi=150);plt.close(fig)
                    occupied=int((masks.sum(0)>0).sum())
                    parts.append(f'<details><summary>Mask overlap and bank use</summary><p>{occupied}/1,024 atoms occur in at least one of these {len(tn)} repaired masks. Every mask has 256 active atoms. This is a descriptive snapshot, not a capacity guarantee.</p><img src="{name}"></details>')
        parts.append(f'<h2>{arm}: paired-seed mean transfer</h2>')
        for key,targets,lower,title in SPECS:
            xx=np.stack([matrix(rr,key,targets) for rr in arm_records]);counts=np.isfinite(xx).sum(0)
            mean=np.divide(np.nansum(xx,0),counts,out=np.full(xx.shape[1:],np.nan),where=counts==2)
            filename=f'xfer_{arm}_mean_{key}.png';plot(mean,targets,key,f'{arm}, both seeds: {title}',ART/filename,limits=bounds[key])
            parts.append(f'<details{" open" if key in ["kid","acceptance","broad_acceptance"] else ""}><summary>{title} — both seeds required per cell</summary><img src="{filename}"></details>')
            delta=mean.copy()
            for j,n in enumerate(targets):
                diag=NAMES.index(n)+1
                for i in range(12):delta[i,j]-=mean[0 if i<diag else diag,j]
            plot(delta,targets,key,f'{arm}: {title}\nChange from initial (forward) / diagonal (backward)',ART/f'xfer_{arm}_delta_{key}.png',True,delta_bounds[key])
            deltafile=f'xfer_{arm}_delta_{key}.png'
            parts.append(f'<details><summary>{title}: change from baseline</summary><img src="{deltafile}"></details>')
    parts.append('<h2>Retention summary</h2><p>Positive drop means worse; negative means improvement. Pending final values are not estimates. Worst drop includes all completed checkpoints from first learning onward. Tables do not combine KID, recognition, and MSE.</p>')
    for key,targets,lower,title in SPECS:
        parts.append(f'<details><summary>{title}: per-run retention</summary><table><tr><th>Run</th><th>Target</th><th>After learning</th><th>Final</th><th>Final drop</th><th>Worst drop so far</th></tr>')
        def fmt(v):return 'pending' if v is None else f'{v:.4f}' if lower else f'{v:.1%}'
        for r in retention:
            if r['metric']==key:
                parts.append(f'<tr><td>{r["run"]}</td><td>{r["target"]}</td>'+''.join(f'<td>{fmt(r[k])}</td>' for k in ['after_learning','final','final_drop','worst_drop_so_far'])+'</tr>')
        parts.append('</table></details>')
    atomic(ART/'retention.json',retention)
    if retention:
        with (ART/'retention.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(retention[0]));writer.writeheader();writer.writerows(retention)
    atomic(ART/'transfer.json',transfer);atomic(ART/'summaries.json',allrecords)
    with (ART/'transfer.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['run','completed_blocks','after','target','relation','metric','value'])
        writer.writeheader()
        for tag,metrics in transfer.items():
            for key,entry in metrics.items():
                for i,row in enumerate(entry['values']):
                    for n,v in zip(entry['targets'],row):
                        if v is None:continue
                        j=NAMES.index(n)
                        relation='initial' if i==0 else 'forward' if i<=j else 'current' if i==j+1 else 'backward'
                        writer.writerow(dict(run=tag,completed_blocks=i,after='initial' if i==0 else NAMES[i-1],
                                             target=n,relation=relation,metric=key,value=v))
    (ART/'report.html').write_text('\n'.join(parts))
    print('REPORT', {k:len(v) for k,v in allrecords.items()},flush=True)

if __name__=='__main__':main()
