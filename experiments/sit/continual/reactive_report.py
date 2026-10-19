"""ReactiveLow results and disjoint-mask audit, generated on the pod."""
import json
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--sd',type=int,default=None)
    args=parser.parse_args()
    suffix='' if args.sd is None else f'_sd{args.sd}'
    threshold=5 if args.sd is None else args.sd
    root=Path('/home/dev/sit_state'); out=Path('/home/dev/workspace/continual/artifacts')/f'reactive_low{suffix}'
    out.mkdir(exist_ok=True)
    shutil.copy2(root/('reactive_protocol.json' if args.sd is None else 'reactive_sweep_protocol.json'),out/'protocol.json')
    runs={}; audits={}
    domains=['photo','monochrome','poster','sketch']
    modes=['cold4','cold40','ic_direct','ic_refine']
    for seed in [123,124,125]:
        name=f'bayes_reactive_low{suffix}_s{seed}'; folder=root/'runs'/name
        if not (folder/'lifecycle.json').exists():
            continue
        history=json.loads((folder/'lifecycle.json').read_text())
        cfg=json.loads((folder/'config.json').read_text())
        records=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()] if (folder/'train.jsonl').exists() else []
        quality=[]
        qroot=folder/'reactive_quality'
        if (qroot/'quality.json').exists():
            quality=json.loads((qroot/'quality.json').read_text())
        else:
            for shard in sorted(qroot.glob('quality_?.json')):
                quality.extend(json.loads(shard.read_text()))
        for q in quality:
            src=Path(q['grid']); dst=out/f'{seed}_{src.name}'
            shutil.copy2(src,dst); q['grid']=dst.name
        blocks=[]; cp=[]
        for block in range(3):
            rr=[r for r in records if r['block']==block]
            if not rr:
                continue
            masks=np.array([r['mask'] for r in rr]); unique=np.unique(masks,axis=0)
            blocks.append({'domain':rr[0]['domain'],'updates':len(rr),'unique_masks':len(unique),
                           'active_atom_union':int(masks.any(0).sum()),'last_mask':masks[-1].tolist(),
                           'changepoints':[r['step'] for r in rr if r['changepoint']],
                           'mask_changes':int(np.any(masks[1:]!=masks[:-1],axis=1).sum())})
            cp.extend({'block':block,'step':r['step'],'loss':r['detector_loss'],
                       'threshold':r['detector_threshold']} for r in rr if r['changepoint'])
        boundaries=np.array([b['last_mask'] for b in blocks])
        overlap=(boundaries@boundaries.T/cfg['k']).tolist() if len(blocks) else []
        audits[str(seed)]={'blocks':blocks,'changepoints':cp,'boundary_overlap':overlap,
                           'replay_losses_observed':sum(r['replay_loss'] is not None for r in records)}
        runs[str(seed)]={'history':history,'quality':quality,'config':cfg}
        if records:
            fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
            steps=[r['block']*cfg['steps']+r['step'] for r in records]
            axes[0].plot(steps,[r['detector_loss'] for r in records],lw=.4,label='pre-update loss')
            axes[0].plot(steps,[np.nan if r['detector_threshold'] is None else r['detector_threshold'] for r in records],lw=.5,label='trigger threshold')
            for r in cp:
                axes[0].axvline(r['block']*cfg['steps']+r['step'],color='red',alpha=.5)
            axes[0].legend(); axes[0].set_ylabel('Velocity MSE'); axes[0].set_title(f'Seed {seed}: loss detector and selected atoms')
            masks=np.array([r['mask'] for r in records])
            axes[1].imshow(masks[::10].T,aspect='auto',interpolation='nearest',cmap='Greys',
                           extent=[0,steps[-1]+1,cfg['H']-.5,-.5],vmin=0,vmax=1)
            axes[1].set_xlabel('Weight update'); axes[1].set_ylabel('Atom index')
            for ax in axes:
                for boundary in [cfg['steps'],2*cfg['steps']]:
                    ax.axvline(boundary,color='tab:orange',ls='--')
            fig.savefig(out/f'detector_masks_{seed}.png',dpi=130); plt.close(fig)
    if not runs:
        raise RuntimeError('No evaluations yet')
    common=min(len(r['history']) for r in runs.values())
    matrices={}
    fig,axes=plt.subplots(1,4,figsize=(20,6),constrained_layout=True)
    for ax,mode in zip(axes,modes):
        values=[]
        for r in runs.values():
            base=[r['history'][0]['domains'][d]['base']['loss'] for d in domains]
            values.append([base]+[[row['domains'][d][mode]['loss'] for d in domains] for row in r['history'][:common]])
        matrix=np.mean(values,axis=0); matrices[mode]=matrix.tolist()
        im=ax.imshow(matrix,vmin=.70,vmax=1.0,cmap='YlOrRd')
        labels=['frozen pretrained base','initial adapter']+[f'after {d}' for d in domains[1:common]]
        ax.set_yticks(range(len(labels)),labels); ax.set_xticks(range(4),domains,rotation=22,ha='right')
        ax.set_title(mode)
        for i in range(matrix.shape[0]):
            for j in range(4):
                ax.text(j,i,f'{matrix[i,j]:.4f}',ha='center',va='center',color='white' if matrix[i,j]>.88 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes,label='Velocity MSE; lower is better',shrink=.65)
    fig.savefig(out/'xfer_reactive_low.png',dpi=140); plt.close(fig)
    final_stats={}; rows=[]
    for mode in modes+['stored']:
        cells=[]; final_stats[mode]={}
        for d in domains:
            vv=[r['history'][-1]['domains'][d][mode]['loss'] for r in runs.values() if mode in r['history'][-1]['domains'][d]]
            if not vv:
                cells.append('—'); continue
            mean=float(np.mean(vv)); sd=float(np.std(vv,ddof=1)) if len(vv)>1 else 0.
            final_stats[mode][d]={'mean':mean,'sd':sd,'n':len(vv)}
            cells.append(f'{mean:.5f} ± {sd:.5f}')
        rows.append('<tr><td>'+mode+'</td><td>'+'</td><td>'.join(cells)+'</td></tr>')
    kid_stats={}; qrows=[]
    for mode in modes+['stored']:
        cells=[]; kid_stats[mode]={}
        for d in domains:
            vv=[q['kid'] for r in runs.values() for q in r['quality'] if q['mode']==mode and q['domain']==d]
            if len(vv)!=3:
                cells.append('—' if mode=='stored' and d=='photo' else 'pending'); continue
            mean=float(np.mean(vv)); sd=float(np.std(vv,ddof=1))
            kid_stats[mode][d]={'mean':mean,'sd':sd,'n':3}; cells.append(f'{mean:.5f} ± {sd:.5f}')
        qrows.append('<tr><td>'+mode+'</td><td>'+'</td><td>'.join(cells)+'</td></tr>')
    forgetting={mode:{} for mode in modes+['stored']}
    if common==4:
        for mode in forgetting:
            for index,d in enumerate(domains[1:],start=1):
                after=[r['history'][index]['domains'][d][mode]['loss'] for r in runs.values()]
                final=[r['history'][-1]['domains'][d][mode]['loss'] for r in runs.values()]
                forgetting[mode][d]={'after_learning':float(np.mean(after)),'final':float(np.mean(final)),
                                     'degradation':float(np.mean(np.array(final)-after))}
    selected_overlap={mode:{d:float(np.mean([
        np.dot(r['history'][-1]['domains'][d][mode]['mask'],audits[s]['blocks'][-1]['last_mask'])/r['config']['k']
        for s,r in runs.items() if audits[s]['blocks']])) for d in domains} for mode in modes}
    summary={'final_loss':final_stats,'kid':kid_stats,'forgetting':forgetting,'mask_audit':audits,
             'inference_overlap_with_final_training_mask':selected_overlap,
             'matrices':matrices,'shared_completed_stages':common}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'results.json').write_text(json.dumps(runs,indent=2))
    hdr='<tr><th>Method</th>'+''.join(f'<th>{d}</th>' for d in domains)+'</tr>'
    audit_html=''.join(f'<h3>Seed {s}</h3><p>Change points: {a["changepoints"]}</p><pre>{json.dumps(a["boundary_overlap"],indent=2)}</pre><img src="detector_masks_{s}.png">' for s,a in audits.items())
    outcome=f'<p>This report contains {common} shared completed lifecycle evaluations across {len(runs)} training seeds.</p>'
    if common==4 and len(runs)==3 and all(len(a['changepoints'])==1 for a in audits.values()):
        outcome=f'''<p><strong>The unchanged SD{threshold} detector did not produce disjoint masks or prevent forgetting in this SiT experiment.</strong> Every seed recorded only initialization, missing both domain switches. Final masks from all three blocks have overlap1: the same16 active atoms were trained through successive domains after startup.</p>
<p>IC-direct monochrome loss rose from {forgetting['ic_direct']['monochrome']['after_learning']:.4f} just after learning to {final_stats['ic_direct']['monochrome']['mean']:.4f} at the end; poster rose from {forgetting['ic_direct']['poster']['after_learning']:.4f} to {final_stats['ic_direct']['poster']['mean']:.4f}. Cold40 ended at {final_stats['cold40']['monochrome']['mean']:.4f} and {final_stats['cold40']['poster']['mean']:.4f}, mostly recovering pretrained-base behavior rather than the learned gains. IC refinement reduced the damage but did not preserve the earlier improvements.</p>
<p>IC still helps sparse recall of the latest domain: sketch MSE is {final_stats['ic_direct']['sketch']['mean']:.4f} with direct4 versus {final_stats['cold4']['sketch']['mean']:.4f} with cold4. This does not establish continual retention. The failure here is that the original reactive detector never allocated a new mask; it does not show that genuinely disjoint masks would fail to protect their atoms.</p>'''
    body=f'''<!doctype html><html><meta charset="utf-8"><title>SiT BayesLoRAReactiveLow</title><style>
body{{font:17px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;background:#f6f8f7;color:#183331}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{padding:9px;border-bottom:1px solid #bfcfca;text-align:left}}img{{max-width:100%}}article{{background:white;padding:15px;margin:15px 0}}select{{font-size:17px;margin:8px}}pre{{font-size:13px}}</style>
<h1>BayesLoRAReactiveLow, SD{threshold}: replay-free SiT, with and without IC</h1>
<p><a href="protocol.json">Exact source settings and architecture mapping</a> · <a href="summary.json">Numeric summary and mask audit</a> · <a href="results.json">Per-example results</a> · <a href="../reactive_sweep/report.html">Threshold comparison</a> · <a href="../report.html">Original study</a></p>
<div id="outcome">{outcome}</div>
<h2>What is being tested</h2><p>The source BayesLoRAReactiveLow preset uses one posterior step with learning rate zero and deterministic masks on stable batches. The loss detector ({threshold} standard deviations, EMA 0.9, source warmup expression) alone triggers 30-step stochastic cold refits. SD5 is the unchanged preset; SD3 and SD1 are user-authorized threshold ablations. The first stable batch installs the deterministic top-k of the refitted logits, which can differ from the stochastic mask used on the trigger batch; subsequent stable masks stay fixed. We do not force task-boundary refits, reserve atoms, or enforce disjoint masks.</p>
<p>All Bayesian parameters are read directly from a byte-identical snapshot of bayesian_adv/config.py. Training uses the inherited learning rate 3e-4 and 1500 updates/domain. The SiT backbone, 64-atom bank, k=16, rank=1, alpha=4, data, batch size and domain order match the preceding SiT study. Thus this is an exact reactive-behavior test on the existing SiT architecture, not the literal toy-model configuration with 4096 atoms, k=256 and rank4. The previous SiT run used learning rate1e-4, so differences from it cannot be attributed solely to the mask schedule.</p>
<p><strong>No replay images are stored or used, and no IC mask repair is performed.</strong> IC uses the same clean-latent PCA representation and known-block archive schedule as the preceding SiT experiment: rank8, reconstruction threshold0.5, four reference images. Training priors and novelty triggers are disabled. Therefore cold and IC inference are paired evaluations of the same trained weights, avoiding an unnecessary duplicate training run. This IC adaptation does not test the source toy router's rank2/threshold0.1 or automatic change-point archive schedule.</p>
<h2>Transfer matrices, including pre-training baselines</h2><p>The first row is the frozen pretrained model with zero adapter. The second is the initial adapter before post-training. Later rows follow monochrome → poster → sketch. Columns are evaluation domains; photo is never post-trained. IC direct retrieves an archived mask; IC refine uses 30 additional support-loss steps. Both use four references and fall back to cold4 when no route matches.</p><img src="xfer_reactive_low.png">
<h2>Final held-out loss</h2><p>Mean ± sample SD across three seeds, 200 held-out images per domain; lower is better. Stored masks use diagnostic domain identity, while cold and IC do not receive domain identity.</p><table>{hdr}{''.join(rows)}</table>
<h2>Final generation quality</h2><p>CleanFID Inception KID, 200 generated images versus400 held-out real images/domain; same seed99233, class labels, Heun50 and CFG4 in every condition. Identical masks reuse the same samples. These small-sample results are not a statistical superiority test.</p><table>{hdr}{''.join(qrows)}</table>
<h2>Matched final-model demo</h2><label>Seed<select id="seed"><option>123</option><option>124</option><option>125</option></select></label><label>Domain<select id="domain"><option>monochrome</option><option>poster</option><option>sketch</option><option>photo</option></select></label><div id="cards"></div>
<h2>Did the masks become disjoint?</h2><p>Overlap matrices use final training masks from successive blocks, with overlap |intersection|/k. Zero means disjoint; one means identical. Orange dashed lines mark the known domain boundaries only for plotting. Red lines mark actual detector triggers. Black pixels indicate selected atoms.</p>{audit_html}
<h2>Limitations</h2><p>Three seeds, synthetic style shifts, ten ImageNet classes, one fixed support set per budget, and a much smaller bank than the original toy configuration. Four references span four classes; forty span ten. Frozen-base recovery through unused atoms is not evidence that learned-domain improvements survived. Known-block IC archives give the router extra scheduling information even though mask search never receives the domain identity or forced boundaries.</p>
<script>const data={json.dumps(runs)};function draw(){{let s=document.getElementById('seed').value,d=document.getElementById('domain').value,r=data[s];if(!r)return;let v=r.history.at(-1).domains[d];document.getElementById('cards').innerHTML=['cold4','cold40','ic_direct','ic_refine','stored'].filter(m=>v[m]).map(m=>{{let q=r.quality.find(x=>x.domain===d&&x.mode===m);return `<article><strong>${{m}}</strong> — MSE ${{v[m].loss.toFixed(5)}}${{q?' · KID '+q.kid.toFixed(5):''}}<br>${{q?'<img src="'+q.grid+'">':'Generation pending'}}</article>`}}).join('')}}document.getElementById('seed').onchange=draw;document.getElementById('domain').onchange=draw;draw();</script></html>'''
    (out/'report.html').write_text(body)
    print(json.dumps({'stages':common,'final_loss':final_stats,'kid':kid_stats,'mask_audit':audits},indent=2))


if __name__=='__main__':
    main()
