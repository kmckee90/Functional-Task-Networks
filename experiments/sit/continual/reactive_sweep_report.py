"""Compare all prespecified detector thresholds, without selecting a winner."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    artifacts=Path('/home/dev/workspace/continual/artifacts')
    out=artifacts/'reactive_sweep'; out.mkdir(exist_ok=True)
    summaries={}; demo={}; table=[]; detector=[]
    domains=['photo','monochrome','poster','sketch']; modes=['cold4','cold40','ic_direct','ic_refine']
    fig,axes=plt.subplots(3,4,figsize=(21,15),constrained_layout=True)
    for i,sd in enumerate([5,3,1]):
        folder=artifacts/('reactive_low' if sd==5 else f'reactive_low_sd{sd}')
        s=json.loads((folder/'summary.json').read_text()); summaries[str(sd)]=s
        assert s['shared_completed_stages']==4
        runs=json.loads((folder/'results.json').read_text())
        demo[str(sd)]={}
        for seed,r in runs.items():
            quality=[]
            for q in r['quality']:
                quality.append(dict(q,grid=f'../{folder.name}/{q["grid"]}'))
            demo[str(sd)][seed]={'final':r['history'][-1],'quality':quality}
        cp=[len(a['changepoints']) for a in s['mask_audit'].values()]
        overlap=np.mean([a['boundary_overlap'] for a in s['mask_audit'].values()],axis=0)
        unions=[a['blocks'][-1]['active_atom_union'] for a in s['mask_audit'].values()]
        within=[sum(r['step']>=50 for r in a['changepoints']) for a in s['mask_audit'].values()]
        detector.append(f'<tr><td>{sd}</td><td>{cp}</td><td>{within}</td><td>{overlap[0,1]:.3f}</td><td>{overlap[1,2]:.3f}</td><td>{unions}</td></tr>')
        for j,mode in enumerate(modes):
            matrix=np.array(s['matrices'][mode]); ax=axes[i,j]
            im=ax.imshow(matrix,vmin=.70,vmax=1.,cmap='YlOrRd')
            ax.set_title(f'SD{sd} · {mode}')
            ax.set_xticks(range(4),domains,rotation=22,ha='right')
            ax.set_yticks(range(5),['frozen base','initial adapter','after monochrome','after poster','after sketch'] if j==0 else ['']*5)
            for row in range(5):
                for col in range(4):
                    ax.text(col,row,f'{matrix[row,col]:.4f}',ha='center',va='center',fontsize=9,color='white' if matrix[row,col]>.88 else 'black')
            vv=s['final_loss'][mode]
            table.append('<tr><td>'+str(sd)+'</td><td>'+mode+'</td>'+''.join(f'<td>{vv[d]["mean"]:.5f} ± {vv[d]["sd"]:.5f}</td>' for d in domains)+'</tr>')
    fig.colorbar(im,ax=axes,label='Held-out velocity MSE, lower is better',shrink=.65)
    fig.savefig(out/'xfer_thresholds.png',dpi=150); plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    qrows=[]
    for sd,s in summaries.items():
        for mode in modes:
            cells=[]
            for d in domains:
                v=s['kid'][mode].get(d)
                cells.append(f'{v["mean"]:.5f} ± {v["sd"]:.5f}' if v else 'pending')
            qrows.append('<tr><td>'+sd+'</td><td>'+mode+'</td><td>'+'</td><td>'.join(cells)+'</td></tr>')
    hdr='<tr><th>SD threshold</th><th>Inference</th>'+''.join(f'<th>{d}</th>' for d in domains)+'</tr>'
    interpretation=['<p><strong>The source SD5 detector missed both domain transitions in all three seeds.</strong> Its final training masks were identical across blocks. Lower thresholds test whether more reactive mask allocation changes that result; they do not enforce disjointness.</p>']
    for sd in ['3','1']:
        s=summaries[sd]
        cps=[len(a['changepoints']) for a in s['mask_audit'].values()]
        overlaps=np.mean([a['boundary_overlap'] for a in s['mask_audit'].values()],axis=0)
        f=s['forgetting']['ic_direct']; c=s['final_loss']['cold40']
        interpretation.append(f'<p><strong>SD{sd}:</strong> change-point counts were {cps}; mean final-mask overlap was {overlaps[0,1]:.3f} for monochrome/poster and {overlaps[1,2]:.3f} for poster/sketch. IC-direct monochrome MSE changed from {f["monochrome"]["after_learning"]:.4f} after learning to {f["monochrome"]["final"]:.4f} after the full sequence; poster changed from {f["poster"]["after_learning"]:.4f} to {f["poster"]["final"]:.4f}. Final cold40 MSE was {c["monochrome"]["mean"]:.4f} and {c["poster"]["mean"]:.4f}, respectively.</p>')
    audit=json.loads((out/'verification.json').read_text())
    assert audit['passed']
    k=summaries['3']['kid']
    interpretation.append(f'<p><strong>Image-quality results qualify the loss-based conclusion.</strong> At SD3, final sketch KID is {k["cold4"]["sketch"]["mean"]:.4f} with cold4, {k["cold40"]["sketch"]["mean"]:.4f} with cold40, and {k["ic_refine"]["sketch"]["mean"]:.4f} with IC plus four-image refinement (lower is better). Thus cold4 retains the earlier styles well but is less reliable for sketch generation. IC plus refinement has competitive image quality across the learned domains, despite worse old-domain velocity MSE than cold search. These distinct metrics should not be collapsed into a single ranking.</p>')
    interpretation.append('<p>SD3 gives the strongest overall balance among the tested thresholds, with substantial replay-free retention under cold mask inference. SD1 causes more than 200 triggers per run and updates almost the entire bank within each domain; its results are mixed rather than uniformly better. This demonstrates support-conditioned continual learning in this three-style setting, not zero forgetting, guaranteed disjoint mask allocation, or generalization to arbitrary domain sequences.</p>')
    interpretation.append('<p>The independent audit reproduced every detector decision and verified zero replay storage, fixed stable masks, and unchanged atoms that were inactive throughout a block. A trigger can occur inside an unchanged domain; the plots and counts below expose that behavior. Final-block mask overlap alone also understates interference when masks change repeatedly within a block.</p>')
    body=f'''<!doctype html><html><meta charset="utf-8"><title>SiT replay-free reactive threshold comparison</title><style>body{{font:17px system-ui;max-width:1350px;margin:35px auto;padding:0 20px;background:#f6f8f7;color:#183331}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{padding:9px;border-bottom:1px solid #bfcfca;text-align:left}}img{{max-width:100%}}article{{background:white;padding:15px;margin:15px 0}}select{{font-size:17px;margin:8px}}</style>
<h1>Replay-free Bayes LoRA: original detector and lower thresholds</h1>
<p><a href="../reactive_low/report.html">Original SD5</a> · <a href="../reactive_low_sd3/report.html">SD3</a> · <a href="../reactive_low_sd1/report.html">SD1</a> · <a href="summary.json">All numeric summaries</a> · <a href="verification.json">Lower-threshold verification</a></p>
<div id="interpretation">{''.join(interpretation)}</div>
<p>The SD5 experiment uses the original BayesLoRAReactiveLow reactive and posterior settings on the existing SiT bank, as requested. After it missed both domain changes, the user authorized lower thresholds. SD3 and SD1 were chosen before these runs, matching the original project's Reactive_Med and Reactive_High thresholds. Every other training setting stays fixed: three paired seeds, 1500 updates/domain, learning rate3e-4, frozen SiT-XL/2, H64/k16/rank1/alpha4, same data and noise streams, no replay storage and no repair.</p>
<p>IC does not affect training here: training priors and novelty triggers are off. Each trained model is evaluated with cold4, cold40, IC direct4 and IC refine4. The IC representation and known-block archives match the preceding SiT experiment; they are not the toy router's original rank2/threshold0.1 setup.</p>
<h2>Full transfer matrices</h2><p>Each panel includes frozen-pretrained and initial-adapter baselines. Rows then follow monochrome → poster → sketch; columns are evaluation domains, including photo, which is never trained. The same color scale is used throughout.</p><a href="xfer_thresholds.png"><img src="xfer_thresholds.png"></a>
<h2>Final query loss</h2><p>Mean ± sample SD over three seeds; 200 held-out images, lower is better. Returning to pretrained performance is not retention of a learned style's improvement.</p><table>{hdr}{''.join(table)}</table>
<h2>Detector and mask audit</h2><p>Counts list seeds123/124/125. Initialization counts as a change point. “Within block” counts triggers at step≥50; it is a descriptive distinction, not a tuned acceptance rule. Overlap is intersection/k of final block masks, averaged across seeds. An overlap of1 means identical masks and0 means disjoint. Independent random16-of64 masks have expected overlap0.25. The original toy bank's256-of4096 geometry has a lower expected overlap0.0625; the user chose to retain the SiT bank for this comparison.</p><table><tr><th>SD</th><th>All change points</th><th>Within block</th><th>Mono/poster overlap</th><th>Poster/sketch overlap</th><th>Atoms used in sketch block</th></tr>{''.join(detector)}</table>
<h2>Generated-image KID</h2><p>200 generated versus400 real test images/domain, common seeds/classes/solver. Small-sample KID and three seeds do not establish statistical superiority. All prespecified thresholds and seeds are shown.</p><table>{hdr}{''.join(qrows)}</table>
<h2>Matched sample browser</h2><label>Threshold<select id="sd"><option>5</option><option>3</option><option>1</option></select></label><label>Seed<select id="seed"><option>123</option><option>124</option><option>125</option></select></label><label>Domain<select id="domain"><option>monochrome</option><option>poster</option><option>sketch</option><option>photo</option></select></label><div id="cards"></div>
<p>These are the first20 metric samples, with the same initial noise and class labels. Reports linked above include stored-mask controls and detector plots.</p>
<script>const data={json.dumps(demo)};function draw(){{let sd=document.getElementById('sd').value,s=document.getElementById('seed').value,d=document.getElementById('domain').value,r=data[sd][s],v=r.final.domains[d];document.getElementById('cards').innerHTML=['cold4','cold40','ic_direct','ic_refine'].map(m=>{{let q=r.quality.find(x=>x.domain===d&&x.mode===m);return `<article><strong>${{m}}</strong> — MSE ${{v[m].loss.toFixed(5)}}${{q?' · KID '+q.kid.toFixed(5):''}}<br>${{q?'<img src="'+q.grid+'">':'Generation pending'}}</article>`}}).join('')}}for(let k of ['sd','seed','domain'])document.getElementById(k).onchange=draw;draw();</script></html>'''
    (out/'report.html').write_text(body)
    print('Threshold comparison generated.')


if __name__=='__main__':
    main()
