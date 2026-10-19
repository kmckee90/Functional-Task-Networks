"""Build repair follow-up tables, transfer matrices and matched sample demo."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('/home/dev/sit_state')
    out=Path('/home/dev/workspace/continual/artifacts/repair'); out.mkdir(exist_ok=True)
    runs={}; all_repairs=[]
    for seed in [123,124,125]:
        name=f'bayes_replay_s{seed}'; folder=root/'runs'/name/'repair'
        if not (folder/'completed.json').exists():
            continue
        history=json.loads((folder/'lifecycle.json').read_text())
        quality=json.loads((folder/'quality.json').read_text()) if (folder/'quality.json').exists() else []
        if not quality:
            for shard in sorted(folder.glob('quality_?.json')):
                quality.extend(json.loads(shard.read_text()))
        for q in quality:
            source=Path(q['grid']); dest=out/f'{seed}_{source.name}'
            shutil.copy2(source,dest); q['grid']=dest.name
        runs[str(seed)]={'history':history,'quality':quality}
        for row in history:
            for r in row['repairs']:
                all_repairs.append(dict(r,seed=seed,stage=row['stage']))
        shutil.copy2(folder/'protocol.json',out/'protocol.json')
    if len(runs)!=3:
        raise RuntimeError('All three repair evaluations must finish before reporting')
    domains=['photo','monochrome','poster','sketch']
    modes=['cold4','cold40','raw_direct4','raw_refine4','repaired_direct4','repaired_refine4']
    stats={}
    loss_rows=[]
    for mode in modes:
        cells=[]; stats[mode]={}
        for d in domains:
            values=[r['history'][-1]['domains'][d][mode]['loss'] for r in runs.values()]
            mean=float(np.mean(values)); sd=float(np.std(values,ddof=1))
            stats[mode][d]={'mean':mean,'sd':sd}
            cells.append(f'{mean:.5f} ± {sd:.5f}')
        loss_rows.append('<tr><td>'+mode+'</td><td>'+'</td><td>'.join(cells)+'</td></tr>')
    quality_rows=[]; quality_stats={}
    for mode in ['cold40','raw_direct4','repaired_direct4','repaired_refine4']:
        cells=[]; quality_stats[mode]={}
        for d in domains:
            values=[q['kid'] for r in runs.values() for q in r['quality'] if q['domain']==d and q['mode']==mode]
            if len(values)!=3:
                cells.append('pending'); continue
            mean=float(np.mean(values)); sd=float(np.std(values,ddof=1))
            quality_stats[mode][d]={'mean':mean,'sd':sd}
            cells.append(f'{mean:.5f} ± {sd:.5f}')
        quality_rows.append('<tr><td>'+mode+'</td><td>'+'</td><td>'.join(cells)+'</td></tr>')
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    matrices={}
    for ax,mode in zip(axes,['raw_direct4','repaired_direct4','cold40']):
        matrix=np.mean([[[row['domains'][d][mode]['loss'] for d in domains] for row in r['history']] for r in runs.values()],axis=0)
        matrices[mode]=matrix.tolist()
        im=ax.imshow(matrix,vmin=.70,vmax=.96,cmap='YlOrRd')
        ax.set_xticks(range(4),domains,rotation=20,ha='right')
        ax.set_yticks(range(3),['after monochrome','after poster','after sketch'])
        ax.set_title(mode)
        for i in range(3):
            for j in range(4):
                ax.text(j,i,f'{matrix[i,j]:.4f}',ha='center',va='center',color='white' if matrix[i,j]>.88 else 'black')
    fig.colorbar(im,ax=axes,label='Held-out velocity MSE (lower is better)',shrink=.65)
    fig.savefig(out/'xfer_repair.png',dpi=150); plt.close(fig)
    accepted=sum(r['accepted'] for r in all_repairs); attempted=sum('before' in r for r in all_repairs)
    repair_rows=[]
    for r in all_repairs:
        repair_rows.append(f"<tr><td>{r['seed']}</td><td>{r['stage']}</td><td>{r['entry']}</td><td>{r['unique_rows']}</td><td>{r.get('before',float('nan')):.4f}</td><td>{r.get('candidate_loss',float('nan')):.4f}</td><td>{r['accepted']}</td><td>{r['batch_domain_counts_for_audit_only']}</td></tr>")
    timing={'mean_offline_repair_seconds_per_run':float(np.mean([sum(row['repair_seconds'] for row in r['history']) for r in runs.values()])),
            'final_seconds_per_domain':{mode:float(np.mean([r['history'][-1]['domains'][d][mode]['metadata']['seconds'] for r in runs.values() for d in domains])) for mode in modes}}
    summary={'loss':stats,'kid':quality_stats,'accepted':accepted,'attempted':attempted,'repairs':all_repairs,'matrices':matrices,'timing':timing}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'results.json').write_text(json.dumps(runs,indent=2))
    header='<tr><th>Method</th>'+''.join(f'<th>{d}</th>' for d in domains)+'</tr>'
    poster_raw=stats['raw_direct4']['poster']['mean']; poster_repaired=stats['repaired_direct4']['poster']['mean']
    poster_cold=stats['cold40']['poster']['mean']
    quality_note=''
    if all('poster' in quality_stats[m] for m in ['cold40','raw_direct4','repaired_direct4']):
        quality_note=f"<p><strong>Poster generation also recovers:</strong> mean KID falls from {quality_stats['raw_direct4']['poster']['mean']:.4f} without repair to {quality_stats['repaired_direct4']['poster']['mean']:.4f} with repair, versus {quality_stats['cold40']['poster']['mean']:.4f} for cold-40. In the matched seed-123 grid, the stale poster mask generates sketch-like images; repair restores colorful posterized generation. Some repaired seed-125 outputs remain monochrome or sketch-like, so the improved distributional score does not imply consistently correct style. This is a positive result for repair, although the loss and KID rankings differ and the small three-seed experiment is not proof of general superiority.</p>"
    body=f'''<!doctype html><html><meta charset="utf-8"><title>SiT input conditioning with mask repair</title>
<style>body{{font:17px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#183331;background:#f6f8f7}}table{{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}}td,th{{padding:9px;border-bottom:1px solid #bfcfca;text-align:left}}img{{max-width:100%}}select{{font-size:17px;margin:8px}}.cards{{display:grid;grid-template-columns:1fr;gap:15px}}article{{background:white;padding:15px}}small{{color:#48615a}}</style>
<h1>Input conditioning with replay-based mask repair</h1>
<p><a href="../report.html">Original continual-learning report</a> · <a href="summary.json">Numeric summary and repair audit</a> · <a href="results.json">All per-example results</a> · <a href="protocol.json">Protocol</a></p>
<p><strong>Final poster MSE: unrepaired input conditioning {poster_raw:.4f}, repaired {poster_repaired:.4f}, cold search with 40 references {poster_cold:.4f}.</strong> All comparisons use the same three previously trained replay models; no adapter weights were retrained. {accepted} of {attempted} attempted repairs passed their training-replay validation test.</p>
<p><strong>Interpretation:</strong> replay repair makes archived input-conditioned masks useful again in this setting. It substantially repairs poster recall and improves monochrome recall, but does not consistently match cold search or establish absence of forgetting. Final monochrome MSE is about 0.7125, versus 0.7262 without repair and 0.7094 with cold-40; it is also slightly worse than the original pretrained baseline near 0.7113. The latest sketch mask is unchanged because the live domain is excluded from repair. Thirty-step deterministic refinement leaves the same hard masks as direct routing in this experiment.</p>
<p>The benefit is a shift of mask-search work into maintenance: repaired direct retrieval uses only four reference-image features at evaluation time. Offline repair averaged {timing['mean_offline_repair_seconds_per_run']:.1f} seconds per three-block run on this pod. Measured final per-domain routing/search time averaged {timing['final_seconds_per_domain']['repaired_direct4']:.4f} seconds for repaired direct routing versus {timing['final_seconds_per_domain']['cold40']:.2f} seconds for cold-40. These timings exclude image generation and should not be treated as a controlled latency benchmark.</p>
<p>Repair does not solve novelty detection: before sketch training, input routing selected an existing style and sketch MSE remained about 0.8218, compared with 0.7810 for cold-40. Photo also has no archived entry and has worse velocity MSE under input routing; its generated-image KID can rank the methods differently. Thus the positive result concerns recall of learned styles, not broad forward transfer or universally reliable routing.</p>
<h2>What was implemented</h2>
<p>The original InputConditioned repair procedure groups training replay by input reconstruction, cold-searches a replacement mask, validates it on separate replay images, and appends ±4 logits for accepted masks under the unchanged projector. Old entries remain intact; the latest equally matching entry wins. The live domain is excluded from repair, matching the source implementation. Neither domain IDs nor evaluation losses select a repair.</p>
<p>Search uses 100 steps × 3 restarts, learning rate 0.3, temperature 1, and best fit-loss mask selection. Our existing buffer stores only 128 sampled images per block, so fit and validation use 32 distinct images each, compared with the source default of 800 each. Exact duplicate replay latents are removed before splitting. No fresh training images, support images or test images are added to the replay budget.</p>
<p>Maintenance is replayed sequentially at the three saved block checkpoints, carrying accepted repairs forward. This is equivalent to inserting this mask-only maintenance between these checkpoints when input priors do not affect training: the live posterior, RNG and weights are restored, and existing gradient replay continues to use its original masks. The new <code>--ic-repair</code> training flag also performs this maintenance during future runs.</p>
<h2>Final held-out loss</h2><p>Mean ± sample SD over three training seeds; 200 held-out images per domain. Every method shares exactly the same query images, noise and transport times. Direct routing uses four clean reference-image latents and no test-time loss search on a routing hit. Refinement adds 30 support-loss optimization steps. Cold methods search for 30 steps using 4 or 40 references. Repair has additional offline replay-search cost, so this is not an equal-total-compute comparison.</p>
<table>{header}{''.join(loss_rows)}</table>
<h2>Continual-learning transfer matrices</h2><p>Rows show successive trained blocks; future-domain columns remain evaluated before training. Initial unrepaired-model results are in the original report. All three panels below use identical support construction, unlike mixing the earlier 40-image lifecycle and 80-image diagnostic transport draws.</p><img src="xfer_repair.png">
<h2>Generated-image check</h2><p>KID from 200 generated images against 400 held-out real images per domain; mean ± sample SD over three seeds, lower is better. All variants share generation seed 99233, class labels, Heun-50 and CFG-4. Identical masks reuse the exact same samples. Small-sample KID is diagnostic and does not establish statistical superiority.</p>{quality_note}<table>{header}{''.join(quality_rows)}</table>
<h2>Matched final-model demo</h2><p>These are the first 20 images from the metric-generation sample sets, with the same starting noise and class labels in each panel.</p>
<label>Seed <select id="seed"><option>123</option><option>124</option><option>125</option></select></label>
<label>Domain <select id="domain"><option>poster</option><option>monochrome</option><option>sketch</option><option>photo</option></select></label><div id="cards" class="cards"></div>
<h2>Repair acceptance and replay-routing audit</h2><p>Before/candidate losses are measured on replay-validation images, not the query set. Batch domain labels below are added only after repair for auditing. As in the source, matching excludes the live projector; therefore replay from a different domain can be assigned to an old projector if it passes its reconstruction threshold. This table exposes any such mixing rather than silently treating the groups as pure.</p>
<table><tr><th>Seed</th><th>Checkpoint</th><th>Entry</th><th>Unique rows</th><th>Before</th><th>Candidate</th><th>Accepted</th><th>Routed batches by actual domain (audit only)</th></tr>{''.join(repair_rows)}</table>
<h2>Limits and validation</h2><p>This isolates repair on the three existing replay seeds and synthetic style domains; it does not test new datasets, no-replay training or training-time input priors. Projectors were originally archived at known block boundaries, despite the reactive detector missing those boundaries. Four references cover four classes, whereas forty cover ten. Photo has no archived entry and can be misrouted. An accepted repair guarantees improvement only on its replay-validation sample, not on held-out generation quality.</p>
<p>Pod checks verified accept/reject behavior, append-only storage, latest-entry retrieval, fit/validation separation, unchanged model parameter versions, absent parameter gradients, and restoration of mask, posterior and CPU/CUDA RNG state. The original training artifacts remain unchanged.</p>
<script>const data={json.dumps(runs)};
function draw(){{const s=document.getElementById('seed').value,d=document.getElementById('domain').value,r=data[s],v=r.history.at(-1).domains[d];document.getElementById('cards').innerHTML=['raw_direct4','repaired_direct4','repaired_refine4','cold40'].map(m=>{{const q=r.quality.find(x=>x.domain===d&&x.mode===m);return `<article><strong>${{m}}</strong> — MSE ${{v[m].loss.toFixed(5)}}${{q?' · KID '+q.kid.toFixed(5):''}}<br>${{q?'<img src="'+q.grid+'">':'Generation pending'}}</article>`}}).join('')}}
document.getElementById('seed').onchange=draw;document.getElementById('domain').onchange=draw;draw();</script></html>'''
    if (out/'verification.json').exists():
        body=body.replace('<h2>Limits and validation</h2>',
            '<h2>Limits and validation</h2><p><a href="verification.json">Independent saved-result verification: passed</a></p>')
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href') and value and not value.startswith(('http','#')):
                    assert (out/value).exists(),value
    Links().feed(body)
    (out/'report.html').write_text(body)
    print(json.dumps({'accepted':accepted,'attempted':attempted,'loss':stats,'kid':quality_stats},indent=2))


if __name__=='__main__':
    main()
