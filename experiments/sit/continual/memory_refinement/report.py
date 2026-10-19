"""Incremental, compact paired-refinement report."""
import csv
import sys
from datetime import datetime,timezone
import json
if '--conservative' in sys.argv:
    from .conservative import ART,MODES
else:
    from .run import ART,MODES
from ..memory_budget.common import NAMES,FILTERS,FACES

def fmt(x,percent=False):return '—' if x is None else f'{x:.1%}' if percent else f'{x:.4f}'

def main():
    ART.mkdir(exist_ok=True)
    body=['<!doctype html><meta charset="utf-8"><title>Direct4 versus refinement</title>',
        '<style>body{font:16px system-ui;max-width:1400px;margin:25px auto;padding:20px}img{max-width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}table{border-collapse:collapse}details{margin:18px 0}</style>',
        '<h1>Final retained-memory checkpoints: direct4 versus refinement</h1>',
        '<p>One fixed four-image support draw per target. Same final bank, direct4 initializer, and generation noise across every comparison. Direct4; 10 steps seed0; 30 steps seed0, seed1, seed2. No generator updates, no readout refitting, no selection using test metrics. This is a first-pass comparison, not replication across support sets or training seeds.</p>',
        '<p>Refinement uses the existing softened-mask Bayesian search: four corruptions of the four support images, learning rate .3, KL .05, temperature1. Masks contain 256 of 1024 atoms. Ten- and thirty-step seed0 runs share their optimization prefix. Generation: 200 samples/filter or 64 samples/face or object; paired seed99233.</p>',
        '<p><b>Coverage distance ↓</b>: average distance from each held-out real image to its nearest generated image, in normalized DINO features. <b>Pair distance</b>: mean distance between generated images; more distance can indicate diversity or unwanted drift. These are descriptive feature-space proxies, not calibrated diversity/recall rates. Interpret with acceptance/KID and paired grids. Reference sets: 40 filter images, 24 face/BMW images, 23 anteater images.</p>',
        '<p>Support MSE uses fresh fixed corruptions, separate from search. Test MSE uses the original fixed queries. DINO/style nearest-retained-image distances are available in the CSV; small distance is not proof of memorization. Direct samples reuse the original saved batch, while all feature diagnostics use saved PNGs consistently.</p>',
        '<p>Updated '+datetime.now(timezone.utc).isoformat()+'. <a href="status.json">Status</a> · <a href="results.csv">All numbers</a>. Unfinished source runs are skipped; rerunning the launcher adds them after completion.</p>']
    if '--conservative' in sys.argv:
        body.insert(3,'<p><b>Conservative refinement:</b> original IC refinement parameters, initialized with actual direct4 readout scores; learning rate .3, KL1, temperature1, stochastic mask sampling disabled. Four support images, four fixed transport corruptions per seed. <a href="../artifacts/report.html">Exploratory repair-style comparison</a>.</p>')
        body=[s.replace('Refinement uses the existing softened-mask Bayesian search: four corruptions of the four support images, learning rate .3, KL .05, temperature1.', 'Refinement uses the original IC settings described above.') for s in body]
    else:
        body.append('<p><a href="../artifacts_conservative/report.html">Conservative refinement with the original stronger prior</a> is evaluated separately. The exploratory search here is intentionally more permissive.</p>')
    flat=[]
    for b in [50,25,10,5]:
        tag=f'replay_p{b:02d}_s125';p=ART/tag/'results.json';rows=json.loads(p.read_text()) if p.exists() else []
        body.append(f'<h2>{b}% retained memory: {len(rows)}/55 conditions</h2>')
        if not rows:body.append('<p>Pending.</p>');continue
        body.append(f'<p><a href="{tag}/protocol.json">Exact protocol and source checkpoint</a></p>')
        # Aggregate only tasks with all five comparisons completed, so columns are matched.
        ready=[n for n in NAMES if sum(r['target']==n for r in rows)==len(MODES)]
        body.append('<h3>Matched completed targets</h3><table><tr><th>Mode</th><th>Filter KID ↓</th><th>Face acceptance ↑</th><th>Object acceptance ↑</th><th>Broad object acceptance ↑</th></tr>')
        for mode,steps,seed in MODES:
            rs=[r for r in rows if r['target'] in ready and r['mode']==mode]
            def mean(names,key):
                vs=[r['metrics'][key] for r in rs if r['target'] in names and key in r['metrics']]
                return sum(vs)/len(vs) if vs else None
            body.append('<tr><td>'+mode+'</td><td>'+fmt(mean(FILTERS,'kid'))+'</td><td>'+fmt(mean(FACES,'acceptance'),True)+'</td><td>'+fmt(mean(NAMES[-2:],'acceptance'),True)+'</td><td>'+fmt(mean(NAMES[-2:],'broad_acceptance'),True)+'</td></tr>')
        body.append('</table><p>Completed targets included: '+', '.join(ready)+'.</p>')
        for n in NAMES:
            rs=[r for r in rows if r['target']==n]
            if not rs:continue
            body.append(f'<details open><summary><b>{n}</b></summary><table><tr><th>Mode</th><th>Acceptance ↑</th><th>KID ↓</th><th>Query MSE ↓</th><th>Support MSE ↓</th><th>Atoms replaced</th><th>Coverage distance ↓</th><th>Pair distance</th></tr>')
            for mode,steps,seed in MODES:
                r=next((r for r in rs if r['mode']==mode),None)
                if r is None:continue
                d=r['diagnostics'];m=r['metrics']
                values=[fmt(m.get('acceptance'),True),fmt(m.get('kid')),fmt(r['query_mse']),fmt(r['support_mse']),str(r['atoms_replaced']),fmt(d['dino_real_to_generated_nn']),fmt(d['dino_pair_distance'])]
                body.append('<tr><td>'+mode+'</td>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
                flat.append(dict(budget=b,target=n,mode=mode,steps=r['steps'],seed=r['refinement_seed'],atoms_replaced=r['atoms_replaced'],query_mse=r['query_mse'],support_mse=r['support_mse'],**m,**d))
            body.append('</table>')
            if (ART/tag/n/'paired_comparison.png').exists():
                body.append(f'<p>Rows: direct, refine10 seed0, refine30 seed0, refine30 seed1, refine30 seed2. Columns use paired generation noise.</p><img loading="lazy" src="{tag}/{n}/paired_comparison.png"><details><summary>Four support images</summary><img loading="lazy" src="{tag}/{n}/support4.png"></details>')
            else:
                for r in rs:body.append(f'<p>{r["mode"]}</p><img loading="lazy" src="{tag}/{r["grid"]}">')
            body.append('</details>')
    if flat:
        fields=list(dict.fromkeys(k for row in flat for k in row))
        with (ART/'results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    tmp=ART/'report.tmp';tmp.write_text('\n'.join(body));tmp.replace(ART/'report.html')
    print('REPORT',len(flat),flush=True)

if __name__=='__main__':main()
