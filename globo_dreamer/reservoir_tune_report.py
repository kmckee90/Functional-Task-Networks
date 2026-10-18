"""Publish isolated reservoir memory tuning results from pod artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/dev/globo_dreamer_state/diagnostics/reservoir_tuning')
LAGS=(0,1,2,4,8,16,32,64)


def main():
    complete=json.loads((ROOT/'complete.json').read_text())
    verified=json.loads((ROOT/'verified.json').read_text())
    held=complete['heldout'];checks=verified['results']
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for row,c in enumerate((64,128)):
        for col,(domain,key,title) in enumerate((('noise','image','Independent image recall'),
                ('noise','tag','Independent feedback recall'),('real_tagged','tag','Feedback recall among real Globo frames'))):
            ax=axes[row,col]
            for mode,color in (('original','#777777'),('sparse','#167cba'),('normalized_dense','#e57923')):
                data=[r['controlled'] for r in checks if r['channels']==c and r['mode']==mode] if mode=='normalized_dense' else [r for r in held if r['channels']==c and r['baseline']==(mode=='original')]
                values=np.array([[r[domain]['scores'][f'{key}_{l}']['r2'] for l in LAGS] for r in data])
                ax.plot(LAGS,values.mean(0),'o-',label=mode.replace('_',' '),color=color)
                ax.fill_between(LAGS,values.min(0),values.max(0),alpha=.15,color=color)
            ax.set_title(f'{c} channels: {title}');ax.set_xlabel('Delay in observations');ax.set_ylabel('Held-out R²')
            ax.set_xscale('symlog',linthresh=1);ax.set_xticks(LAGS,labels=LAGS);ax.set_ylim(-.15,1.04)
            ax.axhline(0,color='gray',ls='--',lw=1);ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(ROOT/'memory.png',dpi=150);fig.savefig(ROOT/'memory.svg');plt.close(fig)
    params=''.join(f'<tr><td>{r["channels"]}</td><td>{r["params"]["density"]:.0%}</td><td>{r["params"]["input"]}</td><td>{r["params"]["rho"]}</td><td>{r["params"]["leak"]}</td></tr>' for r in complete['winners'])
    real=[];summary=[]
    for c in (64,128):
        for mode in ('original','sparse','normalized_dense'):
            rows=[r for r in checks if r['channels']==c and r['mode']==mode]
            r2=np.mean([r['real_scores']['image_0']['r2'] for r in rows]);sat=np.mean([r['saturation'] for r in rows]);echo=np.max([r['echo'][-1]['rms'] for r in rows])
            real.append(f'<tr><td>{c}</td><td>{mode.replace("_"," ")}</td><td>{r2:.3f}</td><td>{sat:.2%}</td><td>{echo:.6f}</td></tr>')
            summary.append(dict(channels=c,mode=mode,current_real_image_r2=float(r2),saturation=float(sat),max_echo_rms_256=float(echo)))
    trials=[json.loads(l) for f in ('trials.jsonl','refinement.jsonl') for l in (ROOT/f).read_text().splitlines()]
    density_rows=[]
    for d in (.01,.02,.05,.1,.2,.3,1.):
        vals=[max(r['objective'] for r in trials if r['channels']==c and r['params'].get('density')==d) for c in (64,128)]
        density_rows.append(f'<tr><td>{d:.0%}</td><td>{vals[0]:.3f}</td><td>{vals[1]:.3f}</td></tr>')
    html=f'''<!doctype html><meta charset="utf-8"><title>Sparse reservoir tuning</title>
<style>body{{font:17px system-ui;max-width:1400px;margin:30px auto;padding:20px;color:#222}}img{{width:100%}}table{{border-collapse:collapse}}th,td{{padding:8px 16px;border-bottom:1px solid #ddd;text-align:right}}a{{color:#167cba}}</style>
<h1>Isolated reservoir memory tuning</h1><p><a href="../continuous/">Live agents</a> · <a href="../reservoir/">Original capacity probe</a></p>
<p>All computation ran on GPU 3 after pausing the scaled 0.25 novelty-weight agent at 57,700 observations.
The other seven agents continue unchanged. No new reservoir has been inserted into a live agent.</p>
<h2>Selected sparse settings</h2><table><tr><th>Channels</th><th>Connections retained</th><th>Input gain</th><th>Recurrent radius</th><th>Leak</th></tr>{params}</table>
<p>Input weights have unit L2 norm per output channel. Recurrent weights are randomly masked, row-normalized,
then scaled by the exact spectral radius of the circular convolution on the 4×9 hidden grid.
The gains are not directly comparable to the original unnormalized weights. Leak 1 replaces the state each step with tanh(input + recurrence).
These are masked dense convolutions; sparsity is a connectivity change, not a sparse-kernel speed claim.</p>
<h2>Held-out recall</h2><img src="memory.png">
<p>The 128-channel sparse setting reaches image recall R² 0.982 at delay 8 and 0.692 at delay 16;
independent feedback recall among real Globo frames reaches 0.619 at delay 16. Original reservoirs have no useful recall at those delays.
Normalized dense controls are almost identical: image R² 0.690 and real-input feedback R² 0.619 at delay 16.
The evidence supports gain normalization as the main improvement; it does not establish a memory advantage from sparsity itself.
Most signals still have little useful recall at delays 32–64.</p>
<p>Means and ranges over reservoir seeds 7, 19 and 101. Original: deployed reservoir. Sparse: selected parameters.
Normalized dense: same selected gains and leak with all recurrent connections retained. Zero R² is the test-mean baseline.
Negative values are clipped below −0.15 in the plot but preserved in JSON.</p>
<p>Independent-input tests use fresh random 4×4 image features, actions, density and model-feedback values.
The realistic memory test uses recorded Globo images, actions and density, replacing only model feedback with an independent random tag.
This prevents visual autocorrelation from masquerading as memory. Readouts are linear ridge regressions, with training-set feature standardization and regularization selected on validation data.
The production Q decoder does not currently use this feature standardization; its learning with the smaller-amplitude states remains a deployment check.</p>
<h2>Unmodified replay input check</h2><table><tr><th>Channels</th><th>Reservoir</th><th>Current image R²</th><th>|h| &gt; 0.99</th><th>Worst initial-state RMS after 256 steps</th></tr>{''.join(real)}</table>
<p>Recorded model feedback is reconstructed using the saved normalization approximation; images, actions and density are unchanged.
Here image targets are coarse 4×4 grayscale features. This measures representation fidelity, not pixel-perfect reconstruction.
The initial-state check runs two states through identical inputs and measures their remaining difference; it is distinct from useful input recall.</p>
<h2>Search coverage</h2><p>{len(trials)} baseline/candidate evaluations, plus cross-seed confirmation and held-out tests.
Connections retained: 1%, 2%, 5%, 10%, 20%, 30%, with dense controls. Input gain: 0.01–1;
recurrent radius: 0.75–1.25; leak: 0.25, 0.5, 1. A coarse sweep was followed by focused refinement.
The objective balances image/feedback/action recall at delays 4, 8, 16, 32 and penalizes poor current-input decoding.
Final sparse selection uses validation scores and initial-state convergence across three reservoir seeds.</p>
<table><tr><th>Connections retained</th><th>Best 64-channel validation score</th><th>Best 128-channel validation score</th></tr>{''.join(density_rows)}</table>
<p>This table is the best seed-7 score seen at each density, before the three-seed stability selection; it is not a confidence interval or proof that one sparsity is universally optimal.
Test streams do not select parameters. Search uses 8 training / 4 validation streams and 12 real chunks.
Final probes use 24 training / 4 validation / 4 test streams, fresh IID inputs and 32 real chunks disjoint from the search chunks.
Reset-crossing target windows are excluded. A 64-step warmup precedes scored states.</p>
<p>These tests establish improved decodability under the tested inputs, not improved exploration reward or navigation.
Changing a frozen reservoir changes replay features, so deployment needs a fresh agent/replay or explicit replay migration.</p>
<p><a href="complete.json">Final results</a> · <a href="verified.json">Replay and dense controls</a> · <a href="refinement.jsonl">Refinement trials</a> ·
<a href="trials.jsonl">Initial trials</a> · <a href="real_dataset.json">Replay provenance</a> · <a href="memory.svg">Export SVG</a></p>'''
    (ROOT/'index.html').write_text(html)
    (ROOT/'report_summary.json').write_text(json.dumps(summary,indent=2))
    link=Path('/home/dev/globo_dreamer_state/reviews/config24/reservoir-tuning')
    if not link.exists():link.symlink_to(ROOT,target_is_directory=True)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
