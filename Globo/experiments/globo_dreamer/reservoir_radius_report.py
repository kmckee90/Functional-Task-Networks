"""Render the fine-radius experiment, verifying achieved recurrent radii."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/dev/globo_dreamer_state/diagnostics/reservoir_radius')
LAGS=(0,1,2,4,8,16,32,64)


def main():
    s=json.loads((ROOT/'complete.json').read_text());rows=s['heldout']
    summaries=[];radius=[]
    for c in (64,128):
        for mode in ('previous','tuned'):
            selected=[r for r in rows if r['channels']==c and r['mode']==mode]
            summary=dict(channels=c,mode=mode,params=selected[0]['params'],
                long_memory=float(np.mean([r['long_memory'] for r in selected])),
                worst_echo=float(max(r['echo'][-1]['rms'] for r in selected)))
            for domain,key,lag in [('noise','image',0),('real_raw','image',0),('noise','image',16),
                    ('noise','image',32),('noise','density',16),('noise','density',32),
                    ('noise','tag',16),('real_tagged','tag',16),('real_tagged','tag',32)]:
                summary[f'{domain}_{key}_{lag}']=float(np.mean([r[domain]['scores'][f'{key}_{lag}']['r2'] for r in selected]))
            summaries.append(summary)
        for seed in (7,19,101,37,71):
            checkpoint=torch.load(ROOT/f'winner_c{c}_seed{seed}.pt',weights_only=True)
            w=checkpoint['state_dict']['cnn2.weight'].cuda()*checkpoint['params']['rho']
            kernel=torch.zeros(c,c,4,9,device='cuda')
            for i in range(3):
                for j in range(3):kernel[:,:,(i-1)%4,(j-1)%9]+=w[:,:,i,j]
            achieved=float(torch.linalg.eigvals(torch.fft.fft2(kernel).permute(2,3,0,1)).abs().max())
            radius.append(dict(channels=c,seed=seed,target=checkpoint['params']['rho'],achieved=achieved))
    (ROOT/'summary.json').write_text(json.dumps(dict(metrics=summaries,radii=radius),indent=2))
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for i,c in enumerate((64,128)):
        for j,(domain,key,title) in enumerate((('noise','image','Image recall'),('noise','density','Density feedback recall'),('real_tagged','tag','Independent feedback amid real frames'))):
            ax=axes[i,j]
            for mode,color in (('previous','#777777'),('tuned','#157bb9')):
                values=np.array([[r[domain]['scores'][f'{key}_{l}']['r2'] for l in LAGS] for r in rows if r['channels']==c and r['mode']==mode])
                ax.plot(LAGS,values.mean(0),'o-',label=mode,color=color)
                ax.fill_between(LAGS,values.min(0),values.max(0),color=color,alpha=.15)
            ax.set_title(f'{c} channels: {title}');ax.set_xscale('symlog',linthresh=1)
            ax.set_xticks(LAGS,labels=LAGS);ax.set_ylim(-.2,1.04);ax.set_xlabel('Delay in observations');ax.set_ylabel('Held-out R²');ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(ROOT/'memory.png',dpi=150);fig.savefig(ROOT/'memory.svg');plt.close(fig)
    params=''.join(f'<tr><td>{r["channels"]}</td><td>{r["mode"]}</td><td>{r["params"]["rho"]}</td><td>{r["params"]["input"]}</td><td>{r["params"]["density"]:.0%}</td></tr>' for r in summaries)
    table=''.join(f'<tr><td>{r["channels"]}</td><td>{r["mode"]}</td><td>{r["noise_image_0"]:.4f}</td><td>{r["real_raw_image_0"]:.4f}</td><td>{r["noise_image_16"]:.3f}</td><td>{r["real_tagged_tag_16"]:.3f}</td><td>{r["real_tagged_tag_32"]:.3f}</td></tr>' for r in summaries)
    error=max(abs(r['achieved']-r['target']) for r in radius)
    html=f'''<!doctype html><meta charset="utf-8"><title>Fine spectral radius tuning</title>
<style>body{{font:17px system-ui;max-width:1400px;margin:30px auto;padding:20px}}img{{width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}</style>
<h1>Fine spectral radius tuning</h1><p><a href="../reservoir-tuning/">Previous sparse/gain search</a> · <a href="../continuous/">Live agents</a></p>
<p>Isolated experiment on reserved GPU 3. No live agents or replay features changed.</p>
<h2>Selected settings</h2><table><tr><th>Channels</th><th>Version</th><th>Radius</th><th>Input gain</th><th>Connections retained</th></tr>{params}</table>
<p>All input weights row-normalized; recurrence scaled by exact circular-convolution spectral radius; leak fixed at 1.
Achieved radii recomputed from saved weights. Maximum absolute error: {error:.2g}.</p>
<h2>Held-out results</h2><img src="memory.png"><table><tr><th>Channels</th><th>Version</th><th>Current noise image R²</th><th>Current real image R²</th><th>Noise image recall at 16</th><th>Real-background feedback recall at 16</th><th>At 32</th></tr>{table}</table>
<p>Mean and shaded range over reservoir seeds 7, 19, 101, 37, 71. Seeds 37 and 71 were not used for parameter selection.
Image targets are coarse 4×4 features. In the realistic memory probe, recorded frames/actions/density are accompanied by an independent tag replacing model feedback, to avoid mistaking visual autocorrelation for memory.
The current-real-image check uses original feedback (model normalization reconstructed approximately from replay).</p>
<p>Search: radii 0.98, 0.985, 0.99, 0.995, 0.9975, 1.0, 1.0025; input gains 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15.
Connectivity fixed at the previous winners (64 channels:5%; 128 channels:20%). Objective emphasizes delays 16 and 32.
Current-input decoding must stay within 0.002 R² of baseline for noise images and 0.005 for other noise targets and real images on every validation seed.
Initial-state RMS must converge below 0.001 at observation256. The previous setting remains eligible.</p>
<p>Selection uses only the first 12 recorded chunks and independent search streams. Final readouts use fresh IID input/tag seed20261009 and 32 disjoint real chunks, split24 train/4 validation/4 test.
Readouts standardize features using training data; ridge regularization uses validation data. Reset-crossing target windows excluded.
These are representation probes, not evidence of better navigation or exploration. The production Q decoder does not currently standardize features.</p>
<p><a href="summary.json">Summary and achieved radii</a> · <a href="complete.json">Held-out metrics</a> · <a href="selection.json">Cross-seed selection</a> · <a href="trials.jsonl">All trials</a> · <a href="memory.svg">Export SVG</a></p>'''
    (ROOT/'index.html').write_text(html)
    link=Path('/home/dev/globo_dreamer_state/reviews/config24/reservoir-radius')
    if not link.exists():link.symlink_to(ROOT,target_is_directory=True)
    print(json.dumps(dict(metrics=summaries,max_radius_error=error),indent=2))


if __name__=='__main__':main()
