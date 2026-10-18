"""Render saved reservoir diagnostic results on the pod."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/home/dev/globo_dreamer_state/diagnostics/reservoir_capacity')
s=json.loads((root/'summary.json').read_text());lags=s['lags']
fig,axes=plt.subplots(1,4,figsize=(15,3.8))
for ax,key,title in zip(axes,('image','density','model','action'),('Image features','Density feedback','Model feedback','Action feedback')):
    for channels,color in ((64,'#377eb8'),(128,'#e66a17')):
        rows=[r for r in s['iid'] if r['channels']==channels]
        metric='accuracy' if key=='action' else 'r2'
        a=np.array([[r['scores'][f'{key}_lag{lag}'][metric] for lag in lags] for r in rows])
        ax.plot(lags,a.mean(0),'o-',label=f'{channels} channels',color=color)
        ax.fill_between(lags,a.min(0),a.max(0),alpha=.15,color=color)
    ax.axhline(.04 if key=='action' else 0,color='gray',ls='--',lw=1)
    ax.set_xscale('symlog',linthresh=1);ax.set_xticks(lags,labels=lags)
    ax.set_ylim(-.12,1.03);ax.set_xlabel('Delay in observations');ax.set_title(title)
    ax.set_ylabel('Test accuracy' if key=='action' else 'Test R²');ax.grid(alpha=.2)
axes[0].legend();fig.suptitle('Independent-input recall: mean and range across three reservoir seeds')
fig.tight_layout();fig.savefig(root/'recall.png',dpi=160);fig.savefig(root/'recall.svg');plt.close(fig)
html='''<!doctype html><meta charset="utf-8"><title>Reservoir memory diagnostics</title>
<style>body{font:17px system-ui;max-width:1300px;margin:30px auto;padding:20px}img{width:100%}td,th{padding:8px}a{color:#2675ad}</style>
<h1>64 versus 128 reservoir channels</h1>
<p>Identical frozen-reservoir implementation, feedback mapping and matched inputs. Reservoir seeds 7, 19, 101.
Inputs are independent random low-frequency images, actions, density and model-feedback values.
24 training, 4 validation and 4 test streams; 400 scored observations per stream after 64-step warmup.
Ridge strength selected by validation separately for each target and delay. Shading is the range across seeds.</p>
<img src="recall.png"><p>128 channels improve current input decoding and short-delay recall.
Neither width provides useful recall at the tested delays of 4–64 observations. This is a controlled probe,
not an upper bound on all information recoverable by any decoder or on task-specific memory.</p>
<p>A two-hidden-layer, 512-wide nonlinear readout was also tested for seed 7 with validation-selected epochs;
it did not recover long-delay information. Real Globo frames were replayed through both reservoirs on matched
chunks: coarse current-image decoding R² was approximately 0.83 versus 0.88. This real-frame probe uses an
approximation of model feedback normalization and is not a test of the trained Q decoder.</p>
<p>About 98.5% of activations have absolute value above 0.99 in both widths. Few features are permanently constant;
saturated binary-like patterns still carry information. Initial hidden-state differences persist for 128 steps,
more strongly at 128 channels; this persistence should not be confused with recoverable input history.</p>
<p><a href="summary.json">Three-seed linear results</a> · <a href="nonlinear_c64.json">64-channel nonlinear probe</a> ·
<a href="nonlinear_c128.json">128-channel nonlinear probe</a> · <a href="real_dataset.json">Replay provenance</a> ·
<a href="recall.svg">Export SVG</a></p>'''
(root/'index.html').write_text(html)
link=Path('/home/dev/globo_dreamer_state/reviews/config24/reservoir')
if not link.exists():link.symlink_to(root,target_is_directory=True)
print(root/'index.html')
