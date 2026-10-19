"""Summarize measured retention/transfer without converting proxies into claims."""
import argparse
import html
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .prepare import DOMAINS
from .artifacts import matrix_plot


def mean_sd(values):
    a = np.array(values, dtype=float)
    return {'mean': float(a.mean()), 'sd': float(a.std(ddof=1)) if len(a)>1 else None, 'n': len(a)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/dev/sit_state')
    p.add_argument('--output', default='/home/dev/workspace/continual/artifacts')
    args = p.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    runs = {}
    for folder in sorted((root/'runs').iterdir()):
        if folder.name == 'smoke' or not (folder/'completed.json').exists():
            continue
        cfg = json.loads((folder/'config.json').read_text())
        history = json.loads((folder/'lifecycle.json').read_text())
        results = {'config': cfg, 'domains': {}}
        order = cfg['order'].split(',')
        for d in DOMAINS:
            mode = f'cold{cfg["support"]}'
            initial = history[0]['domains'][d][mode]['loss']
            final = history[-1]['domains'][d][mode]['loss']
            entry = {'initial': initial, 'final': final, 'gain_from_initial': initial-final,
                     'final_cold4': history[-1]['domains'][d]['cold4']['loss'],
                     'final_random': history[-1]['domains'][d]['random']['loss']}
            if d in order:
                acquired = history[order.index(d)+1]['domains'][d][mode]['loss']
                stored_then = history[order.index(d)+1]['domains'][d]['stored']['loss']
                stored_now = history[-1]['domains'][d]['stored']['loss']
                entry.update(after_learning=acquired, forgetting=final-acquired,
                    stored_after_learning=stored_then, stored_final=stored_now,
                    stored_forgetting=stored_now-stored_then,
                    inference_gap_to_stored=final-stored_now)
                gain = initial-acquired
                entry['retained_fraction_of_gain'] = (initial-final)/gain if gain>0 else None
                entry['prospective_changes'] = [history[i]['domains'][d][mode]['loss']-initial
                                                for i in range(1, order.index(d)+1)]
            results['domains'][d] = entry
        records = [json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
        results['training'] = {'updates': len(records),
            'replay_fraction': sum(r['replay_loss'] is not None for r in records)/max(1,len(records)),
            'changepoints': sum(r['changepoint'] for r in records)}
        quality_path = folder/'quality.json'
        if quality_path.exists():
            quality = json.loads(quality_path.read_text())
            results['quality'] = quality
            shutil.copy2(quality_path, output/f'{folder.name}_quality.json')
            qmatrix = [[next((r['kid'] for r in quality if r['stage']==row['stage'] and r['domain']==d and r['mode']==mode), np.nan)
                        for d in DOMAINS] for row in history]
            fig, ax = plt.subplots(figsize=(8,4))
            im = ax.imshow(qmatrix, cmap='viridis_r')
            ax.set_xticks(range(4), DOMAINS)
            plt.setp(ax.get_xticklabels(), rotation=22, ha='right', fontsize=10)
            ax.set_yticks(range(len(history)), [r['stage'] for r in history])
            for i,row in enumerate(qmatrix):
                for j,v in enumerate(row):
                    if np.isfinite(v):
                        ax.text(j,i,f'{v:.4f}',ha='center',va='center',bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
            ax.set_title(folder.name+' — generated-image KID')
            fig.colorbar(im, ax=ax, label='KID (lower is better); 200 generated / 400 real')
            fig.tight_layout()
            fig.savefig(output/f'{folder.name}_xfer_kid.png', dpi=180)
            plt.close(fig)
        if (folder/'diagnostics.json').exists():
            results['diagnostics'] = json.loads((folder/'diagnostics.json').read_text())
            shutil.copy2(folder/'diagnostics.json', output/f'{folder.name}_diagnostics.json')
        runs[folder.name] = results
    aggregate = {}
    for condition in ['bayes_replay', 'bayes_noreplay']:
        names = [f'{condition}_s{s}' for s in [123,124,125] if f'{condition}_s{s}' in runs]
        if not names:
            continue
        aggregate[condition] = {d: {metric: mean_sd([runs[n]['domains'][d][metric] for n in names])
            for metric in ['initial','after_learning','final','forgetting','stored_forgetting','final_cold4']}
            for d in DOMAINS[1:]}
    summary = {'runs': runs, 'aggregate': aggregate,
               'interpretation': 'Positive forgetting = worse held-out velocity MSE after later domains. Prospective changes below zero = improved mask-only adaptation before weight training on that domain. KID is a small-sample generated-distribution diagnostic.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    rows = []
    for condition, domains in aggregate.items():
        for d, vals in domains.items():
            cells = [condition,d]
            for metric in ['initial','after_learning','final','forgetting','stored_forgetting','final_cold4']:
                v = vals[metric]
                cells.append(f'{v["mean"]:.5f}' + (f' ± {v["sd"]:.5f}' if v['sd'] is not None else ''))
            rows.append('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in cells)+'</tr>')
    details = []
    for name,result in runs.items():
        details.append(f'<h2>{html.escape(name)}</h2><p>Training diagnostics: {html.escape(json.dumps(result["training"]))}</p>')
        details.append(f'<img src="{name}/xfer_cold40_change.png">')
        if 'quality' in result:
            details.append(f'<img src="{name}_xfer_kid.png">')
        if 'diagnostics' in result:
            details.append('<table><tr><th>Domain</th><th>Inference method</th><th>Held-out MSE</th></tr>')
            for d,variants in result['diagnostics'].items():
                for mode,v in variants.items():
                    details.append(f'<tr><td>{d}</td><td>{mode}</td><td>{v["loss"]:.5f}</td></tr>')
            details.append('</table>')
    report = '''<!doctype html><meta charset="utf-8"><title>SiT continual-learning report</title>
<style>body{font:16px system-ui;max-width:1350px;margin:35px auto;padding:20px;color:#162a32;background:#f7f6f1}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #bcc8ca;padding:8px}img{max-width:850px;width:100%}h1,h2{color:#164f60}a{color:#006b83}</style>
<h1>Continual post-training of SiT with Bayes LoRA FTN</h1>
<p><a href="index.html">Open complete transfer matrices and generated-image gallery</a> · <a href="summary.json">Machine-readable results</a></p>
<h2>Question and protocol</h2>
<p>Can one frozen pretrained SiT-XL/2 with a shared bank of learned low-rank atoms acquire successive image domains, retain them, and recover them with the existing Bayesian mask search? The unchanged inference mixin selects 16 of 64 rank-one atoms shared across 112 attention/MLP linear layers. Training uses two persistent updates per batch, loss-triggered 30-step refits, and 1500 weight updates per domain. Mask-filtered replay selects stored batches with overlap in (0.2, 0.8] and evaluates them under their original masks. No per-domain adapter allocation, forced disjoint masks, or trainable router is used.</p>
<p>Imagenette supplies ten original ImageNet class labels. Each domain has 2560 training, 80 support, and 400 test images. Original image identities are disjoint between these splits, and paired across photo/monochrome/poster/sketch transformations. Mask inference receives 4 or 40 reference images; query metrics use 200 held-out images. The original photo domain is never post-trained. Primary runs compare replay/no replay across three paired seeds, plus a matched-active-rank ordinary LoRA control and one reversed-order run. Parameters remain float32 with bfloat16 autocast; no EMA is used.</p>
<h2>Measured retention</h2>
<p>Mean ± sample standard deviation across available seeds. Lower loss is better. “Forget” is final loss minus loss immediately after learning the domain. “Stored forget” measures the same change under the final training mask saved at that domain boundary. The last-trained domain has no later exposure and therefore is not a forgetting test. A small forgetting value only matters if learning first improved over initialization.</p>
<table><tr><th>Condition</th><th>Domain</th><th>Initial MSE</th><th>After learning</th><th>Final MSE</th><th>Forget</th><th>Stored forget</th><th>Final, 4 support</th></tr>'''
    report += ''.join(rows)+'</table>'+'\n'.join(details)
    report += '''<h2>How to interpret these results</h2>
<p>Cold search and stored masks answer different questions: cold search measures recoverable function after support-based inference; a saved mask measures whether that particular function survived shared-weight updates. Exact mask overlap is diagnostic, not a definition of correct recall: multiple masks may work. Future-domain columns measure support-assisted adaptation before weight training on that domain; compare each with its own initial baseline to assess transfer.</p>
<p>Generated-image KID uses CleanFID Inception features, 200 generated images and 400 held-out transformed real images, balanced across the ten classes. It is an unbiased finite-sample MMD estimate and may be slightly negative. Subsample variation is not a confidence interval over independent training runs. Qualitative grids use fixed seeds, class labels, 50-step Heun ODE sampling, and the repository’s original CFG=4 behavior.</p>
<h2>Scope and limitations</h2>
<p>This is a controlled synthetic style-shift experiment on a pretrained generator, not evidence about arbitrary natural-image domains or new categories. The Bayesian posterior is the original factorized-Bernoulli KL surrogate over hard top-k masks, not a calibrated posterior over weights. Bank size, target layers and replay budget are practical transformer-scale choices. The training schedule resets Adam moments at known block boundaries as in the original runner, while the loss detector alone triggers mask resets. Frozen-base recovery is guaranteed by construction and must not be counted as retention of newly acquired styles. No test examples enter weight training, replay, or mask search. Optional input-conditioned priors, when reported, are post-hoc frozen-model diagnostics and are separate from the primary training runs.</p>
<p>Sources: <a href="https://github.com/willisma/SiT">official SiT implementation/checkpoint</a>; <a href="https://github.com/fastai/imagenette">Imagenette</a>; local bayesian_adv/bayes_lora_ftn.py, bayes_reconfig.py and FTN_methods.tex. Runtime environment, manifests, checkpoints, detailed logs, generated images, and raw metrics remain on the pod under /home/dev/sit_state.</p>'''
    (output/'report.html').write_text(report)
    print(json.dumps(aggregate,indent=2), flush=True)


if __name__ == '__main__':
    main()
