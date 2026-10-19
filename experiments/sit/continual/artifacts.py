"""Create lifecycle matrices, learning curves and an HTML gallery on the pod."""
import argparse
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .prepare import DOMAINS


def matrix_plot(values, rows, title, destination, relative=False):
    values = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, max(3.2, .8 * len(rows))))
    if relative:
        bound = max(.01, np.nanmax(np.abs(values)))
        im = ax.imshow(values, cmap='RdBu_r', vmin=-bound, vmax=bound)
    else:
        im = ax.imshow(values, cmap='viridis_r')
    ax.set_xticks(range(len(DOMAINS)), DOMAINS)
    plt.setp(ax.get_xticklabels(), rotation=22, ha='right', fontsize=10)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i,j]):
                ax.text(j, i, f'{values[i,j]:+.3f}' if relative else f'{values[i,j]:.3f}',
                        ha='center', va='center', color='black',
                        bbox=dict(facecolor='white', alpha=.65, edgecolor='none', pad=1))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Velocity MSE change (negative = better)' if relative else 'Held-out velocity MSE (lower = better)')
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/dev/sit_state')
    p.add_argument('--output', default='/home/dev/workspace/continual/artifacts')
    args = p.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sections = []
    if (root/'domain_examples.png').exists():
        shutil.copy2(root/'domain_examples.png', output/'domain_examples.png')
    summaries = {}
    for run in sorted((root/'runs').iterdir()):
        if not (run/'lifecycle.json').exists():
            continue
        cfg = json.loads((run/'config.json').read_text())
        history = json.loads((run/'lifecycle.json').read_text())
        dest = output / run.name
        dest.mkdir(exist_ok=True)
        shutil.copy2(run/'lifecycle.json', dest/'lifecycle.json')
        shutil.copy2(run/'config.json', dest/'config.json')
        rows = [r['stage'].replace('after_', '').replace('_', ' ') for r in history]
        sections.append(f'<h2>{html.escape(run.name)}</h2><p>{html.escape(json.dumps(cfg))}</p>')
        for mode in ['cold4', f'cold{cfg["support"]}', 'stored', 'current', 'base', 'ic_direct', 'ic_refine']:
            values = [[r['domains'][d].get(mode, {}).get('loss', np.nan) for d in DOMAINS] for r in history]
            if not np.isfinite(values).any():
                continue
            name = f'xfer_{mode}.png'
            matrix_plot(values, rows, f'{run.name}: {mode}', dest/name)
            sections.append(f'<img class="matrix" src="{run.name}/{name}">')
            if mode.startswith('cold') and history[0]['stage'] == 'initial':
                relative = np.array(values)-np.array(values)[0]
                name = f'xfer_{mode}_change.png'
                matrix_plot(relative, rows, f'{run.name}: change from initial, {mode}', dest/name, True)
                sections.append(f'<img class="matrix" src="{run.name}/{name}">')
        if (run/'train.jsonl').exists():
            records = [json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
            fig, ax = plt.subplots(figsize=(10,3))
            for bi, domain in enumerate(cfg['order'].split(',')):
                vals = [r['loss'] for r in records if r['block']==bi]
                if not vals:
                    continue
                w = min(20, len(vals))
                smooth = np.convolve(vals, np.ones(w)/w, mode='valid')
                ax.plot(np.arange(len(smooth))+bi*cfg['steps']+w-1, smooth, label=domain)
            ax.set(xlabel='Weight-update step', ylabel='Training velocity MSE', title=run.name)
            ax.legend()
            fig.tight_layout()
            fig.savefig(dest/'learning.png', dpi=170)
            plt.close(fig)
            sections.append(f'<img src="{run.name}/learning.png">')
            if cfg['method'] == 'bayes' and records:
                masks = np.array([r['mask'] for r in records])
                width = 30
                pooled = np.stack([masks[i:i+width].mean(0) for i in range(0,len(masks),width)])
                fig, ax = plt.subplots(figsize=(10,4))
                im = ax.imshow(pooled.T, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1,
                               extent=(0,len(masks),0,masks.shape[1]))
                for boundary in [cfg['steps'],2*cfg['steps']]:
                    if boundary < len(masks):
                        ax.axvline(boundary,color='cyan',linestyle='--',linewidth=1)
                ax.set(xlabel='Weight-update step', ylabel='Atom index', title=run.name+' — mask usage')
                fig.colorbar(im,ax=ax,label='Fraction of updates active (30-step bins)')
                fig.tight_layout()
                fig.savefig(dest/'mask_usage.png',dpi=170)
                plt.close(fig)
                sections.append(f'<img src="{run.name}/mask_usage.png">')
        if (run/'samples').exists():
            shutil.copytree(run/'samples', dest/'samples', dirs_exist_ok=True)
            for stage in sorted((dest/'samples').iterdir()):
                sections.append(f'<details><summary>Samples: {html.escape(stage.name)}</summary>')
                for path in sorted(stage.glob('*.png')):
                    sections.append(f'<p>{html.escape(path.stem)}</p><img src="{run.name}/samples/{stage.name}/{path.name}">')
                sections.append('</details>')
        summaries[run.name] = {'stages': len(history), 'completed': (run/'completed.json').exists()}
    (output/'status.json').write_text(json.dumps(summaries, indent=2))
    (output/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>SiT Bayes LoRA FTN lifecycle</title>
<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;background:#f5f5f2;color:#222}img{max-width:100%;background:white;margin:12px 0}.matrix{width:600px}h1,h2{color:#183c49}summary{cursor:pointer;font-weight:600}p{overflow-wrap:anywhere}</style>
<h1>SiT + Bayes LoRA FTN: continual-learning lifecycle</h1>
<p>Pretrained SiT-XL/2; ten Imagenette classes. Block order: monochrome, poster, sketch unless noted.
Columns include both already-trained and future domains. Photo is never post-trained.
Cold inference uses disjoint reference images, freezes all weights, and searches the shared mask.
Saved masks measure retention without search. Negative change from initial means improvement.
Velocity MSE is a diagnostic, not a substitute for generated-image quality.</p>
<p>Examples below: photo, monochrome, poster, sketch, one row each.</p><img src="domain_examples.png">
''' + '\n'.join(sections))
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
