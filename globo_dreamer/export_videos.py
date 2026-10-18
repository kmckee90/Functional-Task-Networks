"""Copy selected trial videos into a stable pod-side review folder."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=int,required=True)
    args=p.parse_args()
    state=Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))
    out=state/'reviews'/f'config{args.config}'; out.mkdir(parents=True,exist_ok=True)
    sections=[]; manifest=[]
    for seed in (7,19,101):
        run=state/'runs'/f's40_c{args.config:06d}_s{seed}'
        cards=[]
        for source in sorted((run/'environment').glob('behavior_*.mp4')):
            step=int(source.stem.split('_')[-1]); name=f'seed{seed}_step{step}.mp4'
            shutil.copy2(source,out/name)
            manifest.append(dict(seed=seed,step=step,file=name,source=str(source)))
            cards.append(f'<article><p>Observation {step:,}</p><video controls preload="metadata" '
                         f'src="{html.escape(name)}"></video><p><a href="{name}" download>Download MP4</a></p></article>')
        sections.append(f'<h2>Seed {seed}</h2><div class="clips">'+''.join(cards)+'</div>')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Globo exploration videos</title><style>body{font:16px system-ui;background:#141820;color:#eee;'
        'max-width:1200px;margin:32px auto;padding:16px}a{color:#9bd0ff}.clips{display:flex;flex-wrap:wrap;gap:20px}'
        'article{width:360px}video{width:100%;background:#000}</style>'
        f'<h1>Globo explorer · configuration {args.config}</h1>'
        '<p>Recorded behavior samples, approximately 20 seconds each at playback speed. '
        'These are not full trajectories or world-model forecasts. Trials ran for 40,000 observations.</p>'
        +'<h2>Dreamer world-model fidelity</h2><p>Real frames, reconstructions, and open-loop predictions '
        'on fresh withheld trajectories, including untrained comparisons: '
        + ' · '.join(f'<a href="world_seed{s}/">Seed {s}</a>' for s in (7,19,101))+'</p>'
        +''.join(sections))
    print(json.dumps(dict(directory=str(out),clips=len(manifest))))


if __name__=='__main__': main()
