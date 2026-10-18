"""Pod-only, periodically publish completed clips without interrupting training."""
import html
import json
import os
from pathlib import Path
import shutil
import time
import av

STATE=Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))
OUT=STATE/'reviews/config24/progress'


def update():
    db=json.loads((STATE/'search60k/state.json').read_text())
    ranks=json.loads((STATE/'search60k/leaderboard.json').read_text())['rankings']
    selected={0,6}|{r['config'] for r in ranks[:3]}|{t['config'] for t in db['trials'] if t['status']=='running'}
    OUT.mkdir(parents=True,exist_ok=True)
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Live Globo progress</title><style>body{font:16px system-ui;background:#141820;color:#eee;'
        'max-width:1200px;margin:24px auto;padding:16px}a{color:#9bd0ff}video{width:100%;max-width:768px}'
        '.clips{display:flex;flex-wrap:wrap;gap:16px}.clip{width:350px}section{border-bottom:1px solid #555;padding:16px 0}</style>'
        '<h1>60k exploration search — progress videos</h1>'
        f'<p>Updated {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}. '
        'Refreshes once a minute while videos are not playing. Clips are recorded every 10k observations; '
        'only finalized videos are published. The two latest clips per trial appear below.</p>'
        '<p>Config numbers here belong to the 60k search. '
        '<a href="../latest.html">Previous 40k config24 reference gallery</a>.</p>']
    keep=set()
    parts.append('<h2>World-model comparisons</h2><p>Actual | reconstruction | 16-step open-loop forecast. '
                 'These are labeled checkpoint snapshots, not a live model feed.</p>')
    for folder in sorted((STATE/'diagnostics').glob('search60_c*_s*')):
        if not (folder/'index.html').exists(): continue
        for suffix in ('ckpt.mp4','ckpt_h16.jpg','scores.json'):
            dest=OUT/f'{folder.name}_{suffix}';keep.add(dest.name)
            if not dest.exists():shutil.copy2(folder/suffix,dest)
        parts.append(f'<h3>{html.escape(folder.name)} · final 60k checkpoint</h3>'
            f'<video controls playsinline preload="metadata" poster="{folder.name}_ckpt_h16.jpg" '
            f'src="{folder.name}_ckpt.mp4"></video>'
            f'<p><a href="{folder.name}_scores.json">Fidelity metrics</a></p>')
    parts.append('<h2>Explorer behavior</h2>')
    for c in db['configs']:
        if c['id'] not in selected:continue
        params={k:c['params'][k] for k in ('disagreement','prediction_error','disagreement_power','actor_weight','intrinsic_horizon')}
        parts.append(f'<section><h2>Config {c["id"]}</h2><p>{html.escape(str(params))}</p>')
        for t in db['trials']:
            if t['config']!=c['id'] or t['status'] not in ('done','running'):continue
            root=STATE/'runs'/t['name'];status=root/'status.json'
            st=json.loads(status.read_text()) if status.exists() else {}
            parts.append(f'<h3>Seed {t["seed"]} · {t["status"]} · {st.get("step",0):,} observations</h3><div class="clips">')
            videos=[]
            for src in sorted((root/'environment').glob('behavior_*.mp4'),reverse=True):
                dest=OUT/f'{t["name"]}_{src.name}'
                if not dest.exists():
                    try:
                        # In-progress MP4s lack their final stream metadata.
                        with av.open(str(src)) as container:
                            if not container.streams.video[0].frames:continue
                        tmp=dest.with_suffix('.tmp');shutil.copy2(src,tmp);tmp.replace(dest)
                    except (OSError,av.error.FFmpegError):continue
                keep.add(dest.name);videos.append((src,dest))
                if len(videos)==2:break
            for src,dest in reversed(videos):
                step=int(src.stem.split('_')[-1])
                parts.append(f'<div class="clip"><p>Starting at observation {step:,}</p>'
                    f'<video controls playsinline preload="metadata" src="{dest.name}"></video></div>')
            parts.append('</div>')
        parts.append('</section>')
    parts.append('<script>setInterval(()=>{if([...document.querySelectorAll("video")].every(v=>v.paused))'
                 'location.reload()},60000)</script>')
    tmp=OUT/'index.tmp';tmp.write_text(''.join(parts));tmp.replace(OUT/'index.html')
    for path in OUT.iterdir():
        if path.suffix in ('.mp4','.jpg','.json') and path.name not in keep:path.unlink()
    print(time.strftime('%H:%M:%S'),len(keep),'media files',flush=True)


if __name__=='__main__':
    while True:
        try:update()
        except Exception as exc:print(type(exc).__name__,str(exc),flush=True)
        time.sleep(60)
