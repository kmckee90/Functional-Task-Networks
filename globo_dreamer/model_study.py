"""Eight fixed model-only replay experiments and an updating artifact gallery."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .search import atomic, alive
from .search60 import FIXED

STATE=Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))
ROOT=STATE/'model_study'
OUT=STATE/'reviews/config24/models'


def gallery(trials):
    import av
    import shutil
    OUT.mkdir(parents=True,exist_ok=True)
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>World-model learning study</title><style>body{background:#141820;color:#eee;font:16px system-ui;'
        'max-width:1200px;margin:24px auto;padding:16px}a{color:#9bd0ff}video{width:100%;max-width:768px}'
        'video.actual-observation{width:50%;max-width:384px}'
        'section{border-bottom:1px solid #555;padding:20px 0}table{border-collapse:collapse}td,th{padding:8px}</style>'
        '<h1>World-model-only learning: replay volume × warmup</h1>'
        '<p>All runs: seed 7, same density explorer; Dreamer actor/critic disabled. '
        'Left: actual. Middle: reconstruction. Right: 16 context frames, then 16 predictions using actions only. '
        'Fixed withheld data never enters training replay. Videos every 2,000 observations plus initialization and exit.</p>'
        f'<p>Page updated {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}. '
        'Refreshes every minute while videos are paused. Delayed runs remain untrained through observation 20,000.</p>']
    for t in trials:
        run=STATE/'runs'/t['name'];st={}
        try:st=json.loads((run/'status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):pass
        step=st.get('step',0);updates=st.get('updates',0)
        parts.append(f'<section><h2>GPU {t["gpu"]} · ratio {t["ratio"]} · training starts {t["start"]:,}</h2>'
            f'<p>{t["name"]} · {t["status"]} · current observations <b>{step:,}</b> · model updates <b>{updates:,}</b></p>')
        record=st.get('environment') or {}
        parts.append(f'<h3>Actual exploration — latest recorded observations</h3><p>Life {record.get("life","?")}; '
            f'{record.get("cells_lifetime","?")} spatial cells visited across lives. '
            'Recorded every 10k observations and after process restarts; these are snapshots, not a live stream.</p>')
        count=0
        for source in sorted((run/'environment').glob('behavior_*.mp4'),reverse=True):
            dest=OUT/f'{t["name"]}_{source.name}'
            if not dest.exists():
                try:
                    with av.open(str(source)) as container:
                        if not container.streams.video[0].frames:continue
                    tmp=dest.with_suffix('.tmp');shutil.copy2(source,tmp);tmp.replace(dest)
                except (OSError,av.error.FFmpegError):continue
            clip_step=int(source.stem.split('_')[-1])
            parts.append(f'<p>Actual observations starting at <b>{clip_step:,}</b></p>'
                f'<video class="actual-observation" controls playsinline preload="metadata" src="{dest.name}"></video>')
            count+=1
            if count==2:break
        if not count:parts.append('<p>No finalized observation video yet.</p>')
        trajectory=run/'trajectory_progress';trajectory_link=OUT/f'{t["name"]}_trajectory'
        if trajectory.exists() and not trajectory_link.exists():trajectory_link.symlink_to(trajectory,target_is_directory=True)
        if (trajectory/'latest.json').exists():
            td=json.loads((trajectory/'latest.json').read_text())
            prefix=f'{trajectory_link.name}/{td["folder"]}'
            parts.append('<h3>World model on this agent’s recent real trajectory</h3>'
                f'<p>Model checkpoint: <b>{td["step"]:,} observations; {td["model_updates"]:,} updates</b>. '
                'Most recent suitable sequence in saved replay; may have been used for training. '
                'Actual | reconstruction | open-loop prediction. Refreshes with saved checkpoints, approximately every ten minutes.</p>'
                f'<video controls playsinline preload="metadata" poster="{prefix}/recent_h16.jpg" src="{prefix}/recent.mp4"></video>'
                f'<p><a href="{prefix}/textured.mp4">Textured recent sequence</a> · '
                f'<a href="{prefix}/metadata.json">Frame provenance and snapshot details</a></p>')
        else:parts.append('<h3>World model on recent real trajectory</h3><p>Preparing the first checkpoint rendering.</p>')
        parts.append('<h3>World-model evaluation — fixed recorded trajectory</h3>'
            '<p>The evaluation frames below are reused at every checkpoint. Rendering them does not reset the training agent.</p>')
        folder=run/'model_progress';link=OUT/t['name']
        if folder.exists() and not link.exists():link.symlink_to(folder,target_is_directory=True)
        latest=folder/'latest.json'
        if latest.exists():
            d=json.loads(latest.read_text());prefix=f'{t["name"]}/{d["folder"]}'
            parts.append(f'<p>Video checkpoint: <b>{d["step"]:,} observations, {d["model_updates"]:,} updates</b>. '
                f'Reconstruction MSE {d["posterior_mse"]:.5f}; 16-step prediction MSE {d["h16_mse"]:.5f}; '
                f'persistence {d["h16_persistence_mse"]:.5f}.</p>'
                f'<video controls playsinline preload="metadata" poster="{prefix}/model_h16.jpg" '
                f'src="{prefix}/model.mp4"></video><p><a href="{prefix}/fixed.mp4">Fixed probe video</a> · '
                f'<a href="{prefix}/scores.json">Fidelity and novelty metrics</a></p><details><summary>Earlier checkpoints</summary>')
            for p in sorted(folder.glob('step*/scores.json')):
                parts.append(f'<a href="{t["name"]}/{p.parent.name}/model.mp4">{p.parent.name}</a> · ')
            parts.append('</details>')
        else:parts.append('<p>Initializing the model and first video.</p>')
        parts.append('</section>')
    parts.append('<script>setInterval(()=>{if([...document.querySelectorAll("video")].every(v=>v.paused))location.reload()},60000)</script>')
    tmp=OUT/'index.tmp';tmp.write_text(''.join(parts));tmp.replace(OUT/'index.html')


def main():
    import fcntl
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'supervisor.pid').write_text(str(os.getpid()))
    path=ROOT/'state.json'
    if path.exists():trials=json.loads(path.read_text())
    else:
        trials=[dict(name=f'wm_r{ratio}_start{start}_s7',ratio=ratio,start=start,gpu=gpu,status='pending',attempts=0)
            for gpu,(start,ratio) in enumerate((s,r) for s in (0,20000) for r in (256,512,1024,2048))]
    stopping=[False];children={}
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    try:
        while not stopping[0]:
            for t in trials:
                if t['name'] in children:children[t['name']].poll()
                if t['status']=='running' and not alive(t):
                    try:st=json.loads((STATE/'runs'/t['name']/'status.json').read_text())
                    except (FileNotFoundError,json.JSONDecodeError):st={}
                    t['status']='done' if st.get('step',0)>=60000 and not st.get('running',True) else 'pending'
                if t['status']=='pending' and t['attempts']<3:
                    run=STATE/'runs'/t['name'];run.mkdir(parents=True,exist_ok=True)
                    cmd=[sys.executable,'-u','-m','globo_dreamer.train','--name',t['name'],'--seed','7',
                        '--spawn','12,5.1,120','--steps','60000','--model-free','--world-model-only',
                        '--train-ratio',str(t['ratio']),'--model-train-start',str(t['start']),
                        '--model-video-every','2000','--video-every','10000','--checkpoint-seconds','600',
                        '--mf-replay-capacity','1000000','--mf-goal-capacity','100000','--recode-capacity','10000',
                        '--dreamer-replay-capacity','60000','--disagreement','10','--actor-weight','.5']
                    for k,v in dict(FIXED,dreamer_action_probability=0.).items():cmd.extend(['--'+k.replace('_','-'),str(v)])
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(t['gpu']),OMP_NUM_THREADS='4',LP_NUM_THREADS='2',
                        PYTHONDONTWRITEBYTECODE='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                        GLOBO_ENGINE_ROOT=f'/tmp/globo_model_engines/{t["name"]}')
                    cpus=sorted(os.sched_getaffinity(0));cpu=cpus[t['gpu']*4:t['gpu']*4+4]
                    if len(cpu)==4:cmd=['taskset','-c',','.join(map(str,cpu)),*cmd]
                    with (run/'console.log').open('a') as f:
                        p=subprocess.Popen(cmd,env=env,cwd=STATE,stdout=f,stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,start_new_session=True)
                    children[t['name']]=p;t.update(pid=p.pid,status='running',attempts=t['attempts']+1)
                    atomic(path,trials)
                elif t['status']=='pending' and t['attempts']>=3:t['status']='failed'
            atomic(path,trials)
            try:gallery(trials)
            except Exception as e:print('Gallery error',repr(e),flush=True)
            time.sleep(20)
    finally:
        for t in trials:
            if t['status']=='running' and alive(t):os.kill(t['pid'],signal.SIGTERM)
        atomic(path,trials)


if __name__=='__main__':main()
