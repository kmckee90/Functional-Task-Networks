"""Reduced-action FTN-M2E vs the six archived Phase 4 full-action runs.

Run on the pod; checkpoints, logs, videos and report stay outside source sync.
"""
import argparse
import functools
import html
import json
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .phase2 import STATE, read, atomic, detach, publish
from .phase4 import experiments as phase4_experiments, SHARED as ARCHIVE

SHARED = Path('/mnt/obelisk-shared/kmckee90/globo_actions')
OUT = SHARED / 'monitor'
FULL = 'FTN-M2E · full 25 actions (Phase 4)'
REDUCED = 'FTN-M2E · reduced 14 actions'
COLORS = {FULL: '#72b7ff', REDUCED: '#7de1bf'}
INTRO = None
CURRENT_TRIALS = None


def experiments():
    trials = []
    for i in range(4):
        trials.append(dict(name=f'actions14_conv_r{i+1}_s19', model=REDUCED,
            phase='actions', pair=i+1, seed=19, agent_seed=1901+i,
            pod='globo-actions', gpu=i, spawn='12.0,39.1,120.0',
            status='pending', attempts=0, frontier_rotation=False,
            achievements=True, event_capture=True, render_size=256, software_renderer=True,
            weight=0., visitation=True, model_free_only=True, scaled=True,
            reservoir_variant='tuned_v1', mf_replay_dtype='float16',
            m2e_ftn='conv', m2e_ftn_k=296, reduced_actions=True,
            clip_settings=str(SHARED/'clip_settings.json')))
    return trials


def archived():
    return [dict(t, model=FULL, status='archived') for t in phase4_experiments()
            if t.get('m2e_ftn') == 'conv']


def launch():
    SHARED.mkdir(parents=True, exist_ok=True)
    (SHARED/'runs').mkdir(exist_ok=True)
    terrain = SHARED/'terrain'
    if not terrain.exists():terrain.symlink_to(ARCHIVE/'terrain', target_is_directory=True)
    for t in archived():
        src = ARCHIVE/'runs'/t['name']
        spec = read(src/'architecture.json')
        if spec.get('k') != 296 or spec.get('neurons') != 2368:
            raise RuntimeError(f'Unexpected reference architecture: {t["name"]}')
        dst = SHARED/'runs'/t['name']
        if not dst.exists():dst.symlink_to(src, target_is_directory=True)
    atomic(SHARED/'experiments.json', dict(current=experiments(), reference=archived()))
    if not (SHARED/'clip_settings.json').exists():atomic(SHARED/'clip_settings.json', dict(frames=192))
    root = STATE/'continuous_runs';root.mkdir(parents=True, exist_ok=True)
    path = root/'state.json'
    if path.exists():raise RuntimeError('Run list exists; use existing supervisor/resume instead of fresh launch')
    if any((STATE/'runs'/t['name']).exists() for t in experiments()):
        raise RuntimeError('Fresh comparison refused: trial directory already exists')
    atomic(path, experiments())
    for module, args, log in [('globo_dreamer.continuous_runs', (), 'continuous_runs/supervisor.log'),
            (__name__, ('publish',), 'actions_publish.log'),
            (__name__, ('monitor',), 'actions_monitor.log')]:
        # __name__ is __main__ when invoked with -m.
        module = 'globo_dreamer.action_comparison' if module == '__main__' else module
        print(module, detach(module, log, *args), flush=True)


def update():
    from . import phase4_monitor as charts, terrain_maps
    from .achievement_charts import achievement_checklist, checked_achievements
    from .survival import panel as survival_panel
    OUT.mkdir(parents=True, exist_ok=True)
    charts.SHARED = SHARED;charts.COLORS = COLORS
    current = (CURRENT_TRIALS() if CURRENT_TRIALS else
               read(SHARED/'pods/globo-actions.json').get('trials', experiments()))
    trials = archived() + current
    data = {t['name']: charts.history(t) for t in trials}
    ready = all(data.values())
    cap = max([0]+[rows[-1]['step'] for rows in data.values() if rows])
    clipped = {name:[r for r in rows if r['step']>0] for name,rows in data.items()}
    terrain_maps.SHARED = SHARED
    maps={}
    for seed in sorted({t['seed'] for t in current}):
        maps.update(terrain_maps.publish(seed,[t for t in current if t['seed']==seed]))
    parts = ['''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Globo · reduced actions</title><style>
body{background:#10151d;color:#e5eaf0;font:16px system-ui;max-width:1450px;margin:auto;padding:24px}
a{color:#9bd0ff}section{margin:28px 0;padding:22px;background:#171d27;border-radius:12px}
.charts,.media{display:grid;grid-template-columns:1fr 1fr;gap:24px}svg,img,video{width:100%;height:auto}
svg text{fill:#b9c6d4;font-size:13px}.legend{display:flex;gap:16px;flex-wrap:wrap}.muted{color:#a5b3c5}
.curve.hidden{display:none}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}
@media(max-width:800px){.charts,.media{grid-template-columns:1fr}body{padding:10px}}</style></head><body>
<h1>FTN-M2E · Does a smaller action space improve exploration?</h1>
<p>World 19. Four fresh 14-action runs (agent seeds 1901–1904) versus all six archived Phase 4 FTN-M2E runs with 25 actions (1901–1906).
The first four seeds match. Same spawn, tuned 128-channel reservoir, 2M replay, 500k frontier, 100k RECODE centroids, and density learning settings.
FTN retains 2,368 neurons with 296 active; only its action outputs shrink. No Dreamer, warm start, extrinsic reward, or step limit.</p>
<p>Reduced actions retain idle, movement/look, dig/attack and place/interact. Inventory selection and both sneak actions are excluded.
All axes extend to the furthest run. Each condition’s mean and bounds stop at its shortest run. Thick lines: means. Bands: minimum–maximum, not confidence intervals.</p>''']
    if INTRO:
        parts[0] = parts[0].split('<h1>')[0] + INTRO
    parts.append(f'<p class="muted">Updated {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}. Axis endpoint: {cap:,} observations.</p>')
    if not ready:parts.append('<p>Waiting for the first logged observations from all four new runs.</p>')
    parts.append('<p class="muted">The mean/range comparison begins only where all selected runs have recorded observations.</p>')
    parts.append('<div class="legend">')
    for t in trials:
        parts.append(f'<label style="color:{COLORS[t["model"]]}"><input class="toggle" data-run="{t["name"]}" type="checkbox" checked>{t["model"]} · {t["pair"]}</label>')
    parts.append('</div><p id="selection-stats"></p><div class="charts">')
    for key,title,ylabel in [('total_cells','Spatial progress','Discovered 4 m × 4 m cells'),
            ('achievement_checked','Unique achievements','Achievement boxes checked')]:
        parts.append(charts.chart(trials,clipped,key,title,ylabel,xmax=cap).replace('mean of six runs','mean of group'))
    from .survival import comparison_data
    lifespan_trials=[t for t in trials if not t.get('survival')]
    lifespan_cap=cap
    lifespan=comparison_data(lifespan_trials,SHARED,lifespan_cap,extend=False,raw=True)
    if lifespan_trials:
        parts.append('</div><h2>Lifespan over training · exploration runs</h2><p>Each point is one completed life, plotted at its ending step. Lines connect individual lives within each run; there is no smoothing or averaging across runs. Survival forks have separate plots below. Ongoing lives, timeouts and interrupted lives are excluded; no zero is invented before a first death. Game time excludes training wall-clock time.</p><div class="charts">')
        for key,title,ylabel,column in [('lifespan_minutes','Individual lifespan · game time','Completed lifespan (game minutes)',1),
                                        ('lifespan_observations','Individual lifespan · interactions','Completed lifespan (interactions)',2)]:
            graphic=charts.chart(lifespan_trials,lifespan,key,title,ylabel,xmax=cap,aggregate=False).replace('mean of six runs','mean of group')
            graphic=graphic.replace('class="comparison-chart"',f'class="comparison-chart" data-source="survival" data-raw="true" data-column="{column}"')
            graphic=graphic.replace('opacity=".35" stroke-width="1.5"','opacity=".85" stroke-width="1.5"')
            parts.append(graphic)
        if not all(lifespan.values()):parts.append('<p>Some runs have no completed life within the shared interval yet; their lifespan comparison is pending.</p>')
    survival_trials=[t for t in current if t.get('survival')]
    if survival_trials:
        from .survival import fork_charts
        parts.append('</div><section id="survival-lifespans"><h2>Survival runs · lifespan after the fork</h2><p>Survival-reward runs only. Each point is one completed life, with no smoothing or averaging. The horizontal axis starts at zero at each survival fork and ends at the furthest of these survival runs; each line extends to its own latest completed life. Game time excludes training time. Ongoing and interrupted lives are excluded.</p>')
        parts.append(fork_charts(survival_trials,SHARED))
        parts.append('</section><div>')
    parts.append('</div><h2>Current agents</h2><p>Videos show the real trajectory, frontier goal and FTN activity. Updated clips overwrite their predecessors.</p><label>Clip length <select id="frames"><option value="64">10.7 s</option><option value="128">21.3 s</option><option value="192">32 s</option><option value="256">42.7 s</option></select></label> <button id="apply">Apply</button> <span id="message"></span>')
    for t in current:
        root = SHARED/'runs'/t['name'];live=root/'live'
        s=read(root/'status.json');e=s.get('environment') or {};v=read(live/'video.json')
        link=OUT/t['name']
        if not link.exists():link.symlink_to(live,target_is_directory=True)
        stamp=v.get('step',0)
        clip='obs.mp4'
        if t.get('m2e_ftn'):
            clip='obs_display.mp4'
            consolidated=read(live/'clip_consolidation.json')
            stamp=consolidated.get('step',0)
            # Frozen older experiments have compact clips at the original path.
            if not (live/clip).exists() and consolidated.get('revision')==3 and stamp==v.get('step'):
                clip='obs.mp4'
        parts.append(f'<section><h2>{html.escape(t["model"])} · seed {t["agent_seed"]}</h2><p>{t.get("pod", "")} GPU {t["gpu"]} · {s.get("step",0):,} observations · {e.get("cells_lifetime",0):,} cells · {checked_achievements(e)} achievements · {t["status"]} · status age {max(0,time.time()-s.get("time",time.time())):.0f}s</p><div class="media"><div>')
        if (live/clip).exists():
            parts.append(f'<video controls playsinline preload="metadata" src="{t["name"]}/{clip}?v={stamp}"></video><p>Observation, goal, FTN activity · video through step {stamp:,}.</p>')
        else:parts.append('<p>Collecting the first trajectory video.</p>')
        parts.append(f'</div><div><img src="maps/{t["name"]}.png?v={int(time.time())}" alt="Coverage map"></div></div>')
        parts.append(survival_panel(read(live/'survival.json'))+achievement_checklist(e))
        parts.append(f'<p><a href="{t["name"]}/history.jsonl">Progress history</a> · <a href="{t["name"]}/events.jsonl">Stagnation and achievement events</a></p><details><summary>Configuration and diagnostics</summary><pre>{html.escape(json.dumps(dict(config=t,status=s,progress=read(live/"progress.json")),indent=2))}</pre></details></section>')
    lives=[]
    from .life_video_retention import threshold
    minimum=threshold(SHARED/'life_settings.json')
    # The recorder applies the threshold when a new life finishes. Browsing the
    # monitor must never remove or hide videos that have already been retained.
    for t in current:
        folder=SHARED/'runs'/t['name']/'life_videos'
        if not folder.exists():continue
        link=OUT/'lives'/t['name'];link.parent.mkdir(exist_ok=True)
        if not link.exists():link.symlink_to(folder,target_is_directory=True)
        for path in folder.glob('*.json'):
            info=read(path)
            if info.get('file'):lives.append((info['end_step'],t,info))
    death_records={t['name']:read(SHARED/'runs'/t['name']/'live/survival.json').get('deaths',[]) for t in current}
    from .life_events import index as life_event_index, dropdown as life_event_dropdown
    life_event_logs={}
    for t in current:
        clips_root=STATE/'runs'/t['name']/'event_clips'
        if not clips_root.exists():clips_root=SHARED/'runs'/t['name']/'event_clips'
        life_event_logs[t['name']]=life_event_index(SHARED/'runs'/t['name'],clips_root)
    if any(t.get('life_capture') for t in current) or lives:
        parts.append(f'<section id="whole-lives"><h2>Entire-life videos</h2><p>Every observation from spawn to death, with goal and FTN activity, played at 6 FPS. Videos appear after the life ends; interrupted lives are labeled partial.</p><label>Keep new lives longer than <input id="life-minutes" type="number" min="0" step="1" value="{minimum//60}"> game minutes (0 = keep all)</label> <button id="life-apply">Apply</button> <span id="life-message"></span><p>Applies only when a new life video finishes recording, including a life currently in progress. Previously saved videos remain available regardless of this setting.</p>')
        for _,t,v in sorted(lives,key=lambda x:x[0],reverse=True):
            src=html.escape(f'lives/{t["name"]}/{v["file"]}',quote=True)
            label='Complete life' if v['complete'] else 'Partial life'
            cause=v.get('death_reason')
            if v.get('end_reason')=='death':
                if not cause:
                    matches=[r for r in death_records[t['name']] if r.get('life')==v['life'] and abs(r.get('seconds',-1)-v['game_seconds'])<0.1]
                    cause=matches[-1]['reason'] if matches else 'unknown (telemetry pending or unavailable)'
                cause='Cause of death: '+cause.replace('_',' ')
            else:cause='No recorded death: '+v.get('end_reason','recording interrupted')
            parts.append(f'<details><summary>{html.escape(t["model"])} · seed {t["agent_seed"]} · {label} {v["life"]} · steps {v["start_step"]:,}–{v["end_step"]:,} · {v["game_seconds"]/60:.2f} game minutes · {v["frames"]:,} frames · {html.escape(cause)}</summary>'+life_event_dropdown(life_event_logs[t['name']],v)+f'<video controls playsinline preload="none" src="{src}"></video><p><a href="{src}">Open video</a></p></details>')
        if not lives:parts.append('<p>No finalized life videos meeting the current threshold yet.</p>')
        parts.append('</section>')
    clips=[]
    for t in current:
        folder=STATE/'runs'/t['name']/'event_clips'
        if not folder.exists():folder=SHARED/'runs'/t['name']/'event_clips'
        if not folder.exists():continue
        link=OUT/'events'/t['name'];link.parent.mkdir(exist_ok=True)
        if not link.exists():link.symlink_to(folder,target_is_directory=True)
        for path in folder.glob('*.json'):
            info=read(path)
            if not info.get('file'):continue
            labels=', '.join(e['kind'].replace('_',' ')+
                (': '+str(e['detail']).split(':')[-1].replace('_',' ') if e.get('detail') else '')
                for e in info.get('events',[]))
            clips.append((path.stat().st_mtime,t,info,labels))
    parts.append(f'<section id="event-clips"><h2>Saved event clips ({len(clips)})</h2><p>Phase 4 event triggers: animal interactions, significant achievements and sustained exploration. Clips retain up to 80 observations before and after each event, played at 6 FPS (about 27 seconds). Events before recording was enabled are not recoverable.</p>')
    def clip_condition(t):
        gamma=t.get('mf_gamma')
        return (format(gamma,'.5g') if gamma is not None else 'Unspecified',
                str(t.get('mf_offline_updates','Unspecified')))
    parts.append('<div class="legend">')
    for index,key,label in [(0,'discount','Discount factor'),(1,'updates','Offline updates per round')]:
        values=sorted({clip_condition(t)[index] for t in current},
                      key=lambda v:float(v) if v!='Unspecified' else float('inf'))
        parts.append(f'<label>{label} <select id="clips-{key}"><option value="">All</option>')
        for value in values:
            parts.append(f'<option value="{html.escape(value,quote=True)}">{html.escape(value)}</option>')
        parts.append('</select></label>')
    parts.append('</div><p id="clips-count" class="muted" aria-live="polite"></p>')
    for _,t,info,label in sorted(clips,key=lambda x:x[0],reverse=True):
        src=html.escape(f'events/{t["name"]}/{info["file"]}',quote=True)
        gamma,updates=clip_condition(t)
        parts.append(f'<details class="saved-event-clip" data-discount="{gamma}" data-updates="{updates}"><summary>γ={gamma} · {updates} updates · Seed {t["agent_seed"]} · step {info["event_step"]:,} · {html.escape(label)}</summary><video controls playsinline preload="none" src="{src}"></video><p><a href="{src}" target="_blank">Open clip</a></p></details>')
    if not clips:parts.append('<p>Waiting for the first triggered clip.</p>')
    parts.append('</section>')
    payload=[dict(name=t['name'],model=t['model'],color=COLORS[t['model']],
        rows=[[r['step'],r.get('total_cells',0),r.get('achievement_checked',0)] for r in clipped[t['name']]]) for t in trials]
    js=Path(__file__).with_name('phase4_curves.js').read_text().replace('phase4-hidden','actions-hidden')
    survival_payload=[dict(name=t['name'],model=t['model'],color=COLORS[t['model']],
        rows=[[r['step'],r['lifespan_minutes'],r['lifespan_observations']] for r in lifespan[t['name']]]) for t in lifespan_trials]
    parts.append('<script>const comparisonPerModelEndpoints=true;const comparisonRuns='+json.dumps(payload)+';const comparisonSurvivalRuns='+json.dumps(survival_payload)+';'+js+'</script>')
    parts.append('''<script>
const clipFilters=['discount','updates'].map(k=>document.getElementById('clips-'+k));
const clipFilterKey='saved-clip-filters:'+location.pathname;
try{const saved=JSON.parse(sessionStorage.getItem(clipFilterKey)||'[]');clipFilters.forEach((el,i)=>{if([...el.options].some(o=>o.value===saved[i]))el.value=saved[i];});}catch(e){}
function filterSavedClips(){
  const [discount,updates]=clipFilters.map(el=>el.value);
  const clips=[...document.querySelectorAll('#event-clips .saved-event-clip')];
  let shown=0;
  clips.forEach(el=>{
    el.hidden=!!((discount&&el.dataset.discount!==discount)||(updates&&el.dataset.updates!==updates));
    if(el.hidden){el.querySelector('video').pause();el.open=false;}else shown++;
  });
  document.getElementById('clips-count').textContent=shown+' of '+clips.length+' saved clips shown'+(shown?'':'. No clips match these filters.');
  try{sessionStorage.setItem(clipFilterKey,JSON.stringify([discount,updates]));}catch(e){}
}
clipFilters.forEach(el=>el.addEventListener('change',filterSavedClips));
filterSavedClips();
fetch('/api/clip-settings').then(r=>r.json()).then(s=>document.getElementById('frames').value=s.frames);
const lifeApply=document.getElementById('life-apply');
if(lifeApply)lifeApply.onclick=async()=>{const minutes=Number(document.getElementById('life-minutes').value);if(!Number.isSafeInteger(minutes)||minutes<0){document.getElementById('life-message').textContent='Enter a nonnegative integer.';return;}const r=await fetch('/api/life-settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({minimum_minutes:minutes})});document.getElementById('life-message').textContent=r.ok?'Applied to newly completed recordings. Existing videos are preserved.':'Invalid setting';};
document.getElementById('apply').onclick=async()=>{const r=await fetch('/api/clip-settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frames:Number(document.getElementById('frames').value)})});document.getElementById('message').textContent=r.ok?'Applied to upcoming clips.':'Invalid setting';};
setInterval(()=>{if(![...document.querySelectorAll("video")].some(v=>!v.paused&&!v.ended))location.reload()},60000)</script></body></html>''')
    temp=OUT/'index.tmp';temp.write_text(''.join(parts));temp.replace(OUT/'index.html')


def monitor(port):
    OUT.mkdir(parents=True,exist_ok=True)
    from . import experiment_navigation
    experiment_navigation.prepare(OUT)
    from .monitor_settings import read_settings, CLIP_CHOICES
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/api/clip-settings','/api/life-settings'):
                from .life_video_retention import threshold
                value=(dict(minimum_minutes=threshold(SHARED/'life_settings.json')//60)
                       if self.path=='/api/life-settings' else read_settings(SHARED/'clip_settings.json'))
                data=json.dumps(value).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            else:
                path=Path(self.translate_path(self.path))
                if path.is_dir():path=path/'index.html'
                if path.name=='index.html' and path.is_file():
                    body=experiment_navigation.decorate(path,path.read_text()).encode()
                    self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
                    self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                else:super().do_GET()
        def do_POST(self):
            if self.path not in ('/api/clip-settings','/api/life-settings'):self.send_error(404);return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=1024:raise ValueError()
                value=json.loads(self.rfile.read(n))
                if self.path=='/api/life-settings':
                    minutes=value['minimum_minutes']
                    if type(minutes) is not int or minutes<0:raise ValueError()
                    atomic(SHARED/'life_settings.json',dict(minimum_minutes=minutes))
                else:
                    frames=value['frames']
                    if type(frames) is not int or frames not in CLIP_CHOICES:raise ValueError()
                    atomic(SHARED/'clip_settings.json',dict(frames=frames))
            except (ValueError,KeyError,TypeError):self.send_error(400);return
            self.send_response(204);self.end_headers()
    server=ThreadingHTTPServer(('0.0.0.0',port),functools.partial(Handler,directory=str(OUT)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    while True:
        try:update()
        except Exception:traceback.print_exc()
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('launch','publish','monitor'))
    p.add_argument('--port',type=int,default=8014);a=p.parse_args()
    if a.command=='launch':launch()
    elif a.command=='publish':publish('globo-actions','actions',SHARED)
    else:monitor(a.port)
