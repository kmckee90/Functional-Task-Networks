"""Pod-side multi-pod monitor; reads immutable publications, never agent buffers."""
import html
import json
import time
import traceback
from pathlib import Path
from .phase2 import STATE,SHARED,MODELS,read,atomic,experiments
from .achievement_charts import checked_achievements,achievement_checklist,COLORS,discovery_label
from .achievement_catalog import LABELS
from .terrain_maps import publish as maps
from .survival import chart as survival_chart,panel as survival_panel,duration

OUT=SHARED/'monitor'
escape=html.escape

def rows(t):
    root=SHARED/'runs'/t['name'];points={}
    try:
        for line in (root/'live/history.jsonl').read_text().splitlines():
            try:r=json.loads(line);points[r['step']]=r
            except (ValueError,KeyError):pass
    except OSError:pass
    s=read(root/'status.json');e=s.get('environment',{})
    if s.get('step') and e:points[s['step']]=dict(e,step=s['step'],total_cells=e.get('cells_lifetime',0))
    return [dict(r,achievement_checked=checked_achievements(r)) for _,r in sorted(points.items())]

def chart(trials,metric,title,ylabel):
    series=[(t,rows(t)) for t in trials];xmax=max([1]+[r['step'] for _,rs in series for r in rs]);ymax=max([1]+[r.get(metric,0) for _,rs in series for r in rs])
    p=[f'<div><h3>{title}</h3><svg viewBox="0 0 600 290" role="img" aria-label="{title}">']
    for i in range(5):
        x=70+510*i/4;y=230-205*i/4
        p.append(f'<path d="M70 {y} H580" stroke="#354151"/><text x="62" y="{y+4}" text-anchor="end">{ymax*i/4:,.0f}</text><text x="{x}" y="250" text-anchor="middle">{xmax*i/4:,.0f}</text>')
    for t,rs in series:
        path=''
        for r in rs:
            if metric=='achievement_checked' and 'achievement_counts' not in r:continue
            x=70+510*r['step']/xmax;y=230-205*r.get(metric,0)/ymax
            path+=f'M{x:.2f} {y:.2f}' if not path else f'H{x:.2f} V{y:.2f}'
        p.append(f'<path class="curve" data-run="{t["name"]}" d="{path}" fill="none" stroke="{COLORS[MODELS.index(t["model"])]}" stroke-width="2.5"/>')
    p.append(f'<text x="320" y="280" text-anchor="middle">Environment observations</text><text transform="translate(16 135) rotate(-90)" text-anchor="middle">{ylabel}</text></svg></div>')
    return ''.join(p)

def update():
    OUT.mkdir(parents=True,exist_ok=True)
    link=STATE/'reviews/config24/phase2'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    trials=experiments();manifests={}
    for pod in {t['pod'] for t in trials}:
        m=read(SHARED/'pods'/f'{pod}.json');manifests[pod]=m
        byname={t['name']:t for t in m.get('trials',[])}
        for t in trials:
            if t['name'] in byname:t.update(byname[t['name']])
    atomic(SHARED/'experiments.json',trials)
    p=['''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Globo · Phase 2</title><style>
body{background:#10151d;color:#e5eaf0;font:16px system-ui;max-width:1400px;margin:auto;padding:24px}a{color:#9bd0ff}h1,h2,h3{line-height:1.25}section{margin:32px 0;padding:22px;background:#171d27;border-radius:12px}.charts,.media{display:grid;grid-template-columns:1fr 1fr;gap:24px}svg,img,video{width:100%;height:auto}svg text{fill:#b9c6d4;font-size:12px}.legend{display:flex;gap:20px;flex-wrap:wrap}.legend label{cursor:pointer}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{text-align:left;vertical-align:top;padding:12px;border:1px solid #354151}td ul{padding-left:17px;margin:0}td li{margin-bottom:7px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}summary{cursor:pointer;margin:12px 0}.muted{color:#a5b3c5}.stat{line-height:1.8}.smallvideo{max-width:320px}.media{align-items:start}@media(max-width:800px){.charts,.media{grid-template-columns:1fr}body{padding:10px}section{padding:12px}table{font-size:12px}td,th{padding:5px}}.curve.hidden{display:none}</style></head><body>
<style>.comparison-charts{grid-template-columns:repeat(3,minmax(0,1fr))}.comparison-charts h3{min-height:2.5em;margin-bottom:0}.comparison-charts>div{min-width:0}@media(max-width:800px){.comparison-charts{grid-template-columns:1fr}}</style>
<h1>Globo · Phase 2</h1><p><a href="/continuous/">Phase 1 — retained agents</a> · Fresh agents, unlimited exploration. All runs use 256×256 rendering with downsampled agent observations. The MLP-FTN pair is deferred.</p>''']
    p.append(f'<p class="muted">Updated {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}. Auto-refresh pauses while a video plays. Videos and checkpoints overwrite their previous versions.</p>')
    p.append('<label>Video length <select id="frames"><option value="64">10.7 s</option><option value="128">21.3 s</option><option value="192" selected>32 s</option><option value="256">42.7 s</option></select></label> <button id="apply">Apply to phase 2</button> <span id="message"></span>')
    groups=[(seed,[t for t in trials if t['seed']==seed]) for seed in (19,101)]
    for seed,ts in groups:
        p.append(f'<section><h2>Seed {seed}</h2><div class="legend">')
        for t in ts:p.append(f'<label style="color:{COLORS[MODELS.index(t["model"])]}"><input class="toggle" type="checkbox" checked data-run="{t["name"]}"> {t["model"]}</label>')
        survival=[(t['name'],t['model'],COLORS[MODELS.index(t['model'])],read(SHARED/'runs'/t['name']/'live/survival.json')) for t in ts]
        p.append('</div><div class="charts comparison-charts">'+chart(ts,'total_cells','Spatial progress','Discovered 4 m × 4 m cells')+chart(ts,'achievement_checked','Cumulative unique achievements','Achievement boxes checked')+'<div><h3>Survival over training</h3>'+survival_chart(survival)+'</div></div>')
        p.append('<p class="muted">Survival: mean lifespan of the last 20 recorded deaths (or all available, if fewer). Ongoing and interrupted lives are excluded. The same legend controls all three charts.</p>')
        p.append('<table><thead><tr><th>Survival</th>'+''.join(f'<th>{t["model"]}</th>' for t in ts)+'</tr></thead><tbody>')
        for label,field in [('Current life','current'),('Mean completed life','mean'),('Median completed life','median'),('Longest completed life','best'),('Recent mean (up to 20)','recent'),('Recorded deaths','deaths')]:
            p.append('<tr><th>'+label+'</th>')
            for _,_,_,su in survival:
                if field=='current':value=duration((su.get('current') or {}).get('seconds'))+(' (ended)' if (su.get('current') or {}).get('ended') else '')
                elif field=='deaths':value=str(su.get('recorded_deaths','—'))
                elif field=='recent':value=duration((su.get('recent',{}).get('seconds') or {}).get('mean'))
                else:value=duration((su.get('all',{}).get('seconds') or {}).get(field))
                p.append('<td>'+value+'</td>')
            p.append('</tr>')
        p.append('</tbody></table>')
        p.append('<h3>Achievements gained</h3><p class="muted">Each checked category or distinct discovery counts once; repetitions do not raise the curves.</p><table><thead><tr>'+''.join(f'<th>{t["model"]}</th>' for t in ts)+'</tr></thead><tbody><tr>')
        for t in ts:
            e=read(SHARED/'runs'/t['name']/'status.json').get('environment',{})
            labels=[label for key,label in LABELS.items() if e.get('achievement_counts',{}).get(key,0)>0]+[discovery_label(k) for k in e.get('achievement_discoveries',{})]
            p.append('<td><ul>'+''.join('<li>'+escape(label)+'</li>' for label in labels)+'</ul></td>')
        p.append('</tr></tbody></table></section>')
    for seed,ts in groups:
        info=maps(seed,ts)
        p.append(f'<h2>Seed {seed} · live runs</h2><p class="muted">Maps share the same terrain, bounds and intensity scale. Orange/red = more visits (log scale, 1–{info["max_visits"]:,} observations per cell); white = recent path/current heading; blue square = spawn. Terrain {info["tiles_ready"]}/{info["tiles_requested"]} tiles ready; dark areas await mapping. Background shows the original seeded surface, including canopy, rather than modifications by individual agents.</p>')
        for t in ts:
            root=SHARED/'runs'/t['name'];live=root/'live';s=read(root/'status.json');progress=read(live/'progress.json');v=read(live/'video.json');e=s.get('environment',{})
            dest=OUT/t['name']
            if live.exists() and not dest.exists():dest.symlink_to(live,target_is_directory=True)
            age=int(time.time()-s['time']) if s.get('time') else None;pod_age=int(time.time()-manifests[t['pod']].get('time',0))
            status=t['status'] if pod_age<120 else 'publisher pending / unavailable'
            p.append(f'<section id="{t["name"]}"><h2>{t["model"]} · seed {seed}</h2><p class="stat">{status} · {t["pod"]}, GPU {t["gpu"]} · <b>{s.get("step",0):,}</b> observations · {s.get("updates",0):,} updates · status age {age if age is not None else "—"} s · starts {t["attempts"]}</p>')
            if progress:p.append(f'<p>Cells {progress.get("total_cells",0):,}; new in last window {progress.get("new_cells",0):,}; steps since discovery {progress.get("steps_since_discovery",0):,}; stagnant: {progress.get("stalled")}; first stagnation {progress.get("first_stagnation")}; longest stall {progress.get("longest_stall",0):,} observations.</p>')
            p.append('<div class="media"><div>')
            prefix=t['name'];stamp=v.get('step',0)
            if v:
                p.append(f'<p>Current trajectory: {v.get("frame_start",0):,}–{v.get("frame_end",0):,} observations · {v.get("playback_seconds",0):.1f} seconds</p>')
                if t.get('m2e_ftn'):
                    p.append(f'<video controls playsinline preload="metadata" src="{prefix}/obs.mp4?v={stamp}"></video><p class="muted">Observation left; frontier goal above the concurrent FTN mask on the right.</p><p><a href="{prefix}/mask.html" target="_blank">Mask variability diagnostic ↗</a></p>')
                elif t.get('model_free_only'):
                    src='human.mp4' if (live/'human.mp4').exists() else 'obs.mp4'
                    p.append(f'<video class="smallvideo" controls playsinline preload="metadata" src="{prefix}/{src}?v={stamp}"></video>')
                    if (live/'offline_goal.mp4').exists():p.append(f'<p>Offline frontier goal</p><video class="smallvideo" controls playsinline preload="none" src="{prefix}/offline_goal.mp4?v={stamp}"></video>')
                    if (live/'agent_input.mp4').exists():p.append(f'<details><summary>Agent RGB / feedback input</summary><video class="smallvideo" controls playsinline preload="none" src="{prefix}/agent_input.mp4?v={stamp}"></video></details>')
                else:
                    p.append(f'<video class="smallvideo" controls playsinline preload="metadata" src="{prefix}/obs.mp4?v={stamp}"></video><p>Observation · reconstruction · context / predictive rollout</p><video controls playsinline preload="none" src="{prefix}/recent.mp4?v={stamp}"></video><p class="muted">Reconstruction MSE {v.get("reconstruction_mse",0):.5f}; forecast MSE {v.get("forecast_mse",0):.5f}.</p>')
            else:p.append('<p>Waiting for the first trajectory clip.</p>')
            p.append(f'</div><div><img src="maps/{t["name"]}.png?v={int(time.time())}" alt="Comparable terrain and visitation heatmap"><p class="muted">Visitation through observation {info["steps"].get(t["name"]) or 0:,}</p></div></div>')
            p.append(survival_panel(read(live/'survival.json')))
            p.append(achievement_checklist(e))
            p.append('<details><summary>All live metrics and configuration</summary><pre>'+escape(json.dumps(dict(status=s,progress=progress,configuration=t,architecture=read(root/'architecture.json')),indent=2))+'</pre></details>')
            p.append(f'<p><a href="{prefix}/history.jsonl">Progress history</a> · <a href="{prefix}/events.jsonl">Achievements / stagnation events</a> · <a href="{prefix}/video.json">Video provenance</a></p></section>')
    p.append('''<p class="muted">Stagnation: no new spatial cell for 5,000 observations. This is a diagnostic, not a stopping condition. Terrain and positions are evaluation-only and never enter agent observations or rewards.</p><script>
let hidden=JSON.parse(localStorage.getItem('phase2-hidden')||'{}');
for(const c of document.querySelectorAll('.toggle')){c.checked=!hidden[c.dataset.run];const update=()=>{document.querySelectorAll('.curve').forEach(p=>{if(p.dataset.run===c.dataset.run)p.classList.toggle('hidden',!c.checked)});hidden[c.dataset.run]=!c.checked;localStorage.setItem('phase2-hidden',JSON.stringify(hidden))};c.onchange=update;update()}
fetch('/phase2/api/clip-settings').then(r=>r.json()).then(s=>document.getElementById('frames').value=s.frames).catch(()=>{});
document.getElementById('apply').onclick=async()=>{try{const r=await fetch('/phase2/api/clip-settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frames:Number(document.getElementById('frames').value)})});if(!r.ok)throw Error(await r.text());document.getElementById('message').textContent='Applied; upcoming clips will use this length.'}catch(e){document.getElementById('message').textContent=e.message}};
setInterval(()=>{if(![...document.querySelectorAll('video')].some(v=>!v.paused&&!v.ended))location.reload()},60000);
</script></body></html>''')
    tmp=OUT/'index.tmp';tmp.write_text(''.join(p));tmp.replace(OUT/'index.html')

def main():
    import fcntl,os
    os.nice(10)
    with (STATE/'phase2_monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:update()
            except Exception:traceback.print_exc()
            time.sleep(30)

if __name__=='__main__':main()
