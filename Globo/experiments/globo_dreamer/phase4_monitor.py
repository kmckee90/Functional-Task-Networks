"""Phase 4: paired replicates, aggregate curves and live per-run evidence."""
import bisect,fcntl,html,json,statistics,time,traceback
from pathlib import Path
from .phase4 import SHARED,STATE,experiments
from .phase2 import read,atomic
from .achievement_charts import checked_achievements,achievement_checklist,discovery_label
from .achievement_catalog import LABELS
from .survival import summary,panel as survival_panel
from . import terrain_maps

OUT=SHARED/'monitor';COLORS={'M2E':'#72b7ff','M2E+FTN':'#7de1bf'}

def history(t):
    root=SHARED/'runs'/t['name'];points={}
    try:
        for line in (root/'live/history.jsonl').read_text().splitlines():
            try:r=json.loads(line);points[r['step']]=r
            except (ValueError,KeyError):pass
    except OSError:pass
    s=read(root/'status.json');e=s.get('environment',{})
    if s.get('step') and e:points[s['step']]=dict(e,step=s['step'],total_cells=e.get('cells_lifetime',0))
    return [dict(r,achievement_checked=checked_achievements(r)) for _,r in sorted(points.items())]

def at(rows,step,key):
    i=bisect.bisect_right([r['step'] for r in rows],step)-1
    return rows[i].get(key,0) if i>=0 else 0

def chart(trials,data,key,title,ylabel):
    xmax=max([1]+[r['step'] for rows in data.values() for r in rows]);ymax=max([1]+[r.get(key,0) for rows in data.values() for r in rows])*1.05
    def xy(x,y):return 70+655*x/xmax,315-230*y/ymax
    p=[f'<div><h3>{title}</h3><svg class="comparison-chart" data-metric="{key}" data-xmax="{xmax}" data-ymax="{ymax}" viewBox="0 0 760 390" role="img" aria-label="{title}">']
    for i in range(5):
        x,y=xy(xmax*i/4,ymax*i/4)
        p.append(f'<path d="M70 {y}H725" stroke="#354151"/><text x="60" y="{y+4}" text-anchor="end">{ymax*i/4:,.0f}</text><text x="{x}" y="340" text-anchor="middle">{xmax*i/4:,.0f}</text>')
    for t in trials:
        pts=[xy(r['step'],r.get(key,0)) for r in data[t['name']]]
        path=' '.join(('M' if i==0 else 'L')+f'{x:.1f},{y:.1f}' for i,(x,y) in enumerate(pts))
        p.append(f'<path class="curve" data-run="{t["name"]}" d="{path}" fill="none" stroke="{COLORS[t["model"]]}" opacity=".35" stroke-width="1.5"><title>{t["model"]} · pair {t["pair"]}</title></path>')
    p.append('<g class="aggregate">')
    complete=all(data[t['name']] for t in trials)
    if complete:
        cap=min(data[t['name']][-1]['step'] for t in trials)
        for model,color in COLORS.items():
            group=[data[t['name']] for t in trials if t['model']==model]
            start=max(rs[0]['step'] for rs in group)
            xs=sorted({start,cap}|{r['step'] for rs in group for r in rs if start<=r['step']<=cap})
            means=[statistics.mean(at(rs,x,key) for rs in group) for x in xs]
            upper=[xy(x,max(at(rs,x,key) for rs in group)) for x in xs]
            lower=[xy(x,min(at(rs,x,key) for rs in group)) for x in reversed(xs)]
            points=' '.join(f'{x:.1f},{y:.1f}' for x,y in upper+lower)
            p.append(f'<polygon points="{points}" fill="{color}" opacity=".09"/>')
            path=' '.join(('M' if i==0 else 'L')+f'{xy(x,y)[0]:.1f},{xy(x,y)[1]:.1f}' for i,(x,y) in enumerate(zip(xs,means)))
            p.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4"><title>{model}: mean of six runs</title></path>')
    p.append(f'</g><text x="400" y="375" text-anchor="middle">Environment observations</text><text transform="translate(18 205) rotate(-90)" text-anchor="middle">{ylabel}</text></svg></div>')
    return ''.join(p)

def update():
    OUT.mkdir(parents=True,exist_ok=True)
    link=STATE/'reviews/config24/phase4'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    trials=experiments()
    for pod in {t['pod'] for t in trials}:
        byname={t['name']:t for t in read(SHARED/'pods'/f'{pod}.json').get('trials',[])}
        for t in trials:
            if t['name'] in byname:t.update(byname[t['name']])
    data={t['name']:history(t) for t in trials}
    terrain_maps.SHARED=SHARED;maps=terrain_maps.publish(19,trials)
    p=['''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Globo · Phase 4</title><style>
body{background:#10151d;color:#e5eaf0;font:16px system-ui;max-width:1450px;margin:auto;padding:24px}a{color:#9bd0ff}section{margin:28px 0;padding:22px;background:#171d27;border-radius:12px}.charts,.media{display:grid;grid-template-columns:1fr 1fr;gap:24px}svg,img,video{width:100%;height:auto}svg text{fill:#b9c6d4;font-size:13px}.legend{display:flex;gap:16px;flex-wrap:wrap}.muted{color:#a5b3c5}table{width:100%;border-collapse:collapse}td,th{border:1px solid #354151;padding:10px;text-align:left;vertical-align:top}.scroll{overflow:auto}.achievements{min-width:1800px;table-layout:fixed;font-size:13px}.achievements ul{padding-left:16px}li{margin-bottom:6px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}.smallvideo{max-width:384px}.curve.hidden{display:none}@media(max-width:800px){.charts,.media{grid-template-columns:1fr}body{padding:10px}section{padding:12px}}</style></head><body><h1>Phase 4 · Continual learning and exploration</h1><p>World 19 · six fresh M2E runs and six fresh M2E+FTN runs. Within each pair, agent seed, world, spawn, reservoir settings, buffers, and exploration hyperparameters match. FTN uses the convolutional policy decoder. Frontier exhaustion switching disabled. No Dreamer, warm-start, or step limit.</p><p><a href="/phase3/">Phase 3 archive</a> · <a href="/continuous/">Phase 1 archive</a> · <a href="/phase2/">Phase 2 archive</a></p>''']
    p.append(f'<p class="muted">Updated {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}. Thin lines: selected individual runs. Thick lines: means of selected runs, ending at their shared observation count. Shading: selected-run minimum–maximum, not a confidence interval.</p><div class="legend">')
    for t in trials:p.append(f'<label style="color:{COLORS[t["model"]]}"><input class="toggle" data-run="{t["name"]}" type="checkbox" checked>{t["model"]} · {t["pair"]}</label>')
    p.append('</div><p class="muted" id="selection-stats"></p><div class="charts">'+chart(trials,data,'total_cells','Spatial progress','Discovered 4 m × 4 m cells')+chart(trials,data,'achievement_checked','Unique achievements','Achievement boxes checked')+'</div>')
    if all(data[t['name']] for t in trials):
        cap=min(rs[-1]['step'] for rs in data.values())
        p.append(f'<h2>All six pairs at {cap:,} observations</h2><p class="muted">This reference table always includes all pairs; checkbox selection applies to the charts above.</p><table><tr><th>Pair / agent seed</th><th>M2E cells / achievements</th><th>M2E+FTN cells / achievements</th><th>FTN − M2E cells / achievements</th></tr>')
        for pair in range(1,7):
            group=[t for t in trials if t['pair']==pair];values=[[at(data[t['name']],cap,k) for k in ('total_cells','achievement_checked')] for t in group]
            p.append(f'<tr><td>{pair} / {group[0]["agent_seed"]}</td><td>{values[0][0]:,} / {values[0][1]}</td><td>{values[1][0]:,} / {values[1][1]}</td><td>{values[1][0]-values[0][0]:+,} / {values[1][1]-values[0][1]:+}</td></tr>')
        p.append('</table>')
    achievement_trials=sorted(trials,key=lambda t:(t['model']!='M2E',t['pair']))
    p.append('<h2>Achievements gained · latest per run</h2><div class="scroll"><table class="achievements"><tr>'+''.join(f'<th>{t["model"]}<br>Pair {t["pair"]}</th>' for t in achievement_trials)+'</tr><tr>')
    for t in achievement_trials:
        e=read(SHARED/'runs'/t['name']/'status.json').get('environment',{})
        labels=[v for k,v in LABELS.items() if e.get('achievement_counts',{}).get(k,0)>0]+list(e.get('achievement_discoveries',{}))
        p.append('<td><ul>'+''.join(f'<li>{html.escape(discovery_label(v))}</li>' for v in labels)+'</ul></td>')
    p.append('</tr></table></div><h2>Current behavior</h2><p class="muted">Maps use identical terrain, bounds, and visit-intensity scale. Orange/red: repeated visits. Thin white line: recent path. Blue square: spawn. Map positions are evaluation-only.</p><label>Clip length <select id="frames"><option value="64">10.7 s</option><option value="128">21.3 s</option><option value="192" selected>32 s</option><option value="256">42.7 s</option></select></label> <button id="apply">Apply</button> <span id="message"></span>')
    for t in trials:
        root=SHARED/'runs'/t['name'];live=root/'live';s=read(root/'status.json');e=s.get('environment',{});v=read(live/'video.json');progress=read(live/'progress.json')
        prefix=t['name'];link=OUT/prefix
        if not link.exists():link.symlink_to(live,target_is_directory=True)
        age=time.time()-s.get('time',time.time());stamp=v.get('step',s.get('step',0))
        p.append(f'<section><h2>{t["model"]} · pair {t["pair"]}</h2><p>{t["pod"]} GPU {t["gpu"]} · agent seed {t["agent_seed"]} · {s.get("step",0):,} observations · {e.get("cells_lifetime",0):,} cells · {checked_achievements(e)} achievements · {t["status"]} · status age {age:.0f}s</p><div class="media"><div>')
        if (live/'obs.mp4').exists():
            src='obs.mp4' if t.get('m2e_ftn') else 'human.mp4' if (live/'human.mp4').exists() else 'obs.mp4'
            p.append(f'<video controls playsinline preload="metadata" src="{prefix}/{src}?v={stamp}"></video>')
            if t.get('m2e_ftn'):p.append(f'<p>Observation left; goal upper right; FTN mask lower right. <a href="{prefix}/mask.html" target="_blank">Mask diagnostic ↗</a></p>')
            else:
                for file,label in [('offline_goal.mp4','Offline frontier goal'),('agent_input.mp4','Exact policy input')]:
                    if (live/file).exists():p.append(f'<details><summary>{label}</summary><video class="smallvideo" controls playsinline preload="none" src="{prefix}/{file}?v={stamp}"></video></details>')
        else:p.append('<p>Waiting for first trajectory video.</p>')
        p.append(f'</div><div><img src="maps/{t["name"]}.png?v={int(time.time())}" alt="Visitation heatmap"><p>Map through observation {maps["steps"].get(t["name"]) or 0:,}</p></div></div>')
        p.append(survival_panel(read(live/'survival.json'))+achievement_checklist(e))
        p.append(f'<details><summary>Progress, stagnation and configuration</summary><pre>{html.escape(json.dumps(dict(status=s,progress=progress,config=t),indent=2))}</pre></details><p><a href="{prefix}/history.jsonl">Progress history</a> · <a href="{prefix}/events.jsonl">Stagnation / achievement log</a> · <a href="{prefix}/video.json">Video timing</a></p></section>')
    saved_clips=[]
    for t in trials:
        if t.get('event_capture'):
            clips=STATE/'runs'/t['name']/'event_clips'
            if clips.exists():
                event_link=OUT/'events'/t['name'];event_link.parent.mkdir(exist_ok=True)
                if not event_link.exists():event_link.symlink_to(clips,target_is_directory=True)
                files=sorted(clips.glob('*.json'),key=lambda f:f.stat().st_mtime,reverse=True)
                for file in files:
                    info=read(file)
                    label=', '.join(x['kind'].replace('_',' ')+((': '+x['detail'].split(':')[-1].replace('_',' ')) if x.get('detail') else '') for x in info.get('events',[]))
                    src='events/'+t['name']+'/'+info.get('file','')
                    saved_clips.append((file.stat().st_mtime,src,label))
    p.append(f'<section id="event-clips"><h2>Saved event clips ({len(saved_clips)})</h2><p>All captures, newest first. New clips show 80 observations either side of the event at 6 FPS (about 27 seconds). Incomplete context is shortened rather than frozen. Older clips retain their original 15-second timing.</p>')
    for captured,src,label in sorted(saved_clips,reverse=True):
        safe=html.escape(src,quote=True);date=time.strftime('%d %b %H:%M:%S UTC',time.gmtime(captured))
        p.append(f'<details><summary>{html.escape(label.capitalize())} <span class="muted">· {date}</span></summary><video class="smallvideo" controls playsinline preload="none" src="{safe}"></video><p><a href="{safe}" target="_blank">Open clip ↗</a></p></details>')
    if not saved_clips:p.append('<p>No saved clips yet.</p>')
    p.append('</section>')
    payload=[dict(name=t['name'],model=t['model'],color=COLORS[t['model']],rows=[[r['step'],r.get('total_cells',0),r.get('achievement_checked',0)] for r in data[t['name']]]) for t in trials]
    p.append('<script>const comparisonRuns='+json.dumps(payload)+';'+Path(__file__).with_name('phase4_curves.js').read_text()+'</script>')
    p.append('''<script>
fetch('/phase4/api/clip-settings').then(r=>r.json()).then(s=>document.getElementById('frames').value=s.frames).catch(()=>{});document.getElementById('apply').onclick=async()=>{try{const r=await fetch('/phase4/api/clip-settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frames:Number(document.getElementById('frames').value)})});if(!r.ok)throw Error(await r.text());document.getElementById('message').textContent='Applied to upcoming clips.'}catch(e){document.getElementById('message').textContent=e.message}};setInterval(()=>{if(![...document.querySelectorAll('video')].some(v=>!v.paused&&!v.ended))location.reload()},60000);</script></body></html>''')
    tmp=OUT/'index.tmp';tmp.write_text(''.join(p));tmp.replace(OUT/'index.html')

def main():
    import os
    os.nice(10)
    with (STATE/'phase4_monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:update()
            except Exception:traceback.print_exc()
            time.sleep(30)

if __name__=='__main__':main()
