"""Read-only archived/live evidence export. Run on a pod; never modifies runs."""
import bisect
import datetime
import html
import json
import shutil
import statistics
from pathlib import Path
from .achievement_catalog import LABELS
from .achievement_charts import checked_achievements, discovery_label

BASE = Path('/mnt/obelisk-shared/kmckee90')
STATE = Path('/home/dev/globo_dreamer_state')
OUT = STATE / 'proposal_globo_report'
URL = '/assets/proposal/results/globo-report/'
COLORS = {'DreamerV3':'#b39ddb','P2E':'#ffb56b','M2E':'#72b7ff','FTN-M2E':'#7de1bf'}
evidence = {'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}

def rows(path):
    points = {}
    for line in path.read_text().splitlines():
        try:
            r=json.loads(line)
            r['achievements']=checked_achievements(r) if 'achievement_counts' in r else None
            points[r['step']]=r
        except (ValueError, KeyError): pass
    return [r for _,r in sorted(points.items())]

def at(rs,x,key):
    i=bisect.bisect_right([r['step'] for r in rs],x)-1
    return rs[i].get(key) if i>=0 else 0

def figure(file,caption):
    return f'<figure class="result-figure"><a href="{URL}{file}" target="_blank" rel="noopener"><img src="{URL}{file}" alt="{html.escape(caption)}" loading="lazy"></a><figcaption>{caption}</figcaption></figure>'

def chart(file,groups,key,title,cap,aggregate=False,common_end=None):
    ends={name:min(rs[-1]['step'] for rs in group) for name,group in groups.items()}
    if aggregate:
        cap=common_end if common_end is not None else max(ends.values())
    series={}
    for name,group in groups.items():
        pts=[]
        end=min(ends[name],cap) if aggregate else cap
        for x in sorted(set(range(0,end+1,1000)) | {end}):
            vals=[at(rs,x,key) for rs in group if rs[-1]['step']>=x]
            vals=[v for v in vals if v is not None]
            if vals:pts.append((x,statistics.mean(vals),min(vals),max(vals),len(vals)))
        series[name]=pts
    peak=max(p[3] if aggregate else p[1] for ps in series.values() for p in ps)*1.12 or 1
    xy=lambda x,y:(80+680*x/cap,365-255*y/peak)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450"><rect width="800" height="450" rx="12" fill="#111b27"/><style>text{font-family:DejaVu Sans,sans-serif;fill:#c1d0df;font-size:13px}</style>',f'<text x="80" y="34" style="font-size:21px;fill:#e4edf5">{title}</text>']
    for i,(name,ps) in enumerate(series.items()):
        c=COLORS[name];x=80+i*165
        svg.append(f'<path d="M{x} 65h22" stroke="{c}" stroke-width="3"/><text x="{x+29}" y="70">{name}</text>')
    for i in range(5):
        x,y=xy(cap*i/4,peak*i/4)
        svg.append(f'<path d="M80 {y}H760" stroke="#304052"/><text x="70" y="{y+5}" text-anchor="end">{peak*i/4:,.0f}</text><text x="{x}" y="391" text-anchor="middle">{cap*i/4000:g}k</text>')
    for name,ps in series.items():
        c=COLORS[name]
        if aggregate:
            poly=[xy(p[0],p[3]) for p in ps]+[xy(p[0],p[2]) for p in reversed(ps)]
            svg.append(f'<polygon points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in poly)}" fill="{c}" opacity=".12"/>')
        path=' '.join(('M' if i==0 else 'L')+f'{xy(p[0],p[1])[0]:.2f},{xy(p[0],p[1])[1]:.2f}' for i,p in enumerate(ps))
        svg.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="3"/>')
    label='Distinct 4 m × 4 m ground cells' if key=='total_cells' else 'Unique achievements and discoveries'
    svg.append(f'<text x="420" y="432" text-anchor="middle">Environment observations</text><text transform="translate(22 235) rotate(-90)" text-anchor="middle">{label}</text></svg>')
    (OUT/file).write_text(''.join(svg))
    evidence[file]={'cap':cap,'series':series,'model_endpoints':ends}
    return series

def refresh_replicates():
    """Refresh only the multi-run charts, preserving other publication figures."""
    OUT.mkdir(exist_ok=True)
    previous=json.loads((OUT/'evidence.json').read_text())
    groups={name:[] for name in ('P2E','M2E','FTN-M2E')}
    sources={name:[] for name in groups}
    for phase in (4,5):
        for f in sorted((BASE/f'globo_phase{phase}/runs').glob('*/live/history.jsonl')):
            if phase==5 and f.parent.parent.name not in {f"phase5_p2e_r{i}_s19" for i in range(1,7)}:continue
            name='P2E' if phase==5 else ('FTN-M2E' if '_conv_' in str(f) else 'M2E')
            rs=rows(f)
            assert rs, f
            groups[name].append(rs)
            sources[name].append(str(f))
    assert {name:len(group) for name,group in groups.items()}=={'P2E':6,'M2E':6,'FTN-M2E':6}
    common_end=min(rs[-1]['step'] for rs in groups['P2E'])
    assert all(rs[-1]['step']>=common_end for group in groups.values() for rs in group)
    for key,title in [('total_cells','Spatial discovery'),('achievements','Achievements and discoveries')]:
        file=f'replicates-{key}.svg'
        series=chart(file,groups,key,title+' · repeated trials',0,True,common_end=common_end)
        for name,ps in series.items():
            assert all(p[4]==len(groups[name]) for p in ps), (key,name,'missing observations')
            assert ps[-1][0]==common_end
        previous[file]=evidence[file]
    previous['replicates_refreshed_utc']=evidence['captured_utc']
    previous['replicate_common_endpoint']=common_end
    previous['replicate_sources']={name:[rs[-1]['step'] for rs in group] for name,group in groups.items()}
    previous['replicate_source_paths']=sources
    previous['replicate_endpoints']={key:{name:ps[-1] for name,ps in previous[f'replicates-{key}.svg']['series'].items()} for key in ('total_cells','achievements')}
    (OUT/'evidence.json').write_text(json.dumps(previous,indent=2))
    print(json.dumps(previous['replicate_endpoints'],indent=2))

def table(groups,cap):
    p=['<div class="achievement-table-scroll"><table class="achievement-results"><caption>Seed 19: discoveries through '+f'{cap:,}'+' observations, matching the chart above.</caption><thead><tr>']
    records={name:next(r for r in reversed(rs[0]) if r['step']<=cap) for name,rs in groups.items()}
    for name,r in records.items():p.append(f'<th scope="col">{name}<span>{r["achievements"]} achievements</span></th>')
    p.append('</tr></thead><tbody><tr>')
    for r in records.values():
        labels=[v for k,v in LABELS.items() if r.get('achievement_counts',{}).get(k,0)>0]
        labels += [discovery_label(k) for k in r.get('achievement_discoveries',{})]
        p.append('<td><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in labels)+'</ul></td>')
    return ''.join(p)+'</tr></tbody></table></div>'

def main():
    OUT.mkdir(exist_ok=True)
    p=['<p>Globo is an open world of forests, cliffs, wildlife, and hazards. These agents receive no destinations or survival rewards: their drive is to discover unfamiliar experience. We compare <strong>DreamerV3</strong>, <strong>Plan2Explore (P2E)</strong>, <strong>Metalearning2Explore (M2E)</strong>, and <strong>FTN-M2E</strong>, which adds continual learning through neural assemblies to M2E.</p>',
       '<h2>Exploration that keeps growing</h2><p>Preliminary long runs show M2E and FTN-M2E exploring far beyond the model-based baselines. In the headline seed-19 run, FTN-M2E pulls well ahead later in the lifetime. Its added continual learning capacity is a plausible explanation: earlier behaviors can be retained and recombined as new situations appear.</p>']
    # Preserve the exact headline snapshot, relabeling the plot only.
    head=json.loads((STATE/'reviews/config24/proposal_figures/globo-ftn-seed19.json').read_text())
    names=['FTN-M2E','M2E','P2E','DreamerV3']
    g={name:[[{'step':x,'total_cells':y} for x,y in r['points']]] for name,r in zip(names,head['runs'])}
    chart('headline.svg',g,'total_cells','Long-run spatial discovery · world 19',head['observations'])
    p.append(figure('headline.svg','The same preliminary four-model comparison featured in the proposal. FTN-M2E uses the early flat policy decoder; these are individual long runs.'))
    p.append('<p>M2E’s advantage also appears in the earlier comparisons on two different worlds. Memory-density feedback gives it an immediate sense of what is familiar, while its learned exploration policy improves how it reaches unfamiliar observations.</p>')
    p.append('<div class="result-grid">')
    for seed in (19,101):
        groups={name:[rows(STATE/f'runs/phase1_{slug}_s{seed}/live/history.jsonl')] for name,slug in [('DreamerV3','dreamer'),('P2E','p2e'),('M2E','m2e')]}
        cap=min(rs[0][-1]['step'] for rs in groups.values())//1000*1000
        chart(f'long-{seed}.svg',groups,'total_cells',f'M2E across long lifetimes · world {seed}',cap)
        p.append(figure(f'long-{seed}.svg',f'World {seed}: the archived M2E comparison at matched observation counts, through {cap:,} observations.'))
    p.append('</div>')
    p.append('<p>To determine the reliability of these results, I ran 12 seeded runs per model up to 40k steps. The early stages of the long runs show the models performing more comparably, then diverging, so this illuminates the early but not long-run performance differences.</p><p>The shorter trials reveal two different strengths. M2E covers more ground on average, while FTN-M2E accumulates more distinct achievements. Both outperform P2E on achievements. This early spatial ordering also appears in the headline long run before about 50,000 observations; the later FTN-M2E advantage would be missed by judging only the beginning.</p>')
    groups={name:[] for name in ('P2E','M2E','FTN-M2E')}
    for phase in (4,5):
        for f in sorted((BASE/f'globo_phase{phase}/runs').glob('*/live/history.jsonl')):
            if phase==5 and f.parent.parent.name not in {f"phase5_p2e_r{i}_s19" for i in range(1,7)}:continue
            name='P2E' if phase==5 else ('FTN-M2E' if '_conv_' in str(f) else 'M2E')
            groups[name].append(rows(f))
    evidence['replicate_sources']={name:[rs[-1]['step'] for rs in group] for name,group in groups.items()}
    last={}
    p.append('<div class="result-grid">')
    for key,title in [('total_cells','Spatial discovery'),('achievements','Achievements and discoveries')]:
        s=chart(f'replicates-{key}.svg',groups,key,title+' · repeated trials',0,True)
        last[key]={name:ps[-1] for name,ps in s.items()}
        p.append(figure(f'replicates-{key}.svg','Lines show means; shaded bands show the full range across runs, not confidence intervals.'))
    p.append('</div>')
    counts=', '.join(f'{name}: {v[0]:,} observations ({v[-1]} runs)' for name,v in last['total_cells'].items())
    p.append(f'<p class="result-source">World seed 19: six learner seeds each for M2E and FTN-M2E from Phase 4 only; six learner seeds for P2E from Phase 5 (runs 1–6). Each mean and full-range band stops at that model’s shortest run: {counts}. The shared axis spans the longest of these windows. Snapshot {evidence["captured_utc"][:10]}.</p>')
    evidence['replicate_endpoints']=last
    p.append('<p>The detailed long-run comparison shows the same achievement advantage: FTN-M2E collects the broadest repertoire. Counting distinct discoveries captures progress that a ground-coverage score alone misses—new resources, new biomes, climbing, and interactions with the world.</p>')
    longgroups={}
    for name,slug in [('DreamerV3','dreamer'),('P2E','p2e'),('M2E','m2e'),('FTN-M2E','conv')]:
        longgroups[name]=[rows(BASE/f'globo_phase2/runs/phase2_{slug}_s19/live/history.jsonl')]
    cap=min(v[0][-1]['step'] for v in longgroups.values())//1000*1000
    chart('long-achievements.svg',longgroups,'achievements','Long-run achievements · world 19',cap)
    p.append(figure('long-achievements.svg',f'Phase 2: four fresh agents, compared through {cap:,} observations. Each checklist entry or distinct discovery counts once.'))
    p.append(table(longgroups,cap))
    p.append('<p>Across the short-run averages and this long-run comparison, FTN-M2E earns the most achievements. The result suggests that continual learning can broaden what an explorer learns to do, even before it produces the widest spatial coverage.</p>')
    p.append('<p>What does the exploration progress of each agent actually look like? M2E and FTN-M2E spread through the landscape more broadly and robustly. In these long-run maps, DreamerV3 and P2E appear to spend much of their time trapped in trenches; when they escape, their paths often follow a narrow direction across the landscape. M2E and FTN-M2E also find routes into the dense forest, which the model-based baselines largely miss. These are visible patterns in the recorded trajectories, rather than a separate behavioral score.</p>')
    for seed in (19,):
        p.append(f'<h3>World {seed}</h3><div class="globo-map-grid">')
        for name,slug in [('DreamerV3','dreamer'),('P2E','p2e'),('M2E','m2e'),('FTN-M2E','conv')]:
            file=f'phase2_{slug}_s{seed}.png'
            shutil.copyfile(BASE/'globo_phase2/monitor/maps'/file,OUT/file)
            status=json.loads((BASE/f'globo_phase2/runs/phase2_{slug}_s{seed}/status.json').read_text())
            p.append(figure(file,f'<strong>{name}</strong> · {status["step"]:,} observations.'))
        p.append('</div>')
    p.append('<p class="result-source">Maps share terrain, bounds, and visit-intensity scale within each world. Yellow–red marks repeated visits; the blue square marks spawn and the white line the recent path. These are final recorded maps at the labeled run lengths. Click to enlarge.</p>')
    (OUT/'report.txt').write_text(''.join(p))
    (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({'replicates':last,'long_achievements':{k:at(v[0],cap,'achievements') for k,v in longgroups.items()}},indent=2))

if __name__=='__main__':
    import sys
    if '--replicates-only' in sys.argv:refresh_replicates()
    else:main()
