"""Read-only lifespan summaries from recorded episode endings; no learner changes."""
import html
import json
from pathlib import Path
import statistics

_CACHE={}

def completed(root):
    path=Path(root)/'environment/episodes.jsonl'
    try:st=path.stat()
    except OSError:return [],0,False
    signature=(st.st_mtime_ns,st.st_size)
    cached=_CACHE.get(str(path))
    if cached and cached[0]==signature:return cached[1:]
    deaths=[];timeouts=0;seen=set()
    with path.open() as f:
        for line in f:
            if not line.endswith('\n'):continue # Writer may be appending this record.
            try:r=json.loads(line)
            except ValueError:continue
            if not r.get('terminal'):
                timeouts+=int(bool(r.get('truncated')));continue
            if not all(k in r for k in ('step','life_step','game_seconds')):continue
            key=(r.get('life'),r['step'],r['life_step'],r['game_seconds'])
            if key in seen:continue
            seen.add(key)
            deaths.append(dict(step=r['step'],life=r.get('life'),observations=r['life_step'],
                seconds=r['game_seconds'],reason=r.get('death_reason') or r.get('last_damage_type') or 'unknown'))
    result=(deaths,timeouts,True);_CACHE[str(path)]=(signature,*result)
    return result

def summary(root,status=None):
    root=Path(root)
    if status is None:
        try:status=json.loads((root/'status.json').read_text())
        except (OSError,ValueError):status={}
    e=status.get('environment') or {};deaths,timeouts,available=completed(root)
    def stats(rs):
        return {unit:dict(mean=statistics.mean(r[unit] for r in rs),median=statistics.median(r[unit] for r in rs),
                best=max(r[unit] for r in rs)) if rs else None for unit in ('seconds','observations')}
    recent=deaths[-20:];trend=[];highstep=0
    for i,r in enumerate(deaths):
        window=deaths[max(0,i-19):i+1];highstep=max(highstep,r['step'])
        trend.append(dict(step=highstep,mean_seconds=statistics.mean(x['seconds'] for x in window),
            mean_observations=statistics.mean(x['observations'] for x in window),n=len(window)))
    current=None
    if 'game_seconds' in e and 'life_step' in e:
        current=dict(life=e.get('life'),seconds=e['game_seconds'],observations=e['life_step'],ended=bool(e.get('terminal') or e.get('truncated')))
    return dict(version=1,step=status.get('step',0),time=status.get('time'),current=current,
        recorded_deaths=len(deaths),recorded_timeouts=timeouts,episode_log_available=available,
        all=stats(deaths),recent=stats(recent),recent_n=len(recent),deaths=deaths,trend=trend)

def duration(value):
    if value is None:return '—'
    value=max(0,round(value));h,value=divmod(value,3600);m,s=divmod(value,60)
    return f'{h}h {m:02d}m {s:02d}s' if h else f'{m}m {s:02d}s' if m else f'{s}s'

def comparison_data(trials, shared, cap, extend=True, raw=False):
    """Use only completed-life rolling statistics available by the shared cap."""
    data={}
    for t in trials:
        try:s=json.loads((Path(shared)/'runs'/t['name']/'live/survival.json').read_text())
        except (OSError,ValueError):s={}
        rows=[dict(step=r['step'],lifespan_minutes=r['seconds' if raw else 'mean_seconds']/60,
                   lifespan_observations=r['observations' if raw else 'mean_observations'])
              for r in s.get('deaths' if raw else 'trend',[]) if 0 < r['step'] <= cap]
        if extend and rows and rows[-1]['step'] < cap:rows.append(dict(rows[-1],step=cap))
        data[t['name']]=rows
    return data

def fork_charts(trials,shared,xmax=None):
    """Independent post-fork seed traces, without the exploration cohort's cap."""
    colors=['#ff6e82','#8cbcff','#7de1bf','#ffbd66','#d695ff','#d5e76c','#ff96d3','#65d4e8'];series=[];local_endpoints=[]
    for i,t in enumerate(trials):
        try:s=json.loads((Path(shared)/'runs'/t['name']/'live/survival.json').read_text())
        except (OSError,ValueError):s={}
        fork=t['fork_step']
        local_endpoints.append(max(0,s.get('step',fork)-fork))
        rows=[dict(step=r['step'] if xmax else r['step']-fork,minutes=r['seconds']/60,frames=r['observations'])
              for r in s.get('deaths',[]) if r['step']>fork]
        series.append((t,colors[i%len(colors)],rows))
    parts=['<p>'+ ' · '.join(f'<span style="color:{color}">{html.escape(t["model"])} · seed {t["agent_seed"]}</span>' for t,color,_ in series)+'</p><div class="charts">']
    absolute=bool(xmax)
    xmax=xmax or max([1]+local_endpoints+[r['step'] for _,_,rows in series for r in rows])
    for key,title in [('minutes','Individual lifespan (game minutes)'),('frames','Individual lifespan (agent interactions)')]:
        ymax=max([1]+[r[key] for _,_,rows in series for r in rows])
        xy=lambda x,y:(70+655*x/xmax,315-230*y/ymax)
        parts.append(f'<div><h3>{title}</h3><svg viewBox="0 0 760 390" role="img" aria-label="Survival runs: {title}">')
        for i in range(5):
            x,y=xy(xmax*i/4,ymax*i/4)
            parts.append(f'<path d="M70 {y}H725" stroke="#354151"/><text x="60" y="{y+4}" text-anchor="end">{ymax*i/4:,.1f}</text><text x="{x}" y="340" text-anchor="middle">{xmax*i/4:,.0f}</text>')
        for t,color,rows in series:
            path=' '.join(('M' if i==0 else 'L')+f'{xy(r["step"],r[key])[0]:.2f},{xy(r["step"],r[key])[1]:.2f}' for i,r in enumerate(rows))
            parts.append(f'<path d="{path}" stroke="{color}" stroke-width="3" fill="none"><title>Seed {t["agent_seed"]}</title></path>')
            for row in rows:
                x,y=xy(row['step'],row[key]);parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"><title>Seed {t["agent_seed"]} · step {row["step"]:,} · {row[key]:,.2f}</title></circle>')
        parts.append('<text x="400" y="375" text-anchor="middle">'+('Total training observations' if absolute else 'Observations since survival fork')+'</text></svg></div>')
    parts.append('</div>')
    for t,_,rows in series:
        if not rows:parts.append(f'<p>Seed {t["agent_seed"]}: waiting for its first completed life.</p>')
    return ''.join(parts)


def chart(series):
    """series: (run name, label, color, summary); gaps never become fabricated zeros."""
    xmax=max([1]+[s.get('step',0) for _,_,_,s in series]+[r['step'] for _,_,_,s in series for r in s.get('trend',[])])
    ymax=max([1]+[r['mean_seconds']/60 for _,_,_,s in series for r in s.get('trend',[])])
    parts=['<svg viewBox="0 0 600 290" style="width:100%;height:auto" role="img" aria-label="Recent completed lifespan versus environment observations">']
    for i in range(5):
        x=70+510*i/4;y=230-205*i/4
        parts.append(f'<path d="M70 {y} H580" stroke="#354151"/><text x="62" y="{y+4}" fill="#bbc8d7" font-size="12" text-anchor="end">{ymax*i/4:.1f}</text><text x="{x}" y="250" fill="#bbc8d7" font-size="12" text-anchor="middle">{xmax*i/4:,.0f}</text>')
    for name,label,color,s in series:
        path='';rows=s.get('trend',[])
        for r in rows:
            x=70+510*r['step']/xmax;y=230-205*(r['mean_seconds']/60)/ymax
            path+=f'M{x:.2f} {y:.2f}' if not path else f'H{x:.2f} V{y:.2f}'
        safe=html.escape(name,quote=True)
        parts.append(f'<g class="curve" data-run="{safe}"><path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"><title>{html.escape(label)}</title></path>')
        if rows:parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
        parts.append('</g>')
    parts.append('<text x="320" y="280" fill="#bbc8d7" text-anchor="middle">Environment observations</text><text transform="translate(16 135) rotate(-90)" fill="#bbc8d7" font-size="12" text-anchor="middle">Mean lifespan (game minutes)</text></svg>')
    if not any(s.get('trend') for _,_,_,s in series):parts.append('<p>No completed deaths recorded yet; current lives are shown below.</p>')
    return ''.join(parts)

def panel(s,individual_chart=False):
    if not s:return '<p>Survival metrics are being published.</p>'
    current=s.get('current');all_=s.get('all',{});recent=s.get('recent',{})
    p=['<div class="survival"><h3>Survival</h3>']
    if current:p.append(f'<p>{"Latest life (ended)" if current["ended"] else "Current life"} #{current.get("life","—")}: <b>{duration(current["seconds"])}</b> game time · {current["observations"]:,} environment interactions.</p>')
    p.append(f'<p>Recorded deaths: {s.get("recorded_deaths",0):,}. ')
    if all_.get('seconds'):
        a=all_['seconds'];r=recent['seconds'];o=all_['observations']
        p.append(f'Mean completed lifespan: <b>{duration(a["mean"])}</b>; median {duration(a["median"])}; longest {duration(a["best"])}. '
            f'Last {s["recent_n"]} completed lives: <b>{duration(r["mean"])}</b> mean.</p>'
            f'<p>In environment interactions: mean {o["mean"]:,.0f}; median {o["median"]:,.0f}; longest {o["best"]:,}; recent mean {recent["observations"]["mean"]:,.0f}.</p>')
    else:p.append('No completed lifespan statistics yet.</p>')
    p.append('<p class="muted">Age is measured at the latest observation, in game time—not training wall time. Averages include recorded deaths only; ongoing lives, timeouts and interrupted lives are excluded.</p>')
    if individual_chart:p.append(chart([('survival','Recent completed lifespan','#7de1bf',s)]))
    p.append('<details><summary>Recent life endings</summary><table><thead><tr><th>Life</th><th>End observation</th><th>Game time</th><th>Interactions</th><th>Recorded cause</th></tr></thead><tbody>')
    for r in reversed(s.get('deaths',[])[-20:]):
        p.append(f'<tr><td>{r.get("life","—")}</td><td>{r["step"]:,}</td><td>{duration(r["seconds"])}</td><td>{r["observations"]:,}</td><td>{html.escape(r["reason"])}</td></tr>')
    p.append('</tbody></table></details></div>')
    return ''.join(p)
