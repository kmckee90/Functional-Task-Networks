"""Seed-matched achievement counts; missing instrumentation is never zero."""
import html
import json
from .achievement_catalog import LABELS

COLORS=('#f5bd63','#7de1bf','#a9a0ff','#f68da7','#70cfff','#f4f080','#ff8f58','#d4a6ff','#63c4ac','#c7d4e8')


def discovery_label(key):
    if key.startswith('new_resource/'):
        return 'Found '+key.split('/',1)[1].rsplit(':',1)[-1].replace('_',' ')
    if key.startswith('new_biome/'):
        return 'Discovered '+key.split('/',1)[1].rsplit(':',1)[-1].replace('_',' ')
    return key


def checked_achievements(record):
    """Exactly the checked checklist entries, never repeated event counts."""
    return sum(record.get('achievement_counts',{}).get(key,0)>0 for key in LABELS) + len(record.get('achievement_discoveries',{}))


def achievement_checklist(record):
    labels=LABELS.items()
    if 'achievement_counts' not in record:
        return '<h3>Achievements</h3><p>Waiting for instrumented observations; earlier history is unavailable.</p>'
    counts=record['achievement_counts']
    parts=[f'<details class="achievement-checklist"><summary>Achievements: {checked_achievements(record):,} boxes checked</summary>'
        f'<p>Tracking from environment observation {record["achievement_start_step"]:,}; earlier events are not counted.</p><ul>']
    for key,label in labels:
        count=counts.get(key,0);first=record.get('achievement_first',{}).get(key)
        parts.append(f'<li>{"☑" if count else "☐"} {label}: <b>{count:,}</b>'
            +(f' · first at observation {first:,}' if first is not None else '')+'</li>')
    for key,step in sorted(record.get('achievement_discoveries',{}).items()):
        parts.append(f'<li>☑ {html.escape(discovery_label(key))} · first at observation {step:,}</li>')
    parts.append('</ul><p>Expanded tracking begins at deployment; earlier unrecorded events cannot be recovered. '
                 'Recovery requires a measured increase. Milestones use game time and actual positions.</p></details>')
    if record.get('achievement_expanded_start_step') is not None:
        parts[-1]=parts[-1].replace('begins at deployment',f'begins at observation {record["achievement_expanded_start_step"]:,}')
    return ''.join(parts)


def run_charts(rows):
    rows=[dict(r,achievement_checked=checked_achievements(r)) if 'achievement_counts' in r else r for r in rows]
    parts=['<div class="run-charts">']
    for metric,title,ylabel,color in [('total_cells','Spatial progress','Discovered cells','#7de1bf'),
            ('achievement_checked','Cumulative achievements','Achievement boxes checked','#f5bd63')]:
        points=[r for r in rows if metric in r]
        parts.append(f'<div><h3>{title}</h3>')
        if not points:
            parts.append('<p>Waiting for instrumented observations.</p></div>');continue
        xmax=max(1,max(r['step'] for r in rows));ymax=max(1,max(r[metric] for r in points))
        parts.append(f'<svg viewBox="0 0 540 260" style="height:auto" role="img" aria-label="{title}">')
        for i in range(5):
            x=72+450*i/4;y=205-180*i/4
            parts.append(f'<path d="M72 {y} H522" stroke="#343c48"/>'
                f'<text x="64" y="{y+4}" text-anchor="end" fill="#ccc" font-size="12">{ymax*i/4:,.0f}</text>'
                f'<text x="{x}" y="225" text-anchor="middle" fill="#ccc" font-size="12">{xmax*i/4:,.0f}</text>')
        # Step paths retain jumps and never invent pre-instrumentation history.
        path=''
        for r in points:
            x=72+450*r['step']/xmax;y=205-180*r[metric]/ymax
            path+=f'M{x:.2f} {y:.2f}' if not path else f'H{x:.2f} V{y:.2f}'
        parts.append(f'<path d="{path}" stroke="{color}" stroke-width="2" fill="none"/>'
            '<text x="297" y="250" text-anchor="middle" fill="#ddd" font-size="14">Environment observations</text>'
            f'<text transform="translate(17 115) rotate(-90)" text-anchor="middle" fill="#ddd" font-size="13">{ylabel}</text></svg></div>')
    return ''.join(parts)+'</div>'


def achievement_charts(groups,state,label):
    parts=['<h2>Achievements</h2><p>Number of checked achievement boxes, including individual discoveries. '
           'Repeating an achievement does not increase the curve. Kills require a direct player killing blow. '
           'Uninstrumented histories are unavailable, not zero. Expanded tracking starts at its deployment; '
           'older baseline totals cover only the original six event types.</p>']
    for seed,trials in groups:
        series=[];missing=[]
        for i,t in enumerate(trials):
            path=state/'runs'/t['name'];points={}
            try:
                for line in (path/'live/history.jsonl').read_text().splitlines():
                    try:row=json.loads(line)
                    except ValueError:continue
                    if 'achievement_counts' in row:points[row['step']]=dict(row,achievement_checked=checked_achievements(row))
            except OSError:pass
            try:
                status=json.loads((path/'status.json').read_text());row=status['environment']
                if 'achievement_counts' in row:points[status['step']]=dict(row,step=status['step'],achievement_checked=checked_achievements(row))
            except (OSError,ValueError,KeyError):pass
            if points:series.append((label(t),sorted(points.items()),COLORS[i%len(COLORS)]))
            else:missing.append(label(t))
        parts.append(f'<section><h3>Seed {seed}</h3>')
        if not series:parts.append('<p>Achievement instrumentation has not been enabled for these runs.</p>')
        else:
            xmax=max(1,max(x for _,pts,_ in series for x,_ in pts))
            for metric,title in [('achievement_checked','Cumulative achievements — boxes checked')]:
                ymax=max(1,max(row[metric] for _,pts,_ in series for _,row in pts))
                parts.append(f'<h4>{title}</h4><svg viewBox="0 0 900 290" style="height:auto" role="img" aria-label="Seed {seed}: {title}">')
                for i in range(5):
                    x=65+810*i/4;y=235-205*i/4
                    parts.append(f'<path d="M65 {y} H875" stroke="#343c48"/>'
                        f'<text x="55" y="{y+4}" text-anchor="end" fill="#bbb">{ymax*i/4:g}</text>'
                        f'<text x="{x}" y="256" text-anchor="middle" fill="#bbb">{xmax*i/4:,.0f}</text>')
                for name,pts,color in series:
                    # No fabricated line between step zero and late instrumentation.
                    x,row=pts[0];path=f'M{65+810*x/xmax:.2f} {235-205*row[metric]/ymax:.2f}'
                    for x,row in pts[1:]:path+=f' H{65+810*x/xmax:.2f} V{235-205*row[metric]/ymax:.2f}'
                    parts.append(f'<path class="achievement-line" data-run="{seed}:{html.escape(name,quote=True)}" d="{path}" stroke="{color}" stroke-width="2" fill="none"><title>{html.escape(name)}</title></path>')
                parts.append('<text x="460" y="282" text-anchor="middle" fill="#ddd">Observations</text>'
                    f'<text transform="translate(15 135) rotate(-90)" text-anchor="middle" fill="#ddd" font-size="12">{title}</text></svg>')
            for name,pts,color in series:
                row=pts[-1][1]
                parts.append(f'<p style="color:{color}">{html.escape(name)} — tracking from environment observation '
                    f'{row["achievement_start_step"]:,}; {checked_achievements(row):,} boxes checked.</p>'
                    f'<details><summary>First achievements and block/species breakdown</summary><pre>{html.escape(json.dumps(dict(first_environment_steps=row.get("achievement_first",{}),first_unix_time=row.get("achievement_first_time",{}),details=row.get("achievement_details",{})),indent=2))}</pre></details>')
        if missing:parts.append('<p>Unavailable: '+html.escape('; '.join(missing))+'.</p>')
        parts.append('</section>')
    return ''.join(parts)
