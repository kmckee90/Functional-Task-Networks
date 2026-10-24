"""Retroactive life-video event lists from telemetry, never video inference."""
import bisect
import html
import json
from pathlib import Path
from .achievement_catalog import LABELS

_CACHE={}
EXTRA={'collect_valuable':'Collect', 'dig_hole':'Dig a hole',
       'long_run':'Explore new ground', 'climb_tree':'Climb a tree',
       'punch_animal':'Punch an animal', 'animal_hit':'Hit an animal'}

def read(path,lines=False):
    path=Path(path)
    try:
        st=path.stat();signature=(st.st_mtime_ns,st.st_size)
    except OSError:return [] if lines else {}
    key=str(path);cached=_CACHE.get(key)
    if cached and cached[0]==signature:return cached[1]
    try:
        text=path.read_text()
        if lines:
            value=[]
            for line in text.splitlines(keepends=True):
                if not line.endswith('\n'):continue
                try:value.append(json.loads(line))
                except ValueError:continue
        else:value=json.loads(text)
    except (OSError,ValueError):return [] if lines else {}
    _CACHE[key]=(signature,value)
    return value

def index(shared_root,clip_root):
    """Use agent steps from live events; discovery steps are engine steps."""
    root=Path(shared_root);events={};discoveries={}
    for identity,step in read(root/'environment/achievements.json').get('discoveries',{}).items():
        kind,_,detail=identity.partition('/')
        discoveries.setdefault((step,kind),[]).append(detail)
    def add(step,life,kind,count=1,details=()):
        key=(int(step),life,kind)
        item=events.setdefault(key,dict(step=int(step),life=life,kind=kind,count=0,details=set()))
        item['count']=max(item['count'],count)
        item['details'].update(str(v) for v in details if v)
    for row in read(root/'live/events.jsonl',True):
        if row.get('kind')!='achievement':continue
        for kind,count in row.get('delta',{}).items():
            if not count:continue
            details=discoveries.get((row.get('env_step'),kind),[])
            # A newly discovered food was consumed at this exact event; do not
            # guess food types for repeated consumption without detailed logs.
            if kind=='eat_food':details=discoveries.get((row.get('env_step'),'new_food'),[])
            add(row['step'],row.get('life'),kind,count,details)
    for path in Path(clip_root).glob('*.json'):
        row=read(path)
        if 'event_step' not in row:continue
        for event in row.get('events',[]):
            details=event.get('details') or []
            if event.get('detail'):details=[*details,event['detail']]
            add(row['event_step'],row.get('life'),event['kind'],event.get('count',1),details)
    rows=sorted(events.values(),key=lambda r:(r['step'],r['kind']))
    return [r['step'] for r in rows],rows

def dropdown(indexed,video):
    steps,rows=indexed
    rows=[r for r in rows[bisect.bisect_left(steps,video['start_step']):bisect.bisect_right(steps,video['end_step'])]
          if r['life']==video['life']]
    parts=[f'<details class="life-events"><summary>What happened · {len(rows)} logged entries</summary>']
    if not rows:parts.append('<p>No achievements or interactions were logged for this video.</p>')
    else:
        parts.append('<ul>')
        for row in rows:
            kind=row['kind'];label=LABELS.get(kind,EXTRA.get(kind,kind.replace('_',' ').capitalize()))
            if row['details']:
                label+=': '+', '.join(d.rsplit(':',1)[-1].replace('_',' ') for d in sorted(row['details']))
            elif kind=='eat_food':label+=' (food type not recorded)'
            if row['count']>1:label+=f' ×{row["count"]}'
            parts.append(f'<li>Step {row["step"]:,} — {html.escape(label)}</li>')
        parts.append('</ul>')
    parts.append('<p class="muted">Recorded events only. Collecting a food item does not necessarily mean eating it. Food names appear where the logs identify them.</p></details>')
    return ''.join(parts)
