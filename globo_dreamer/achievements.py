"""Persistent evaluation counters, isolated from observations and reward."""
from pathlib import Path
import json
import time
from .env import atomic_json

from .achievement_catalog import KINDS


class Achievements:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'achievements.json'
        self.state=dict(version=1,counts={},details={},first={},first_time={},start_step=None,
                        engine=None,seen={},seen_details={})
        if self.path.exists():self.state.update(json.loads(self.path.read_text()))

    def observe(self,data,engine,step,life,initial=False):
        s=self.state
        before=json.dumps(s,sort_keys=True)
        if s['start_step'] is None:s['start_step']=step
        if int(data.get('version',1))>=2:
            s.setdefault('expanded_start_step',step)
            s['version']=2
        if s['engine']!=engine:
            s.update(engine=engine,seen={},seen_details={})
        counts=data.get('counts') or {};details=data.get('details') or {}
        delta={}
        for key,value in counts.items():
            if key not in KINDS:continue
            change=max(0,int(value)-s['seen'].get(key,0))
            if change and not initial:
                s['counts'][key]=s['counts'].get(key,0)+change
                s['first'].setdefault(key,step);delta[key]=change
                s['first_time'].setdefault(key,time.time())
            s['seen'][key]=int(value)
        for key,value in details.items():
            change=max(0,int(value)-s['seen_details'].get(key,0))
            if not initial and change:
                s['details'][key]=s['details'].get(key,0)+change
                kind,_,detail=key.partition('/')
                discovery=('new_food' if kind=='eat_food' else
                           'new_resource' if kind in ('dig_block','collect_resource') else
                           'new_species' if kind.startswith('kill_') else
                           'new_biome' if kind=='visit_biome' else None)
                if discovery:
                    identity=discovery+'/'+detail.removesuffix('_male') if discovery=='new_species' else discovery+'/'+detail
                    if identity not in s.setdefault('discoveries',{}):
                        s['discoveries'][identity]=step
                        s['counts'][discovery]=s['counts'].get(discovery,0)+1
                        s['first'].setdefault(discovery,step)
                        s['first_time'].setdefault(discovery,time.time())
                        delta[discovery]=delta.get(discovery,0)+1
            s['seen_details'][key]=int(value)
        # Persist changed counters immediately, independently of model saves.
        if json.dumps(s,sort_keys=True)!=before:atomic_json(self.path,s)
        return dict(achievement_counts=s['counts'].copy(),achievement_details=s['details'].copy(),
            achievement_first=s['first'].copy(),achievement_total=sum(s['counts'].values()),
            achievement_first_time=s['first_time'].copy(),
            achievement_discoveries=s.get('discoveries',{}).copy(),
            achievement_expanded_start_step=s.get('expanded_start_step'),
            achievement_distinct=len(s['first']),achievement_start_step=s['start_step'],
            achievement_delta=delta,achievement_env_step=step,achievement_life=life)
