"""Evaluation-only position counters; never an input to the agent."""
from collections import deque
import math
from pathlib import Path
import json
import time

class Visitation:
    def __init__(self,root):
        self.path=Path(root)/'visitation.json'
        try:self.state=json.loads(self.path.read_text())
        except (OSError,ValueError):self.state=dict(version=1,cell_size=4,counts={},last_step=-1,recent=[])
        self.recent=deque(self.state.get('recent',[]),maxlen=512)
        self.dirty=False
    def observe(self,record,step):
        if step<=self.state['last_step']:return
        x,y,z=record['position'];key=f'{math.floor(x/4)},{math.floor(z/4)}'
        counts=self.state['counts'];counts[key]=counts.get(key,0)+1
        self.state.update(last_step=step,position=[x,y,z],yaw=record.get('look_yaw',0),life=record['life'],time=time.time())
        self.state.setdefault('spawn',[x,y,z])
        if step%4==0 or not self.recent:self.recent.append([step,x,z,record['life']])
        self.dirty=True
        if step%100==0:self.save()
    def save(self):
        if not self.dirty:return
        self.state['recent']=list(self.recent)
        tmp=self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path);self.dirty=False
