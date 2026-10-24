"""Evaluation-only event recordings: consecutive observations at six FPS."""
from collections import deque
import bisect,json,math,queue,threading,time,traceback
from pathlib import Path
import numpy as np

FPS=6
CONTEXT=80

class Triggers:
    def __init__(self):
        self.life=None;self.seen_event=0;self.previous_cell=None;self.streak=0;self.last_entry=0;self.details={}

    def observe(self,r):
        now=r['game_seconds'];life=r['life'];out=[]
        if life!=self.life:
            self.life=life;self.seen_event=max([0]+[e['id'] for e in r.get('video_events',[])])
            self.previous_cell=None;self.streak=0
            self.details=r.get('achievement_details',{}).copy()
            return []
        for event in r.get('video_events',[]):
            if event['id']>self.seen_event:out.append(dict(kind=event['kind'],detail=event.get('detail')))
            self.seen_event=max(self.seen_event,event['id'])
        for key,n in r.get('achievement_delta',{}).items():
            if n and (key.startswith('kill_') or key in ('place_block','climb_10m','climb_20m',
                              'harvest_papaya','eat_thorn_fruit','eat_dawn_cactus','heal_goldcap','coffee_energy')):
                details=[k.split('/',1)[1] for k,v in r.get('achievement_details',{}).items()
                         if k.startswith(key+'/') and v>self.details.get(k,0)]
                out.append(dict(kind=key,count=n,details=details))
        self.details=r.get('achievement_details',{}).copy()
        cell=tuple(math.floor(r['position'][axis]/4) for axis in (0,2))
        if cell!=self.previous_cell:
            if now-self.last_entry>10:self.streak=0
            self.streak=self.streak+1 if r.get('new_spatial_cell') else 0
            self.previous_cell=cell;self.last_entry=now
            if self.streak>=6 and self.streak%6==0:out.append(dict(kind='long_run',new_cells=self.streak))
        return out

def sample_frames(rows,event_step):
    center=next(i for i,r in enumerate(rows) if r[1]==event_step)
    return rows[max(0,center-CONTEXT):center+CONTEXT+1]

class EventCapture:
    def __init__(self,root,ftn):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.ftn=ftn;self.frames=deque(maxlen=CONTEXT+1);self.pending=[];self.triggers=Triggers();self.life=None
        self.queue=queue.Queue();self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
        self.serial=time.time_ns();self.last_frame=-1e30

    def log(self,record):
        with (self.root/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**record))+'\n')

    def observe(self,row,record,step):
        now=record['game_seconds'];life=record['life']
        if life!=self.life:
            for pending in self.pending:self.submit(pending,'life ended before post-event window completed')
            self.pending=[];self.frames.clear();self.last_frame=-1e30;self.life=life
        events=self.triggers.observe(record)
        frame=(now,step,{k:np.array(row[k],copy=True) for k in ('log/human','log/m2e_ftn_ema','log/offline_goal','log/offline_goal_step','log/video_hud') if k in row})
        self.frames.append(frame)
        for pending in self.pending:pending['rows'].append(frame)
        if events:
            # Distinct simultaneous triggers share a single centered clip.
            pending=dict(events=events,event_time=now,event_step=step,life=life,rows=list(self.frames),
                         event_index=len(self.frames)-1,layout=np.array(self.ftn.layout,copy=True),grid=self.ftn.grid)
            self.pending.append(pending)
        finished=[p for p in self.pending if len(p['rows'])-p['event_index']-1>=CONTEXT]
        for pending in finished:self.submit(pending,None)
        self.pending=[p for p in self.pending if len(p['rows'])-p['event_index']-1<CONTEXT]

    def submit(self,pending,reason):
        pending['padding_reason']=reason
        if self.queue.qsize()>=16:
            # Spill the render job to pod storage; no silent event loss or unbounded RAM.
            import pickle
            self.serial+=1;path=self.root/f'pending_{self.serial}.pkl'
            with path.open('wb') as f:pickle.dump(pending,f)
            self.queue.put(path)
        else:self.queue.put(pending)

    def work(self):
        while True:
            job=self.queue.get()
            try:
                if job is None:return
                if isinstance(job,Path):
                    import pickle
                    with job.open('rb') as f:pending=pickle.load(f)
                else:pending=job
                self.encode(pending)
                if isinstance(job,Path):job.unlink()
            except Exception:
                self.log(dict(status='encoding_error',error=traceback.format_exc()))
            finally:self.queue.task_done()

    def encode(self,p):
        import av
        from PIL import Image,ImageDraw
        label='+'.join(e['kind'] for e in p['events'])
        stem=f'{p["event_step"]:012d}_{label}_{time.time_ns()}'
        target=self.root/(stem+'.mp4');tmp=self.root/(stem+'.pending.mp4')
        out=av.open(str(tmp),'w');stream=out.add_stream('libx264',rate=FPS)
        from .video_hud import HEIGHT, draw as draw_hud
        stream.width=384;stream.height=HEIGHT;stream.pix_fmt='yuv420p';stream.options={'crf':'18','threads':'1'}
        rows=sample_frames(p['rows'],p['event_step'])
        event_index=next(i for i,r in enumerate(rows) if r[1]==p['event_step'])
        try:
            for i,(_,step,row) in enumerate(rows):
                canvas=Image.new('RGB',(384,HEIGHT),'#171c24');draw=ImageDraw.Draw(canvas)
                canvas.paste(Image.fromarray(row['log/human']).resize((256,256)),(0,24))
                mask=row['log/m2e_ftn_ema'][p['layout']].reshape(*p['grid'],3)
                canvas.paste(Image.fromarray(mask).resize((128,128),Image.Resampling.NEAREST),(256,152))
                if int(row.get('log/offline_goal_step',-1))>=0:
                    canvas.paste(Image.fromarray(row['log/offline_goal']).convert('RGB').resize((128,64),Image.Resampling.NEAREST),(256,52))
                draw.text((5,5),f'Step {step:,}',fill='white');draw.text((260,5),'Goal / mask',fill='white')
                draw_hud(canvas,row.get('log/video_hud'))
                for packet in stream.encode(av.VideoFrame.from_image(canvas)):out.mux(packet)
            for packet in stream.encode():out.mux(packet)
        finally:out.close()
        tmp.replace(target)
        meta=dict(version=2,file=target.name,events=p['events'],event_step=p['event_step'],life=p['life'],
                  fps=FPS,duration=len(rows)/FPS,event_at_seconds=event_index/FPS,time_basis='consecutive observations at 6 FPS',
                  frames=len(rows),observations_before=event_index,observations_after=len(rows)-event_index-1,
                  event_game_time=p['event_time'],source_start=p['rows'][0][0],source_end=p['rows'][-1][0],
                  padding_before=0,padding_after=0,padding_reason=None,truncation_reason=p['padding_reason'],
                  mask_layout='Fixed at trigger; recorded action-mask EMA; no learner reconfiguration')
        (self.root/(stem+'.json')).write_text(json.dumps(meta,indent=2));self.log(dict(status='saved',**meta))

    def close(self):
        for p in self.pending:self.submit(p,'recorder stopped before post-event window completed')
        self.pending=[];self.queue.join();self.queue.put(None);self.thread.join()
