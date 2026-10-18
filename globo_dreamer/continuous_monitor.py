"""Bounded live movies and compact, persistent exploration/stagnation history."""
from collections import Counter,deque
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from .monitor_settings import SETTINGS,DEFAULT_CLIP_FRAMES,MAX_CLIP_FRAMES,read_settings


class ContinuousMonitor:
    def __init__(self,root,agent=None,frames=DEFAULT_CLIP_FRAMES,stall_steps=5000,settings_path=SETTINGS):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.agent=agent;self.frames=frames;self.stall_steps=stall_steps
        self.buffer=deque(maxlen=frames);self.recent=None
        self.state=dict(cells=0,last_discovery=0,stalled=False,first_stagnation=None,
            stall_start=None,longest_stall=0,n=0,deaths=0,low=0,sky=0,density=0.,
            model_novelty=0.,weighted_model_novelty=0.,model_input=0.,model_reward=0.,
            input_saturated=0,reward_clipped=0,actions={},window_start=1)
        self.last_render=0;self.last_render_time=time.time()
        self.report_stream=None
        self.settings_path=settings_path;self.settings_checked=0
        self.poll_settings()
        self.visitation=None
        if os.environ.get('GLOBO_VISITATION')=='1':
            from .visitation import Visitation
            self.visitation=Visitation(self.root)

    def poll_settings(self):
        now=time.monotonic()
        if now-self.settings_checked<5:return
        self.settings_checked=now
        frames=read_settings(self.settings_path)['frames']
        if frames!=self.frames:
            self.frames=frames;self.buffer=deque(self.buffer,maxlen=frames)
            self.recent=list(self.buffer) if len(self.buffer)==frames else None
            self.last_render_time=0
        from .env import atomic_json
        atomic_json(self.root/'clip_status.json',dict(requested_frames=frames,
            collected_frames=len(self.buffer),time=time.time()))

    def save(self):
        if self.visitation:self.visitation.save()
        return self.state.copy()

    def load(self,state):
        self.state.update(state);self.buffer.clear();self.recent=None

    def event(self,kind,step,**kwargs):
        with (self.root/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(kind=kind,step=step,time=time.time(),**kwargs))+'\n')

    def observe(self,transition,record,step,metrics):
        if self.visitation:self.visitation.observe(record,step)
        self.poll_settings()
        if bool(transition['is_first']):
            self.buffer.clear()
        row={k:np.array(v,copy=True) for k,v in transition.items()}
        if 'achievement_counts' in record:
            self.state['achievements']={k:v for k,v in record.items() if k.startswith('achievement_')}
            if record['achievement_delta']:
                self.event('achievement',step,delta=record['achievement_delta'],
                    counts=record['achievement_counts'],env_step=record['achievement_env_step'],
                    life=record['achievement_life'])
        if getattr(self.agent,'ftn_enabled',False):
            # Live-buffer-only diagnostic, not another tensor in replay.
            mask=np.zeros(self.agent.atom_count,np.float32);mask[row['ftn_mask']]=1
            ema=self.state.get('action_mask_ema',np.zeros((self.agent.atom_count,3),np.float32))
            ema=np.maximum(ema*np.array([.95,.30,.975],np.float32),mask[:,None])
            self.state['action_mask_ema']=ema
            row['log/ftn_ema']=np.uint8(np.clip(ema,0,1)*255)
        row.setdefault('consec',np.int32(0));row.setdefault('stepid',np.zeros(20,np.uint8))
        self.buffer.append((step,row))
        if len(self.buffer)==self.frames:self.recent=list(self.buffer)
        s=self.state;cells=record['cells_lifetime']
        if cells>s['cells']:
            if s['stalled']:
                duration=step-s['stall_start'];s['longest_stall']=max(s['longest_stall'],duration)
                self.event('recovered',step,stalled_since=s['stall_start'],duration=duration)
            s.update(cells=cells,last_discovery=step,stalled=False,stall_start=None)
        gap=step-s['last_discovery']
        if gap>=self.stall_steps and not s['stalled']:
            s.update(stalled=True,stall_start=s['last_discovery']+1)
            if s['first_stagnation'] is None:s['first_stagnation']=s['stall_start']
            self.event('stagnation_confirmed',step,onset=s['stall_start'],criterion=f'No new spatial cell for {self.stall_steps} observations')
        if s['stalled']:s['longest_stall']=max(s['longest_stall'],gap)
        s['n']+=1;s['deaths']+=int(record['terminal']);s['low']+=int(record['low_detail'])
        s['sky']+=int(record['sky_fraction']>.85);s['density']+=record['density']
        s['model_novelty']+=metrics.get('model_novelty',0.)
        s['weighted_model_novelty']+=metrics.get('weighted_model_novelty',0.)
        s['model_input']+=metrics.get('model_input',0.)
        s['model_reward']+=metrics.get('model_reward',0.)
        s['input_saturated']+=int(metrics.get('model_input',0.)>=.99)
        s['reward_clipped']+=int(metrics.get('model_reward',0.)>=.299999)
        s['model_input_scale']=metrics.get('model_input_scale',0.)
        s['model_reward_scale']=metrics.get('model_reward_scale',0.)
        if 'ftn_updates' in metrics:
            s['ftn_metrics']={k:v for k,v in metrics.items() if k.startswith('ftn_')}
        action=record['action'];s['actions'][action]=s['actions'].get(action,0)+1
        if step%1000==0:self.window(step)

    def window(self,step):
        from .env import atomic_json
        s=self.state;n=max(1,s['n']);last_cells=s.get('window_cells',0)
        entropy=-sum((v/n)*math.log(v/n) for v in s['actions'].values())
        row=dict(step=step,time=time.time(),window_start=s['window_start'],observations=s['n'],
            new_cells=max(0,s['cells']-last_cells),total_cells=s['cells'],deaths=s['deaths'],
            mean_density_novelty=s['density']/n,mean_model_novelty=s['model_novelty']/n,
            mean_weighted_model_novelty=s['weighted_model_novelty']/n,low_detail_fraction=s['low']/n,
            mean_model_input=s['model_input']/n,mean_model_reward=s['model_reward']/n,
            model_input_scale=s.get('model_input_scale',0.),model_reward_scale=s.get('model_reward_scale',0.),
            model_input_saturation_fraction=s['input_saturated']/n,model_reward_clip_fraction=s['reward_clipped']/n,
            sky_fraction=s['sky']/n,action_entropy=entropy,steps_since_discovery=step-s['last_discovery'],
            stalled=s['stalled'],first_stagnation=s['first_stagnation'],longest_stall=s['longest_stall'])
        if 'achievements' in s:
            row.update({k:v for k,v in s['achievements'].items() if k!='achievement_delta'})
        row.update(s.get('ftn_metrics',{}))
        with (self.root/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        atomic_json(self.root/'progress.json',row)
        s.update(n=0,deaths=0,low=0,sky=0,density=0.,model_novelty=0.,weighted_model_novelty=0.,
                 model_input=0.,model_reward=0.,input_saturated=0,reward_clipped=0,
                 actions={},window_start=step+1,window_cells=s['cells'])

    def render(self,step,updates,force=False):
        if hasattr(self,'m2e_ftn') and len(self.buffer)>=24:
            self.recent=list(self.buffer)
        if self.recent is None:return
        if not force and step-self.last_render<1000 and time.time()-self.last_render_time<300:return
        if self.agent is None:
            return self.render_observations(step)
        from .world_probe import video
        from .env import atomic_json
        import av
        from PIL import Image,ImageDraw
        rows=self.recent
        keys=list(rows[0][1])
        batch,offset=report_batch(rows)
        # Dreamer's prefetcher requires an endless source. Reuse one stream;
        # constructing a new prefetch thread per video leaks threads and arrays.
        if self.report_stream is None:
            import queue
            self.report_queue=queue.SimpleQueue()
            self.report_stream=iter(self.agent.stream(iter(self.report_queue.get,None)))
        self.report_queue.put(batch)
        prepared=next(self.report_stream)
        _,result=self.agent.report(self.agent.init_report(8),prepared)
        result={k:np.asarray(v) for k,v in result.items() if k!='params/summary'}
        # The compiled inference shape is fixed. Real context ends at its midpoint;
        # padding before that context is reset away, and padded outputs are omitted.
        result['truth']=result['truth'][:,offset:offset+len(rows)]
        result['posterior']=result['posterior'][:,offset:offset+len(rows)]
        result['forecast']=result['forecast'][:,:len(rows)//2]
        if 'log/ftn_ema' in rows[0][1]:
            layout=self.agent.layout.copy()
            result['mask_frames']=np.stack([row['log/ftn_ema'][layout].reshape(self.agent.mask_side,self.agent.mask_side,3) for _,row in rows])
        video(self.root/'recent_pending.mp4',result,index=0)
        (self.root/'recent_pending.mp4').replace(self.root/'recent.mp4')
        for path in self.root.glob('recent_h*.jpg'):path.unlink()
        for path in self.root.glob('recent_pending_h*.jpg'):
            path.replace(path.with_name(path.name.replace('recent_pending','recent')))
        container=av.open(str(self.root/'obs_pending.mp4'),'w');stream=container.add_stream('libx264',rate=6)
        stream.width=256;stream.height=280;stream.pix_fmt='yuv420p'
        for obs_step,row in rows:
            canvas=Image.new('RGB',(256,280),'#171c24')
            canvas.paste(Image.fromarray(row.get('log/human',row['image'])).resize((256,256)),(0,24))
            ImageDraw.Draw(canvas).text((6,5),f'Actual observation {obs_step:,}',fill='white')
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
        container.close();(self.root/'obs_pending.mp4').replace(self.root/'obs.mp4')
        if 'log/ftn_ema' in rows[0][1]:
            container=av.open(str(self.root/'bw_pending.mp4'),'w');stream=container.add_stream('libx264',rate=6)
            stream.width=384;stream.height=408;stream.pix_fmt='yuv420p'
            layout=self.agent.layout.copy()
            for obs_step,row in rows:
                canvas=Image.new('RGB',(384,408),'#171c24')
                raster=row['log/ftn_ema'][layout].reshape(self.agent.mask_side,self.agent.mask_side,3)
                canvas.paste(Image.fromarray(raster).resize((384,384),Image.Resampling.NEAREST),(0,24))
                ImageDraw.Draw(canvas).text((6,5),f'Training mask activity at obs {obs_step:,}',fill='white')
                for packet in stream.encode(av.VideoFrame.from_image(canvas)):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
            container.close();(self.root/'bw_pending.mp4').replace(self.root/'bw.mp4')
        truth=result['truth'][0].astype(np.float32)/255;C=len(truth)//2
        metadata=dict(step=step,model_updates=updates,time=time.time(),frame_start=rows[0][0],frame_end=rows[-1][0],
            frame_age=step-rows[-1][0],frames=len(rows),context=C,prediction_frames=len(rows)-C,
            playback_seconds=len(rows)/6,heldout=False,
            reconstruction_mse=float(np.mean((result['posterior'][0]-truth)**2)),
            forecast_mse=float(np.mean((result['forecast'][0]-truth[C:])**2)))
        if 'mask_frames' in result:
            metadata.update(panels=['observation','reconstruction','prediction','live action mask with EMA'],
                mask_timing='Recorded at each real observation; shared display permutation for the clip',
                reconstruction_mask='Current selected model mask at rendering; historical live masks are shown separately')
        atomic_json(self.root/'video.json',metadata)
        self.last_render=step;self.last_render_time=time.time()

    def render_observations(self,step):
        """Observation-only feed; never import or invoke a world model."""
        if hasattr(self,'m2e_ftn'):
            from .m2e_ftn import render
            render(self.root,self.recent,step,self.m2e_ftn)
            self.last_render=step;self.last_render_time=time.time()
            return
        if 'log/human' in self.recent[0][1]:
            from .human_monitor import render
            render(self.root,self.recent,step)
            self.last_render=step;self.last_render_time=time.time()
            return
        import av
        from PIL import Image,ImageDraw
        from .env import atomic_json
        rows=self.recent
        container=av.open(str(self.root/'obs_pending.mp4'),'w')
        stream=container.add_stream('libx264',rate=6)
        has_goal='log/offline_goal' in rows[0][1]
        width=512 if has_goal else 256
        stream.width=width;stream.height=280;stream.pix_fmt='yuv420p'
        for obs_step,row in rows:
            canvas=Image.new('RGB',(width,280),'#171c24')
            canvas.paste(Image.fromarray(row['image']).resize((256,256)),(0,24))
            ImageDraw.Draw(canvas).text((6,5),f'Actual observation {obs_step:,}',fill='white')
            if has_goal:
                draw=ImageDraw.Draw(canvas)
                draw.text((262,5),'Offline frontier (density view)',fill='white')
                goal_step=int(row['log/offline_goal_step'])
                if goal_step>=0:
                    canvas.paste(Image.fromarray(row['log/offline_goal']).convert('RGB').resize((256,128),Image.Resampling.NEAREST),(256,80))
                    draw.text((262,240),f'Observed at step {goal_step:,}',fill='white')
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
        container.close();(self.root/'obs_pending.mp4').replace(self.root/'obs.mp4')
        atomic_json(self.root/'video.json',dict(step=step,model_updates=0,time=time.time(),
            frame_start=rows[0][0],frame_end=rows[-1][0],frame_age=step-rows[-1][0],
            frames=len(rows),playback_seconds=len(rows)/6,panels=['observation']+(['offline frontier'] if has_goal else []),model_free_only=True,offline_goal=has_goal))
        self.last_render=step;self.last_render_time=time.time()


def report_batch(rows):
    """Place live context against the fixed report midpoint; never predict padding."""
    frames=len(rows)
    if frames%2 or not 2<=frames<=MAX_CLIP_FRAMES:raise ValueError('Invalid report frame count')
    offset=MAX_CLIP_FRAMES//2-frames//2
    batch={}
    for key in rows[0][1]:
        values=np.stack([r[key] for _,r in rows])
        padded=np.zeros((MAX_CLIP_FRAMES,*values.shape[1:]),dtype=values.dtype)
        padded[offset:offset+frames]=values
        batch[key]=np.repeat(padded[None],8,axis=0)
    batch['is_first'][:,offset]=True
    return batch,offset


def bounded_cleanup(root,replay):
    """After a committed synchronous checkpoint, retain only live replay chunks."""
    root=Path(root)
    with replay.rwlock.writing:
        live={chunk.filename for chunk in replay.chunks.values()}
        for path in (root/'replay').glob('*.npz'):
            if path.name not in live:path.unlink()
    bound_logs(root)


def bound_logs(root):
    root=Path(root)
    # Keep compact progress/event histories, but bound verbose implementation logs.
    for name in ('console.log','metrics.jsonl','scores.jsonl','agent_feedback.jsonl'):
        path=root/name
        if path.exists() and path.stat().st_size>8*1024**2:
            with path.open('r+b') as f:
                f.seek(-4*1024**2,2);tail=f.read();tail=tail[tail.find(b'\n')+1:]
                f.seek(0);f.write(tail);f.truncate()
