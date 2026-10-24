"""Bounded-memory streaming recordings of complete lives, outside source sync."""
import json
import time
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw
from .video_hud import HEIGHT, draw as draw_hud, values

class LifeCapture:
    def __init__(self, root, ftn):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.ftn=ftn;self.out=None;self.life=None

    def observe(self,row,record,step):
        life=record['life']
        if self.out is not None and life!=self.life:self.close('life boundary')
        if self.out is None:
            self.life=life;self.start=step;self.frames=0
            self.started_at_spawn=bool(row.get('is_first',False))
            self.stem=f'{step:012d}_life{life}_{time.time_ns()}'
            self.tmp=self.root/(self.stem+'.partial.mp4')
            self.out=av.open(str(self.tmp),'w',options={'movflags':'frag_keyframe+empty_moov+default_base_moof'})
            self.stream=self.out.add_stream('libx264',rate=6)
            self.stream.width=384;self.stream.height=HEIGHT;self.stream.pix_fmt='yuv420p'
            self.stream.options={'crf':'22','preset':'veryfast','threads':'1','g':'36'}
        im=Image.new('RGB',(384,HEIGHT),'#171c24');d=ImageDraw.Draw(im)
        im.paste(Image.fromarray(row['log/human']).resize((256,256)),(0,24))
        mask=np.asarray(row['log/m2e_ftn_ema'])[self.ftn.layout].reshape(*self.ftn.grid,3)
        im.paste(Image.fromarray(mask).resize((128,128),Image.Resampling.NEAREST),(256,152))
        if int(row.get('log/offline_goal_step',-1))>=0:
            im.paste(Image.fromarray(row['log/offline_goal']).convert('RGB').resize((128,64),Image.Resampling.NEAREST),(256,52))
        d.text((5,5),f'Step {step:,} | Life {life}',fill='white');d.text((260,5),'Goal / mask',fill='white')
        draw_hud(im,values(record))
        for packet in self.stream.encode(av.VideoFrame.from_image(im)):self.out.mux(packet)
        self.frames+=1;self.end=step;self.game_seconds=record['game_seconds']
        self.death_reason=(record.get('death_reason') or record.get('last_damage_type') or 'unknown') if bool(row.get('is_terminal',False)) else None
        if bool(row.get('is_last',False)):
            self.close('death' if bool(row.get('is_terminal',False)) else 'episode ended')

    def close(self,reason='recorder stopped'):
        if self.out is None:return
        for packet in self.stream.encode():self.out.mux(packet)
        self.out.close();self.out=None
        from .life_video_retention import threshold
        minimum=threshold()
        if minimum>0 and self.game_seconds<=minimum:
            self.tmp.unlink(missing_ok=True)
            return
        target=self.root/(self.stem+'.mp4');self.tmp.replace(target)
        meta=dict(file=target.name,life=self.life,start_step=self.start,end_step=self.end,
                  frames=self.frames,fps=6,duration=self.frames/6,game_seconds=self.game_seconds,
                  complete=self.started_at_spawn and reason=='death',end_reason=reason,
                  death_reason=self.death_reason,
                  time_basis='Every agent observation, played at 6 FPS; not game-clock timing')
        tmp=self.root/(self.stem+'.json.tmp');tmp.write_text(json.dumps(meta,indent=2))
        tmp.replace(self.root/(self.stem+'.json'))
