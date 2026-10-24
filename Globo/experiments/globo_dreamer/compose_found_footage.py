"""Two pod-rendered cuts of nascent/proposal assets/demo 2; no site changes."""
import json
import math
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/dev/globo_dreamer_state/found_footage')
FPS=24
RED='#ff3030'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
ORDER=['cow fly','defend','plants','place_block','zoom']
SECTIONS=[('cow fly',4,16.5),('defend',0,18),('plants',0,19),
          ('place_block',0,26.8333334),('zoom',0,26.8333334)]


def font(n):return ImageFont.truetype(FONT,n)


def title(base,size,annotated=False):
    im=base.resize(size,Image.Resampling.NEAREST).convert('RGBA')
    im=Image.alpha_composite(im,Image.new('RGBA',size,(35,37,40,195))).convert('RGB')
    d=ImageDraw.Draw(im);cx,cy=size[0]/2,size[1]/2
    d.text((cx,cy-44),'Nascent: Found Footage',font=font(49),fill='white',anchor='mm')
    d.text((cx,cy+24),'Event-triggered clips from later training',font=font(25),fill='white',anchor='mm')
    if annotated:d.text((cx,cy+76),'An annotated selection',font=font(22),fill='white',anchor='mm')
    return im


def arrow(im,a,b):
    d=ImageDraw.Draw(im);d.line((a,b),fill=RED,width=5)
    angle=math.atan2(b[1]-a[1],b[0]-a[0])
    d.polygon([b,(b[0]-20*math.cos(angle-.45),b[1]-20*math.sin(angle-.45)),
              (b[0]-20*math.cos(angle+.45),b[1]-20*math.sin(angle+.45))],fill=RED)


def annotate(frame,heading,body,box=None,point=None,pause=False,slow=False):
    im=Image.new('RGB',(960,820),'black')
    im.paste(frame.resize((960,700),Image.Resampling.NEAREST),(0,120))
    d=ImageDraw.Draw(im)
    d.rounded_rectangle((15,12,945,108),radius=7,fill='black',outline=RED,width=3)
    d.text((30,24),heading,font=font(29),fill='white')
    d.text((30,67),body,font=font(23),fill='white')
    if box:
        x1,y1,x2,y2=box
        d.rectangle((x1*2.5,120+y1*2.5,x2*2.5,120+y2*2.5),outline=RED,width=4)
    if point:arrow(im,(440,110),(point[0]*2.5,120+point[1]*2.5))
    badge='Paused for inspection' if pause else 'Half-speed playback' if slow else None
    if badge:
        w=d.textlength(badge,font=font(18))
        d.rectangle((951-w-14,121,959,151),fill='black',outline=RED,width=2)
        d.text((951-w,125),badge,font=font(18),fill='white')
    return im


class Writer:
    def __init__(self,name,size):
        self.path=ROOT/name;self.count=0;self.chapters=[]
        self.out=av.open(str(self.path),'w',options={'movflags':'+faststart'})
        self.stream=self.out.add_stream('libx264',rate=FPS)
        self.stream.width,self.stream.height=size;self.stream.pix_fmt='yuv420p'
        self.stream.options={'crf':'18','preset':'fast','threads':'2'}

    def emit(self,im,repeats=4):
        for _ in range(repeats):
            f=av.VideoFrame.from_image(im);f.pts=self.count;f.time_base=Fraction(1,FPS)
            for packet in self.stream.encode(f):self.out.mux(packet)
            self.count+=1

    def chapter(self,label,start,**kwargs):
        self.chapters.append(dict(label=label,start=start/FPS,end=self.count/FPS,**kwargs))
        print(self.path.name,self.chapters[-1],flush=True)

    def close(self):
        for packet in self.stream.encode():self.out.mux(packet)
        self.out.close()
        with av.open(str(self.path)) as c:
            c.streams.video[0].thread_count=2
            decoded=sum(1 for _ in c.decode(video=0))
        assert decoded==self.count,(decoded,self.count)
        manifest=dict(file=self.path.name,frames=self.count,fps=FPS,duration=self.count/FPS,
                      chapters=self.chapters,source_fps=6,audio='Sources are silent',
                      notes='Frame repetition preserves source timing; no interpolated or generated motion.')
        self.path.with_suffix('.json').write_text(json.dumps(manifest,indent=2))


def main():
    footage={}
    for name in ORDER:
        with av.open(str(ROOT/'input'/f'{name}.mp4')) as c:
            assert not c.streams.audio,'Unexpected audio: preserve it explicitly before rendering'
            assert c.streams.video[0].average_rate==6
            c.streams.video[0].thread_count=2
            footage[name]=[f.to_image() for f in c.decode(video=0)]
    base=footage['zoom'][54]
    basic=Writer('nascent-found-footage.mp4',(960,700))
    first=title(base,(960,700));first.save(ROOT/'nascent-found-footage.jpg',quality=95)
    basic.emit(first,4*FPS);basic.chapter('Title',0)
    for name,begin,end in SECTIONS:
        start=basic.count
        for im in footage[name][round(begin*6):round(end*6)]:
            basic.emit(im.resize((960,700),Image.Resampling.NEAREST))
        basic.chapter(name,start,source=name+'.mp4',source_start=begin,source_end=min(end,len(footage[name])/6))
    basic.close()

    edited=Writer('nascent-found-footage-annotated.mp4',(960,820))
    pauses={('plants',72):('A closer look','The agent brings the plants into view.',(0,24,180,166),(105,118)),
            ('place_block',84):('Learning to use blocks','Digging and placing blocks in the surrounding terrain.',(50,70,255,250),(145,140)),
            ('defend',72):('Self-defense','An attacking wolf closes in.',(115,119,146,151),(130,122)),
            ('defend',84):('Self-defense','The agent kills the wolf.',(113,101,218,186),(160,132))}
    for name,begin,end in SECTIONS:
        start=edited.count
        for i in range(round(begin*6),min(round(end*6),len(footage[name]))):
            t=i/6;im=footage[name][i]
            if name=='cow fly':
                clean=Image.new('RGB',(960,820),'black')
                clean.paste(im.resize((960,700),Image.Resampling.NEAREST),(0,120))
                if i==90:
                    still=clean.copy();d=ImageDraw.Draw(still)
                    d.rounded_rectangle((15,12,945,108),radius=7,fill='black',outline=RED,width=3)
                    d.text((480,60),'EXPLORE',font=font(52),fill='white',anchor='mm')
                    still.save(ROOT/'nascent-found-footage-annotated.jpg',quality=95)
                    edited.emit(still,4*FPS)
                edited.emit(clean,4)
                continue
            if (name,i) in pauses:
                h,b,rect,point=pauses[(name,i)]
                still=annotate(im,h,b,rect,point,pause=True)
                still.save(ROOT/(name.replace(' ','-')+f'-{i}-annotation.jpg'),quality=95)
                edited.emit(still,2*FPS)
            if name=='zoom':h,b='Long sprints','Sustained bursts of movement across the landscape.'
            elif name=='plants':h,b='Exploring the undergrowth','A closer look at the plants and ground cover.'
            elif name=='place_block':h,b='Learning to use blocks','Digging and placing blocks in the surrounding terrain.'
            elif name=='defend':
                h='Self-defense'
                b='An attacking wolf closes in.' if t<13.5 else 'The wolf is killed; the agent moves on.'
            else:
                h='An unexpected shortcut'
                b='A cow becomes a launchpad: exploiting a game bug.' if t<11.5 else 'The landscape falls away below.'
            edited.emit(annotate(im,h,b),4)
        edited.chapter(name,start,source=name+'.mp4',source_start=begin,source_end=end,
                       pauses=[15] if name=='cow fly' else [i/6 for n,i in pauses if n==name],half_speed_source=None)
    edited.close()


if __name__=='__main__':main()
