"""Render the annotated Nascent montage on the pod, without publishing it."""
import bisect
import json
import math
from fractions import Fraction
from pathlib import Path
import av
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/dev/globo_dreamer_state/media_review')
RED='#ff3030'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FPS=24

def text_box(im,xy,lines,size=25):
    d=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,size)
    x,y=xy;h=size+10;width=max(d.textlength(s,font=font) for s in lines)
    d.rounded_rectangle((x-12,y-10,x+width+12,y+h*len(lines)+2),radius=7,fill='#111820',outline=RED,width=3)
    for i,s in enumerate(lines):d.text((x,y+i*h),s,font=font,fill='white')

def arrow(im,a,b,width=5):
    d=ImageDraw.Draw(im);d.line((a,b),fill=RED,width=width)
    theta=math.atan2(b[1]-a[1],b[0]-a[0])
    d.polygon([b,(b[0]-18*math.cos(theta-.45),b[1]-18*math.sin(theta-.45)),
               (b[0]-18*math.cos(theta+.45),b[1]-18*math.sin(theta+.45))],fill=RED)

def intro(base,part):
    im=base.copy();d=ImageDraw.Draw(im)
    if part==1:
        d.rectangle((3,70,637,709),outline=RED,width=5)
        text_box(im,(660,415),['256×256 frames','downsample to','RGB 64×64','for input.'],23)
        arrow(im,(680,400),(638,340))
    elif part==2:
        d.rectangle((641,140,957,300),outline=RED,width=5)
        text_box(im,(35,112),['A compressed, unusual observation','in memory as the goal state'],25)
        arrow(im,(565,215),(640,215))
    else:
        d.rectangle((641,389,957,709),outline=RED,width=5)
        text_box(im,(30,400),["Agent's active brain state (FTN):",'1 white pixel = 1 active neuron.','Purple: Recent trace of activity.'],25)
        arrow(im,(560,548),(640,548))
    return im

def wolf(im,phase):
    # Final replay already contains half-speed motion and its initial goal pause.
    text_box(im,(25,535),["Agent's brain state learns",'to represent goals, changes','when a goal is accomplished.'],25)
    width=5
    ImageDraw.Draw(im).rectangle((641,389,957,709),outline=RED,width=width)
    arrow(im,(404,582),(641,550),width)
    return im

def wandering(im):
    d=ImageDraw.Draw(im)
    d.rectangle((6,15,250,58),outline=RED,width=3)
    text_box(im,(280,22),['Natural behavior within minutes of gameplay.'],25)
    return im

def frames(path,start,end):
    with av.open(str(path)) as c:
        c.streams.video[0].thread_count=2
        c.seek(max(0,int(start/FPS*1e6)))
        for f in c.decode(video=0):
            index=round(float(f.time)*FPS)
            if index<start:continue
            if index>=end:break
            yield index,f

def main():
    mapping=json.loads((ROOT/'smooth-climbing-preview.json').read_text())['source_frame_indices']
    source=ROOT/'ftn-m2e-montage-smooth-climbing-preview.mp4'
    base=next(frames(source,198,199))[1].to_image()
    title=Image.alpha_composite(base.convert('RGBA'),Image.new('RGBA',base.size,(45,48,52,190))).convert('RGB')
    d=ImageDraw.Draw(title);font=ImageFont.truetype(FONT,41)
    for i,line in enumerate(['Nascent:','Exploration with Continual Learning','in the Globo Open World']):
        d.text((480,292+i*62),line,font=font,anchor='mm',fill='white')
    title.save(ROOT/'nascent-annotated-preview.jpg',quality=95)
    intros=[intro(base,i) for i in (1,2,3)]
    for i,im in enumerate(intros,1):im.save(ROOT/f'intro-panel-{i}.jpg',quality=95)
    output=ROOT/'nascent-annotated-preview-v3.mp4';chapters=[];count=0
    with av.open(str(output),'w',options={'movflags':'+faststart'}) as out:
        stream=out.add_stream('libx264',rate=FPS);stream.width=960;stream.height=720;stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'fast','threads':'2'}
        def emit(image,repeats=1):
            nonlocal count
            f=av.VideoFrame.from_image(image) if isinstance(image,Image.Image) else image
            for _ in range(repeats):
                f.pts=count;f.time_base=Fraction(1,FPS)
                for packet in stream.encode(f):out.mux(packet)
                count+=1
        def segment(label,start,end,edit=False):
            begin=count
            a=bisect.bisect_left(mapping,start);b=bisect.bisect_left(mapping,end)
            brain_frames=0;last_brain=None
            for i,f in frames(source,a,b):
                original=mapping[i]
                # The requested 4:25 frame in v1 maps to this smoothed frame.
                if i==2792:
                    im=f.to_image();text_box(im,(55,150),['Learns to jump off the wolf?'],29)
                    arrow(im,(280,205),(220,510),6)
                    im.save(ROOT/'tree-wolf-callout-v2.jpg',quality=95)
                    pause_start=count;emit(im,4*FPS)
                    chapters.append(dict(label='Learns to jump off the wolf?',start=pause_start/FPS,end=count/FPS))
                if edit and 5890<=original<5994:
                    last_brain=f.to_image();im=wolf(last_brain.copy(),brain_frames/FPS)
                    if brain_frames==0:im.save(ROOT/'wolf-brain-callout-v2.jpg',quality=95)
                    emit(im);brain_frames+=1
                elif edit and any(s<=original<s+6*FPS for s in (198,460,826)):
                    im=wandering(f.to_image())
                    if original==198:im.save(ROOT/'wandering-callout.jpg',quality=95)
                    emit(im)
                elif 4520<=original<4568 or 3630<=original<3678:
                    im=f.to_image()
                    box=(378,138,605,181) if original>=4520 else (203,108,759,156)
                    ImageDraw.Draw(im).rounded_rectangle(box,radius=7,outline=RED,width=3)
                    emit(im)
                else:emit(f)
            if brain_frames:
                # Only 4⅓ seconds of half-speed footage remain in the recording.
                # Keep its speed intact, holding the final frame for the rest
                # of the requested six-second callout; do not invent motion.
                while brain_frames<6*FPS:
                    emit(wolf(last_brain.copy(),brain_frames/FPS));brain_frames+=1
            chapters.append(dict(label=label,start=begin/FPS,end=count/FPS))
            print(chapters[-1],flush=True)
        emit(title,4*FPS)
        chapters.append(dict(label='Nascent title',start=0,end=4))
        for i,im in enumerate(intros,1):
            begin=count;emit(im,8*FPS)
            chapters.append(dict(label=f'Panel explanation {i}',start=begin/FPS,end=count/FPS))
        segment('Early wandering',96,1856,True)
        segment('Interacting with wildlife',3872,5994,True)
        emit(Image.new('RGB',(960,720)),6)
        segment('Learning to climb trees',1856,3866)
        for packet in stream.encode():out.mux(packet)
    with av.open(str(output)) as c:
        c.streams.video[0].thread_count=2
        assert sum(1 for _ in c.decode(video=0))==count
    (ROOT/'nascent-annotated-preview-v3.json').write_text(json.dumps(dict(fps=FPS,frames=count,duration=count/FPS,
        chapters=chapters,first_wolf_playback_speed=1,final_wolf_replay_speed=.5,
        final_wolf_callout_seconds=6,final_wolf_end_hold_seconds=40/24,
        tree_pause_seconds=4,wandering_annotation_seconds_per_clip=6),indent=2))
    print('Validated',count,'frames;',count/FPS,'seconds',flush=True)

if __name__=='__main__':main()
