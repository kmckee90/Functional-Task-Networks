"""Pod-side annotated replay of the wolf sighting; retain the original recording."""
import math,json
from pathlib import Path
import imageio_ffmpeg as ff
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('/home/dev/proposal_state/new_ftn')
def main():
    reader=ff.read_frames(str(ROOT/'339k_find-wolf.mp4'),pix_fmt='rgb24')
    meta=next(reader);frames=[Image.frombytes('RGB',meta['size'],raw) for raw in reader]
    assert meta['fps']==6 and meta['size']==(384,280)
    scale=3;size=(1152,840);font=ImageFont.truetype(str(ROOT/'DejaVuSans.ttf'),26)
    def enlarged(i):return frames[i].resize(size,Image.Resampling.NEAREST)
    def label(im,text,xy):
        d=ImageDraw.Draw(im);box=d.textbbox(xy,text,font=font,anchor='mm')
        d.rounded_rectangle((box[0]-10,box[1]-7,box[2]+10,box[3]+7),radius=6,fill='#111820')
        d.text(xy,text,font=font,anchor='mm',fill='white')
    def annotated(i):
        im=enlarged(i);d=ImageDraw.Draw(im);red='#ff3030'
        d.rectangle((258*3,52*3,382*3,117*3),outline=red,width=10)
        def arrow(a,b):
            a=tuple(v*3 for v in a);b=tuple(v*3 for v in b)
            d.line((a,b),fill=red,width=8)
            angle=math.atan2(b[1]-a[1],b[0]-a[0]);length=25
            d.polygon([b,(b[0]-length*math.cos(angle-.45),b[1]-length*math.sin(angle-.45)),(b[0]-length*math.cos(angle+.45),b[1]-length*math.sin(angle+.45))],fill=red)
        arrow((258,82),(137,80))
        arrow((320,117),(320,195))
        label(im,'Exploration goal',(585,210))
        label(im,'Goal-Policy subnetwork',(960,440))
        label(im,'Wolf sighting',(180,790))
        return im
    output=ROOT/'339k_find-wolf-annotated-replay.mp4'
    writer=ff.write_frames(str(output),size,fps=24,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,
        ffmpeg_log_level='error',output_params=['-crf','18','-threads','2','-movflags','+faststart'])
    writer.send(None);n=0
    def emit(im,repeats):
        nonlocal n
        raw=im.tobytes()
        for _ in range(repeats):writer.send(raw);n+=1
    try:
        for i in range(180,len(frames)):emit(enlarged(i),4)
        emit(Image.new('RGB',size,'black'),6)
        for i in range(225,len(frames)):
            im=enlarged(i);label(im,'Replay · ½ speed',(190,790))
            if i==243:
                still=annotated(i);still.save(ROOT/'339k_find-wolf-annotated-replay.jpg',quality=95)
                emit(still,72)
            emit(im,8)
    finally:writer.close()
    (ROOT/'339k_find-wolf-annotated-replay.json').write_text(json.dumps(dict(source='339k_find-wolf.mp4',start=30,replay_start=37.5,pause_source_time=40.5,pause_duration=3,replay_speed=.5,duration=n/24),indent=2))
    print(output,n/24,'seconds')

if __name__=='__main__':main()
