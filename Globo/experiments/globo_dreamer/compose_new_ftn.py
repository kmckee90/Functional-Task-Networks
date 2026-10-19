"""Render proposal montage on a pod; source clips and outputs are runtime data."""
from pathlib import Path
import subprocess
import json
import imageio_ffmpeg as ff
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/dev/proposal_state/new_ftn')
FPS=24
W,H=960,720
EXE=ff.get_ffmpeg_exe()
CLIPS=[('3k_wander.mp4',0,None),('6k_obs.mp4',15,30),
       ('7k_look-seek.mp4',0,None),('327k_wolf-run.mp4',0,None),
       ('339k_find-wolf.mp4',15,None),('drown.mp4',0,None)]

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    reader=ff.read_frames(str(ROOT/CLIPS[0][0]),pix_fmt='rgb24')
    meta=next(reader)
    frame=Image.frombytes('RGB',meta['size'],next(reader));reader.close()
    scale=min(W/frame.width,H/frame.height)
    frame=frame.resize((round(frame.width*scale),round(frame.height*scale)),Image.Resampling.NEAREST)
    title=Image.new('RGB',(W,H),'#111820')
    title.paste(frame,((W-frame.width)//2,(H-frame.height)//2))
    title=Image.alpha_composite(title.convert('RGBA'),Image.new('RGBA',(W,H),(45,48,52,170))).convert('RGB')
    draw=ImageDraw.Draw(title)
    draw.line((420,245,540,245),fill='#7de1bf',width=3)
    for text,y,size in [('Smart Exploration with',285,40),('Brain-Like Continual Learning',345,40),('on Globo',410,32)]:
        font=ImageFont.truetype(str(ROOT/'DejaVuSans.ttf'),size)
        draw.text(((W-draw.textlength(text,font=font))/2,y),text,font=font,fill='#f1f4f7')
    title.save(ROOT/'ftn-agent-montage.jpg',quality=94)
    writer=ff.write_frames(str(ROOT/'ftn-agent-montage.mp4'),(W,H),fps=FPS,codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,
        ffmpeg_log_level='error',output_params=['-crf','18','-threads','2','-movflags','+faststart'])
    writer.send(None);count=0;chapters=[]
    def emit(raw,n=1):
        nonlocal count
        for _ in range(n):writer.send(raw);count+=1
    try:
        emit(title.tobytes(),4*FPS)
        for index,(name,start,end) in enumerate(CLIPS):
            if index:emit(bytes(W*H*3),FPS//4)
            begin=count/FPS
            cmd=[EXE,'-v','error','-threads','2','-i',str(ROOT/name),'-ss',str(start)]
            if end is not None:cmd+=['-t',str(end-start)]
            cmd+=['-an','-vf',f'scale={W}:{H}:force_original_aspect_ratio=decrease:flags=neighbor,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x111820,setsar=1,fps={FPS}',
                  '-threads','2','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE)
            while True:
                raw=proc.stdout.read(W*H*3)
                if not raw:break
                if len(raw)!=W*H*3:raise RuntimeError('Incomplete frame')
                emit(raw)
            if proc.wait():raise RuntimeError('Clip decode failed: '+name)
            chapters.append(dict(file=name,source_start=start,source_end=end,start=begin,end=count/FPS))
            print(chapters[-1],flush=True)
    finally:writer.close()
    (ROOT/'ftn-agent-montage.json').write_text(json.dumps(dict(fps=FPS,duration=count/FPS,title_seconds=4,black_transition_seconds=.25,clips=chapters),indent=2))

if __name__=='__main__':main()
