"""Three-section FTN+M2E montage; execute only on the media pod."""
import json
import math
import subprocess
from PIL import Image, ImageDraw, ImageFont
from .compose_new_ftn import ROOT, FPS, W, H, EXE, ff

SECTIONS = [
    ('Early wandering', [('3k_wander.mp4', 0, None), ('6k_obs.mp4', 15, 30),
                         ('7k_look-seek.mp4', 0, None)]),
    ('Learning to climb trees', [('tree climb 0.mp4', 0, None),
                                 ('climb tree get nut.mp4', 0, None),
                                 ('climb tree success.mp4', 0, None),
                                 ('in-tree-get-nut.mp4', 0, None)]),
    ('Interacting with wildlife', [('wolf-punch.mp4', 0, None),
                                   ('327k_wolf-run.mp4', 0, None),
                                   ('339k_find-wolf.mp4', 15, None),
                                   ('339k_find-wolf-annotated-replay.mp4', None, None)]),
]

def annotation(raw, name):
    im=Image.frombytes('RGB',(W,H),raw);d=ImageDraw.Draw(im);red='#ff3030'
    # Source panels are 384x280, fitted to 960x720 with a 10px top margin.
    def xy(x,y):return (x*2.5,y*2.5+10)
    if name=='in-tree-get-nut.mp4':
        d.ellipse((*xy(106,173),*xy(154,221)),outline=red,width=6)
        text='Gathering health-regenerating nuts';label=(W/2,130)
        a=(480,158);b=xy(140,172)
    else:
        text='Punched wolf';label=(490,160)
        a=(470,188);b=xy(133,137)
    d.line((a,b),fill=red,width=6)
    angle=math.atan2(b[1]-a[1],b[0]-a[0]);length=20
    d.polygon([b,(b[0]-length*math.cos(angle-.45),b[1]-length*math.sin(angle-.45)),
               (b[0]-length*math.cos(angle+.45),b[1]-length*math.sin(angle+.45))],fill=red)
    font=ImageFont.truetype(str(ROOT/'DejaVuSans.ttf'),30)
    box=d.textbbox(label,text,font=font,anchor='mm')
    d.rounded_rectangle((box[0]-12,box[1]-9,box[2]+12,box[3]+9),radius=6,fill='#111820')
    d.text(label,text,font=font,anchor='mm',fill='white')
    im.save(ROOT/(name.replace('.mp4','')+'-annotation.jpg'),quality=95)
    return im.tobytes()

def title_frame(name, lines):
    reader=ff.read_frames(str(ROOT/name),pix_fmt='rgb24');meta=next(reader)
    im=Image.frombytes('RGB',meta['size'],next(reader));reader.close()
    im.thumbnail((W,H),Image.Resampling.NEAREST)
    # Upscale as well as downscale, preserving the full source panel layout.
    scale=min(W/meta['size'][0],H/meta['size'][1])
    im=im.resize((round(meta['size'][0]*scale),round(meta['size'][1]*scale)),Image.Resampling.NEAREST)
    title=Image.new('RGB',(W,H),'#111820');title.paste(im,((W-im.width)//2,(H-im.height)//2))
    title=Image.alpha_composite(title.convert('RGBA'),Image.new('RGBA',(W,H),(45,48,52,170))).convert('RGB')
    draw=ImageDraw.Draw(title);font=ImageFont.truetype(str(ROOT/'DejaVuSans.ttf'),44)
    draw.line((420,260,540,260),fill='#7de1bf',width=3)
    for i,line in enumerate(lines):draw.text((W/2,H/2+(i-(len(lines)-1)/2)*60),line,font=font,anchor='mm',fill='#f1f4f7')
    return title

def frames(name, start, end):
    cmd = [EXE, '-v', 'error', '-threads', '2', '-i', str(ROOT/name), '-ss', str(start)]
    if end is not None: cmd += ['-t', str(end-start)]
    cmd += ['-an', '-vf', f'scale={W}:{H}:force_original_aspect_ratio=decrease:flags=neighbor,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x111820,setsar=1,fps={FPS}',
            '-threads', '2', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            raw = proc.stdout.read(W*H*3)
            if not raw: break
            if len(raw) != W*H*3: raise RuntimeError('Incomplete frame')
            yield raw
        if proc.wait(): raise RuntimeError('Decode failed: '+name)
    finally:
        proc.stdout.close()
        if proc.poll() is None: proc.terminate(); proc.wait()

def main():
    # Original normal-speed segment, followed by its quarter-second black gap.
    reader = ff.read_frames(str(ROOT/'339k_find-wolf.mp4'))
    meta = next(reader); reader.close()
    replay_start = meta['duration'] - 30 + .25
    # The source is 256 frames at 6 fps; avoid rounded container duration.
    count, _ = ff.count_frames_and_secs(str(ROOT/'339k_find-wolf.mp4'))
    replay_start = (count / meta['fps'] - 30) + .25
    output = ROOT/'ftn-m2e-montage.mp4'
    writer = ff.write_frames(str(output), (W,H), fps=FPS, codec='libx264',
        pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1,
        ffmpeg_log_level='error', output_params=['-crf','18','-threads','2','-movflags','+faststart'])
    writer.send(None); n = 0; chapters = []
    def emit(raw, repeats=1):
        nonlocal n
        for _ in range(repeats): writer.send(raw); n += 1
    try:
        main_title=title_frame('3k_wander.mp4',['Exploration with Continual Learning','in the Globo Open World'])
        main_title.save(ROOT/'ftn-m2e-montage.jpg',quality=94)
        chapters.append(dict(title='Exploration with Continual Learning in the Globo Open World',start=0,end=4))
        emit(main_title.tobytes(),4*FPS)
        for section, clips in SECTIONS:
            if n: emit(bytes(W*H*3), FPS//4)
            title=title_frame(clips[0][0],[section])
            chapters.append(dict(title=section,start=n/FPS,end=n/FPS+4))
            emit(title.tobytes(),4*FPS)
            for i,(name,start,end) in enumerate(clips):
                if i: emit(bytes(W*H*3), FPS//4)
                if start is None: start = replay_start
                begin = n/FPS
                pause_at={'in-tree-get-nut.mp4':172,'wolf-punch.mp4':552}.get(name)
                for frame_index,raw in enumerate(frames(name,start,end)):
                    if frame_index==pause_at:
                        emit(annotation(raw,name),2*FPS)
                        chapters.append(dict(annotation=name,source_time=start+frame_index/FPS,start=n/FPS-2,end=n/FPS))
                    emit(raw)
                chapters.append(dict(section=section,file=name,source_start=start,source_end=end,start=begin,end=n/FPS))
                print(chapters[-1],flush=True)
    finally: writer.close()
    (ROOT/'ftn-m2e-montage.json').write_text(json.dumps(dict(duration=n/FPS,fps=FPS,title_seconds=4,black_transition_seconds=.25,chapters=chapters),indent=2))
    print(output, n/FPS, 'seconds', flush=True)

if __name__ == '__main__': main()
