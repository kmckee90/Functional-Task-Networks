"""Render the selected presentation montage on the pod. No training dependencies."""
from pathlib import Path
import json
import subprocess
import imageio_ffmpeg as ff
from PIL import Image, ImageDraw, ImageFont

SOURCE = Path(__file__).resolve().parent.parent / 'proposal assets/exploration/globo'
OUT = Path('/home/dev/proposal_state/media')
OUT.mkdir(parents=True, exist_ok=True)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BG = '#111820'
FPS = 12

def text_center(image, text, y, size=32, fill='#f1f4f7'):
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(FONT,size) if Path(FONT).exists() else ImageFont.load_default(size=size)
    width=draw.textlength(text,font=font)
    draw.text(((960-width)/2,y),text,font=font,fill=fill)

def first_frame(path):
    if Path(path).suffix.lower()=='.gif':
        with Image.open(path) as image:return image.convert('RGB')
    reader=ff.read_frames(str(path),pix_fmt='rgb24')
    try:
        metadata=next(reader)
        return Image.frombytes('RGB',metadata['size'],next(reader))
    finally:reader.close()

def card(title, subtitle, background=None):
    image=Image.new('RGB',(960,720),BG)
    if background is not None:
        view=background.convert('RGB')
        view.thumbnail((960,720),Image.Resampling.LANCZOS)
        # Scale up while preserving the full frame and aspect ratio.
        scale=min(960/view.width,720/view.height)
        view=view.resize((round(view.width*scale),round(view.height*scale)),Image.Resampling.NEAREST)
        image.paste(view,((960-view.width)//2,(720-view.height)//2))
        image=Image.alpha_composite(image.convert('RGBA'),Image.new('RGBA',(960,720),(45,48,52,170))).convert('RGB')
    ImageDraw.Draw(image).line((420,265,540,265),fill='#7de1bf',width=3)
    text_center(image,title,305,40)
    text_center(image,subtitle,370,26,'#e0e5eb')
    return image

def main():
    # User explicitly requested this small derivative alongside the original.
    crop=SOURCE/'fighting cropped.mp4'
    pending=OUT/'fighting-crop-pending.mp4'
    subprocess.run([ff.get_ffmpeg_exe(),'-v','error','-y','-i',str(SOURCE/'fighting.mp4'),
        '-vf','crop=256:280:0:0','-an','-c:v','libx264','-crf','18','-threads','2',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(pending)],check=True)
    # Source and runtime can be separate filesystems.
    import shutil
    shutil.copyfile(pending,crop);pending.unlink()
    clips=[
        ('wander.mp4','Naturalistic behavior','Wandering and looking around'),
        ('cliff.mp4','Cautious exploration','Following a cliff edge'),
        ('climb scan 0.mp4','Cautious exploration','Surveying from a hill'),
        ('climb tree.mp4','Advanced skills: Climbing','Climbing a tree'),
        ('fighting cropped.mp4','Advanced skills: Fighting','Interacting with a wolf'),
    ]
    target=OUT/'globo-m2e-pending.mp4'
    writer=ff.write_frames(str(target),(960,720),fps=FPS,codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,
        ffmpeg_log_level='error',output_params=['-crf','20','-threads','2','-movflags','+faststart'])
    writer.send(None);frames=0;chapters=[]
    def emit(image, repeat=1):
        nonlocal frames
        data=image.tobytes()
        for _ in range(repeat):writer.send(data);frames+=1
    try:
        opening=card('Meta-Learning to Explore:','Faster, smarter exploration.',first_frame(SOURCE/'wander.mp4'))
        opening.save(OUT/'globo-m2e.jpg',quality=92)
        emit(opening,4*FPS)
        for filename,title,subtitle in clips:
            start=frames/FPS
            emit(card(title,subtitle,first_frame(SOURCE/filename)),int(2.5*FPS))
            reader=ff.read_frames(str(SOURCE/filename),pix_fmt='rgb24')
            metadata=next(reader);assert metadata['size']==(256,280)
            assert metadata['fps']==6
            for raw in reader:
                image=Image.new('RGB',(960,720),BG)
                view=Image.frombytes('RGB',(256,280),raw)
                # Retain the recorded observation labels and pixel structure.
                image.paste(view.resize((512,560),Image.Resampling.NEAREST),(224,100))
                text_center(image,title,30,28)
                text_center(image,subtitle,676,19,'#b7c4cf')
                emit(image,2)  # Preserve the supplied playback speed, no interpolation.
            chapters.append(dict(file=filename,title=title,start=start,end=frames/FPS))
            print(filename,'complete',flush=True)
    finally:writer.close()
    target.replace(OUT/'globo-m2e.mp4')
    shutil.copyfile(OUT/'globo-m2e.mp4',SOURCE.parent.parent/'globo-m2e.mp4')
    (OUT/'globo-m2e.json').write_text(json.dumps(dict(frames=frames,fps=FPS,
        duration=frames/FPS,chapters=chapters,playback='Original recorded speed',
        fighting_source='Earlier joint Dreamer/explorer run; observation pane only'),indent=2))
    print('Saved',OUT/'globo-m2e.mp4',frames/FPS,'seconds',flush=True)

if __name__=='__main__':main()
