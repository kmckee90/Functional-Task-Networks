"""Pod-only maze and locomotion montages; copy deliverables to proposal assets."""
from pathlib import Path
import json
import shutil
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg as ff
from compose_globo import card, text_center, BG, OUT, FONT, first_frame

ASSETS=Path(__file__).resolve().parent.parent/'proposal assets'

def maze_legend(image):
    image=image.copy();draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(FONT,16) if Path(FONT).exists() else ImageFont.load_default(size=16)
    entries=[('#ffe34d','latest trajectory'),('#ff5252','visited location'),('#00ffff','current goal location')]
    widths=[22+draw.textlength(label,font=font) for _,label in entries]
    x=(960-sum(widths)-48)/2
    for (color,label),width in zip(entries,widths):
        draw.ellipse((x,109,x+9,118),fill=color)
        draw.text((x+18,104),label,font=font,fill='#c6d0d9')
        x+=width+24
    return image

class Movie:
    def __init__(self,name,fps,background):
        self.name=name;self.fps=fps;self.count=0;self.chapters=[]
        self.temp=OUT/f'{name}-pending.mp4'
        self.writer=ff.write_frames(str(self.temp),(960,720),fps=fps,
            codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,
            ffmpeg_log_level='error',output_params=['-crf','20','-threads','2','-movflags','+faststart'])
        self.writer.send(None)
        opening=card('Meta-Learning to Explore:','Faster, smarter exploration.',background)
        opening.save(OUT/f'{name}.jpg',quality=92)
        self.emit(opening,4*fps)

    def emit(self,image,count):
        data=image.tobytes()
        for _ in range(count):self.writer.send(data);self.count+=1

    def frame(self,view,title,caption):
        image=Image.new('RGB',(960,720),BG)
        view=view.convert('RGB');view.thumbnail((560,560),Image.Resampling.LANCZOS)
        image.paste(view,((960-view.width)//2,100+(560-view.height)//2))
        text_center(image,title,30,28)
        text_center(image,caption,72,22,'#b7c4cf')
        return maze_legend(image) if self.name=='maze-m2e' else image

    def finish(self,provenance):
        self.writer.close();target=OUT/f'{self.name}.mp4';self.temp.replace(target)
        metadata=dict(duration=self.count/self.fps,fps=self.fps,frames=self.count,
            chapters=self.chapters,**provenance)
        (OUT/f'{self.name}.json').write_text(json.dumps(metadata,indent=2))
        shutil.copyfile(target,ASSETS/target.name)
        print(target,metadata['duration'],'seconds',flush=True)

def maze():
    movie=Movie('maze-m2e',10,first_frame(ASSETS/'exploration/maze/fixed.webm'))
    for filename,title in [('fixed.webm','Fixed recurring environments'),
            ('random.webm','Randomized environments'),
            ('continual.webm','Large, continual environments')]:
        movie.emit(card(title,'Progress over 100,000 steps',first_frame(ASSETS/'exploration/maze'/filename)),25)
        reader=ff.read_frames(str(ASSETS/'exploration/maze'/filename),pix_fmt='rgb24')
        meta=next(reader);assert meta['fps']==5
        # Decode sequentially, selecting quarter windows and the final four seconds.
        windows=[(round(meta['duration']*5*q/4),q*25000) for q in range(4)]
        windows.append((round(meta['duration']*5)-20,95000))
        for index,raw in enumerate(reader):
            for start,steps in windows:
                if start<=index<start+20:
                    if index==start:
                        movie.chapters.append(dict(source=filename,source_seconds=start/5,
                            step=steps,start=movie.count/movie.fps,seconds=4))
                    view=Image.frombytes('RGB',meta['size'],raw)
                    movie.emit(movie.frame(view,title,f'{steps:,} steps'),2)
                    break
            if index>=windows[-1][0]+19:break
        reader.close();print(filename,'complete',flush=True)
    movie.finish(dict(step_labels='Quarter labels inferred from a 100k-step recording; final four seconds labeled 95k as requested',
        playback='Original recorded speed; four seconds at each quarter and at the end'))

def body():
    paths=sorted((ASSETS/'exploration/body/best').glob('*.gif'),key=lambda p:int(p.stem.split('_')[-1]))
    background=first_frame(paths[0])
    movie=Movie('body-m2e',50,background)
    movie.emit(card('Native locomotion skills','from pure exploration',background),150)
    for path in paths:
        humanoid,step=path.stem.rsplit('_',1)
        start=movie.count/movie.fps
        with Image.open(path) as gif:
            for index in range(gif.n_frames):
                gif.seek(index)
                duration=gif.info.get('duration',100)
                assert duration%20==0
                movie.emit(movie.frame(gif.copy(),'Native locomotion skills from pure exploration',
                    f'{humanoid.replace("humanoid","Humanoid ")} · Step {int(step)*300:,}'),duration//20)
        movie.chapters.append(dict(source=path.name,step=int(step)*300,start=start,end=movie.count/movie.fps))
        print(path.name,'complete',flush=True)
    movie.finish(dict(step_labels='Numeric suffix from each GIF filename multiplied by 300 interactions, as specified by the experiment author',
        playback='Original GIF frame durations; clips are from different humanoid IDs'))

if __name__=='__main__':maze();body()
