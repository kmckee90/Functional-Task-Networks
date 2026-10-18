"""Compose the supplied FTN robotics GIFs on the pod, excluding IoU panels."""
from pathlib import Path
import json
import shutil
from PIL import Image
import imageio_ffmpeg as ff
from compose_globo import card, text_center, OUT, BG

ASSETS=Path(__file__).resolve().parent.parent/'proposal assets'
SOURCE=ASSETS/'continual learning/Robotics/demos'
CLIPS=['plate-slide-side-v3_ftn.gif','drawer-open-v3_ftn.gif',
       'button-press-topdown-v3_ftn.gif','peg-insert-side-v3_ftn.gif','faucet-close-v3_ftn.gif']

def ftn_title(background, subtitle):
    title=card('Functional Task Networks:',
        'Neural Assemblies for Continual Learning and Knowledge Transfer',background)
    text_center(title,subtitle,425,28,'#7de1bf')
    return title

def main():
    with Image.open(SOURCE/CLIPS[0]) as gif:
        background=gif.convert('RGB').crop((0,0,640,350))
    title=ftn_title(background,'Flexible robotic arm control')
    title.save(OUT/'ftn-robotics.jpg',quality=92)
    pending=OUT/'ftn-robotics-pending.mp4'
    writer=ff.write_frames(str(pending),(960,720),fps=50,codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,
        ffmpeg_log_level='error',output_params=['-crf','18','-threads','2','-movflags','+faststart'])
    writer.send(None);count=0;chapters=[]
    def emit(image,n):
        nonlocal count
        data=image.tobytes()
        for _ in range(n):writer.send(data);count+=1
    try:
        emit(title,200)
        for filename in CLIPS:
            start=count/50
            with Image.open(SOURCE/filename) as gif:
                assert gif.size==(960,350)
                for i in range(gif.n_frames):
                    gif.seek(i);duration=gif.info.get('duration',60)
                    assert duration%20==0
                    crop=gif.convert('RGB').crop((0,0,640,350))
                    frame=Image.new('RGB',(960,720),BG)
                    frame.paste(crop.resize((960,525),Image.Resampling.NEAREST),(0,97))
                    emit(frame,duration//20)
            chapters.append(dict(source=filename,start=start,end=count/50))
            print(filename,'complete',flush=True)
    finally:writer.close()
    pending.replace(OUT/'ftn-robotics.mp4')
    shutil.copyfile(OUT/'ftn-robotics.mp4',ASSETS/'ftn-robotics.mp4')
    (OUT/'ftn-robotics.json').write_text(json.dumps(dict(duration=count/50,
        frames=count,fps=50,crop=[0,0,640,350],chapters=chapters,
        playback='Original GIF durations; one opening title, no interstitials'),indent=2))
    print('Saved',count/50,'seconds',flush=True)

def classification():
    source=ASSETS/'misc/demo_live_orth_1k_full.gif'
    with Image.open(source) as gif:
        title=ftn_title(gif.convert('RGB'),'Learning and recall of classifiers')
    title.save(OUT/'ftn-classification.jpg',quality=92)
    pending=OUT/'ftn-classification-pending.mp4'
    writer=ff.write_frames(str(pending),(960,720),fps=20,codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,
        ffmpeg_log_level='error',output_params=['-crf','18','-threads','2','-movflags','+faststart'])
    writer.send(None);count=0
    try:
        for _ in range(80):writer.send(title.tobytes());count+=1
        with Image.open(source) as gif:
            for i in range(gif.n_frames):
                gif.seek(i);duration=gif.info.get('duration',50)
                assert duration==50
                view=gif.convert('RGB');view.thumbnail((960,720),Image.Resampling.LANCZOS)
                frame=Image.new('RGB',(960,720),BG)
                frame.paste(view,((960-view.width)//2,(720-view.height)//2))
                writer.send(frame.tobytes());count+=1
    finally:writer.close()
    pending.replace(OUT/'ftn-classification.mp4')
    shutil.copyfile(OUT/'ftn-classification.mp4',ASSETS/'ftn-classification.mp4')
    (OUT/'ftn-classification.json').write_text(json.dumps(dict(source=source.name,
        duration=count/20,frames=count,fps=20,playback='Original GIF timing; full three panes'),indent=2))
    print('Classification saved',count/20,'seconds',flush=True)

def planning():
    source=ASSETS/'planning and representation/CSCG'
    with Image.open(source/'diagram.png') as im:diagram=im.convert('RGB')
    title=ftn_title(diagram,'Learning abstract actions for planning')
    title.save(OUT/'ftn-planning.jpg',quality=92)
    pending=OUT/'ftn-planning-pending.mp4'
    writer=ff.write_frames(str(pending),(1440,1080),fps=25,codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,
        ffmpeg_log_level='error',output_params=['-crf','18','-threads','2','-movflags','+faststart'])
    writer.send(None);count=0;chapters=[]
    def emit(view,n):
        nonlocal count
        view=view.convert('RGB');view.thumbnail((1440,1080),Image.Resampling.LANCZOS)
        frame=Image.new('RGB',(1440,1080),BG)
        frame.paste(view,((1440-view.width)//2,(1080-view.height)//2))
        raw=frame.tobytes()
        for _ in range(n):writer.send(raw);count+=1
    try:
        emit(title.resize((1440,1080),Image.Resampling.LANCZOS),100)
        emit(diagram,375)
        chapters.append(dict(source='diagram.png',start=4,end=19))
        square='square.gif' if (source/'square.gif').exists() else 'demo_ftn.gif'
        for name in (square,'maze.gif'):
            start=count/25
            with Image.open(source/name) as gif:
                for i in range(gif.n_frames):
                    gif.seek(i);duration=gif.info.get('duration',80)
                    assert duration%40==0
                    emit(gif.copy(),duration//40)
            chapters.append(dict(source=name,start=start,end=count/25))
            print(name,'complete',flush=True)
    finally:writer.close()
    pending.replace(OUT/'ftn-planning.mp4')
    shutil.copyfile(OUT/'ftn-planning.mp4',ASSETS/'ftn-planning.mp4')
    (OUT/'ftn-planning.json').write_text(json.dumps(dict(duration=count/25,frames=count,
        fps=25,chapters=chapters,playback='Diagram held 15 seconds; GIFs retain original durations and all panes'),indent=2))
    print('Planning saved',count/25,'seconds',flush=True)

if __name__=='__main__':main();classification();planning()
