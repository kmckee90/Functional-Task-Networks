"""Bounded human render and literal reservoir-input diagnostics."""
import time
import numpy as np
import av
from PIL import Image, ImageDraw
from .env import atomic_json


def render(root, rows, step):
    # RGB and feedback are concatenated on CHANNELS in the actual model (4x32x64).
    # Stack their visualizations vertically to produce a 64x64 display tile.
    inputs=np.stack([r['log/m2e_input'] for _,r in rows])
    feedback=inputs[:,3]
    limit=max(1.,float(np.abs(feedback).max()))
    target=root/'obs_pending.mp4'
    container=av.open(str(target),'w');stream=container.add_stream('libx264',rate=6)
    has_goal='log/offline_goal' in rows[0][1]
    width=768 if has_goal else 512
    stream.width=width;stream.height=320;stream.pix_fmt='yuv420p'
    stream.options={'crf':'18'}
    for (obs_step,row),x in zip(rows,inputs):
        canvas=Image.new('RGB',(width,320),'#171c24')
        canvas.paste(Image.fromarray(row['log/human']),(0,32))
        rgb=np.uint8(np.clip(x[:3].transpose(1,2,0)*255,0,255).round())
        gray=np.uint8(np.clip((x[3]/limit+1)*127.5,0,255).round())
        tile=np.concatenate([rgb,np.repeat(gray[...,None],3,axis=2)],axis=0)
        canvas.paste(Image.fromarray(tile).resize((256,256),Image.Resampling.NEAREST),(256,32))
        draw=ImageDraw.Draw(canvas)
        draw.text((6,8),f'Step {obs_step:,}',fill='white')
        draw.text((262,8),'M2E: RGB above / feedback below',fill='white')
        draw.text((6,294),'256 -> 64 -> 32x64 RGB',fill='white')
        draw.text((262,294),f'Feedback gray: -{limit:.2f} to +{limit:.2f}',fill='white')
        if has_goal:
            draw.text((518,8),'Offline frontier (density view)',fill='white')
            goal_step=int(row['log/offline_goal_step'])
            if goal_step>=0:
                canvas.paste(Image.fromarray(row['log/offline_goal']).convert('RGB').resize((256,128),Image.Resampling.NEAREST),(512,96))
                draw.text((518,294),f'Observed at step {goal_step:,}',fill='white')
        for packet in stream.encode(av.VideoFrame.from_image(canvas)):container.mux(packet)
    for packet in stream.encode():container.mux(packet)
    container.close();target.replace(root/'obs.mp4')
    # Exact float values and vector inputs, before visualization / video compression.
    with (root/'inputs_pending.npz').open('wb') as f:
        np.savez_compressed(f,steps=np.array([s for s,_ in rows]),reservoir_input=inputs,
            feedback_vector=np.stack([r['log/m2e_feedback'] for _,r in rows]))
    (root/'inputs_pending.npz').replace(root/'inputs.npz')
    atomic_json(root/'video.json',dict(step=step,model_updates=0,time=time.time(),
        frame_start=rows[0][0],frame_end=rows[-1][0],frame_age=step-rows[-1][0],
        frames=len(rows),playback_seconds=len(rows)/6,model_free_only=True,
        panels=['human 256x256','actual RGB 32x64 + projected feedback 32x64'],
        reservoir_input_shape=[4,32,64],feedback_display_range=[-limit,limit],
        exact_inputs='inputs.npz',render_size=256,offline_goal=has_goal))
