"""Publish separate views without restarting the running agents."""
import json
import av
import numpy as np
from PIL import Image, ImageDraw


def publish(root, metadata):
    marker=root/'separate_video.json'
    names=('human','agent_input','offline_goal') if metadata.get('offline_goal') else ('human','agent_input')
    version=[metadata['step'],metadata['time'],3]
    try:
        saved=json.loads(marker.read_text())
        if saved['version']==version and all((root/f'{name}.mp4').exists() for name in names):
            return
    except (OSError,ValueError,KeyError):pass
    outputs=[]
    try:
        with av.open(str(root/'obs.mp4')) as source:
            assert source.streams.video[0].width==256*len(names)
            height=source.streams.video[0].height
            for name in names:
                container=av.open(str(root/f'{name}_pending.mp4'),'w')
                stream=container.add_stream('libx264',rate=6)
                stream.width=256;stream.height=height;stream.pix_fmt='yuv420p'
                stream.options={'crf':'18','threads':'1'}
                outputs.append((name,container,stream))
            with np.load(root/'inputs.npz') as data:
                steps=data['steps'].copy()
            for frame_index,frame in enumerate(source.decode(video=0)):
                pixels=frame.to_ndarray(format='rgb24')
                for i,(_,container,stream) in enumerate(outputs):
                    crop=av.VideoFrame.from_ndarray(pixels[:,i*256:(i+1)*256].copy(),format='rgb24')
                    if i==0:
                        view=crop.to_image()
                        draw=ImageDraw.Draw(view)
                        draw.rectangle((0,0,255,31),fill='#171c24')
                        draw.text((6,8),f'Step {int(steps[frame_index]):,}',fill='white')
                        crop=av.VideoFrame.from_image(view)
                    for packet in stream.encode(crop):container.mux(packet)
            for _,container,stream in outputs:
                for packet in stream.encode():container.mux(packet)
    finally:
        for _,container,_ in outputs:container.close()
    for name,_,_ in outputs:(root/f'{name}_pending.mp4').replace(root/f'{name}.mp4')
    temp=marker.with_suffix('.tmp');temp.write_text(json.dumps(dict(version=version)))
    temp.replace(marker)
