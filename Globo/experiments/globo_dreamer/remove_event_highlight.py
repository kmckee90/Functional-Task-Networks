"""Remove the old single-frame marker from pod-only event clips."""
import json,os
from pathlib import Path
import av
import numpy as np
from .phase2 import STATE

def main():
    for i in range(1,5):
        root=STATE/'runs'/f'phase3_conv_r{i}_s19'/'event_clips'
        for path in sorted(root.glob('*.json')):
            meta=json.loads(path.read_text())
            if meta.get('highlight_removed'):continue
            video=path.with_suffix('.mp4')
            if not video.exists():continue
            with av.open(str(video)) as source:frames=[f.to_ndarray(format='rgb24') for f in source.decode(video=0)]
            if len(frames)!=180:raise RuntimeError(f'Unexpected frame count: {video}')
            # The original two-pixel border overwrote source pixels. Restore
            # only its edge strips from the immediately preceding frame.
            frame=frames[90];yellow=np.array([255,207,87])
            if np.abs(frame[24,100].astype(int)-yellow).sum()<110:
                previous=frames[89]
                frame[24:26,:256]=previous[24:26,:256];frame[278:280,:256]=previous[278:280,:256]
                frame[24:280,:2]=previous[24:280,:2];frame[24:280,254:256]=previous[24:280,254:256]
                backup=root/'original_highlights';backup.mkdir(exist_ok=True)
                if not (backup/video.name).exists():os.link(video,backup/video.name)
                temp=video.with_suffix('.clean.mp4');out=av.open(str(temp),'w')
                stream=out.add_stream('libx264',rate=12);stream.width=384;stream.height=280
                stream.pix_fmt='yuv420p';stream.options={'crf':'18','threads':'1'}
                for frame in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
                out.close();temp.replace(video)
                print('Cleaned',video.name,flush=True)
            meta['highlight_removed']=True
            path.write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
