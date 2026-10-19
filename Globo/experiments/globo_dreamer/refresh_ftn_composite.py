"""Pod-only migration of a saved three-panel clip using recorded replay masks."""
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw
from .ftn_spatial_permute import spatial_permutation


def main():
    root=Path('/home/dev/globo_dreamer_state/runs/ftn_scaled_w4p0_s19')
    live=root/'live';meta=json.loads((live/'video.json').read_text())
    source=av.open(str(live/'recent.mp4'))
    if source.streams.video[0].width==1024:
        source.close();print('Already four panels');return
    rows={}
    for path in sorted((root/'replay').glob('*.npz')):
        with np.load(path) as data:
            for index,mask in zip(data['ftn_index'],data['ftn_mask']):rows[int(index)]=mask.copy()
    start,end=meta['frame_start'],meta['frame_end']
    assert all(i in rows for i in range(start,end+1))
    ema=np.zeros((2304,3),np.float32);rasters=[];masks=[]
    for index in sorted(rows):
        if index>end:break
        mask=np.zeros(2304,np.float32);mask[rows[index]]=1
        ema=np.maximum(ema*np.array([.95,.30,.975]),mask[:,None])
        if index>=start:rasters.append(np.uint8(ema*255));masks.append(mask)
    order=spatial_permutation(np.array(masks[::8][-16:]),(48,48),sweeps=1)
    output=av.open(str(live/'four_pending.mp4'),'w');stream=output.add_stream('libx264',rate=6)
    stream.width=1024;stream.height=280;stream.pix_fmt='yuv420p'
    count=0
    for frame in source.decode(video=0):
        canvas=Image.new('RGB',(1024,280),'#171c24');canvas.paste(frame.to_image(),(0,0))
        canvas.paste(Image.fromarray(rasters[count][order].reshape(48,48,3)).resize((256,256),Image.Resampling.NEAREST),(768,24))
        ImageDraw.Draw(canvas).text((774,5),'Live mask + EMA trails',fill='white')
        for packet in stream.encode(av.VideoFrame.from_image(canvas)):output.mux(packet)
        count+=1
    assert count==meta['frames']
    for packet in stream.encode():output.mux(packet)
    source.close();output.close();(live/'four_pending.mp4').replace(live/'recent.mp4')
    meta.update(panels=['observation','reconstruction','prediction','live action mask with EMA'],
        mask_timing='Recorded at each real observation; shared display permutation for the clip',
        reconstruction_mask='Current selected model mask at rendering; historical live masks are shown separately')
    pending=live/'video.pending.json';pending.write_text(json.dumps(meta,indent=2));pending.replace(live/'video.json')
    print('Four-panel clip:',count,'frames',1024,280)

if __name__=='__main__':main()
