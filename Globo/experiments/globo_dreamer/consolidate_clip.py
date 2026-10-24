"""Display-only postprocessing for already-running workers; no model access."""
import json
import os
import shutil
from pathlib import Path
import time
import numpy as np
import av
from PIL import Image,ImageDraw
from .ftn_spatial_permute import spatial_permutation

STATE=Path('/home/dev/globo_dreamer_state')
REVISION=5


def process(live, dry_run=False):
    metadata=json.loads((live/'video.json').read_text())
    if not metadata.get('m2e_ftn'):return
    stamp=live/'clip_consolidation.json'
    previous=json.loads(stamp.read_text()) if stamp.exists() else {}
    display=live/'obs_display.mp4'
    if previous.get('step')==metadata['step'] and previous.get('revision',1)>=REVISION and display.exists():return
    source=live/'obs.mp4';signature=source.stat().st_mtime_ns
    with av.open(str(source)) as video:
        frames=[frame.to_ndarray(format='rgb24') for frame in video.decode(video=0)]
    # Adopt a clip compacted by the previous publisher without transforming it
    # again. Subsequent worker clips remain raw and have a separate display file.
    if frames and frames[0].shape[1]==384:
        if previous.get('step')!=metadata['step']:return
        if not dry_run:
            pending=live/'clip_obs_pending.mp4';shutil.copyfile(source,pending)
            if source.stat().st_mtime_ns!=signature:
                pending.unlink(missing_ok=True);return
            pending.replace(display)
            previous['revision']=REVISION
            tmp=stamp.with_suffix('.tmp');tmp.write_text(json.dumps(previous));tmp.replace(stamp)
        return
    height,width=metadata['grid']
    # Center samples avoid boundaries in the nearest-neighbor mask raster.
    ys=np.floor((np.arange(height)+.5)*256/height).astype(int)+24
    xs=np.floor((np.arange(width)+.5)*256/width).astype(int)+256
    tiles=np.stack([frame[ys[:,None],xs[None,:]].reshape(-1,3) for frame in frames])
    # White is current activity; green decays to ~76 on the next inactive frame.
    # Source is compressed video, so this is a display reconstruction, not raw masks.
    masks=np.unique(tiles[:,:,1]>220,axis=0)
    # For exponential trails, (exp(-t/tau))**2 = exp(-t/(tau/2)).
    # White active cells stay white; only the recorded display trail is shortened.
    if not (previous.get('step')==metadata['step'] and previous.get('ema_time_scale')==.5):
        tiles=np.rint(255*np.square(tiles.astype(np.float32)/255)).astype(np.uint8)
    order=spatial_permutation(masks,(height,width),previous=np.arange(height*width),
        sweeps=8,initialization='random_blobs',center=.25,roundness=2.,radius=7,temperature=.05,seed=0)
    pending=live/'clip_obs_pending.mp4';mask_pending=live/'clip_bw_pending.mp4'
    with av.open(str(pending),'w') as output, av.open(str(mask_pending),'w') as mask_output:
        stream=output.add_stream('libx264',rate=6);stream.width=384;stream.height=frames[0].shape[0];stream.pix_fmt='yuv420p'
        mask_stream=mask_output.add_stream('libx264',rate=6);mask_stream.width=256;mask_stream.height=280;mask_stream.pix_fmt='yuv420p'
        for frame,tile in zip(frames,tiles):
            remapped=np.asarray(Image.fromarray(tile[order].reshape(height,width,3)).resize((256,256),Image.Resampling.NEAREST))
            before=frame.copy();frame[24:280,256:512]=remapped
            assert np.array_equal(frame[:,:256],before[:,:256])
            assert np.array_equal(frame[:,512:],before[:,512:])
            canvas=Image.new('RGB',(384,frame.shape[0]),'#171c24')
            canvas.paste(Image.fromarray(frame[:,:256]),(0,0))
            draw=ImageDraw.Draw(canvas);draw.text((262,5),'Goal / mask',fill='white')
            if frame.shape[1]>=768:
                canvas.paste(Image.fromarray(frame[24:280,512:768]).resize((128,128),Image.Resampling.LANCZOS),(256,24))
            canvas.paste(Image.fromarray(remapped).resize((128,128),Image.Resampling.NEAREST),(256,152))
            if frame.shape[0]>280:
                canvas.paste(Image.fromarray(frame[280:,:384]),(0,280))
            assert np.array_equal(np.asarray(canvas)[:,:256],before[:,:256])
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):output.mux(packet)
            for packet in mask_stream.encode(av.VideoFrame.from_ndarray(frame[:280,256:512].copy(),format='rgb24')):mask_output.mux(packet)
        for packet in stream.encode():output.mux(packet)
        for packet in mask_stream.encode():mask_output.mux(packet)
    info=dict(step=metadata['step'],revision=REVISION,ema_time_scale=.5,layout='obs-left_goal-top-right_mask-bottom-right',frames=len(frames),distinct_masks=len(masks),
        scope='all distinct masks in this clip',source='compressed video cell-center samples',time=time.time())
    unchanged=(source.stat().st_mtime_ns==signature and
        json.loads((live/'video.json').read_text())['step']==metadata['step'])
    if not dry_run and unchanged:
        pending.replace(display);mask_pending.replace(live/'bw.mp4')
        mask_stamp=live/'mask_video.json';mask_tmp=live/'clip_mask_stamp.tmp'
        mask_tmp.write_text(json.dumps(dict(step=metadata['step'],frame_start=metadata['frame_start'],frame_end=metadata['frame_end'])))
        mask_tmp.replace(mask_stamp)
        tmp=stamp.with_suffix('.tmp');tmp.write_text(json.dumps(info));tmp.replace(stamp)
    else:
        pending.unlink(missing_ok=True);mask_pending.unlink(missing_ok=True)
    print(json.dumps(dict(run=live.parent.name,published=not dry_run and unchanged,**info)),flush=True)


def main():
    import argparse,fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--check',type=Path);args=parser.parse_args()
    if args.check:process(args.check,True);return
    os.nice(10)
    with (STATE/'clip_consolidation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            trials=json.loads((STATE/'continuous_runs/state.json').read_text())
            for trial in trials:
                if trial.get('m2e_ftn') and trial.get('status')=='running':
                    try:process(STATE/'runs'/trial['name']/'live')
                    except Exception as exc:print(trial['name'],repr(exc),flush=True)
            time.sleep(15)


if __name__=='__main__':main()
