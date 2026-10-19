"""Pod-only duplicate-frame analysis and separate montage preview export."""
import argparse
import json
from fractions import Fraction
from pathlib import Path
import av
import numpy as np

ROOT=Path('/home/dev/globo_dreamer_state/media_review')
# Original 24fps montage chapter boundaries. Deliberate titles and annotations
# are excluded from deduplication, including the nut-gathering annotation.
SPANS=[('wander',198,454),('wander2',460,820),('wander3',826,1850),
       ('climb1',1952,2720),('climb2',2726,3086),('climb3',3092,3452),
       ('climb4a',3458,3630),('climb4b',3678,3866)]

def signature(frame):
    return frame.reformat(width=240,height=180,format='rgb24').to_ndarray().astype(np.int16)

def analyze():
    result={}
    with av.open(str(ROOT/'original.mp4')) as src:
        src.streams.video[0].thread_count=2
        for i,f in enumerate(src.decode(video=0)):
            section=next((s for s in SPANS if s[1]<=i<s[2]),None)
            if not section:continue
            name,start,end=section
            s=signature(f)
            item=result.setdefault(name,dict(diffs=[],changes=[],runs=[],start=start,end=end))
            if i==start:
                anchor=s;last=s;run=1
            else:
                d=np.abs(s-anchor);mad=float(d.mean());changed=float((d>8).mean())
                item['diffs'].append(mad);item['changes'].append(changed)
                if mad>.3 or changed>.004:
                    item['runs'].append(run);run=1;anchor=s
                else:run+=1
            if i==end-1:item['runs'].append(run)
    for name,r in result.items():
        print(name,'diff_quantiles',np.quantile(r['diffs'],[0,.25,.5,.75,.9,1]).round(3),
              'holds',dict(zip(*np.unique(r['runs'],return_counts=True))),flush=True)
    (ROOT/'frame-analysis.json').write_text(json.dumps(result))

def render():
    data=json.loads((ROOT/'frame-analysis.json').read_text())
    reference=data['wander']['runs']
    cadence=int(np.median(reference))
    assert 1<=cadence<=8,cadence
    repeats={};removed={};added={};all_indices=[]
    for name,start,end in SPANS:
        if not name.startswith('climb'):continue
        pos=start;removed[name]=0;added[name]=0
        for length in data[name]['runs']:
            for offset in range(cadence):
                index=pos+min(offset,length-1)
                repeats[index]=repeats.get(index,0)+1
            removed[name]+=max(0,length-cadence)
            added[name]+=max(0,cadence-length)
            pos+=length
        assert pos==end
    output=ROOT/'ftn-m2e-montage-smooth-climbing-preview.mp4'
    n=0
    with av.open(str(ROOT/'original.mp4')) as src,av.open(str(output),'w',options={'movflags':'+faststart'}) as dst:
        src.streams.video[0].thread_count=2
        out=dst.add_stream('libx264',rate=24)
        out.width=960;out.height=720;out.pix_fmt='yuv420p'
        out.options={'crf':'18','preset':'fast','threads':'2'}
        assert not src.streams.audio,'Source audio requires separate retiming'
        for i,f in enumerate(src.decode(video=0)):
            editing=any(start<=i<end for name,start,end in SPANS if name.startswith('climb'))
            for _ in range(repeats.get(i,0) if editing else 1):
                f.pts=n;f.time_base=Fraction(1,24)
                for packet in out.encode(f):dst.mux(packet)
                all_indices.append(i);n+=1
        for packet in out.encode():dst.mux(packet)
    report=dict(reference_hold_frames=cadence,removed_by_clip=removed,added_by_clip=added,
                output_frames=n,duration=n/24,source_frame_indices=all_indices)
    (ROOT/'smooth-climbing-preview.json').write_text(json.dumps(report,indent=2))
    with av.open(str(output)) as check:assert sum(1 for _ in check.decode(video=0))==n
    print({k:v for k,v in report.items() if k!='source_frame_indices'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['analyze','render']);a=p.parse_args()
    analyze() if a.mode=='analyze' else render()
