"""Fixed diagnostic objective; does not feed privileged geometry to the agent."""
import base64
import math
import numpy as np
from .report import records, progress_windows


def score_run(root, steps=40000, tail_windows=10):
    # After a crash/restart, use the last record for any repeated observation index.
    rows = {r['step']:r for r in records(root/'environment/trace.jsonl') if r['step']<=steps}
    rows = [rows[k] for k in sorted(rows)]
    if len(rows)<steps:
        raise ValueError(f'Incomplete trial: {len(rows)}/{steps} observations')
    windows=progress_windows(rows,1000)
    archive=np.empty((steps//10+1,192),np.float32)
    size=0; per_window=[0]*len(windows)
    for i,row in enumerate(rows):
        if 'view_signature' not in row or row.get('low_detail',True) or row.get('sky_fraction',1)>.85:
            continue
        emb=np.frombuffer(base64.b64decode(row['view_signature']),np.uint8).astype(np.float32)/255
        if size:
            diff=archive[:size]-emb
            distance=np.einsum('ij,ij->i',diff,diff).min()/192
            if distance<.06**2: continue
        archive[size]=emb; size+=1; per_window[i//1000]+=1
    for w,v in zip(windows,per_window): w['new_visual_views']=v
    tail=windows[-tail_windows:]
    spatial=np.mean([w['new_4m_cells'] for w in tail])
    visual=np.mean([w['new_visual_views'] for w in tail])
    low=np.mean([w.get('low_detail_fraction',1) for w in tail])
    sky=np.mean([w.get('sky_dominated_fraction',1) for w in tail])
    productive=np.mean([w['new_4m_cells']>=2 or w['new_visual_views']>=3 for w in tail])
    # Late discovery dominates. Historical breadth has only a small weight.
    value=(.45*math.log1p(spatial)+.45*math.log1p(visual)+.5*productive
           +.10*math.log1p(windows[-1]['cumulative_4m_cells']/(steps/1000))
           -1.5*low-.5*sky)
    return dict(score=float(value),steps=len(rows),spatial_cells=windows[-1]['cumulative_4m_cells'],
        visual_views=size,late_spatial_per_1k=float(spatial),late_visual_per_1k=float(visual),
        late_productive_fraction=float(productive),late_low_detail=float(low),late_sky=float(sky),
        windows=windows,objective_version=1,
        limitation='Visual distance and position are discovery proxies, not semantic achievements; '
                   'no cross-seed claim until all three seeds finish.')
