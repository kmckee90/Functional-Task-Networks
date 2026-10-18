"""Isolated CUDA RECODE lookup benchmark; synthetic vectors, no agent mutation."""
import json
import os
from pathlib import Path
import statistics
import time
import torch
from .legacy.recode.recode import adaptive_knn_reward

torch.set_num_threads(4)
torch.manual_seed(7)
device='cuda:0'
root=Path('/home/dev/globo_dreamer_state/benchmarks')
root.mkdir(parents=True,exist_ok=True)
results=[]
for size in (1000,10000,100000):
    memory=torch.rand(size,512,device=device)
    # Fix total support above adaptive k's threshold in every arm (k=64).
    counts=torch.full((size,),100000./size,device=device)
    d2=torch.tensor(1.,device=device)
    for batch in (1,64,128,8192):
        queries=torch.rand(batch,512,device=device)
        def lookup():
            # Identical 128-query chunking to install_scalable_sampling().
            return torch.cat([adaptive_knn_reward(x,memory,counts,d2) for x in queries.split(128)])
        with torch.no_grad():
            for _ in range(5):lookup()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            elapsed=[]
            repeats=30 if batch<8192 else 10
            for _ in range(repeats):
                torch.cuda.synchronize();start=time.perf_counter()
                value=lookup()
                torch.cuda.synchronize();elapsed.append((time.perf_counter()-start)*1000)
            row=dict(centroids=size,queries=batch,dimension=512,k=64,repeats=repeats,
                median_ms=statistics.median(elapsed),min_ms=min(elapsed),max_ms=max(elapsed),
                p90_ms=sorted(elapsed)[int(.9*(len(elapsed)-1))],
                queries_per_second=batch/(statistics.median(elapsed)/1000),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                finite=bool(torch.isfinite(value).all()))
        results.append(row);print(json.dumps(row),flush=True)
        del queries,value
    del memory,counts,d2
(root/'recode_lookup.json').write_text(json.dumps(dict(
    device=torch.cuda.get_device_name(),dtype='float32',seed=7,results=results),indent=2))
