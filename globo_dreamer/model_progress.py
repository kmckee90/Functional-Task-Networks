"""Read-only held-out inference and pod-side video snapshots during training."""
import itertools
import json
from pathlib import Path
import numpy as np


def target_updates(step, start, batch_steps, ratio):
    # No warmup debt: start counting only after collection/replay warmup.
    return int(max(0, step-max(start,batch_steps-1))*ratio/batch_steps)


class ModelProgress:
    def __init__(self, agent, dataset, root):
        self.agent=agent;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        data=dict(np.load(dataset))
        self.batches=[{k:v[i:i+8].copy() for k,v in data.items()} for i in range(0,len(data['image']),8)]
        self.stream=iter(agent.stream(itertools.cycle(self.batches)))

    def write(self,step,updates):
        from .env import atomic_json
        from .world_probe import video
        import time
        results=[]
        for _ in self.batches:
            _,r=self.agent.report(self.agent.init_report(8),next(self.stream))
            results.append({k:np.asarray(v) for k,v in r.items() if k!='params/summary'})
        r={k:np.concatenate([a[k] for a in results]) for k in results[0]}
        out=self.root/f'step{step:07d}';out.mkdir(exist_ok=True)
        video(out/'model.mp4',results[0])
        video(out/'fixed.mp4',results[0],index=0)
        truth=r['truth'].astype(np.float32)/255;C=truth.shape[1]//2
        target=truth[:,C:];persist=truth[:,C-1:C]
        scores=dict(step=step,model_updates=updates,time=time.time(),sequences=len(truth),
            posterior_mse=float(np.mean((r['posterior']-truth)**2)),
            body_mse=float(np.mean((r['predicted_body']-r['true_body'])**2)))
        for h in (1,4,8,16):
            scores[f'h{h}_mse']=float(np.mean((r['forecast'][:,h-1]-target[:,h-1])**2))
            scores[f'h{h}_persistence_mse']=float(np.mean((persist[:,0]-target[:,h-1])**2))
            scores[f'h{h}_wrong_action_mse']=float(np.mean((r['wrong_actions'][:,h-1]-target[:,h-1])**2))
        error=r['novelty_error'].ravel();dis=r['novelty_disagreement'].ravel()
        corr=float(np.corrcoef(error,dis)[0,1]) if min(error.std(),dis.std())>1e-9 else None
        scores.update(novelty_error_mean=float(error.mean()),novelty_disagreement_mean=float(dis.mean()),
                      novelty_error_correlation=corr)
        atomic_json(out/'scores.json',scores)
        with (self.root/'history.jsonl').open('a') as f:f.write(json.dumps(scores)+'\n')
        atomic_json(self.root/'latest.json',dict(scores,folder=out.name))
