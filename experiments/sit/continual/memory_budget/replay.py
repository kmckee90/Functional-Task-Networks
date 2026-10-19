"""Anonymous mask/example replay: 20% < overlap <= 85%, no task labels."""
import hashlib
import numpy as np
import torch
from ..model import transport_batch


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def pool_entries(memory):
    """Drop dataset metadata before handing memory to the replay policy."""
    pool=[]
    for e in memory:
        if '_replay_pool' not in e:
            keys=[fingerprint(z) for z in e['z']]
            e['_replay_pool']=[dict(mask=e['masks'][r['mask']],mask_key=fingerprint(e['masks'][r['mask']]),
                z=e['z'][r['indices']],y=e['y'][r['indices']],
                keys=[keys[i] for i in r['indices']]) for r in e['entries']]
        pool.extend(e['_replay_pool'])
    return pool


class CoverageReplay:
    def __init__(self,seed):
        self.rng=np.random.default_rng(seed)
        self.generator=torch.Generator(device='cuda').manual_seed(seed)
        self.counts={};self.last={};self.atom_counts=np.zeros(1024)
        self.image_counts={};self.loss_ema={};self.loss_start={}
        self.attempts=0;self.hits=0

    def state(self):
        return dict(rng=self.rng.bit_generator.state,generator=self.generator.get_state(),
                    **{k:getattr(self,k) for k in ['counts','last','atom_counts','image_counts','loss_ema','loss_start','attempts','hits']})

    def restore(self,state):
        self.rng.bit_generator.state=state['rng'];self.generator.set_state(state['generator'])
        for k,v in state.items():
            if k not in ['rng','generator']:setattr(self,k,v)

    def prune(self,pool):
        """Do not retain per-example state for discarded active-block images."""
        images={k for e in pool for k in e['keys']};masks={e['mask_key'] for e in pool}
        self.image_counts={k:v for k,v in self.image_counts.items() if k in images}
        for field in ['counts','last','loss_ema','loss_start']:
            setattr(self,field,{k:v for k,v in getattr(self,field).items() if k in masks})

    def backward(self,model,pool):
        self.attempts+=1
        mask=model.m.detach().cpu().numpy();groups={}
        for index,e in enumerate(pool):
            key=e['mask_key']
            if key not in groups:
                shared=e['mask'].numpy()*mask;overlap=float(shared.sum()/256)
                if not .2<overlap<=.85:continue
                groups[key]=dict(mask=e['mask'],overlap=overlap,shared=shared,examples={})
            # Deduplicate examples with the same latent under the same mask.
            for i,image in enumerate(e['keys']):groups[key]['examples'].setdefault(image,(index,i))
        if not groups:return None
        keys=sorted(groups);weights=[]
        for key in keys:
            g=groups[key]
            coverage=float((g['shared']/(1+self.atom_counts)).sum())
            age=1+min(100,self.attempts-self.last.get(key,0))/100
            deterioration=np.clip(self.loss_ema.get(key,1)/max(self.loss_start.get(key,1),1e-8),.5,3)
            weights.append(coverage*age*deterioration/np.sqrt(1+self.counts.get(key,0)))
        weights=np.asarray(weights);weights=weights/weights.sum()
        # Exploration is uniform over distinct eligible masks, never over tasks.
        q=int(self.rng.choice(len(keys),p=.8*weights+.2/len(keys)))
        key=keys[q];g=groups[key];images=sorted(g['examples'])
        counts=np.array([self.image_counts.get(k,0) for k in images])
        order=np.lexsort((self.rng.random(len(images)),counts));chosen=[images[i] for i in order[:4]]
        if len(chosen)<4:chosen=(chosen*4)[:4]
        refs=[g['examples'][k] for k in chosen]
        z=torch.stack([pool[p]['z'][i] for p,i in refs]).cuda()
        y=torch.stack([pool[p]['y'][i] for p,i in refs]).cuda()
        # Original classifier-free dropout; null labels remain null automatically.
        x,target=transport_batch(z,y,self.generator,dropout=.1)
        loss=model.loss(model(x,m=g['mask'].cuda()),target);loss.backward();value=float(loss.detach())
        self.loss_start.setdefault(key,value)
        self.loss_ema[key]=.9*self.loss_ema.get(key,value)+.1*value
        self.counts[key]=self.counts.get(key,0)+1;self.last[key]=self.attempts
        self.atom_counts+=g['shared']
        for k in chosen:self.image_counts[k]=self.image_counts.get(k,0)+1
        self.hits+=1
        return dict(mask_key=key,image_keys=chosen,loss=value,overlap=g['overlap'],
                    shared_atoms=int(g['overlap']*256),distinct_images=len(set(chosen)))
