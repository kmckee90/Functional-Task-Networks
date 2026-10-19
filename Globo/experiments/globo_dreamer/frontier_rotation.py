"""Bounded dwell on novelty frontiers; no changes to the density estimator."""
import math
import random
import torch


class FrontierRotation:
    def __init__(self,seed=721):
        self.rng=random.Random(seed);self.current=None;self.since=0;self.checked=0
        self.recent=[];self.switches=0;self.last_probability=0.

    @staticmethod
    def similar(a,b):
        return a['id']==b['id'] or float((a['emb']-b['emb']).square().mean().sqrt())<.03

    def choose(self,candidates,step,current_valid=True):
        # Candidates carry stable observation IDs, not recyclable ring indices.
        candidates=sorted(candidates,key=lambda c:c['novelty'],reverse=True)
        distinct=[]
        for candidate in candidates:
            if not any(self.similar(candidate,other) for other in distinct):distinct.append(candidate)
            if len(distinct)==5:break
        alternatives=[c for c in distinct if self.current is None or not self.similar(c,self.current)]
        age=max(0,step-self.since);old_age=max(0,self.checked-self.since)
        # Weibull survival exp(-(age/600)^2): ~94% switched by 1000,
        # followed by a hard cap when an alternative exists. Uses env time.
        self.last_probability=1-math.exp(-max(0,age*age-old_age*old_age)/(600.**2))
        switch=self.current is None or not current_valid or (alternatives and
            (age>=1000 or self.rng.random()<self.last_probability))
        self.checked=step
        if switch:
            pool=alternatives or (distinct if self.current is None or not current_valid else [])
            if pool:
                fresh=[c for c in pool if not any(self.similar(c,old) for old in self.recent[-4:])]
                if not fresh:
                    # Prefer the least recently used of the remaining top five.
                    def recency(c):
                        return max([-1]+[i for i,old in enumerate(self.recent) if self.similar(c,old)])
                    oldest=min(map(recency,pool));fresh=[c for c in pool if recency(c)==oldest]
                chosen=self.rng.choice(fresh)
                if self.current is not None:self.recent=(self.recent+[self.current])[-4:];self.switches+=1
                self.current=dict(chosen,emb=chosen['emb'].detach().cpu().clone());self.since=step
        return self.current

    def save(self):return dict(current=self.current,since=self.since,checked=self.checked,
        recent=self.recent,switches=self.switches,rng=self.rng.getstate())

    def load(self,state):
        if not state:return
        self.rng.setstate(state['rng'])
        for key in ('current','since','checked','recent','switches'):setattr(self,key,state[key])

    def metrics(self,step):return dict(frontier_age=max(0,step-self.since),
        frontier_switches=self.switches,frontier_switch_probability=self.last_probability)


@torch.no_grad()
def candidates(agent,buffer,source,scan):
    scan=scan.unique();novelty=agent._novelty(buffer.emb[scan])
    # Diversify the best candidates on GPU, transferring at most five vectors.
    order=novelty.argsort(descending=True);selected=[];result=[]
    for _ in range(5):
        if not len(order):break
        j=int(order[0]);index=int(scan[j]);emb=buffer.emb[index]
        result.append(dict(source=source,index=index,id=int(buffer.model_step[index]),
            emb=emb.detach().cpu().clone(),novelty=float(novelty[j])))
        distance=(buffer.emb[scan[order]]-emb).square().mean(-1).sqrt()
        order=order[distance>=.03]
    return result
