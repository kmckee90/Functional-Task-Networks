"""Pod-only scheduler tests, without starting an environment."""
import torch
from .frontier_rotation import FrontierRotation


def main():
    pool=[dict(id=i,emb=torch.full((512,),i/5),novelty=10-i,index=i,source='memory') for i in range(5)]
    r=FrontierRotation(5);chosen=r.choose(pool,0);last=chosen['id'];last_step=0;visited={last}
    for step in range(50,10001,50):
        chosen=r.choose(pool,step)
        if chosen['id']!=last:
            assert step-last_step<=1000
            visited.add(chosen['id']);last=chosen['id'];last_step=step
    assert len(visited)==5
    restored=FrontierRotation();restored.load(r.save())
    for step in range(10050,13001,50):
        assert r.choose(pool,step)['id']==restored.choose(pool,step)['id']
    # Duplicate/near-identical frames must not masquerade as alternative goals.
    r=FrontierRotation();r.choose(pool[:1],0)
    duplicates=[dict(pool[0],id=20+i) for i in range(10)]
    assert r.choose(duplicates,2000)['id']==0
    chosen=r.choose(pool,2050);assert chosen['id']!=0
    # Constant top five: exhausting a recent-choice cycle visits each once.
    r=FrontierRotation();ids=[r.choose(pool,step)['id'] for step in range(0,5000,1000)]
    assert len(set(ids))==5
    r=FrontierRotation();r.load(None);assert r.current is None
    print('PASS: capped dwell, no immediate repeats, visual deduplication, all-five rotation, exact checkpoint/RNG recovery, old-checkpoint default')


if __name__=='__main__':main()
