"""One real SiT update and post-block retention check; not an experiment run."""
import gc
import torch
from . import common as C
from .selection import retain
from .replay import CoverageReplay,pool_entries
from ..model import transport_batch

def main():
    m=C.model(125);train,fs=C.open_block(C.NAMES[0])
    gen=torch.Generator(device='cuda').manual_seed(711)
    x,y=transport_batch(train['z'][:4],train['y'][:4],gen,dropout=.1)
    m.reconfigure_changepoint(x,y)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=3e-4)
    loss=m.loss(m(x),y);loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    mask=m.m.detach().cpu().clone()
    entry,report=retain(m,C.NAMES[0],train,fs,5,[mask],{})
    assert len(entry['ids'])==128 and len(set(entry['ids']))==128
    C.close_block(C.NAMES[0],entry)
    del train,fs,x,y,loss,opt;gc.collect()
    pool=pool_entries(list(C.MEMORY.values()));replay=CoverageReplay(1234)
    assert replay.backward(m,pool) is None  # identical masks excluded
    active=torch.where(mask==1)[0];inactive=torch.where(mask==0)[0]
    changed=mask.clone();changed[active[:128]]=0;changed[inactive[:128]]=1
    m.m.copy_(changed.cuda())
    rep=replay.backward(m,pool)
    assert rep is not None and rep['overlap']==.5 and rep['distinct_images']==4
    replay.prune(pool)
    assert set(replay.image_counts)<={k for e in pool for k in e['keys']}
    assert all(len(C.features(C.NAMES[0],'train',b))==128 for b in ['dino','style','face'])
    try:C.open_block(C.NAMES[0])
    except AssertionError:pass
    else:raise AssertionError('Reopened old full training set')
    C.atomic(C.ART/'integration.json',dict(passed=True,retained=128,replay_overlap=rep['overlap'],selection=report,
        checks=['real SiT training gradient','real feature selection','retained-only feature access','historical full-set reopen rejected','real replay gradient','85-percent gate','pruned replay state']))
    print('INTEGRATION_PASSED',report['coverage'],flush=True)

if __name__=='__main__':main()
