"""Pod-only architecture, isolation, mask search, replay and checkpoint checks."""
import json
import tempfile
from pathlib import Path
import torch
from .model_free import ModelFreeBehavior
from .m2e_ftn import M2EFTN


def main():
    torch.set_num_threads(4)
    results=[]
    for variant in ('flat','conv'):
        torch.manual_seed(19)
        with tempfile.TemporaryDirectory(prefix='m2e-ftn-test-',dir='/home/dev/globo_dreamer_state') as tmp:
            b=ModelFreeBehavior('cuda:0',19,0.,128,128,128,Path(tmp)/'replay',
                reservoir_channels=128,reservoir_variant='tuned_v1',model_feedback=True)
            f=M2EFTN(b.agent,variant);b.ftn=f
            a=b.agent
            assert len(f.q.weights)==9 and len(f.q.biases)==9  # 8 hidden + scalar output
            assert f.q.H==8*f.q.k and abs(f.q.spec['relative_error'])<.01
            assert all(not p.requires_grad for p in a.reservoir.parameters())
            assert all(p.shape[0]==f.q.H for p in f.q.parameters())
            z=torch.randn(4,*a.hidden_shape,device=a.device)*.05
            mask=f.selector.m.clone()
            full=f.q.contributions(z).mul(mask[None,:,None]).sum(1)
            assert torch.allclose(full,f.q(z,mask),atol=2e-5)
            # Inactive bank changes cannot influence a selected assembly.
            saved=f.q(z,mask).detach().clone()
            inactive=torch.nonzero(mask==0).flatten()
            with torch.no_grad():f.q.biases[-1][inactive]+=1.
            assert torch.equal(saved,f.q(z,mask))
            for i in range(64):
                v=torch.randn(*a.hidden_shape,device=a.device)*.05
                emb=torch.rand(512,device=a.device)
                a.density.update(emb)
                a.memory.add(v,[i%25],0.,.1,emb,v*.9,i%31==0,i%31==0,model_step=i)
            idx=torch.arange(8,device=a.device)
            data=f.data(a.memory,idx,False)
            weights=[p.detach().clone() for p in f.q.parameters()]
            f.search(data)
            assert f.selector.last_reconfig_stats['steps']==2
            assert f.selector.m.sum()==f.q.k
            assert torch.isfinite(f.selector.mask_logits).all()
            assert all(torch.equal(x,p) for x,p in zip(weights,f.q.parameters()))
            # Optimizer modifies active neurons only, including moment state.
            opt=a.optimizer;opt.zero_grad();f.loss(data,mask).backward()
            opt.step(torch.nonzero(mask).flatten())
            assert all(torch.equal(p[inactive],old[inactive]) for p,old in zip(f.q.parameters(),weights))
            assert all(not torch.any(st['square_avg'][inactive]) for st in opt.state.values())
            loss=f.update(a.memory,idx,False)
            assert torch.isfinite(torch.tensor(loss))
            # Test overlapping replay explicitly with 50% shared active neurons.
            current=f.selector.m.clone();ids=torch.nonzero(current).flatten()
            rest=torch.nonzero(current==0).flatten()
            other=torch.zeros_like(current);other[ids[:f.q.k//2]]=1;other[rest[:f.q.k-f.q.k//2]]=1
            f.replay=[dict(mask=other,data=data)]
            original=f.search;f.search=lambda _:None
            f.update(a.memory,idx,False);f.search=original
            assert f.replay_updates==1
            b.updates=f.updates
            state=b.save();torch.save(state,Path(tmp)/'state.pt')
            before=f.q(z).detach().clone();f.selector.mask_logits.add_(1)
            b.load(torch.load(Path(tmp)/'state.pt',weights_only=False))
            assert torch.equal(before,f.q(z))
            assert f.replay_updates==1 and a.memory.size==64
            assert torch.isfinite(torch.tensor(a.train_round(3)))
            rotation=a.frontier_rotation
            assert rotation.current is not None and rotation.current['source']=='goal'
            assert int(a.goal.model_step[rotation.current['index']])==rotation.current['id']
            snapshot=f.save();chosen=rotation.current['id'];rotation.current=None
            f.load(snapshot);assert rotation.current['id']==chosen
            # Earlier run checkpoints lack the scheduler; their model is valid.
            snapshot.pop('frontier_rotation');f.load(snapshot)
            result=dict(f.q.spec,loss=loss,free_vram_gib=torch.cuda.mem_get_info()[0]/2**30)
            results.append(result);print(json.dumps(result),flush=True)
            del b,a,f
    print('PASS: both architectures, parameter budgets, exact-k Bayes search, neuron isolation, filtered replay, checkpoint restore',flush=True)


if __name__=='__main__':main()
