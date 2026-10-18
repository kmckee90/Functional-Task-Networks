"""Pod-only checks of retained goal identity, stable rewards, routing and replay."""
import io
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .model_free import ModelFreeBehavior
from .frontier_ftn import FrontierFTN


def main():
    torch.set_num_threads(4)
    for mode in ('frontier','waypoints'):
        with tempfile.TemporaryDirectory(dir='/home/dev/globo_dreamer_state',prefix='goal-ftn-test-') as tmp:
            root=Path(tmp)
            b=ModelFreeBehavior('cuda:0',101,0.,128,128,128,root/'replay',
                reservoir_channels=128,reservoir_variant='tuned_v1',model_feedback=True)
            f=FrontierFTN(b.agent,mode,root);b.ftn=f;a=b.agent
            image=io.BytesIO();Image.fromarray(np.full((256,256,3),128,np.uint8)).save(image,format='JPEG')
            for i in range(64):
                z=torch.randn(*a.hidden_shape,device=a.device)*.05
                emb=torch.rand(512,device=a.device)
                a.density.update(emb)
                a.memory.add(z,[i%25],0.,.1,emb,z,False,i%16==15,model_step=i)
                f.images[i]=image.getvalue()
            d=f.data(a.memory,torch.arange(16,device=a.device),True)
            goal=f.register(d,15,True);assert goal==1
            other=f.register(f.data(a.memory,torch.arange(16,32,device=a.device),True),15,True)
            assert other==2
            o=f.options[goal]
            f.restore(goal);y=f.labels(d)
            assert y[-1,1]==1 and y[-1,3]==0  # stop bootstrap exactly at goal
            # Density drift changes priority, never the reaching objective.
            original=a._novelty;a._novelty=lambda x:torch.rand(len(x),device=x.device)*100
            assert torch.equal(y,f.labels(d));a._novelty=original
            old={k:v.clone() for k,v in o['posterior'].items()}
            f.update_option(other,f.options[other]['supports'][0])
            assert all(torch.equal(o['posterior'][k],v) for k,v in old.items())
            f.restore(goal);assert torch.equal(f.selector.m,old['m'])
            f.update_option(goal,o['supports'][0])
            f.active_goal=goal;f.restore(goal)
            f.current_image=torch.rand(3,32,64,device=a.device)
            state=b.save();torch.save(state,root/'state.pt')
            b.load(torch.load(root/'state.pt',weights_only=False))
            assert f.active_goal==goal and len(f.options)==3
            assert f.options[goal]['image']==image.getvalue()
            assert torch.equal(f.selector.m,f.options[goal]['posterior']['m'])
            # Explicitly check directed composition and absence of invented edges.
            f.options[goal]['updates']=f.options[other]['updates']=2
            f.edges={goal:{other:1}}
            distances,paths=f.shortest_paths(goal)
            assert paths[other]==[other] and goal not in f.shortest_paths(other)[1]
            assert f.reached(goal,d['emb'][-1],d['z_next'][-1])
            f.final_goal=other;f.publish()
            assert (root/'live/active_goal.jpg').exists()
            assert json.loads((root/'live/goal_status.json').read_text())['active_goal']==goal
            # Real scheduler: new frontier registration + two searches per batch.
            value=f.train_round(5)
            assert np.isfinite(value) and f.active_goal==goal
            assert f.selector.last_reconfig_stats['steps']==2
            from .m2e_ftn import render
            from .mask_diagnostics import publish
            import av
            rows=[(i,{'log/human':np.full((256,256,3),128,np.uint8),
                'log/m2e_ftn_ema':f.frame(),'log/goal_id':goal,'log/frontier_id':other}) for i in range(24)]
            render(root/'live',rows,24,f)
            metadata=json.loads((root/'live/video.json').read_text())
            publish(root/'live',metadata,mode,{'step':24})
            for name,width in [('obs.mp4',768),('bw.mp4',256)]:
                with av.open(str(root/'live'/name)) as video:
                    frames=list(video.decode(video=0))
                    assert len(frames)==24 and frames[0].width==width
            assert (root/'live/mask.html').exists()
            print(json.dumps(dict(mode=mode,loss=value,goals=len(f.options)-1,
                replay_updates=f.replay_updates,mask_searches=f.mask_searches)),flush=True)
            del b,a,f
    print('PASS: goal reward stationarity, independent posteriors, exact retrieval, terminating goal targets, directed routes, checkpoint and scheduler',flush=True)


if __name__=='__main__':main()
