"""Pod-only integration checks against saved, held-out-tested reservoir weights."""
import json
from pathlib import Path
import torch
from .legacy.models import Conv2dReservoir
from .legacy.agent import GloboExplorerDQN


def main():
    torch.set_num_threads(4)
    root=Path('/home/dev/globo_dreamer_state/diagnostics/reservoir_radius')
    results=[]
    for channels in (64,128):
        for seed in (7,19,101):
            saved=torch.load(root/f'winner_c{channels}_seed{seed}.pt',weights_only=True)
            res=Conv2dReservoir(4,num_channels=channels,seed=seed).cuda()
            cfg=res.configure_tuned(32,64,seed)
            for key,value in res.state_dict().items():
                torch.testing.assert_close(value.cpu(),saved['state_dict'][key],rtol=0,atol=1e-6)
            assert (res.input_scale,res.recur_scale,res.tc)==(.04,.99,1.)
            assert all(not p.requires_grad for p in res.parameters())
            results.append(dict(channels=channels,seed=seed,weights_match=True,config=cfg))
    for variant in ('legacy','tuned_v1'):
        agent=GloboExplorerDQN(num_actions=25,in_h=32,in_w=64,reservoir_channels=128,
            reservoir_variant=variant,memory_length=64,goal_memory_length=64,
            recode_memory_size=64,device='cuda:0',seed=7,model_feedback=True)
        sd=agent.state_dict()
        agent.load_state_dict(sd)
        if variant=='legacy':
            del sd['reservoir_config'];agent.load_state_dict(sd)
        mismatched={'reservoir_config':{'version':'legacy' if variant=='tuned_v1' else 'tuned_v1'}}
        try:agent.load_state_dict(mismatched)
        except ValueError:pass
        else:raise AssertionError('Incompatible reservoir checkpoint was accepted')
        # Real Q decoder must receive a finite nonzero training signal even with
        # the lower-amplitude reservoir. Production training is checked by live run.
        x=torch.rand(8,4,32,64,device='cuda')
        hidden=agent.reservoir.init_hidden(8,32,64,'cuda')
        for _ in range(32):hidden=agent.reservoir(x,hidden)
        before=next(agent.q_net.parameters()).detach().clone()
        q=agent.q_net(hidden)[0]
        loss=(q-torch.linspace(-1,1,25,device='cuda')).square().mean()
        agent.optimizer.zero_grad();loss.backward()
        grad=next(agent.q_net.parameters()).grad
        assert torch.isfinite(grad).all() and grad.norm()>0
        agent.optimizer.step()
        assert not torch.equal(before,next(agent.q_net.parameters()))
        results.append(dict(variant=variant,checkpoint_roundtrip=True,mismatch_rejected=True,
            decoder_loss=float(loss),gradient_norm=float(grad.norm())))
        del agent,sd
    (root/'integration.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
