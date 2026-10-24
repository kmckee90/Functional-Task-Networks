import io
import unittest
from dataclasses import asdict
from unittest.mock import patch
import torch

from agents.agent_ftn import RDDPG_Dual, FTNDecoder, IsolatedRMSprop, FTN_DEFAULTS
from experiment_ftn import Config, Hyperparameters
from experiment_recode import Hyperparameters as OriginalHyperparameters


def make_agent(device='cpu', **extra):
    cfg={**asdict(Config()),**asdict(Hyperparameters())}
    cfg.update(device=device,hidden_size=8,minibatch_size=4,memory_length=32,
        goal_memory_length=32,sample_memory_length=16,n_optimizer_steps=2,n_optimizer_epochs=1,
        ftn_k=2,ftn_hidden_size=8,ftn_width=3,ftn_depth=2,ftn_support_size=4,
        ftn_replay_every=1,ftn_overlap_low=0.,ftn_overlap_high=1.)
    cfg.update(extra)
    return RDDPG_Dual(2,2,(torch.full((2,),-1.,device=device),torch.ones(2,device=device)),**cfg).to(device)


class FTNTests(unittest.TestCase):
    def test_config_preserves_original_and_exposes_ftn(self):
        cfg=asdict(Hyperparameters())
        for k,v in asdict(OriginalHyperparameters()).items():self.assertEqual(cfg[k],v)
        for k,v in FTN_DEFAULTS.items():self.assertEqual(cfg[k],v)

    def test_selected_forward_and_inactive_weights(self):
        cfg=dict(FTN_DEFAULTS,ftn_k=2,ftn_hidden_size=8,ftn_width=3,ftn_depth=2)
        decoder=FTNDecoder(4,2,100,False,cfg)
        opt=IsolatedRMSprop(decoder.parameters(),.01)
        x=torch.randn(3,4)
        full=(decoder.contributions(x)*decoder.mask[None,:,None]).sum(1)
        torch.testing.assert_close(decoder(x),full)
        # Accumulate gradients from two masks. Update only their union, even
        # when other atoms already have RMSprop history from a prior step.
        decoder(x,torch.ones(8)).square().mean().backward();opt.step(torch.ones(8,dtype=torch.bool))
        opt.zero_grad(set_to_none=True)
        old=[p.clone() for p in decoder.parameters()]
        old_avg=[opt.state[p]['square_avg'].clone() for p in decoder.parameters()]
        a=torch.tensor([1,1,0,0,0,0,0,0.]);b=torch.tensor([0,1,1,0,0,0,0,0.])
        (decoder(x,a).square().mean()+decoder(x,b).square().mean()).backward()
        active=(a+b).bool();opt.step(active)
        for p,before,avg in zip(decoder.parameters(),old,old_avg):
            torch.testing.assert_close(p[~active],before[~active],rtol=0,atol=0)
            torch.testing.assert_close(opt.state[p]['square_avg'][~active],avg[~active],rtol=0,atol=0)

    def test_replay_search_gradients_and_checkpoint(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                agent=make_agent(device,a=.25)
                for buffer,length in [(agent.memory,8),(agent.goal_memory,24)]:
                    for i in range(length):
                        buffer.remember(torch.randn(2,device=device),torch.randn(agent.feedback_size,device=device),
                            torch.randn(2,device=device),torch.randn(agent.feedback_size,device=device),
                            torch.rand(2,device=device)*2-1,torch.tensor([.3],device=device),
                            torch.tensor([i%4==3],device=device))
                idx=torch.arange(4,device=device)
                weights=[p.clone() for p in agent.act_pred.parameters()]
                reservoirs=[p.clone() for m in (agent.state_model_o,agent.state_model_f) for p in m.parameters()]
                hidden=[m.hidden.clone() for m in (agent.state_model_o,agent.state_model_f)]
                with patch.object(agent,'get_reward',side_effect=lambda obs:torch.ones(len(obs),device=device)),patch.object(agent,'_prioritized_recall',side_effect=lambda buf,mode:buf.recall(idx)):
                    agent.replay()
                self.assertEqual(agent.ftn_updates,2)
                self.assertEqual(agent.ftn_replay_updates,1)
                self.assertTrue(any(not torch.equal(a,p) for a,p in zip(weights,agent.act_pred.parameters())))
                for before,p in zip(reservoirs,[p for m in (agent.state_model_o,agent.state_model_f) for p in m.parameters()]):torch.testing.assert_close(before,p,rtol=0,atol=0)
                for h,m in zip(hidden,(agent.state_model_o,agent.state_model_f)):torch.testing.assert_close(h,m.hidden)
                for k,posterior in agent.posteriors.items():
                    self.assertEqual(int(posterior.m.sum()),2)
                    self.assertTrue(torch.isfinite(posterior.mask_logits).all())
                    for name in agent._names(k):torch.testing.assert_close(getattr(agent,name).mask,posterior.m)
                agent.cur_step=8;agent.epsilon=0.
                with torch.no_grad():out=agent(torch.zeros(2,device=device),torch.zeros(agent.feedback_size,device=device))
                self.assertEqual(agent.ftn_searches,3)
                self.assertTrue(torch.isfinite(out).all() and (out.abs()<=1).all())
                saved=io.BytesIO();torch.save(agent.state_dict(),saved);saved.seek(0)
                restored=make_agent(device,a=.25)
                restored.load_state_dict(torch.load(saved,map_location=device,weights_only=False))
                self.assertEqual(restored.ftn_updates,agent.ftn_updates)
                self.assertEqual(len(restored.ftn_archive),2)
                for k in agent.posteriors:torch.testing.assert_close(restored.posteriors[k].mask_logits,agent.posteriors[k].mask_logits)
                # Whole-object serialization is also used by the experiment.
                saved=io.BytesIO();torch.save(agent,saved);saved.seek(0)
                whole=torch.load(saved,map_location=device,weights_only=False)
                self.assertEqual(whole.ftn_updates,2)

    def test_changepoint_and_disabled_stream(self):
        agent=make_agent(OCP=False,ftn_changepoint_warmup=0,ftn_changepoint_sd=0.)
        d=dict(o=torch.randn(4,agent.rnn_size_o),f=torch.randn(4,agent.rnn_size_f),
            no=torch.randn(4,agent.rnn_size_o),nf=torch.randn(4,agent.rnn_size_f),
            action=torch.randn(4,2),ro=torch.ones(4,1),rf=torch.ones(4,1)*10,
            done=torch.ones(4,1,dtype=torch.bool),goal=False)
        old=[p.clone() for p in agent.q_pred_o.parameters()]
        with patch.object(agent.posteriors['f'],'reconfigure_changepoint',wraps=agent.posteriors['f'].reconfigure_changepoint) as search:
            agent._train_support(d)
            search.assert_called_once()
        for a,p in zip(old,agent.q_pred_o.parameters()):torch.testing.assert_close(a,p,rtol=0,atol=0)
        self.assertEqual(agent.ftn_changepoints,1)

if __name__=='__main__':unittest.main()
