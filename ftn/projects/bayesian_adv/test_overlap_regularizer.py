"""Run on the pod: python -m unittest test_overlap_regularizer -v."""
import copy
import unittest
import torch
from overlap_regularizer import OverlapRegularizer
from ftn import FTN, BayesFTN
from lora_ftn import LoRAFTN
from bayes_lora_ftn import BayesLoRAFTN
from bayes_stack_ftn import BayesRNNResidualFTN
from bayes_lora_rnn_ftn import BayesLoRARNNFTN


def model(cls, **kwargs):
    pars=dict(input_size=3,output_size=2,hidden_size=16,square_dim=4,
              k=4,reconfig_steps=3,reconfig_sample_size=8)
    if cls in (FTN,BayesFTN,BayesRNNResidualFTN):
        pars.update(layers=2,inner_size=4)
    else:
        pars.update(width=8,rank=1)
        if cls in (LoRAFTN,BayesLoRAFTN):pars['n_layers']=2
    if cls in (BayesRNNResidualFTN,BayesLoRARNNFTN):pars['n_stack']=2
    if cls not in (FTN,LoRAFTN):
        pars.update(bayes_reconfig_steps=3,bayes_changepoint_reconfig_steps=3,
                    bayes_reconfig_take_best=True)
    return cls(**pars,**kwargs)


CLASSES=(FTN,LoRAFTN,BayesFTN,BayesLoRAFTN,BayesRNNResidualFTN,BayesLoRARNNFTN)


class OverlapTests(unittest.TestCase):
    def test_config_bundles(self):
        from config import Default
        c=Default()
        c.RECONFIG_OVERLAP_MODE='gated';c.RECONFIG_OVERLAP_LAMBDA=2.
        for bundle in (c.FTN_PARS,c.BAYES_FTN_PARS,c.LORA_PARS):
            self.assertEqual(bundle['overlap_mode'],'gated')
            self.assertEqual(bundle['overlap_lambda'],2.)
            self.assertEqual(bundle['overlap_temperature'],.2)

    def test_shapes_gradients_and_fixed_reference(self):
        r=OverlapRegularizer();r.m=torch.tensor([1.,1.,0.,0.])
        for mode in ['bump','gated']:
            r.configure_overlap(mode,1.,.5,.2)
            ref=r._overlap_reference();r.m.zero_()
            self.assertEqual(ref.sum().item(),2)
            for value,positive in [(.1,True),(.9,False)]:
                m=torch.full((4,),value,requires_grad=True)
                cost=r._overlap_penalty(m,ref);cost.backward()
                self.assertEqual(m.grad[0].item()>0,positive)
                self.assertEqual(m.grad[2].item(),0)
            r.m.copy_(ref)
        r.configure_overlap('bump',1.)
        self.assertEqual(r._overlap_penalty(ref,ref).item(),0.)
        self.assertEqual(r._overlap_penalty(1-ref,ref).item(),0.)
        # Total activity normalization works for concatenated recurrent masks.
        a=torch.tensor([1.,0.,1.,0.]);b=torch.tensor([1.,0.,0.,1.])
        self.assertEqual(r._overlap_penalty(a,b).item(),.25)

    def test_validation(self):
        for args in [('bad',0,.5,.2),('bump',-1,.5,.2),('bump',1,2,.2),('gated',1,.5,0)]:
            with self.assertRaises(ValueError):OverlapRegularizer().configure_overlap(*args)

    def test_all_searches_and_cold_eval(self):
        x=torch.randn(8,3);y=torch.eye(2)[torch.arange(8)%2]
        for cls in CLASSES:
            for mode in ['bump','gated']:
                with self.subTest(cls=cls.__name__,mode=mode):
                    m=model(cls,overlap_mode=mode,overlap_lambda=2.)
                    m.m.reshape(-1,16)[:,:4]=1
                    before=[p.detach().clone() for p in m.parameters()]
                    m.reconfigure(x,y);m.reconfigure_changepoint(x,y)
                    if cls in (FTN,LoRAFTN):m.fast_reconfigure(x,y)
                    self.assertTrue(torch.all(m.m.reshape(-1,16).sum(-1)==4))
                    for p,old in zip(m.parameters(),before):
                        self.assertTrue(torch.equal(p,old));self.assertIsNone(p.grad)
                    # Cold inference remains independent of prior training mask.
                    a=model(cls,overlap_mode=mode,overlap_lambda=2.)
                    b=model(cls,overlap_mode=mode,overlap_lambda=2.)
                    a.load_state_dict(m.state_dict());b.load_state_dict(m.state_dict())
                    a.m.zero_();b.m.fill_(1)
                    torch.manual_seed(19);am=a.cold_reconfigure(x,y,steps=2).clone()
                    torch.manual_seed(19);bm=b.cold_reconfigure(x,y,steps=2).clone()
                    self.assertTrue(torch.equal(am,bm))

    def test_zero_weight_independent_of_mode_and_reference(self):
        x=torch.randn(8,3);y=torch.eye(2)[torch.arange(8)%2]
        for cls in CLASSES:
            a=model(cls);b=copy.deepcopy(a)
            a.m[:4]=1;b.m[-4:]=1
            b.configure_overlap('gated',0.,.8,.4)
            torch.manual_seed(29);am=a.reconfigure(x,y).clone()
            torch.manual_seed(29);bm=b.reconfigure(x,y).clone()
            self.assertTrue(torch.equal(am,bm),cls.__name__)

    def test_bayes_best_candidate_includes_penalty(self):
        x=torch.randn(8,3);y=torch.eye(2)[torch.arange(8)%2]
        for cls in (BayesFTN,BayesLoRAFTN):
            for mode in ['bump','gated']:
                m=model(cls,overlap_mode=mode,overlap_lambda=1.)
                m.m[:4]=1
                partial=torch.zeros(16);partial[[0,4,5,6]]=1
                disjoint=torch.zeros(16);disjoint[4:8]=1
                masks=iter([partial,disjoint,partial])
                m._sample_mask=lambda *args,**kw: next(masks)
                m.loss=lambda pred,y: pred.sum()*0
                result=m.reconfigure_changepoint(x,y)
                self.assertTrue(torch.equal(result,disjoint))
                self.assertEqual(m.last_reconfig_stats['overlap_penalty'],0)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
