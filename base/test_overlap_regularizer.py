"""Root-project overlap integration checks; execute only on the pod."""
import copy
import unittest
from unittest.mock import patch
import contextlib
import io
import torch
from config import Default, Overlap
from ftn import FTN
from lora_ftn import LoRAFTN
from lora_rnn_ftn import LoRARNNFTN
from stack_ftn import StackFTN
from hml_ftn import HMLFTN
from hml_ftn_stable import HMLFTN as StableHMLFTN
from wave_ftn import WaveFTN

CASES=[(FTN,{}),(FTN,dict(backbone='quad',quad_dim=2)),
       (FTN,dict(use_second_mask=True,k2=4,reconfig_steps2=2,reconfig_sample_size2=8)),
       (LoRAFTN,dict(n_layers=2,width=8,rank=1)),
       (LoRARNNFTN,dict(n_steps=2,width=8,rank=1,io='ftn')),
       (StackFTN,dict(n_stack=2)),
       (StackFTN,dict(n_stack=2,residual=True,sequential=True)),
       (StackFTN,dict(n_stack=2,residual=True,recurrent=True,rnn_recurrent_mask=True)),
       (HMLFTN,dict(depth=1,rank=2)),
       (StableHMLFTN,dict(depth=1,rank=2)),
       (WaveFTN,dict(n_stack=2)),
       (WaveFTN,dict(n_stack=2,step_masks=False)),
       (WaveFTN,dict(n_stack=2,reconfig_bptt=False))]

def make(cls,extra,weight=0.,mode='bump'):
    pars=dict(input_size=3,output_size=2,hidden_size=16,square_dim=4,
              classification=True,k=4,reconfig_steps=2,reconfig_sample_size=8,
              dropout_p=0.,overlap_mode=mode,overlap_lambda=weight)
    if cls in (FTN,StackFTN,WaveFTN):pars.update(layers=2,inner_size=4)
    return cls(**pars,**extra)

def seed_mask(m):
    m.m.zero_();m.m.reshape(-1,16)[:,:4]=1
    m.record_training_mask()

class Tests(unittest.TestCase):
    def test_config(self):
        for cfg in [Default(),Overlap()]:
            for prop in ['FTN_PARS','QUAD_PARS','LORA_PARS','LORARNN_PARS','STACK_PARS','HML_PARS','WAVE_PARS']:
                pars=getattr(cfg,prop)
                self.assertEqual(pars['overlap_lambda'],cfg.RECONFIG_OVERLAP_LAMBDA)
                self.assertEqual(pars['overlap_temperature'],cfg.RECONFIG_OVERLAP_TEMPERATURE)

    def test_search_variants(self):
        x=torch.randn(8,3);y=torch.eye(2)[torch.arange(8)%2]
        for cls,extra in CASES:
            for mode in ['bump','gated']:
                with self.subTest(cls=cls.__name__,extra=extra,mode=mode):
                    m=make(cls,extra,2.,mode);seed_mask(m)
                    archive=m.train_mask_archive.clone()
                    before=[p.detach().clone() for p in m.parameters()]
                    for method in ['reconfigure','fast_reconfigure']:
                        result=getattr(m,method)(x,y)
                        self.assertTrue(torch.isfinite(result).all())
                        sums=result.reshape(-1,16).sum(-1)
                        self.assertTrue(torch.allclose(sums,torch.full_like(sums,4)))
                        self.assertTrue(torch.allclose(result,result.round(),atol=1e-6))
                    if extra.get('use_second_mask'):
                        m.dual_reconfigure(x,y)
                        self.assertTrue(torch.equal(m.m,m.m1*m.m2))
                    if cls in (StackFTN,LoRARNNFTN,WaveFTN) and not extra.get('rnn_recurrent_mask'):
                        arg={'n_masks':1} if cls is LoRARNNFTN else {'n_steps':1}
                        m._search_simultaneous(x,y,True,**arg)
                    for p,old in zip(m.parameters(),before):
                        self.assertTrue(torch.equal(p,old))
                    self.assertTrue(torch.equal(m.train_mask_archive,archive))

    def test_zero_is_bypassed(self):
        x=torch.randn(8,3);y=torch.eye(2)[torch.arange(8)%2]
        for cls,extra in CASES:
            with self.subTest(cls=cls.__name__,extra=extra):
                a=make(cls,extra);b=copy.deepcopy(a);seed_mask(a);seed_mask(b)
                b.configure_overlap('gated',0.,.8,.4)
                b._with_overlap=lambda loss,*args:loss
                for method in ['reconfigure','fast_reconfigure']:
                    torch.manual_seed(7);am=getattr(a,method)(x,y).clone()
                    torch.manual_seed(7);bm=getattr(b,method)(x,y).clone()
                    self.assertTrue(torch.equal(am,bm))

    def test_expansion_preserves_config(self):
        m=make(FTN,{},5.,'gated');seed_mask(m);expanded=m.expand(32)
        self.assertEqual(expanded.overlap_lambda,5.)
        self.assertEqual(expanded.overlap_mode,'seen_logits')
        self.assertTrue(torch.equal(expanded.train_mask_archive[:,:16],m.train_mask_archive))
        self.assertFalse(expanded.train_mask_archive[:,16:].any())

    def test_mean_logits_on_union_of_seen_atoms(self):
        m=make(FTN,{},1.)
        seed_mask(m)
        first=m.m.clone()
        m.m.zero_();m.m[4:8]=1;m.record_training_mask()
        second=m.m.clone()
        self.assertEqual(len(m.train_mask_archive),2)
        for mode in ['seen_logits','seen_mass','bump','gated']:
            m.train_mask_archive=m.train_mask_archive[:2].clone()
            m.configure_overlap(mode,1.,.5,.2)
            ref=m._overlap_reference()
            a=m._overlap_penalty(first,ref)
            b=m._overlap_penalty(second,ref)
            self.assertTrue(torch.equal(a,b))
            self.assertEqual(a.item(),.25)
            patchwork=torch.zeros(16);patchwork[[0,1,4,5]]=1
            self.assertEqual(m._overlap_penalty(patchwork,ref).item(),.25)
            disjoint=torch.zeros(16);disjoint[8:12]=1
            self.assertEqual(m._overlap_penalty(disjoint,ref).item(),0.)
            logits=torch.full((16,),-2.,requires_grad=True)
            penalty=m._overlap_penalty(logits,ref)
            self.assertEqual(penalty.item(),-1.)
            penalty.backward()
            self.assertTrue(torch.equal(logits.grad,ref/16))
            # Reference stays fixed if another training mask is later recorded.
            m.m.copy_(disjoint);m.record_training_mask()
            self.assertEqual(ref.sum().item(),8)
        m.m.copy_(first);self.assertFalse(m.record_training_mask())
        self.assertEqual(len(m.train_mask_archive),3)
        # Duplicate sightings and masks sharing atoms do not multiply cost.
        m.m.zero_();m.m[[0,1,4,5]]=1;m.record_training_mask()
        self.assertEqual(m._overlap_reference().sum().item(),12)

    def test_annealed_search_passes_raw_logits(self):
        x=torch.randn(8,3);y=torch.eye(2)[torch.arange(8)%2]
        for cls,extra in CASES:
            with self.subTest(cls=cls.__name__,extra=extra):
                m=make(cls,extra,1.);seed_mask(m)
                captured=[];penalty=m._overlap_penalty
                def capture(logits,reference):
                    if reference is not None and logits.requires_grad:
                        captured.append(logits.detach().clone())
                    return penalty(logits,reference)
                m._overlap_penalty=capture
                m.reconfigure(x,y)
                self.assertTrue(captured)
                self.assertLess(captured[0].abs().max().item(),.01)

    def test_archive_state_dict_and_old_checkpoint(self):
        m=make(FTN,{},1.);seed_mask(m)
        state=copy.deepcopy(m.state_dict());n=make(FTN,{},1.)
        n.load_state_dict(state)
        self.assertTrue(torch.equal(n.train_mask_archive,m.train_mask_archive))
        self.assertFalse(n.record_training_mask())
        del state['train_mask_archive']
        n.load_state_dict(state)
        self.assertEqual(n.train_mask_archive.numel(),0)

    def test_training_records_without_replay_and_eval_does_not(self):
        import run_experiments as runner
        x=torch.randn(8,3);y=torch.eye(2)[torch.arange(8)%2]
        for replay in [False,True]:
            cfg=Default();cfg.TRAIN_DURATION=3;cfg.RENDER_STEPS=False
            cfg.RECONFIG_REACTIVE=True;cfg.RECONFIG_REACTIVE_SD=-1e6
            cfg.RECONFIG_REACTIVE_EMA=0.;cfg.NOISE_PROBE=False
            cfg.EVAL_BACKWARDS=False;cfg.EVAL_RECONFIG_SAMPLE_SIZE=4
            cfg.N_SAMPLE_PER_TASK=3 if replay else 0
            m=make(FTN,{},1.);opt=torch.optim.Adam(m.parameters(),lr=.001)
            buffer=runner.ReplayBuffer(maxlen=1 if replay else 0)
            recorded=[]
            original=m.record_training_mask
            def record():
                recorded.append(m.m.detach().clone()>0.5)
                return original()
            m.record_training_mask=record
            with patch.object(runner,'Config',cfg,create=True),contextlib.redirect_stdout(io.StringIO()):
                runner.train(m,lambda _: (x,y),[0,1],1,'test',opt,buffer,save_viz=False)
                self.assertEqual(len(recorded),6)
                self.assertTrue(torch.equal(m.train_mask_archive,torch.stack(recorded).unique(dim=0)) or
                                {tuple(r.tolist()) for r in m.train_mask_archive}=={tuple(r.tolist()) for r in recorded})
                archive=m.train_mask_archive.clone()
                runner.eval_reconfigure(m,x,y)
                self.assertTrue(torch.equal(m.train_mask_archive,archive))

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
