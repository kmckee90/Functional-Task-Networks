"""Pod-only output entropy math and mask-search integration checks."""
import copy
import math
import unittest
import torch
from config import Default, Entropy
from ftn import FTN
from test_overlap_regularizer import CASES, make, seed_mask


class Tests(unittest.TestCase):
    def test_entropy_values_and_gradients(self):
        m=make(FTN,{})
        self.assertAlmostEqual(m._output_entropy(torch.zeros(2,2)).item(),math.log(2),places=6)
        z=torch.tensor([[1.,-1.]],requires_grad=True)
        h=m._output_entropy(z);h.backward()
        self.assertLess(z.grad[0,0],0);self.assertGreater(z.grad[0,1],0)
        self.assertTrue(torch.allclose(h,m._output_entropy(z+100)))
        for binary in [False,True]:
            m.binary=binary
            z=torch.tensor([[10000.,-10000.]],requires_grad=True)
            h=m._output_entropy(z);h.backward()
            self.assertLess(h.item(),1e-5)
            self.assertTrue(torch.isfinite(z.grad).all())
        self.assertAlmostEqual(m._output_entropy(torch.zeros(2,2)).item(),math.log(2),places=6)

    def test_objective_and_training_loss(self):
        m=make(FTN,{})
        z=torch.tensor([[1.,-1.]]);y=torch.tensor([[1.,0.]])
        original=m.loss(z,y)
        m.configure_entropy(.3)
        self.assertTrue(torch.equal(m.loss(z,y),original))
        self.assertTrue(torch.allclose(m._reconfig_task_loss(z,y),original+.3*m._output_entropy(z)))
        m.classification=False
        self.assertTrue(torch.equal(m._reconfig_task_loss(z,y),m.loss(z,y)))
        with self.assertRaises(ValueError):m.configure_entropy(float('nan'))

    def test_all_paths_and_disabled_equivalence(self):
        x=torch.randn(8,3);y=torch.eye(2)[torch.arange(8)%2]
        for cls,extra in CASES:
            with self.subTest(cls=cls.__name__,extra=extra):
                m=make(cls,extra);seed_mask(m);m.configure_entropy(.1)
                called=[];entropy=m._output_entropy
                def check(z):
                    called.append(True);return entropy(z)
                m._output_entropy=check
                weights=[p.detach().clone() for p in m.parameters()]
                for method in ['reconfigure','fast_reconfigure']:
                    called.clear();getattr(m,method)(x,y)
                    self.assertTrue(called)
                if extra.get('use_second_mask'):m.dual_reconfigure(x,y)
                for p,old in zip(m.parameters(),weights):self.assertTrue(torch.equal(p,old))
                a=make(cls,extra);b=copy.deepcopy(a)
                b._reconfig_task_loss=b.loss
                for method in ['reconfigure','fast_reconfigure']:
                    torch.manual_seed(13);am=getattr(a,method)(x,y).clone()
                    torch.manual_seed(13);bm=getattr(b,method)(x,y).clone()
                    self.assertTrue(torch.equal(am,bm))

    def test_config_and_expansion(self):
        for cfg in [Default(),Entropy()]:
            for prop in ['FTN_PARS','LORA_PARS','STACK_PARS','HML_PARS','LORARNN_PARS','WAVE_PARS']:
                self.assertEqual(getattr(cfg,prop)['reconfig_entropy_coef'],cfg.RECONFIG_ENTROPY_COEF)
        m=FTN(input_size=3,output_size=2,hidden_size=16,square_dim=4,
              classification=True,k=4,reconfig_entropy_coef=.7)
        self.assertEqual(m.expand(32).reconfig_entropy_coef,.7)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
