"""Run only in the base pod: python -m unittest test_wave_ftn.py"""
import io
import unittest

import torch

from config import Wave, WaveSharedMask, WaveDetach
from run_compose import build_model
from wave_ftn import WaveFTN


class WaveFTNTests(unittest.TestCase):
    def make_model(self, **kwargs):
        torch.manual_seed(7)
        pars = dict(input_size=3,output_size=2,hidden_size=64,square_dim=8,
                    n_stack=3,k=4,layers=1,inner_size=4,wave_substeps=2,
                    reconfig_steps=2,reconfig_sample_size=4)
        pars.update(kwargs)
        return WaveFTN(**pars).eval()

    def test_reference_update_and_trace(self):
        model = self.make_model()
        x = torch.randn(2,3)
        output, trace = model.trace(x)
        r = torch.zeros(2,2,8,8)
        for mask in model._split(model.m):
            z = model.cell(model._cell_input(x,r))*mask
            r[:,0] += model.wave_input_scale*z.reshape(2,8,8)
            for _ in range(2):
                r = model.waves(r)
        torch.testing.assert_close(output,model.readout(z))
        torch.testing.assert_close(trace[-1]['r'],r)
        self.assertEqual(len(trace),1+3*(1+2))
        self.assertEqual(torch.count_nonzero(trace[0]['r']).item(),0)
        for frame in trace:
            if frame['phase']=='inject':
                self.assertEqual(torch.count_nonzero(frame['z'][:,frame['mask']==0]).item(),0)

    def test_reservoir_resets_and_batch_is_independent(self):
        model = self.make_model()
        x = torch.randn(3,3)
        before = model(x)
        model(torch.randn(4,3))
        torch.testing.assert_close(before,model(x))
        torch.testing.assert_close(before,torch.cat([model(row[None]) for row in x]))
        torch.testing.assert_close(before,model(x[:,None].expand(-1,3,-1)))

    def test_search_gradient_modes(self):
        for bptt in (False,True):
            model = self.make_model(reconfig_bptt=bptt)
            mask = model.m.clone().requires_grad_()
            pred = model._forward_stack(torch.randn(4,3),model._split(mask),detach=True)
            pred.square().sum().backward()
            grads = mask.grad.reshape(3,64).abs().sum(1)
            self.assertGreater(grads[-1].item(),0)
            if bptt:
                self.assertTrue((grads[:2]>0).all())
            else:
                self.assertEqual(grads[:2].sum().item(),0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_training_keeps_bptt_in_detach_mode(self):
        model = self.make_model(reconfig_bptt=False)
        mask = model.m.clone().requires_grad_()
        model(torch.randn(4,3),m=mask).square().sum().backward()
        self.assertTrue((mask.grad.reshape(3,64).abs().sum(1)>0).all())
        self.assertGreater(model.cell.weights[0].grad.abs().sum().item(),0)
        self.assertEqual(list(model.waves.parameters()),[])

    def test_reconfigure_masks_replay_and_no_weight_grads(self):
        for shared in (False,True):
            for bptt in (False,True):
                for algorithm in ('annealste','fast'):
                    with self.subTest(shared=shared,bptt=bptt,algorithm=algorithm):
                        model = self.make_model(step_masks=not shared,reconfig_bptt=bptt,
                                                reconfig_algorithm=algorithm)
                        old = model.m.clone()
                        x,y = torch.randn(4,3),torch.randn(4,2)
                        with torch.no_grad():
                            model._reconfig(x,y)
                        self.assertTrue(all(p.grad is None for p in model.parameters()))
                        self.assertTrue(((model.m==0)|(model.m==1)).all())
                        self.assertTrue((model.m.reshape(-1,64).sum(1)==4).all())
                        if not shared and not bptt:
                            torch.testing.assert_close(old[:-64],model.m[:-64])
                        if shared:
                            masks = model._split(model.m)
                            self.assertTrue(all(m.data_ptr()==model.m.data_ptr() for m in masks))
                        explicit = model.m.clone()
                        expected = model(x,m=explicit)
                        model.m.zero_()
                        torch.testing.assert_close(expected,model(x,m=explicit))

    def test_steps_add_no_parameters_and_checkpoint_roundtrip(self):
        models = [self.make_model(n_stack=n,step_masks=separate)
                  for n,separate in ((1,True),(6,True),(6,False))]
        counts = [sum(p.numel() for p in model.parameters()) for model in models]
        self.assertEqual(len(set(counts)),1)
        self.assertEqual([m.m.numel() for m in models],[64,384,64])
        buffer = io.BytesIO()
        torch.save(models[1].state_dict(),buffer)
        buffer.seek(0)
        restored = self.make_model(n_stack=6)
        restored.load_state_dict(torch.load(buffer,weights_only=True))
        x = torch.randn(3,3)
        torch.testing.assert_close(restored(x),models[1](x))

    def test_shared_mask_gradient_and_short_unroll(self):
        model = self.make_model(step_masks=False,reconfig_bptt=False)
        x = torch.randn(4,3)
        mask = model.m.clone().requires_grad_()
        model._forward_stack(x,model._split(mask),detach=True).sum().backward()
        self.assertGreater(mask.grad.abs().sum().item(),0)
        old = model.m.clone()
        output = model.forward_k(x,torch.randn(4,2),k_steps=1)
        self.assertEqual(output.shape,(4,2))
        torch.testing.assert_close(old,model.m)

    def test_match_stored_and_config_dispatch(self):
        cfg = type('TestConfig',(WaveDetach,WaveSharedMask,Wave),{})()
        cfg.SQUARE_DIM,cfg.K,cfg.LAYERS,cfg.INNER_DIM = 8,4,1,4
        cfg.CLASSIFICATION = False
        cfg.WAVE_INPUT_THRESHOLD = .4
        cfg.WAVE_INPUT_FIRST_STEP_ONLY = True
        model = build_model(cfg,3,2)
        self.assertIsInstance(model,WaveFTN)
        self.assertFalse(model.step_masks)
        self.assertFalse(model.reconfig_bptt)
        self.assertEqual(model.wave_input_threshold,.4)
        self.assertTrue(model.wave_input_first_step_only)
        model.reconfig_steps = 0
        model.match_stored = True
        stored = model.m.clone()
        model.set_stored_masks_ref([stored])
        model._reconfig(torch.randn(4,3),torch.randn(4,2))
        torch.testing.assert_close(stored,model.m)

    def test_invalid_configuration(self):
        for pars in ({'wave_rho':1.01},{'wave_c':1.},{'wave_padding':'invalid'},
                     {'wave_substeps':0},{'k':65},{'wave_input_threshold':-.1},
                     {'wave_input_threshold':float('nan')}):
            with self.assertRaises(ValueError):
                self.make_model(**pars)

    def test_threshold_signed_values_boundary_readout_and_gradients(self):
        class FixedCell(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.zeros(64))
                with torch.no_grad():
                    self.values[:6].copy_(torch.tensor([-.8,-.5,-.2,.2,.5,.8]))

            def forward(self,x,detach=False):
                values = self.values.detach() if detach else self.values
                return values.expand(len(x),-1)

        model = self.make_model(n_stack=1,k=6,wave_input_threshold=.5)
        model.cell = FixedCell()
        model.m.zero_()
        model.m[:6] = 1
        inputs = []
        hook = model.waves.register_forward_pre_hook(lambda module,args: inputs.append(args[0]))
        x = torch.randn(2,3)
        output = model(x)
        hook.remove()
        admitted = inputs[0][:,0].flatten(1)/model.wave_input_scale
        expected = torch.zeros_like(admitted)
        expected[:,0],expected[:,5] = -.8,.8
        torch.testing.assert_close(admitted,expected)
        torch.testing.assert_close(output,model.readout(model.cell.values.expand(2,-1)))
        gradient, = torch.autograd.grad(inputs[0].sum(),model.cell.values)
        torch.testing.assert_close(gradient[:6],
            torch.tensor([1.,0.,0.,0.,0.,1.])*2*model.wave_input_scale)
        model.wave_input_threshold = 1.
        _, trace = model.trace(x)
        self.assertTrue(all(torch.count_nonzero(frame['r']).item()==0 for frame in trace))
        torch.testing.assert_close(model(x),output)

    def test_first_step_only_inputs_and_reservoir_feedback(self):
        for enabled in (False,True):
            for per_step in (False,True):
                for search in (False,True):
                    model = self.make_model(wave_input_first_step_only=enabled)
                    x = torch.randn(2,3,3) if per_step else torch.randn(2,3)
                    original = x.clone()
                    seen = []
                    hook = model.cell.register_forward_pre_hook(
                        lambda module,args: seen.append(args[0].detach().clone()))
                    model._forward_stack(x,model._split(model.m),detach=search)
                    hook.remove()
                    for step, cell_input in enumerate(seen):
                        expected = x[:,step] if per_step else x
                        if enabled and step>0:
                            expected = torch.zeros_like(expected)
                        torch.testing.assert_close(cell_input[:,:,:3],expected[:,None,:].expand(-1,64,-1))
                    self.assertGreater(seen[1][:,:,3:].abs().sum().item(),0)
                    torch.testing.assert_close(x,original)
        model = self.make_model(wave_input_first_step_only=True)
        x = torch.randn(2,3,3,requires_grad=True)
        model(x).sum().backward()
        self.assertGreater(x.grad[:,0].abs().sum().item(),0)
        self.assertEqual(x.grad[:,1:].abs().sum().item(),0)

    def test_detach_last_recall_preserves_prefix(self):
        model = self.make_model(reconfig_bptt=False,match_stored=True,reconfig_steps=0)
        old = model.m.clone()
        recalled = torch.roll(old,7)
        model.set_stored_masks_ref([recalled])
        model._reconfig(torch.randn(4,3),torch.randn(4,2))
        torch.testing.assert_close(model.m[:-64],old[:-64])
        torch.testing.assert_close(model.m[-64:],recalled[-64:])


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
