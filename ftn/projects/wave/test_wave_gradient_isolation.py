"""Gradient-isolation audit. Run in base pod:
python -m unittest projects.wave.test_wave_gradient_isolation
"""
from pathlib import Path
from collections import defaultdict
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from wave_ftn import WaveFTN


class WaveGradientIsolationTests(unittest.TestCase):
    def make_model(self, **overrides):
        torch.manual_seed(19)
        args = dict(input_size=3,output_size=2,hidden_size=64,square_dim=8,
                    n_stack=3,layers=1,inner_size=4,k=4,wave_substeps=3,
                    wave_input_scale=.1,reconfig_steps=2,reconfig_sample_size=4)
        args.update(overrides)
        return WaveFTN(**args)

    def masks(self, model, offset=0):
        masks = torch.zeros(model.n_masks,64)
        for step in range(model.n_masks):
            start = offset+4*step
            masks[step,start:start+4] = 1
        return masks.flatten()

    def test_weight_and_per_step_activation_gradients(self):
        # Cover all boundary types, both mask layouts, both input schedules,
        # no/partial/total threshold rejection, and the reconfig-only flag.
        cases = 0
        for padding in ('zeros','circular','reflect'):
            for separate in (False,True):
                for first_only in (False,True):
                    for threshold in (0.,.05,2.):
                        for bptt in (False,True):
                            with self.subTest(padding=padding,separate=separate,
                                              first_only=first_only,threshold=threshold,bptt=bptt):
                                model = self.make_model(wave_padding=padding,step_masks=separate,
                                    wave_input_first_step_only=first_only,
                                    wave_input_threshold=threshold,reconfig_bptt=bptt,dropout_p=.2)
                                mask = self.masks(model)
                                per_step = model._split(mask)
                                union = torch.stack(per_step).bool().any(0)
                                raw = []
                                def capture(module, inputs, output):
                                    output.retain_grad()
                                    raw.append(output)
                                hook = model.cell.register_forward_hook(capture)
                                model.loss(model(torch.randn(4,3),m=mask),torch.randn(4,2)).backward()
                                hook.remove()
                                for activation, active in zip(raw,per_step):
                                    if activation.grad is not None:
                                        self.assertEqual(torch.count_nonzero(activation.grad[:,active==0]).item(),0)
                                for p in list(model.cell.weights)+list(model.cell.biases):
                                    self.assertIsNotNone(p.grad)
                                    self.assertEqual(torch.count_nonzero(p.grad[~union]).item(),0)
                                self.assertGreater(model.cell.biases[-1].grad[union].abs().sum().item(),0)
                                self.assertEqual(torch.count_nonzero(model.readout.weight.grad[:,per_step[-1]==0]).item(),0)
                                self.assertEqual(list(model.waves.parameters()),[])
                                cases += 1
        print(f'PASS: exact zero inactive neuron/parameter gradients in {cases} configurations')

    def test_spatial_reservoir_gradients_do_not_activate_neuron_parameters(self):
        model = self.make_model(step_masks=True)
        mask = self.masks(model)
        union = mask.reshape(3,64).bool().any(0)
        fields = []
        raw = []
        def capture_field(module, inputs, output):
            output.retain_grad()
            fields.append(output)
        def capture_cell(module, inputs, output):
            output.retain_grad()
            raw.append(output)
        h1 = model.waves.register_forward_hook(capture_field)
        h2 = model.cell.register_forward_hook(capture_cell)
        model(torch.randn(4,3),m=mask).square().sum().backward()
        h1.remove()
        h2.remove()
        # Later active neurons read the full reservoir field. That field can
        # receive gradients even at coordinates whose FTN neuron never fired.
        reservoir_outside = sum(f.grad[:,0].flatten(1)[:,~union].abs().sum().item()
                                for f in fields if f.grad is not None)
        self.assertGreater(reservoir_outside,0)
        self.assertGreater(raw[0].grad[:,:4].abs().sum().item(),0)
        for p in model.cell.parameters():
            self.assertEqual(torch.count_nonzero(p.grad[~union]).item(),0)
        print('PASS: reservoir gradients spread spatially; never-active neuron gradients remain zero')

    def test_reconfiguration_does_not_change_or_accumulate_parameter_gradients(self):
        for bptt in (False,True):
            for separate in (False,True):
                for algorithm in ('annealste','fast'):
                    model = self.make_model(reconfig_bptt=bptt,step_masks=separate,
                                            reconfig_algorithm=algorithm)
                    # Preserve existing gradients too: search must neither add
                    # gradients to weights nor silently clear caller gradients.
                    before = []
                    for p in model.parameters():
                        if p.requires_grad:
                            p.grad = torch.randn_like(p)
                            before.append((p,p.detach().clone(),p.grad.clone()))
                    model._reconfig(torch.randn(4,3),torch.randn(4,2))
                    for p,value,gradient in before:
                        torch.testing.assert_close(p,value,rtol=0,atol=0)
                        torch.testing.assert_close(p.grad,gradient,rtol=0,atol=0)
        print('PASS: mask search leaves neuron/readout weights and their existing gradients untouched')

    def test_adam_momentum_is_distinct_from_gradient_leakage(self):
        model = self.make_model(step_masks=False)
        optimizer = torch.optim.Adam(model.parameters(),lr=.01)
        x,y = torch.randn(4,3),torch.randn(4,2)
        first,second = self.masks(model,0),self.masks(model,16)
        optimizer.zero_grad(set_to_none=True)
        model.loss(model(x,m=first),y).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.loss(model(x,m=second),y).backward()
        p = model.cell.biases[-1]
        self.assertEqual(torch.count_nonzero(p.grad[:4]).item(),0)
        before = p.detach().clone()
        optimizer.step()
        movement = (p[:4]-before[:4]).abs().max().item()
        self.assertGreater(movement,0)
        print(f'CONFIRMED: Adam changes previously-active biases by up to {movement:.6g} with zero current gradient')
        # Match run_experiments.py's task-boundary safeguard: resetting Adam
        # removes this drift. This is distinct from changing masks mid-block.
        optimizer.state = defaultdict(dict)
        optimizer.zero_grad(set_to_none=True)
        model.loss(model(x,m=second),y).backward()
        inactive = second == 0
        before = [(p,p.detach().clone()) for p in model.cell.parameters()]
        readout_before = model.readout.weight.detach().clone()
        optimizer.step()
        for p,value in before:
            torch.testing.assert_close(p[inactive],value[inactive],rtol=0,atol=0)
        torch.testing.assert_close(model.readout.weight[:,inactive],
                                   readout_before[:,inactive],rtol=0,atol=0)
        print('PASS: existing task-boundary Adam reset prevents inactive parameter movement')


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
