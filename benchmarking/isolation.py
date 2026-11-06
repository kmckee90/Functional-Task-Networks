"""FTN parameter-isolation policy requested for these benchmark experiments.

Weights have no decay. Every training mask reconfiguration resets Adam's
moments. A step may update only the union of current/replayed hard masks;
moments for every other atom are zeroed before Adam can move it. All trainable
parameters must be neuron-local or atom-local, including readout columns.
"""
from collections import defaultdict
import functools

import torch


def atom_axes(model):
    result={}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('parallel.weights.') or name.startswith('parallel.biases.'):
            axis=0
        elif name=='linear.weight':
            axis=1
        elif name.endswith(('.A','.B','.bias_bank')):
            axis=0
        else:
            raise RuntimeError(f'Unisolated shared trainable parameter: {name} {tuple(p.shape)}')
        if p.shape[axis]!=model.hidden_size:
            raise RuntimeError(f'Atom axis mismatch: {name} {tuple(p.shape)}')
        result[p]=(name,axis)
    if not result:
        raise RuntimeError('No trainable atom-local parameters found.')
    return result


class IsolationAdam(torch.optim.Adam):
    def __init__(self,model,lr):
        self.model=model
        self.axes=atom_axes(model)
        super().__init__(list(self.axes),lr=lr,weight_decay=0.)
        self.active_union=torch.zeros_like(model.m,dtype=torch.bool)
        self.reconfiguration_resets=0
        self.steps_completed=0
        self._search_depth=0
        self._hook=model.register_forward_hook(self._record_mask,with_kwargs=True)
        for name in ['_reconfig','reconfigure','fast_reconfigure','reconfigure_changepoint','cold_reconfigure']:
            if hasattr(model,name):
                self._wrap_search(name)

    def _record_mask(self,module,args,kwargs,output):
        if not module.training or not torch.is_grad_enabled() or not output.requires_grad:
            return
        mask=kwargs.get('m')
        if mask is None and len(args)>2:
            mask=args[2]
        if mask is None:
            mask=module.m
        if mask.requires_grad:
            raise RuntimeError('Weight training received a differentiable search mask.')
        self.active_union.logical_or_(mask.detach().reshape(-1)>0)

    def _wrap_search(self,name):
        original=getattr(self.model,name)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            if self.model.training and self._search_depth==0:
                # Reset moments AND bias-correction step counters. Persistent
                # Bayesian mask logits belong to the model and are preserved.
                self.state=defaultdict(dict)
                self.active_union.zero_()
                self.reconfiguration_resets+=1
            self._search_depth+=1
            try:
                return original(*args,**kwargs)
            finally:
                self._search_depth-=1
        setattr(self.model,name,wrapped)

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:
            raise RuntimeError('FTN benchmark isolation optimizer does not use closures.')
        if any(group['weight_decay']!=0 for group in self.param_groups):
            raise RuntimeError('FTN weight decay must be exactly zero.')
        if not self.active_union.any():
            raise RuntimeError('No hard training/replay mask recorded for this update.')
        for p,(_,axis) in self.axes.items():
            view=[1]*p.ndim
            view[axis]=self.model.hidden_size
            active=self.active_union.reshape(view)
            if p.grad is not None:
                p.grad.mul_(active)
            state=self.state.get(p,{})
            for name in ['exp_avg','exp_avg_sq','max_exp_avg_sq']:
                if name in state:
                    state[name].mul_(active)
        value=super().step()
        self.active_union.zero_()
        self.steps_completed+=1
        return value

    def policy(self):
        return {'optimizer':'Adam','weight_decay':0.,
                'reset_moments':'every training mask reconfiguration, plus native task resets',
                'inactive_moments':'zeroed each update outside union of actual current/replay masks',
                'reconfiguration_resets':self.reconfiguration_resets,
                'optimizer_steps':self.steps_completed,
                'trainable_parameter_axes':{name:axis for name,axis in self.axes.values()}}
