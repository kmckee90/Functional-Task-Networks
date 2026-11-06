"""Load exact FTN source snapshots; isolate each family in its own process.

Only the explicitly requested ViT adaptation replaces a backbone. Mask search,
posterior updates, loss, and replay scheduling come from the original sources.
"""
import importlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from storage import WORKSPACE, bulk, write_json

VARIANTS = {
    'Base': ('ftn', ['Base']),
    'Reactive_Low_NoReplay': ('ftn', ['Reactive_Low', 'NoReplay']),
    'LoRA': ('ftn', ['LoRA']),
    'LoRA_Reactive_Low_NoReplay': ('ftn', ['LoRA', 'Reactive_Low', 'NoReplay']),
    'Bayes': ('bayesian_adv', ['Bayes']),
    'BayesLoRA': ('bayesian_adv', ['BayesLoRA']),
}
VIT_NAME = 'vit_base_patch16_224'
VIT_CHECKPOINT_URL = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'


def pretrained_vit(n_classes=0):
    import timm
    import fcntl
    model=timm.create_model(VIT_NAME,pretrained=False,num_classes=n_classes)
    path=bulk('cache','vit','ViT-B_16.npz')
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            temporary=path.with_suffix('.partial')
            torch.hub.download_url_to_file(VIT_CHECKPOINT_URL,str(temporary))
            temporary.replace(path)
        model.load_pretrained(str(path))
    return model


def load_family(variant):
    family, classes = VARIANTS[variant]
    source = WORKSPACE / 'sources' / family
    if 'config' in sys.modules:
        raise RuntimeError('Run each FTN variant in a fresh process to isolate absolute imports.')
    sys.path.insert(0, str(source))
    module = importlib.import_module('config')
    bases = tuple(getattr(module, name) for name in reversed(classes))
    cfg = type('BenchmarkConfig', bases, {})()
    runner = importlib.import_module('run_experiments')
    return cfg, runner


def config_dict(cfg):
    result = {}
    for key in dir(cfg):
        if key.isupper():
            value = getattr(cfg, key)
            if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
                result[key] = value
    return result


def build(cfg, input_size, n_classes, vit=False, pretrained=True):
    native = importlib.import_module('ftn')
    arch = cfg.ARCHITECTURE
    if arch == 'base':
        cls, pars = native.FTN, cfg.FTN_PARS
    elif arch == 'bayes':
        cls, pars = native.BayesFTN, {**cfg.FTN_PARS, **cfg.BAYES_FTN_PARS}
    elif arch == 'lora':
        cls = importlib.import_module('lora_ftn').LoRAFTN
        pars = cfg.LORA_PARS
    elif arch == 'bayes_lora':
        cls = importlib.import_module('bayes_lora_ftn').BayesLoRAFTN
        pars = {**cfg.LORA_FTN_PARS, **cfg.BAYES_FTN_PARS}
    else:
        raise ValueError(arch)
    if vit:
        if arch not in ('lora', 'bayes_lora'):
            raise ValueError('ViT uses the explicitly authorized LoRA variants.')
        # Reuse the native class's complete inference implementation and state.
        cls = type('ViT' + cls.__name__, (TransformerBackbone, cls), {})
    model = cls(input_size=input_size, output_size=n_classes,
                hidden_size=cfg.HIDDEN_SIZE, **pars)
    if vit:
        model.install_transformer(cfg, n_classes, pretrained=pretrained)
    # Root FTN constructs lateral modules with requires_grad=True even when
    # smoothing is disabled. They never participate in these six defaults;
    # freezing them changes no forward value or update and makes the absence
    # of shared trainable preprocessing explicit to the isolation audit.
    model.isolation_frozen_dormant=[]
    for name,steps in [('lateral',getattr(cfg,'LATERAL_STEPS',0)),
                       ('mask_lateral',getattr(cfg,'MASK_LATERAL_STEPS',0)),
                       ('mask_lateral2',getattr(cfg,'MASK_LATERAL_STEPS_2',0))]:
        module=getattr(model,name,None)
        if module is not None and steps==0:
            if any(p.requires_grad for p in module.parameters()):
                model.isolation_frozen_dormant.append(name)
            module.requires_grad_(False)
    return model


class FusedBankLinear(nn.Module):
    """Native LoRA equation contracted as a weight matrix for token sequences.

    All H atoms remain differentiable to the mask during mask inference.
    The pretrained weight and its original bias are frozen. Atom initialization
    and alpha/k scaling match the native LoRALinear exactly.
    """
    def __init__(self, base, controller, cfg):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.controller = controller
        self.alpha = cfg.LORA_ALPHA
        self.A = nn.Parameter(torch.empty(controller.H, base.in_features, cfg.LORA_RANK))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.B = nn.Parameter(torch.empty(controller.H, cfg.LORA_RANK, base.out_features).normal_(0, cfg.LORA_B0_SCALE))
        self.bias_bank = (nn.Parameter(torch.empty(controller.H, base.out_features).normal_(0, cfg.LORA_B0_SCALE)) if cfg.LORA_BIAS else None)
        self.dropout = nn.Dropout(cfg.LORA_DROPOUT)

    def forward(self, x):
        a, b, bb = self.A, self.B, self.bias_bank
        if self.controller.detach_atoms:
            a, b = a.detach(), b.detach()
            bb = None if bb is None else bb.detach()
        mask = self.controller.mask
        # h i r, h r o -> i o. No active-only gather: every mask logit must
        # receive the same dense gradient as in the source implementation.
        weighted_b = b * mask[:, None, None]
        delta = a.permute(1, 0, 2).flatten(1) @ weighted_b.flatten(0, 1)
        scale = self.alpha / max(1, self.controller.k)
        bias = None if bb is None else scale * (mask @ bb)
        return self.base(x) + F.linear(self.dropout(x), scale * delta.T, bias)


class TransformerBackbone:
    def install_transformer(self, cfg, n_classes, pretrained=True):
        import timm
        self.layers = nn.ModuleList()  # discard the native random MLP backbone
        self.transformer = timm.create_model(VIT_NAME, pretrained=False, num_classes=n_classes)
        if pretrained:
            # Exactly the Google checkpoint used by the native L2P/DualPrompt
            # implementations, rather than a different timm fine-tuned variant.
            import fcntl
            import hashlib
            path = bulk('cache','vit','ViT-B_16.npz')
            with path.with_suffix('.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not path.exists():
                    temporary=path.with_suffix('.partial')
                    torch.hub.download_url_to_file(VIT_CHECKPOINT_URL,str(temporary))
                    temporary.replace(path)
                self.transformer.load_pretrained(str(path))
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
                write_json(bulk('manifests','vit_checkpoint.json'),{'url':VIT_CHECKPOINT_URL,'sha256':h.hexdigest()})
        self.transformer.requires_grad_(False)
        self.insertion_points = []
        # Same bank on attention qkv/projection, MLP fc1/fc2, and classifier.
        modules = list(self.transformer.named_modules())
        for name, layer in modules:
            if isinstance(layer, nn.Linear) and (name.startswith('blocks.') or name == 'head'):
                parent_name, _, child = name.rpartition('.')
                parent = self.transformer.get_submodule(parent_name) if parent_name else self.transformer
                setattr(parent, child, FusedBankLinear(layer, self.controller, cfg))
                self.insertion_points.append(name)

    def _backbone(self, x):
        return self.transformer(x)
