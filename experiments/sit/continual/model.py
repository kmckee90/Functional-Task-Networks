"""Pretrained SiT + the original persistent Bayesian mask inference mixin.

The mixin in bayes_reconfig.py is copied verbatim from bayesian_adv. Only the
backbone, packed transport inputs, and regression loss hooks are new here.
"""
import math
import hashlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models import SiT_models
from .bayes_reconfig import BayesMaskReconfig


@dataclass
class Controller:
    H: int
    k: int
    mask: object = None
    detach_atoms: bool = False


class BankedLinear(nn.Module):
    def __init__(self, base, controller, rank=1, alpha=4., b0_scale=.01):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.controller = controller
        self.alpha = alpha
        H, din, dout = controller.H, base.in_features, base.out_features
        self.A = nn.Parameter(torch.empty(H, din, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.empty(H, rank, dout).normal_(0, b0_scale))
        self.bias_bank = nn.Parameter(torch.empty(H, dout).normal_(0, b0_scale))

    def forward(self, x):
        A, B, bias = self.A, self.B, self.bias_bank
        if self.controller.detach_atoms:
            A, B, bias = A.detach(), B.detach(), bias.detach()
        H, din, rank = A.shape
        # Algebraically identical to compute-all-then-gate einsums; supports
        # both conditioning vectors and (batch, tokens, channels) tensors.
        proj = F.linear(x, A.permute(0, 2, 1).reshape(H * rank, din))
        m = self.controller.mask.to(proj.dtype)
        proj = proj * m.repeat_interleave(rank)
        delta = F.linear(proj, B.reshape(H * rank, -1).T)
        bias_delta = (m[:, None] * bias).sum(0)
        # Preserve the original linear's autocast dtype, including at zero
        # mask; otherwise a float32 bias sum silently changes attention math.
        return self.base(x) + (self.alpha / max(1, self.controller.k)) * (delta + bias_delta.to(delta.dtype))


def pack(z, t, labels):
    return torch.cat([z.flatten(1), t[:, None], labels[:, None].float()], dim=1)


def transport_batch(z, labels, generator=None, dropout=0.):
    z = z.float()
    noise = torch.randn(z.shape, generator=generator, device=z.device)
    t = torch.rand(len(z), generator=generator, device=z.device)
    if dropout:
        labels = labels.clone()
        labels[torch.rand(len(z), generator=generator, device=z.device) < dropout] = 1000
    xt = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * z
    return pack(xt, t, labels), z - noise


class BayesSiT(BayesMaskReconfig, nn.Module):
    def __init__(self, checkpoint, H=64, k=16, rank=1, alpha=4., b0_scale=.01,
                 model_name='SiT-XL/2', amp=True, reactive_low=False):
        super().__init__()
        self.hidden_size, self.k = H, k
        self.reconfig_steps = 30
        self.reconfig_sample_size = None
        self.reconfig_noise_scale = 1.
        self.amp = amp
        self.backbone = SiT_models[model_name](input_size=32, num_classes=1000, learn_sigma=True)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.backbone.load_state_dict(state.get('ema', state))
        self.backbone.requires_grad_(False).eval()
        self.controller = Controller(H, k)
        self.adapted_names = []
        for name, module in list(self.backbone.named_modules()):
            if isinstance(module, nn.Linear) and name.startswith('blocks.') and any(
                name.endswith(s) for s in ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')
            ):
                parent_name, leaf = name.rsplit('.', 1)
                parent = self.backbone.get_submodule(parent_name)
                setattr(parent, leaf, BankedLinear(module, self.controller, rank, alpha, b0_scale))
                self.adapted_names.append(name)
        self.register_buffer('m', torch.zeros(H))
        bayes_pars=dict(bayes_reconfig_steps=2, bayes_reconfig_lr=.3,
                         bayes_kl_weight=1., bayes_temperature=3.,
                         bayes_prior_decay=1., bayes_stochastic_mask=True,
                         bayes_reconfig_take_best=False,
                         bayes_changepoint_reconfig_steps=30,
                         bayes_changepoint_reconfig_lr=.3,
                         bayes_changepoint_prior_decay=0.,
                         bayes_changepoint_kl_weight=0., sample_type='gumbel')
        if reactive_low:
            from .bayesian_adv_config_snapshot import BayesLoRAReactiveLow
            source=BayesLoRAReactiveLow()
            bayes_pars.update(source.BAYES_FTN_PARS)
            self.reconfig_steps=source.RECONFIG_STEPS
            self.reconfig_sample_size=source.RECONFIG_SAMPLE_SIZE
            self.reconfig_noise_scale=source.RECONFIG_NOISE_SCALE
        self._init_bayes(**bayes_pars)

    def _topk_mask(self, scores):
        return torch.zeros_like(scores).scatter(0, scores.topk(self.k).indices, 1.)

    def _run(self, x, mask, detach):
        self.controller.mask = mask
        self.controller.detach_atoms = detach
        try:
            z = x[:, :-2].reshape(-1, 4, 32, 32)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
                return self.backbone(z, x[:, -2], x[:, -1].long()).float()
        finally:
            self.controller.detach_atoms = False

    def forward(self, x, y=None, m=None):
        if m is None:
            if y is not None:
                self._reconfig(x, y)
            m = self.m
        return self._run(x, m, detach=False)

    def loss(self, pred, target):
        return F.mse_loss(pred.float(), target.float())

    def adapter_state(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: v.detach().cpu().clone() for n, v in self.state_dict().items()
                if n in names or n in ('m', 'mask_logits')}

    def load_adapter(self, state):
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys


class InputPrior:
    """Optional fixed PCA keys and archived logits; no task ID at retrieval.

    Natural-image adaptation: PCA receives 4x4 pooled clean VAE latents, not
    pure generation noise. This feature choice is explicit and optional.
    """
    def __init__(self, rank=8, threshold=.5):
        self.rank, self.threshold, self.entries = rank, threshold, []

    @staticmethod
    def features(z):
        return F.adaptive_avg_pool2d(z.float(), (4, 4)).flatten(1)

    def archive(self, z, logits):
        x = self.features(z)
        _, vectors = torch.linalg.eigh(x.T @ x)
        basis = vectors[:, -self.rank:]
        self.entries.append({'projector': (basis @ basis.T).cpu(),
                             'logits': logits.detach().cpu().clone()})

    def retrieve(self, z):
        if not self.entries:
            return None, None, None
        x = self.features(z)
        errors = [((x - x @ e['projector'].to(x)).square().sum() /
                   x.square().sum().clamp_min(1e-12)).item() for e in self.entries]
        best = min(errors)
        index = max(i for i, err in enumerate(errors) if err <= best + .001)
        logits = self.entries[index]['logits'].to(z.device) if errors[index] < self.threshold else None
        return logits, index, errors[index]

    def repair_from_replay(self, model, replay, *, live_projector=None,
                           fit_rows=32, validation_rows=32, steps=100,
                           restarts=3, seed=7401, min_improvement=0.):
        """Append validation-approved mask repairs, following InputConditioned.

        Route replay batches by input reconstruction, with no domain labels.
        Fit and validate on distinct replay images, never evaluation support.
        The source's 800+800 rows are reduced for our 128-image/block buffer.
        Duplicate latent images are removed before splitting (sampling during
        training used replacement). No weights, projectors, or live posterior
        are updated. The latest entry wins ties in subsequent retrieval.
        """
        if min(fit_rows, validation_rows, steps, restarts) < 1:
            raise ValueError('Repair sizes and search budgets must be positive')
        domains = []
        for index in reversed(range(len(self.entries))):
            entry = self.entries[index]
            projector = entry['projector'].cpu()
            if live_projector is not None and torch.linalg.norm(projector-live_projector.cpu()) < .05:
                continue
            if all(torch.linalg.norm(projector-e['projector'].cpu()) >= .05 for _,e in domains):
                domains.append((index,entry))
        grouped = [[] for _ in domains]
        if not domains:
            return []
        for batch_index, (z, labels, _) in enumerate(replay):
            features = self.features(z[:4])
            errors = [((features-features@e['projector'].to(features)).square().sum()/
                       features.square().sum().clamp_min(1e-12)).item() for _,e in domains]
            closest = min(range(len(errors)), key=errors.__getitem__)
            if errors[closest] < self.threshold:
                grouped[closest].append((batch_index,z,labels))
        saved = model.get_reconfig_state()
        controller_mask = model.controller.mask
        records = []
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                for (index,entry), batches in zip(domains,grouped):
                    unique = {}
                    for batch_index,z,labels in batches:
                        for row in range(len(z)):
                            digest = hashlib.sha256(z[row].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                            unique.setdefault(digest,(z[row],labels[row],batch_index))
                    record = {'entry':index,'unique_rows':len(unique),
                              'routed_batches':[b[0] for b in batches], 'accepted':False}
                    records.append(record)
                    if len(unique) < fit_rows+validation_rows:
                        record['skipped']='insufficient_unique_replay'
                        continue
                    keys = list(unique)
                    order = torch.randperm(len(keys)).tolist()[:fit_rows+validation_rows]
                    keys = [keys[i] for i in order]
                    z = torch.stack([unique[h][0] for h in keys])
                    labels = torch.stack([unique[h][1] for h in keys])
                    x,y = transport_batch(z,labels,torch.Generator(device=z.device).manual_seed(seed+index+1))
                    xf,yf,xv,yv=x[:fit_rows],y[:fit_rows],x[fit_rows:],y[fit_rows:]
                    old_mask = model._topk_mask(entry['logits'].to(model.m))
                    with torch.no_grad():
                        before = model.loss(model(xv,m=old_mask),yv).item()
                    model.cold_reconfigure(xf,yf,steps=steps,lr=.3,sample_size=None,
                                           temperature=1.,restarts=restarts,take_best=True)
                    candidate = model.m.detach().clone()
                    with torch.no_grad():
                        candidate_loss=model.loss(model(xv,m=candidate),yv).item()
                    accepted=candidate_loss+min_improvement < before
                    record.update(before=before,candidate_loss=candidate_loss,
                                  after=candidate_loss if accepted else before,accepted=accepted,
                                  fit_hashes=keys[:fit_rows],validation_hashes=keys[fit_rows:],
                                  old_mask=old_mask.cpu().tolist(),candidate_mask=candidate.cpu().tolist())
                    if accepted:
                        logits=torch.where(candidate>.5,4.,-4.)
                        self.entries.append({'projector':entry['projector'].clone(),
                                             'logits':logits.cpu(),'repaired_from':index})
                        record['new_entry']=len(self.entries)-1
        finally:
            model.set_reconfig_state(saved)
            model.controller.mask=controller_mask
        return records

    def infer(self, model, z, x, target, refine=True):
        logits, index, error = self.retrieve(z)
        if logits is None:
            model.cold_reconfigure(x, target, steps=30, take_best=False)
        elif not refine:
            model.m.copy_(model._topk_mask(logits))
        else:
            old_logits = model.mask_logits.clone()
            model.mask_logits.copy_(logits)
            try:
                model._posterior_update(x, target, steps=30, lr=.3, sample_size=None,
                    kl_weight=1., temperature=1., prior_decay=1.,
                    stochastic_mask=False, changepoint=False)
            finally:
                model.mask_logits.copy_(old_logits)
        return {'entry': index, 'error': error, 'hit': logits is not None}
