"""Fast posterior and detector checks; run on the actl pod."""
from types import SimpleNamespace
from dataclasses import replace

import torch
from torch import nn

from bayes_configurer import BayesianPerplexityConfigurer
from config import CFG
CFG = replace(CFG, grid_dim=4, lateral_steps=0)
from lora_bank import MaskController


class TinyModel(nn.Module):
    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.weight = nn.Parameter(torch.arange(ctrl.H, dtype=torch.float32) / ctrl.H)

    def forward(self, input_ids, attention_mask=None, labels=None):
        value = (self.ctrl.mask * self.weight).sum()
        target = input_ids.float().mean() / 10
        return SimpleNamespace(loss=(value - target).square())


def test_posterior():
    ctrl = MaskController(16, 4)
    model = TinyModel(ctrl)
    cfgr = BayesianPerplexityConfigurer(CFG, ctrl)
    ids = torch.tensor([[1, 2, 3]])

    mask, _, _ = cfgr(model, ids, changepoint_steps=3)
    assert cfgr.last_stats["changepoint"]
    assert mask.sum().item() == 4
    assert cfgr.mask_logits.abs().sum().item() > 0
    assert model.weight.requires_grad
    assert model.weight.grad is None

    mask, _, _ = cfgr(model, ids, steps=2)
    assert not cfgr.last_stats["changepoint"]
    assert mask.sum().item() == 4

    # A sufficiently surprising prompt under the current mask triggers the
    # long reset/refit once the detector's warmup window has elapsed.
    cfgr.detector.mean = -1.0
    cfgr.detector.var = 0.0
    cfgr.detector.since = 100
    mask, _, _ = cfgr(model, ids, changepoint_steps=3)
    assert cfgr.last_stats["changepoint"]
    assert cfgr.last_stats["steps"] == 3

    old_mask = ctrl.mask.clone()
    old_logits = cfgr.mask_logits.clone()
    old_detector = cfgr.detector.to_dict()
    running = cfgr.iterate(model, ids, changepoint_steps=3)
    next(running)
    running.close()
    assert torch.equal(ctrl.mask, old_mask)
    assert torch.equal(cfgr.mask_logits, old_logits)
    assert cfgr.detector.to_dict() == old_detector
    assert model.weight.requires_grad

    mask, _, _ = cfgr(model, ids, k=5, changepoint_steps=3)
    assert cfgr.last_stats["changepoint"]
    assert mask.sum().item() == 5
    saved = cfgr.state_dict()
    restored = BayesianPerplexityConfigurer(CFG, ctrl)
    restored.load_state_dict(saved)
    assert torch.equal(restored.mask_logits, cfgr.mask_logits)
    assert torch.equal(restored.controller.mask, ctrl.mask)


def test_routed_refinement():
    ctrl = MaskController(16, 4)
    model = TinyModel(ctrl)
    cfgr = BayesianPerplexityConfigurer(CFG, ctrl)
    ids = torch.tensor([[1, 2, 3]])
    prior = torch.full((16,), -4.0)
    prior[[3, 5, 7, 9]] = 4.0
    expected = torch.zeros(16)
    expected[[3, 5, 7, 9]] = 1.0

    # The first forward mask comes from retrieved logits even though ordinary
    # changepoint search has zero prior retention. Closing rolls everything back.
    running = cfgr.iterate(model, ids, routed_prior=prior, routed_steps=2,
                           routed_kl_weight=1.0)
    assert torch.equal(next(running), expected)
    running.close()
    assert ctrl.mask.sum().item() == 0
    assert cfgr.mask_logits.abs().sum().item() == 0
    assert cfgr.detector.mean is None

    mask, _, _ = cfgr(model, ids, routed_prior=prior, routed_steps=2,
                      routed_kl_weight=1.0)
    assert mask.sum().item() == 4
    assert cfgr.last_stats["routed_refine"]
    assert cfgr.last_stats["steps"] == 2
    assert cfgr.detector.mean is not None
    assert model.weight.requires_grad


if __name__ == "__main__":
    test_posterior()
    test_routed_refinement()
    print("Bayes configurer checks passed")
