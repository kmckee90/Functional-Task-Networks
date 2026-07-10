"""Self-supervised prompt-perplexity reconfiguration — the core FTN mechanism.

Given a batch of prompt token-ids, optimize a binary top-k mask over the atom
bank to MINIMIZE the language-model NLL of those tokens, with the backbone and
bank weights frozen (only the H mask logits receive gradient). AnnealSTE inner
loop (proven best at small inner-loop budgets in the FTN paper) + lateral
smoothing for spatially-coherent blobs. Records the mask at every step so the
app can animate the "search".

Copied verbatim from ftn_chat/perplexity_configurer.py (see ftn_agent/design.md).
"""
from __future__ import annotations

import math

import torch

from config import Config
from lora_bank import MaskController
from spatial import LateralSmoother, kwta


class PerplexityConfigurer:
    def __init__(self, cfg: Config, controller: MaskController):
        self.cfg = cfg
        self.controller = controller
        self.smoother = LateralSmoother(cfg.grid_dim, cfg.lateral_kernel, cfg.torus)

    def _nll(self, model, input_ids, attention_mask) -> torch.Tensor:
        """Mean next-token NLL over the prompt tokens, under the current mask."""
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    labels=input_ids)
        return out.loss

    def iterate(self, model, input_ids, attention_mask=None, steps=None,
                reconfig_lr=None, lateral_steps=None, lateral_polyak=None, k=None,
                init_mask=None, init_a=0.0, init_b=1e-3):
        """Generator form: run the reconfiguration search and YIELD the current
        mask at every step, so the caller can animate the search LIVE (one real
        forward pass per frame). The last yielded mask is the final selection;
        self._last_nll holds its NLL afterwards.

        `steps`, `reconfig_lr`, and `lateral_steps` override the config defaults
        per call (so the app can expose them as live sliders).

        `k` overrides the active-atom count for this search (and is written back
        to the controller, so generation under the resulting mask uses it too).

        Mask-logit init is `init_a * init_mask + init_b * noise` when an
        `init_mask` is given. This lets mid-rollout reconfiguration warm-start
        from the CURRENT mask (a=1, b=0), explore from pure noise (a=0, b=1), or
        start near zero (small a, small b). With no init_mask we fall back to the
        old tiny-random init (`init_b * noise`).
        """
        cfg, ctrl = self.cfg, self.controller
        H = ctrl.H
        k = ctrl.k if k is None else int(k)
        ctrl.k = k                              # generation reuses this active count
        n_steps = cfg.reconfig_steps if steps is None else steps
        lr = cfg.reconfig_lr if reconfig_lr is None else reconfig_lr
        n_lateral = cfg.lateral_steps if lateral_steps is None else lateral_steps
        polyak = cfg.lateral_polyak if lateral_polyak is None else lateral_polyak
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Freeze everything; only the H mask logits get gradient. Disable
        # checkpointing for the small inner-loop graph.
        saved = [(p, p.requires_grad) for p in model.parameters()]
        for p, _ in saved:
            p.requires_grad = False
        ckpt_on = getattr(model, "is_gradient_checkpointing", False)
        if ckpt_on:
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                ckpt_on = False

        dev = ctrl.mask.device
        noise0 = torch.randn(H, device=dev) * init_b
        if init_mask is not None and init_a != 0.0:
            base = init_a * init_mask.detach().float().to(dev) + noise0
        else:
            base = noise0
        logits = base.clone().requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=lr)
        best_mask, best_nll = None, float("inf")
        try:
            with torch.enable_grad():
                for step in range(n_steps):
                    sm = torch.sigmoid(logits)
                    # math.exp keeps the anneal factor a python float -> no stray
                    # CPU tensor that would clash with cuda when run off-thread.
                    tau = logits.detach().std() * math.exp(-step / max(1.0, n_steps / 4))
                    noise = torch.randn_like(logits) * tau
                    # Hard top-k value, straight-through identity gradient.
                    mask = kwta(logits + noise, k) + sm - sm.detach()
                    ctrl.set_mask(mask)

                    loss = self._nll(model, input_ids, attention_mask)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    with torch.no_grad():
                        nll = float(loss.detach())
                        hard = kwta(logits.detach(), k)
                        if nll < best_nll:
                            best_nll, best_mask = nll, hard.clone()
                    del loss
                    yield hard.detach()             # live: current search state
        finally:
            for p, rg in saved:
                p.requires_grad = rg
            if ckpt_on:
                try:
                    model.gradient_checkpointing_enable()
                except Exception:
                    pass

        # Lateral smoothing -> contiguous blobs, then final kwta.
        with torch.no_grad():
            m = best_mask.float()
            for _ in range(n_lateral):
                m_lat = self.smoother.step(m)
                m = polyak * m + (1 - polyak) * m_lat
                yield kwta(m, k)
            final = kwta(m, k)
        ctrl.set_mask(final)
        self._last_nll = best_nll
        yield final                                 # last frame == final mask

    def __call__(self, model, input_ids, attention_mask=None, record=False, steps=None):
        """Non-streaming form (train / routing): drive the search to completion.
        Returns (best_mask, best_nll, frames) for backward compatibility.
        """
        self._last_nll = float("inf")
        frames, last = [], None
        for m in self.iterate(model, input_ids, attention_mask, steps):
            last = m
            if record:
                frames.append(m.clone())
        return last.detach(), self._last_nll, frames
