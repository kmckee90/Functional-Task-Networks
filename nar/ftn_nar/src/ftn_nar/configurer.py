"""Support-conditioned configurer: annealed straight-through top-k over frozen weights
(design section 9), plus SOFT_MIXTURE and DENSE task-vector variants and the optional
local-swap diagnostic."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import ExperimentCfg
from .losses import support_loss
from .masks import hard_topk, mask_from_indices, soft_gate, ste_topk_gate
from .model import Program
from .seeds import torch_seed, STREAM_CONFIGURER


@dataclass
class ConfigResult:
    program: Program  # detached, hard (or soft for SOFT_MIXTURE / z for DENSE)
    selection_loss: float  # hard-mask support loss (restart selection criterion)
    init_loss: float
    logits_ext: np.ndarray | None = None
    logits_red: np.ndarray | None = None
    trajectory: list = field(default_factory=list)

    def masks_np(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.program.gate_ext.detach().cpu().numpy(),
                self.program.gate_red.detach().cpu().numpy())


@contextlib.contextmanager
def params_frozen(model: torch.nn.Module):
    states = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in states:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, rg in states:
            p.requires_grad_(rg)


def _tau(s: int, steps: int, t0: float, t1: float) -> float:
    if steps <= 1:
        return t1
    return t0 * (t1 / t0) ** (s / (steps - 1))


def _init_logit(size: int, std: float, key: tuple, device) -> torch.Tensor:
    g = torch.Generator().manual_seed(torch_seed(STREAM_CONFIGURER, *key))
    return (torch.randn(size, generator=g) * std).to(device).requires_grad_(True)


def configure(model, support_batch, cfg: ExperimentCfg, restart_key: tuple,
              edge_mode: str = "ste", mode: str = "ftn",
              record_trajectory: bool = False) -> ConfigResult:
    """One restart of inner-loop configuration on a frozen model.

    mode='ftn': STE top-k masks. mode='soft': dense softmax gates + entropy penalty.
    mode='dense': 128-d continuous task vector, FiLM-injected.
    Support loss uses final outputs only.
    """
    ccfg = cfg.configurer
    mcfg = model.cfg
    dev = support_batch.node_features.device
    k_ext, k_red = mcfg.active_extender_atoms, mcfg.active_reducer_atoms

    if mode in ("ftn", "soft"):
        a_ext = _init_logit(mcfg.extender_atoms, ccfg.logit_init_std, (*restart_key, "ext"), dev)
        a_red = _init_logit(mcfg.reducer_atoms, ccfg.logit_init_std, (*restart_key, "red"), dev)
        params = [a_ext, a_red]
    else:
        z = _init_logit(mcfg.latent_dim, ccfg.logit_init_std, (*restart_key, "z"), dev)
        params = [z]

    opt = torch.optim.SGD(params, lr=ccfg.learning_rate, momentum=0.0, weight_decay=0.0)
    traj = []
    init_loss = float("nan")

    with params_frozen(model):
        for s in range(ccfg.inner_steps):
            tau = _tau(s, ccfg.inner_steps, ccfg.tau_start, ccfg.tau_end)
            extra = None
            if mode == "ftn":
                prog = Program(gate_ext=ste_topk_gate(a_ext, k_ext, tau),
                               gate_red=ste_topk_gate(a_red, k_red, tau))
            elif mode == "soft":
                ge, gr = soft_gate(a_ext, k_ext), soft_gate(a_red, k_red)
                prog = Program(gate_ext=ge, gate_red=gr)
                ent = -(torch.softmax(a_ext, -1) * torch.log_softmax(a_ext, -1)).sum() \
                      - (torch.softmax(a_red, -1) * torch.log_softmax(a_red, -1)).sum()
                target = np.log(k_ext) + np.log(k_red)
                extra = ccfg.soft_entropy_weight * (ent - target) ** 2
            else:
                prog = Program(z=z)

            out = model(support_batch, prog, edge_mode=edge_mode, return_trace=False)
            logits = params if mode != "dense" else [z]
            loss = support_loss(out, support_batch, cfg.loss, logits=logits, extra=extra)
            if s == 0:
                init_loss = float(loss.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, ccfg.gradient_clip)
            opt.step()
            if record_trajectory:
                traj.append({"step": s, "tau": tau, "loss": float(loss.detach()),
                             "grad_norm": float(gn)})

        # Project and score strictly hard (design 9.8): selection never sees queries.
        with torch.no_grad():
            if mode == "ftn":
                m_ext = hard_topk(a_ext, k_ext)
                m_red = hard_topk(a_red, k_red)
                assert int(m_ext.sum()) == k_ext and int(m_red.sum()) == k_red
                prog = Program(gate_ext=m_ext, gate_red=m_red)
            elif mode == "soft":
                prog = Program(gate_ext=soft_gate(a_ext, k_ext).detach(),
                               gate_red=soft_gate(a_red, k_red).detach())
            else:
                prog = Program(z=z.detach())
            out = model(support_batch, prog, edge_mode="hard", return_trace=False)
            sel_loss = float(support_loss(out, support_batch, cfg.loss))

    return ConfigResult(
        program=prog, selection_loss=sel_loss, init_loss=init_loss,
        logits_ext=a_ext.detach().cpu().numpy() if mode in ("ftn", "soft") else None,
        logits_red=a_red.detach().cpu().numpy() if mode in ("ftn", "soft") else None,
        trajectory=traj)


def configure_with_restarts(model, support_batch, cfg: ExperimentCfg, base_key: tuple,
                            n_restarts: int, edge_mode: str = "ste", mode: str = "ftn",
                            record_trajectory: bool = False) -> tuple[ConfigResult, list[ConfigResult]]:
    results = [configure(model, support_batch, cfg, (*base_key, r), edge_mode=edge_mode,
                         mode=mode, record_trajectory=record_trajectory)
               for r in range(n_restarts)]
    best = min(results, key=lambda r: r.selection_loss)
    return best, results


def local_swap(model, support_batch, cfg: ExperimentCfg, res: ConfigResult,
               max_swaps: int = 4) -> ConfigResult:
    """Optional diagnostic (design 9.9): greedy one-for-one swaps, alternating banks.
    Reported separately as FTN + LOCAL_SWAP; never part of the primary method."""
    m_ext, m_red = [torch.as_tensor(m).clone() for m in res.masks_np()]
    dev = support_batch.node_features.device
    m_ext, m_red = m_ext.to(dev), m_red.to(dev)
    logits = {"ext": res.logits_ext, "red": res.logits_red}
    best_loss = res.selection_loss

    def loss_of(me, mr):
        with torch.no_grad(), params_frozen(model):
            out = model(support_batch, Program(gate_ext=me, gate_red=mr),
                        edge_mode="hard", return_trace=False)
            return float(support_loss(out, support_batch, cfg.loss))

    accepted = 0
    bank = "ext"
    while accepted < max_swaps:
        m = m_ext if bank == "ext" else m_red
        lg = logits[bank]
        act = np.flatnonzero(m.cpu().numpy() > 0.5)
        ina = np.flatnonzero(m.cpu().numpy() < 0.5)
        act_low = act[np.argsort(lg[act])][:4]
        ina_high = ina[np.argsort(-lg[ina])][:8]
        best_swap, best_swap_loss = None, best_loss
        for a in act_low:
            for b in ina_high:
                m2 = m.clone()
                m2[a], m2[b] = 0.0, 1.0
                l = loss_of(m2 if bank == "ext" else m_ext, m2 if bank == "red" else m_red)
                if l < best_swap_loss:
                    best_swap, best_swap_loss = (int(a), int(b)), l
        if best_swap is None:
            if bank == "red":
                break
            bank = "red"
            continue
        a, b = best_swap
        m[a], m[b] = 0.0, 1.0
        best_loss = best_swap_loss
        accepted += 1
        bank = "red" if bank == "ext" else "ext"

    return ConfigResult(program=Program(gate_ext=m_ext, gate_red=m_red),
                        selection_loss=best_loss, init_loss=res.init_loss,
                        logits_ext=res.logits_ext, logits_red=res.logits_red)
