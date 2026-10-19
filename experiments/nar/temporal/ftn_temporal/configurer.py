"""Trace configurers: how a program (mask sequence + length) is selected for
a task, given only a labeled support batch and frozen weights.

Route A — LibraryBFS ("plan over learned operators", hard states):
    Consolidated atom masks (one per trained 1-step operator + identity) are
    the action set. Breadth-first search over atom sequences, executing on a
    small support batch with hard re-encoding and MEMOIZING on the produced
    support state — traces that realize the same group element collapse to
    one node, so the search space is the (tiny) group, not the 3^T tree.
    Halting is free: BFS returns the SHORTEST trace that matches the support.

Route B — UnrolledAnnealSTE ("gradient trace search", soft states):
    The abelian AnnealSTE configurer unrolled through time: logits (T, H),
    per-step top-k with straight-through relaxation, annealed noise, loss =
    CE of the soft rollout's final state on the full support batch. Length is
    chosen externally by sweeping T (warm-starting T from the best T-1 rows)
    and keeping the best support loss — halting by verification.
"""

import torch
import torch.nn as nn

from ftn_core.configurer import kwta
from ftn_core.model import sequence_ce


class StepAnnealSTE:
    """Single-stage mask configurer (one step = one operator)."""

    def __init__(self, k, lr, L, p):
        self.k, self.lr, self.L, self.p = k, lr, L, p

    def __call__(self, model, support, steps):
        x, y = support
        was_training = model.training
        model.eval()
        with torch.no_grad():
            z = model.stage.bank(x)
        s = nn.Parameter(torch.randn(model.H, device=x.device) * 1e-4)
        opt = torch.optim.Adam([s], lr=self.lr)
        best, best_loss = None, float("inf")
        for i in range(steps):
            tau = s.detach().std() * torch.exp(
                torch.tensor(-i / max(steps / 3, 1.0), device=x.device))
            hard = kwta(s + torch.randn_like(s) * tau, self.k)
            sg = torch.sigmoid(s)
            m = hard + sg - sg.detach()
            loss = sequence_ce(model.stage.forward_from_z(z, m), y, self.L, self.p)
            if loss.item() < best_loss:
                best, best_loss = hard.detach().clone(), loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if was_training:
            model.train()
        return best, best_loss


class LibraryBFS:
    """Trace search over consolidated atom masks with state memoization."""

    def __init__(self, library: dict, T_max: int, L: int, p: int):
        """library: {atom_name: (H,) binary mask}"""
        self.names = list(library.keys())
        self.masks = [library[n] for n in self.names]
        self.T_max, self.L, self.p = T_max, L, p

    @torch.no_grad()
    def __call__(self, model, support):
        """support: (x_onehot, y_int) small search batch.
        Returns (trace_names, trace_masks, support_exact, support_pos)."""
        was_training = model.training
        model.eval()
        x, y = support
        n = x.shape[0]

        def state_key(seq_int):
            return seq_int.reshape(-1).cpu().numpy().tobytes()

        def score(seq_int):
            pos = (seq_int == y).float().mean().item()
            exact = (seq_int == y).all(-1).float().mean().item()
            return pos, exact

        x_int = x.reshape(n, self.L, self.p).argmax(-1)
        start_key = state_key(x_int)
        visited = {start_key}
        frontier = [([], x)]                       # (trace names, state one-hot)
        best = ([], score(x_int)[0], score(x_int)[1])   # identity trace baseline

        for depth in range(1, self.T_max + 1):
            if best[2] > 0.999:
                break
            new_frontier = []
            H = self.masks[0].shape[0]
            n_atoms = len(self.masks)
            atom_masks = torch.stack(self.masks)                      # (A, H)
            for trace, s in frontier:
                # one batched forward for all atoms from this node
                S = s.unsqueeze(0).expand(n_atoms, n, -1).reshape(n_atoms * n, -1)
                M = atom_masks.unsqueeze(1).expand(-1, n, -1).reshape(n_atoms * n, H)
                logits = model.step(S, M).reshape(n_atoms, n, self.L, self.p)
                nxt_int = logits.argmax(-1)
                for j, name in enumerate(self.names):
                    key = state_key(nxt_int[j])
                    if key in visited:
                        continue
                    visited.add(key)
                    pos, exact = score(nxt_int[j])
                    cand = trace + [name]
                    if exact > best[2] or (exact == best[2] and pos > best[1]):
                        best = (cand, pos, exact)
                    oh = nn.functional.one_hot(nxt_int[j], self.p).float()
                    new_frontier.append((cand, oh.reshape(n, -1), pos))
                    if exact > 0.999:
                        if was_training:
                            model.train()
                        lib = dict(zip(self.names, self.masks))
                        return cand, [lib[t] for t in cand], exact, pos
            # beam safeguard: with an imperfect model, states can drift off the
            # group orbit and defeat memoization — keep the best 128 nodes
            new_frontier.sort(key=lambda e: -e[2])
            frontier = [(c, s) for c, s, _ in new_frontier[:128]]
            if not frontier:
                break
        if was_training:
            model.train()
        lib = dict(zip(self.names, self.masks))
        return best[0], [lib[t] for t in best[0]], best[2], best[1]


class GreedyGrow:
    """Route C — library-free hill-climbing in program space (hard states).

    Extend the trace one step at a time: execute the frozen prefix hard on
    the support to get the current state, then configure ONLY the new step's
    free mask (a single-stage AnnealSTE toward the FINAL target y). Keep the
    extension, halt when support exact-match saturates or stops improving.
    Cheap (each extension is one single-step configuration), inherently
    order-aware, and free to discover non-atomic 'macro-op' masks that jump
    several group elements in one step. Its known failure mode — greedy
    branches that never reduce distance to y — is part of what we measure.
    """

    def __init__(self, k, lr, L, p, patience: int = 2):
        self.k, self.lr, self.L, self.p = k, lr, L, p
        self.patience = patience

    def __call__(self, model, support, T_max, steps):
        from .model import RecurrentFTN  # noqa: F401  (type only)
        x, y = support
        was_training = model.training
        model.eval()
        step_cfg = StepAnnealSTE(self.k, self.lr, self.L, self.p)
        masks, state = [], x
        best = ([], -1.0, float("inf"))
        stall = 0
        for T in range(1, T_max + 1):
            m, loss = step_cfg(model, (state, y), steps)
            masks = masks + [m]
            with torch.no_grad():
                logits = model.step(state, m)
                pred = logits.reshape(-1, self.L, self.p).argmax(-1)
                exact = (pred == y).all(-1).float().mean().item()
                state = nn.functional.one_hot(pred, self.p).float().reshape(
                    x.shape[0], -1)
            if exact > best[1]:
                best = (list(masks), exact, loss)
                stall = 0
            else:
                stall += 1
            if exact > 0.999 or stall > self.patience:
                break
        if was_training:
            model.train()
        return best[0], len(best[0]), best[2]


class UnrolledAnnealSTE:
    """Gradient trace search: per-step masks optimized jointly through a soft
    rollout; length chosen by sweeping T with warm starts."""

    def __init__(self, k, lr, L, p, tau_exec=1.0):
        self.k, self.lr, self.L, self.p = k, lr, L, p
        self.tau_exec = tau_exec

    def _configure_T(self, model, support, T, steps, init_rows=None):
        x, y = support
        s = torch.randn(T, model.H, device=x.device) * 1e-4
        if init_rows is not None:
            rows = min(init_rows.shape[0], T)
            s[:rows] = init_rows[:rows]
        s = nn.Parameter(s)
        opt = torch.optim.Adam([s], lr=self.lr)
        best, best_loss = None, float("inf")
        for i in range(steps):
            tau = s.detach().std() * torch.exp(
                torch.tensor(-i / max(steps / 3, 1.0), device=x.device))
            noisy = s + torch.randn_like(s) * tau
            masks = []
            sg = torch.sigmoid(s)
            for t in range(T):
                hard = kwta(noisy[t], self.k)
                masks.append(hard + sg[t] - sg[t].detach())
            loss = sequence_ce(model.execute_soft(x, masks, self.tau_exec),
                               y, self.L, self.p)
            if loss.item() < best_loss:
                best = [kwta(noisy[t].detach(), self.k).clone() for t in range(T)]
                best_loss = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return best, best_loss, s.detach()

    def __call__(self, model, support, T_max, steps, restarts: int = 3):
        """Best-of-restarts over a T sweep with warm starts; inner budget
        scales with T. T is selected by HARD-execution exact-match on the
        support (soft CE is not comparable across lengths), ties broken by
        loss then brevity. Returns (masks, T, loss)."""
        was_training = model.training
        model.eval()
        x, y = support
        best = (None, 0, float("inf"), -1.0)   # masks, T, loss, hard exact
        for _ in range(restarts):
            warm = None
            for T in range(1, T_max + 1):
                steps_T = int(steps * (0.5 + 0.5 * T))
                masks, loss, rows = self._configure_T(model, support, T,
                                                      steps_T, init_rows=warm)
                warm = rows
                with torch.no_grad():
                    pred = model.execute_hard(x, masks).reshape(
                        -1, self.L, self.p).argmax(-1)
                    exact = (pred == y).all(-1).float().mean().item()
                if exact > best[3] or (exact == best[3] and loss < best[2]):
                    best = (masks, T, loss, exact)
                if exact > 0.999:
                    break
            if best[3] > 0.999:
                break
        if was_training:
            model.train()
        return best[0], best[1], best[2]
