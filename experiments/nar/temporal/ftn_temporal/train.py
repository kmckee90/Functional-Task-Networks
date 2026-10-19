"""Two trainers for the recurrent FTN, both block-sequential and both denied
task identity (support sets only). Curriculum: tasks ordered by shortest-word
length within every cycle (identity and generators first), shuffled within a
level — 'primitives first' made deliberate this time.

V1 — library/EM (hard states):
    Atom blocks (l <= 1) train exactly like single FTN tasks: per-batch
    StepAnnealSTE reconfiguration + a weight step. Composite blocks alternate
    E-steps (LibraryBFS trace search on a support batch, refreshed every
    trace_refresh batches) with M-steps (BPTT through the soft rollout with
    the trace's masks FIXED — credit assignment through a known program).

V2 — BPTT-configured (soft states, fully general):
    Every batch: UnrolledAnnealSTE sweeps trace length and configures all
    per-step masks by gradient, then a weight step backpropagates through the
    chosen soft rollout. No atom library, no privileged 1-step treatment.

Both use mask-overlap-filtered replay over stored (x, y, trace) slots and
optimizer resets at block boundaries.
"""

import random

import torch

from ftn_core.model import sequence_ce
from .configurer import LibraryBFS, StepAnnealSTE, UnrolledAnnealSTE
from .model import RecurrentFTN, trace_accuracy


def curriculum_order(tasks, lengths, seed, cycles):
    by_level = {}
    for t in tasks:
        by_level.setdefault(lengths[t], []).append(t)
    visits = []
    for cyc in range(cycles):
        rng = random.Random(seed + 31 * cyc)
        for lvl in sorted(by_level):
            block = list(by_level[lvl])
            rng.shuffle(block)
            visits.extend(block)
    return visits


class TraceReplay:
    def __init__(self, capacity, rho_lo, rho_hi, n_samples):
        self.capacity, self.rho_lo, self.rho_hi = capacity, rho_lo, rho_hi
        self.n_samples = n_samples
        self.slots = []

    @staticmethod
    def _union(masks):
        return torch.clamp(torch.stack(masks).sum(0), 0, 1) if masks else None

    def insert(self, x, y, masks):
        self.slots.append((x, y, [m.detach() for m in masks], self._union(masks)))
        if len(self.slots) > self.capacity:
            self.slots.pop(0)

    def sample(self, cur_masks, rng):
        u_cur = self._union(cur_masks)
        if u_cur is None:
            return []
        eligible = []
        for x, y, masks, u in self.slots:
            if u is None:
                continue
            rho = (u * u_cur).sum().item() / max(u.sum().item(), u_cur.sum().item())
            if self.rho_lo < rho <= self.rho_hi:
                eligible.append((x, y, masks))
        if not eligible:
            return []
        return rng.sample(eligible, min(self.n_samples, len(eligible)))


def consolidate_library(model, tasks_obj, cfg, restarts=None):
    """Consensus atom masks for the identity + each generator, against the
    CURRENT weights (the recurrent analogue of post-stream consensus masks)."""
    from ftn_core.configurer import kwta
    restarts = restarts or cfg.lib_restarts
    step_cfg = StepAnnealSTE(cfg.k, cfg.configurer_lr, cfg.L, cfg.p)
    atoms = {"id": (1, 0)}
    atoms.update({name: g for name, g in cfg.generators})
    library = {}
    for name, g in atoms.items():
        acc = torch.zeros(model.H, device=cfg.device)
        for _ in range(restarts):
            xs, ys = tasks_obj.batch(g, cfg.support_size, "train")
            m, _ = step_cfg(model, (xs, ys), cfg.s_eval)
            acc += m
        library[name] = kwta(acc / restarts, cfg.k)
    return library


def train_v1(model: RecurrentFTN, tasks_obj, cfg, log_fn=print):
    """Library/EM trainer."""
    step_cfg = StepAnnealSTE(cfg.k, cfg.configurer_lr, cfg.L, cfg.p)
    replay = TraceReplay(cfg.replay_capacity, cfg.rho_lo, cfg.rho_hi,
                         cfg.replay_samples)
    rng = random.Random(cfg.seed + 2)
    visits = curriculum_order(tasks_obj.train_tasks, tasks_obj.length,
                              cfg.seed + 3, cfg.cycles)
    history = []
    model.train()

    for block, g in enumerate(visits):
        lvl = tasks_obj.length[g]
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        trace_masks, trace_names = None, None

        for step in range(cfg.steps_per_task):
            if lvl <= 1:
                # atom block: single-step FTN training
                xs, ys = tasks_obj.batch(g, cfg.support_size, "train")
                m, _ = step_cfg(model, (xs, ys), cfg.s_train)
                masks = [m]
            else:
                # composite block: refresh the trace periodically (E-step)
                if trace_masks is None or step % cfg.trace_refresh == 0:
                    library = consolidate_library(model, tasks_obj, cfg, restarts=1)
                    bfs = LibraryBFS(library, cfg.T_max, cfg.L, cfg.p)
                    xs, ys = tasks_obj.batch(g, cfg.search_support, "train")
                    trace_names, trace_masks, _, _ = bfs(model, (xs, ys))
                    if not trace_masks:      # identity trace found; skip M-step
                        trace_masks = [library["id"]]
                        trace_names = ["id"]
                masks = trace_masks

            x, y = tasks_obj.batch(g, cfg.batch_size, "train")
            loss = sequence_ce(model.execute_soft(x, masks), y, cfg.L, cfg.p)
            for xr, yr, mr in replay.sample(masks, rng):
                loss = loss + sequence_ce(model.execute_soft(xr, mr), yr,
                                          cfg.L, cfg.p)
            opt.zero_grad()
            loss.backward()
            opt.step()

            xb, yb = tasks_obj.batch(g, cfg.replay_batch, "train")
            replay.insert(xb, yb, masks)

            if (step + 1) % 100 == 0:
                pos, exact = trace_accuracy(model, masks,
                                            *tasks_obj.eval_tensors(g, "test"))
                history.append(dict(block=block, task=g, level=lvl,
                                    step=step + 1, loss=loss.item(),
                                    pos=pos, exact=exact,
                                    trace=trace_names))
                log_fn(f"[V1 block {block:03d} task {g} l={lvl}] step {step+1}: "
                       f"loss={loss.item():.4f} pos={pos:.3f} exact={exact:.3f} "
                       f"trace={trace_names}")
    return history


def train_v2(model: RecurrentFTN, tasks_obj, cfg, log_fn=print):
    """BPTT-configured trainer (fully general: no library, no atom privilege)."""
    tracer = UnrolledAnnealSTE(cfg.k, cfg.configurer_lr, cfg.L, cfg.p)
    replay = TraceReplay(cfg.replay_capacity, cfg.rho_lo, cfg.rho_hi,
                         cfg.replay_samples)
    rng = random.Random(cfg.seed + 2)
    visits = curriculum_order(tasks_obj.train_tasks, tasks_obj.length,
                              cfg.seed + 3, cfg.cycles)
    history = []
    model.train()

    for block, g in enumerate(visits):
        lvl = tasks_obj.length[g]
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        T_cap = cfg.v2_T_train   # fixed for all blocks: no length hint leaks

        for step in range(cfg.steps_per_task):
            xs, ys = tasks_obj.batch(g, cfg.support_size, "train")
            masks, T, _ = tracer(model, (xs, ys), T_cap, cfg.v2_s_train,
                                 restarts=1)

            x, y = tasks_obj.batch(g, cfg.batch_size, "train")
            loss = sequence_ce(model.execute_soft(x, masks), y, cfg.L, cfg.p)
            for xr, yr, mr in replay.sample(masks, rng):
                loss = loss + sequence_ce(model.execute_soft(xr, mr), yr,
                                          cfg.L, cfg.p)
            opt.zero_grad()
            loss.backward()
            opt.step()

            xb, yb = tasks_obj.batch(g, cfg.replay_batch, "train")
            replay.insert(xb, yb, masks)

            if (step + 1) % 100 == 0:
                pos, exact = trace_accuracy(model, masks,
                                            *tasks_obj.eval_tensors(g, "test"))
                history.append(dict(block=block, task=g, level=lvl,
                                    step=step + 1, loss=loss.item(),
                                    pos=pos, exact=exact, T=T))
                log_fn(f"[V2 block {block:03d} task {g} l={lvl}] step {step+1}: "
                       f"loss={loss.item():.4f} pos={pos:.3f} exact={exact:.3f} "
                       f"T={T}")
    return history
