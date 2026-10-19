"""Evaluation battery for the temporal experiment.

For every task (trained and held-out, grouped by shortest-word length l):
support batch -> infer a program (both routes) with weights frozen -> execute
hard on the 50% held-out input sequences -> score against the generator.
Held-out tasks with l > train_max_len are the scale-OOD axis: programs longer
than any ever trained.

Also: inferred length vs true shortest length (halting quality), step-level
faithfulness of V1 traces (does each executed step match its atom's ground
truth?), and the two controls (random bank; joint task-conditioned MLP).
"""

import torch
import torch.nn as nn

from ftn_core.model import sequence_ce
from .configurer import GreedyGrow, LibraryBFS, UnrolledAnnealSTE
from .model import RecurrentFTN, trace_accuracy
from .train import consolidate_library


def eval_tasks(model, tasks_obj, cfg, task_list,
               methods=("library", "gradient", "greedy"), log_fn=print,
               search_include_id=False):
    """search_include_id: BFS halts by depth, so the identity atom is never
    needed in its action set — and because the learned id is only ~97%
    faithful, including it lets slightly-perturbed duplicate states defeat
    memoization and pads traces with spurious ids (measured: this is the
    dominant failure at l>=6). Excluded by default; the id atom remains in
    the library for consolidation and faithfulness metrics."""
    rows = []
    library = consolidate_library(model, tasks_obj, cfg) if "library" in methods else None
    if library is not None:
        search_lib = library if search_include_id else \
            {k: v for k, v in library.items() if k != "id"}
    bfs = LibraryBFS(search_lib, cfg.T_max, cfg.L, cfg.p) if library else None
    tracer = UnrolledAnnealSTE(cfg.k, cfg.configurer_lr, cfg.L, cfg.p) \
        if "gradient" in methods else None
    greedy = GreedyGrow(cfg.k, cfg.configurer_lr, cfg.L, cfg.p) \
        if "greedy" in methods else None

    for g in task_list:
        lvl = tasks_obj.length[g]
        x, y = tasks_obj.eval_tensors(g, "test")
        row = dict(task=g, level=lvl, shortest_word=tasks_obj.word[g])

        if bfs is not None:
            xs, ys = tasks_obj.batch(g, cfg.search_support, "train")
            names, masks, sup_exact, _ = bfs(model, (xs, ys))
            pos, exact = trace_accuracy(model, masks, x, y)
            row["library"] = dict(pos=pos, exact=exact, trace=names,
                                  T=len(names), support_exact=sup_exact)

        if tracer is not None:
            xs, ys = tasks_obj.batch(g, cfg.support_size, "train")
            masks, T, sup_loss = tracer(model, (xs, ys), cfg.T_max, cfg.s_eval)
            pos, exact = trace_accuracy(model, masks, x, y)
            row["gradient"] = dict(pos=pos, exact=exact, T=T,
                                   support_loss=sup_loss)

        if greedy is not None:
            xs, ys = tasks_obj.batch(g, cfg.support_size, "train")
            masks, T, sup_loss = greedy(model, (xs, ys), cfg.T_max, cfg.s_eval)
            if masks:
                pos, exact = trace_accuracy(model, masks, x, y)
            else:
                pos, exact = trace_accuracy(model, [], x, y)
            row["greedy"] = dict(pos=pos, exact=exact, T=T,
                                 support_loss=sup_loss)

        rows.append(row)
        msg = f"[eval] task {g} l={lvl}"
        if "library" in row:
            msg += (f" | lib: pos={row['library']['pos']:.3f} "
                    f"T={row['library']['T']} trace={row['library']['trace']}")
        if "gradient" in row:
            msg += (f" | grad: pos={row['gradient']['pos']:.3f} "
                    f"T={row['gradient']['T']}")
        if "greedy" in row:
            msg += (f" | greedy: pos={row['greedy']['pos']:.3f} "
                    f"T={row['greedy']['T']}")
        log_fn(msg)
    return rows


def step_faithfulness(model, tasks_obj, cfg, rows):
    """For each V1-evaluated task: execute the inferred trace on test inputs
    and check every intermediate step against its atom's ground-truth
    semantics. Answers: is the execution trace really running the named
    program, or merely arriving at the right terminal state?"""
    atoms = {"id": (1, 0)}
    atoms.update({name: g for name, g in cfg.generators})
    was_training = model.training
    model.eval()
    library = consolidate_library(model, tasks_obj, cfg)
    out = []
    x_pool = tasks_obj.test_seqs[:2000]
    x = tasks_obj.encode(x_pool)
    for row in rows:
        if "library" not in row or not row["library"]["trace"]:
            continue
        names = row["library"]["trace"]
        masks = [library[n] for n in names]
        with torch.no_grad():
            states = model.execute_hard_states(x, masks)
        per_step = []
        cur_int = x_pool
        for t, name in enumerate(names):
            truth = tasks_obj.apply(cur_int, atoms[name])
            pred_int = states[t + 1].reshape(-1, cfg.L, cfg.p).argmax(-1)
            per_step.append((pred_int == truth).float().mean().item())
            cur_int = pred_int   # faithfulness of step t+1 given actual state
        out.append(dict(task=row["task"], level=row["level"], trace=names,
                        step_faithfulness=per_step))
    if was_training:
        model.train()
    return out


class JointConditionedMLP(nn.Module):
    """The 'well-solved?' baseline again: task (c, a) as one-hot input,
    joint i.i.d. training over the training tasks."""

    def __init__(self, p, L, hidden, layers):
        super().__init__()
        self.p, self.L = p, L
        dims = [L * p + (p - 1) + p] + [hidden] * layers + [L * p]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x, c, a):
        oh_c = nn.functional.one_hot(c - 1, self.p - 1).float()
        oh_a = nn.functional.one_hot(a, self.p).float()
        return self.net(torch.cat([x, oh_c, oh_a], dim=-1))


def train_joint_baseline(tasks_obj, cfg, log_fn=print):
    device = cfg.device
    model = JointConditionedMLP(cfg.p, cfg.L, cfg.joint_hidden,
                                cfg.joint_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_t = torch.tensor(tasks_obj.train_tasks, device=device)

    for step in range(cfg.joint_steps):
        which = torch.randint(0, len(tasks_obj.train_tasks), (cfg.batch_size,),
                              device=device)
        c, a = train_t[which, 0], train_t[which, 1]
        idx = torch.randint(0, tasks_obj.train_seqs.shape[0], (cfg.batch_size,),
                            device=device)
        seqs = tasks_obj.train_seqs[idx]
        y = (c.unsqueeze(1) * seqs + a.unsqueeze(1)) % cfg.p
        loss = sequence_ce(model(tasks_obj.encode(seqs), c, a), y, cfg.L, cfg.p)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 2000 == 0:
            log_fn(f"[joint baseline] step {step + 1}: loss={loss.item():.4f}")

    @torch.no_grad()
    def eval_cells(task_list):
        rows = []
        for (c, a) in task_list:
            x, y = tasks_obj.eval_tensors((c, a), "test")
            cc = torch.full((x.shape[0],), c, device=device, dtype=torch.long)
            aa = torch.full((x.shape[0],), a, device=device, dtype=torch.long)
            pred = model(x, cc, aa).reshape(-1, cfg.L, cfg.p).argmax(-1)
            rows.append(dict(task=(c, a), level=tasks_obj.length[(c, a)],
                             pos=(pred == y).float().mean().item(),
                             exact=(pred == y).all(-1).float().mean().item()))
        return rows

    return eval_cells(tasks_obj.train_tasks), eval_cells(tasks_obj.held_tasks)
