"""
baselines.py — SupSup and Progressive Networks baselines for func_net_kt.

Both methods share the parallel-MLP NeuroModel backbone so the parameter
budget is directly comparable to FTN (H=8192 neuron slots, k=512 active per
task, same per-MLP topology). Each method has its own training path because
they make different assumptions about which parameters move and which are
frozen, and they both use per-task readout heads.

SupSup (Wortsman et al. 2020).
  * Backbone (the parallel-MLP bank) is FROZEN at random init.
  * Per task, learn a binary mask over the H slots via score-based STE
    ("edge-popup"-style): per-slot scores s in R^H, hard mask = topk(s, k),
    gradient flows into s via straight-through.
  * Per task, learn a fresh readout head linear(H, out).
  * At eval (no task identity), scan stored (mask_t, readout_t) pairs and
    pick the pair whose support-set loss is lowest. This is the simpler
    one-hot version of SupSup's GG retrieval and is the closest analog to
    FixedMask + Router.

Progressive Networks (Rusu et al. 2016).
  * Disjoint per-task columns: column t = slots [t*k, (t+1)*k).
  * When training task t, columns 0..t-1 are frozen (no gradient); only
    column t's parallel-MLP weights and the new readout head receive
    updates.
  * Lateral connections: the active mask during task-t training is the
    CUMULATIVE mask of columns 0..t, so the readout head has access to
    all prior columns' outputs alongside the freshly trained column.
  * Per-task readout head linear(H, out).
  * At eval (no task identity), scan stored (cumulative_mask_t, readout_t)
    pairs, pick by support-set loss.

Both methods share the head-start protocol used elsewhere in the project:
at the boundary of a new task, before any weight update on that task, scan
the stored bank for the best (mask, readout) pair on the support set, then
report its loss on a fresh held-out batch from the new task.
"""

from __future__ import annotations

import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from masks import kwta


# ---------------------------------------------------------------------------
# Per-task readout bank — small helper
# ---------------------------------------------------------------------------


def _new_readout(hidden_size: int, output_size: int, device: str) -> nn.Linear:
    layer = nn.Linear(hidden_size, output_size, bias=False)
    nn.init.xavier_uniform_(layer.weight)
    return layer.to(device)


def _loss_fn(y_pred: torch.Tensor, y: torch.Tensor, is_clf: bool) -> torch.Tensor:
    if is_clf:
        target = y.argmax(-1) if y.dim() == 2 else y.long()
        return F.cross_entropy(y_pred, target)
    return F.mse_loss(y_pred.squeeze(), y.squeeze())


def _retrieve(
    model: nn.Module,
    mask_store: dict,
    readout_store: dict,
    support: tuple,
    is_clf: bool,
) -> tuple:
    """Scan stored (mask, readout) pairs, return (best_idx, best_mask, best_readout, best_loss)."""
    x, y = support
    if not mask_store:
        return None, None, None, float("inf")
    with torch.no_grad():
        z = model.parallel(x).detach()
    best_idx, best_loss = None, float("inf")
    best_mask, best_readout = None, None
    device = x.device
    for tid, mask in mask_store.items():
        mask = mask.to(device)
        readout = readout_store[tid].to(device)
        with torch.no_grad():
            y_pred = readout(z * mask.reshape(1, -1))
            loss = _loss_fn(y_pred, y, is_clf).item()
        if loss < best_loss:
            best_loss = loss
            best_idx = tid
            best_mask = mask.clone()
            best_readout = readout
    return best_idx, best_mask, best_readout, best_loss


def _eval_perf(
    task_set,
    model: nn.Module,
    mask: torch.Tensor,
    readout: nn.Linear,
    task_idx: int,
    eval_batch_size: int,
    device: str,
) -> float:
    """Compute task-level performance metric on the test split using a swapped readout."""
    # Swap the model's readout temporarily so task_set.evaluate uses the right head.
    saved = model.linear
    try:
        model.linear = readout
        return task_set.evaluate(
            model,
            mask,
            task_idx,
            split="test",
            batch_size=eval_batch_size,
            device=device,
        )
    finally:
        model.linear = saved


# ---------------------------------------------------------------------------
# SupSup
# ---------------------------------------------------------------------------


def supsup_train(
    model: nn.Module,
    task_set,
    config,
    hidden_size: int,
    k: int,
    output_size: int,
    score_lr: float = 1e-2,
    readout_lr: float = 3e-4,
    verbose: bool = False,
) -> dict:
    """Train SupSup: frozen backbone, per-task (mask, readout)."""
    device = config.device
    model = model.to(device)
    # Freeze backbone (parallel MLP bank). Readout will be discarded; we
    # build per-task readouts.
    for p in model.parallel.parameters():
        p.requires_grad = False
    # Disable dropout to keep the frozen-backbone features deterministic.
    if hasattr(model, "dropout"):
        model.dropout.p = 0.0

    is_clf = getattr(task_set, "is_classification", True)
    n_tasks = task_set.n_tasks

    mask_store: dict[int, torch.Tensor] = {}
    readout_store: dict[int, nn.Linear] = {}
    perf_matrix_reconfig: list[list[float]] = []
    perf_matrix_stored: list[list[float]] = []
    head_start: list[dict] = []

    for task_idx in range(n_tasks):
        # ---- Head-start: best stored (mask, readout) for this task's support
        head_loss = None
        if mask_store:
            sup_x, sup_y = task_set.sample_batch(
                task_idx, config.reconfig_batch_size, device=device,
            )
            best_idx, best_mask, best_readout, _best_sup_loss = _retrieve(
                model, mask_store, readout_store, (sup_x, sup_y), is_clf,
            )
            eval_x, eval_y = task_set.sample_batch(
                task_idx, config.eval_batch_size, split="train", device=device,
            )
            with torch.no_grad():
                z = model.parallel(eval_x).detach()
                y_pred = best_readout(z * best_mask.reshape(1, -1))
                head_loss = float(_loss_fn(y_pred, eval_y, is_clf).item())
        head_start.append({
            "reconfig_initial_loss": None,
            "reconfig_final_loss": None,
            "task_loss_with_mask": head_loss,
        })

        # ---- Train task: fresh scores + fresh readout
        scores = nn.Parameter(0.01 * torch.randn(hidden_size, device=device))
        readout = _new_readout(hidden_size, output_size, device)
        opt = torch.optim.Adam(
            [
                {"params": [scores], "lr": score_lr},
                {"params": readout.parameters(), "lr": readout_lr},
            ],
        )

        model.train()
        for _step in range(config.train_steps):
            x, y = task_set.sample_batch(task_idx, config.batch_size, device=device)
            with torch.no_grad():
                z = model.parallel(x).detach()
            hard = kwta(scores.detach(), k=k)
            mask_ste = hard + scores - scores.detach()  # STE
            y_pred = readout(z * mask_ste.reshape(1, -1))
            loss = _loss_fn(y_pred, y, is_clf)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Final hard mask
        with torch.no_grad():
            final_mask = kwta(scores.detach(), k=k).cpu()
        mask_store[task_idx] = final_mask
        readout_store[task_idx] = copy.deepcopy(readout).to("cpu")

        # ---- Eval all seen tasks under reconfig protocol
        row_reconfig: list[float] = []
        row_stored: list[float] = []
        for eval_idx in range(task_idx + 1):
            sup_x, sup_y = task_set.sample_batch(
                eval_idx, config.reconfig_batch_size, device=device,
            )
            _bid, best_mask, best_readout, _bl = _retrieve(
                model, mask_store, readout_store, (sup_x, sup_y), is_clf,
            )
            perf_recon = _eval_perf(
                task_set, model, best_mask, best_readout, eval_idx,
                config.eval_batch_size, device,
            )
            row_reconfig.append(perf_recon)

            # "Stored" oracle: use the stored mask/readout for that task ID.
            stored_mask = mask_store[eval_idx].to(device)
            stored_readout = readout_store[eval_idx].to(device)
            perf_st = _eval_perf(
                task_set, model, stored_mask, stored_readout, eval_idx,
                config.eval_batch_size, device,
            )
            row_stored.append(perf_st)

        perf_matrix_reconfig.append(row_reconfig)
        perf_matrix_stored.append(row_stored)

        if verbose:
            print(f"  T{task_idx}: reconfig={row_reconfig}, stored={row_stored}")

    return {
        "evals": {
            "stored": perf_matrix_stored,
            "reconfig": perf_matrix_reconfig,
        },
        "mask_store": mask_store,
        "head_start": head_start,
        "first_epoch_loss": [],
        "loss_history": [],
    }


# ---------------------------------------------------------------------------
# Progressive Networks (column bank)
# ---------------------------------------------------------------------------


def _zero_grad_mask_for_old_columns(model: nn.Module, hidden_size: int, k: int, task_idx: int) -> None:
    """Zero gradients for slots outside column task_idx (i.e. old columns).

    Called after backward(), before optimizer.step().
    """
    new_lo = task_idx * k
    new_hi = min((task_idx + 1) * k, hidden_size)
    for w in model.parallel.weights:
        if w.grad is None:
            continue
        # w shape: (H, in, out); zero rows outside [new_lo, new_hi).
        w.grad[:new_lo].zero_()
        if new_hi < hidden_size:
            w.grad[new_hi:].zero_()
    for b in model.parallel.biases:
        if b.grad is None:
            continue
        b.grad[:new_lo].zero_()
        if new_hi < hidden_size:
            b.grad[new_hi:].zero_()


def progressive_train(
    model: nn.Module,
    task_set,
    config,
    hidden_size: int,
    k: int,
    output_size: int,
    readout_lr: float | None = None,
    verbose: bool = False,
) -> dict:
    """Train Progressive Networks: per-task disjoint column + per-task readout."""
    device = config.device
    model = model.to(device)
    if hasattr(model, "dropout"):
        model.dropout.p = 0.0

    is_clf = getattr(task_set, "is_classification", True)
    n_tasks = task_set.n_tasks
    if readout_lr is None:
        readout_lr = config.lr

    mask_store: dict[int, torch.Tensor] = {}
    readout_store: dict[int, nn.Linear] = {}
    perf_matrix_reconfig: list[list[float]] = []
    perf_matrix_stored: list[list[float]] = []
    head_start: list[dict] = []

    for task_idx in range(n_tasks):
        # ---- Head-start: best stored (cumulative_mask, readout)
        head_loss = None
        if mask_store:
            sup_x, sup_y = task_set.sample_batch(
                task_idx, config.reconfig_batch_size, device=device,
            )
            _bid, best_mask, best_readout, _bl = _retrieve(
                model, mask_store, readout_store, (sup_x, sup_y), is_clf,
            )
            eval_x, eval_y = task_set.sample_batch(
                task_idx, config.eval_batch_size, split="train", device=device,
            )
            with torch.no_grad():
                z = model.parallel(eval_x).detach()
                y_pred = best_readout(z * best_mask.reshape(1, -1))
                head_loss = float(_loss_fn(y_pred, eval_y, is_clf).item())
        head_start.append({
            "reconfig_initial_loss": None,
            "reconfig_final_loss": None,
            "task_loss_with_mask": head_loss,
        })

        # ---- Build cumulative mask for tasks 0..task_idx
        cum_mask = torch.zeros(hidden_size, device=device)
        end = min((task_idx + 1) * k, hidden_size)
        cum_mask[: end] = 1.0

        # ---- Fresh readout per task
        readout = _new_readout(hidden_size, output_size, device)
        opt = torch.optim.Adam(
            [
                {"params": model.parallel.parameters(), "lr": config.lr},
                {"params": readout.parameters(), "lr": readout_lr},
            ],
            weight_decay=0.0,
        )

        model.train()
        for _step in range(config.train_steps):
            x, y = task_set.sample_batch(task_idx, config.batch_size, device=device)
            z = model.parallel(x)
            y_pred = readout(z * cum_mask.reshape(1, -1))
            loss = _loss_fn(y_pred, y, is_clf)
            opt.zero_grad()
            loss.backward()
            # Freeze old columns: zero their gradients before stepping.
            _zero_grad_mask_for_old_columns(model, hidden_size, k, task_idx)
            opt.step()

        mask_store[task_idx] = cum_mask.detach().cpu()
        readout_store[task_idx] = copy.deepcopy(readout).to("cpu")

        # ---- Eval all seen tasks
        row_reconfig: list[float] = []
        row_stored: list[float] = []
        for eval_idx in range(task_idx + 1):
            sup_x, sup_y = task_set.sample_batch(
                eval_idx, config.reconfig_batch_size, device=device,
            )
            _bid, best_mask, best_readout, _bl = _retrieve(
                model, mask_store, readout_store, (sup_x, sup_y), is_clf,
            )
            perf_recon = _eval_perf(
                task_set, model, best_mask, best_readout, eval_idx,
                config.eval_batch_size, device,
            )
            row_reconfig.append(perf_recon)

            stored_mask = mask_store[eval_idx].to(device)
            stored_readout = readout_store[eval_idx].to(device)
            perf_st = _eval_perf(
                task_set, model, stored_mask, stored_readout, eval_idx,
                config.eval_batch_size, device,
            )
            row_stored.append(perf_st)

        perf_matrix_reconfig.append(row_reconfig)
        perf_matrix_stored.append(row_stored)

        if verbose:
            print(f"  T{task_idx}: reconfig={row_reconfig}, stored={row_stored}")

    return {
        "evals": {
            "stored": perf_matrix_stored,
            "reconfig": perf_matrix_reconfig,
        },
        "mask_store": mask_store,
        "head_start": head_start,
        "first_epoch_loss": [],
        "loss_history": [],
    }
