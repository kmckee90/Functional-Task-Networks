"""Block-sequential trainer for FTN-LoRA.

Each task arrives as a contiguous block. For each minibatch:
  1. Pull a support batch from the current task.
  2. Run the configurer on (model, support batch) to obtain a binary mask.
  3. Update only the active LoRA atoms on a query minibatch (loss masked to
     answer tokens).
  4. Optionally draw a replay sample whose mask overlaps the current mask in
     a middle band, and add a replay-loss term.

We also collect per-task metrics: pre-update head-start NLL on a held-out
batch (forward transfer) and post-task evaluation accuracy/NLL (retention).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from .lora_bank import MaskController, BankedLinear
from .configurer import (
    AnnealSTEConfigurer, RandomFixedConfigurer, RetrievalRouter, _topk_mask,
    ConfigurerConfig,
)
from .replay import MaskFilteredReplayBuffer, UniformReplayBuffer, ReplayBufferConfig
from .tasks import collate_for_train


@dataclass
class TrainConfig:
    method: str = "FTN"
    # method ∈ {
    #   FTN, FTN_router, FTN_noreplay,
    #   FTN_RandMask,             # FTN-LoRA training pipeline (replay + per-batch mask reconfig)
    #                             # but random per-batch top-k masks instead of the configurer.
    #                             # Eval uses the configurer to match FTN-LoRA's eval protocol.
    #   SeqLoRA, SeqLoRA_ER,
    #   FixedMask, FixedMask_router, RandMask_router,
    # }
    train_steps_per_task: int = 250
    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 2e-4
    max_len: int = 512
    eval_batch_size: int = 16
    head_start_eval_size: int = 64

    # Configurer
    cfg_inner_steps: int = 5
    cfg_lr: float = 0.1

    # Replay
    replay_capacity: int = 200
    replay_rho_low: float = 0.1
    replay_rho_high: float = 0.7
    replay_n_per_step: int = 1


def _model_forward_loss(model, batch, device, controller=None, mask=None):
    """Forward pass with mask applied; returns (loss, n_token, logits[optional])."""
    if controller is not None and mask is not None:
        controller.set_mask(mask)
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    return out.loss


def _support_loss_factory(model, support_batch, device, controller):
    """Returns a closure that computes the support-set loss with the current
    mask in `controller`. Used by the configurer."""
    def fn():
        out = model(
            input_ids=support_batch["input_ids"].to(device),
            attention_mask=support_batch["attention_mask"].to(device),
            labels=support_batch["labels"].to(device),
        )
        return out.loss
    return fn


def configure_mask_for_method(
    method: str,
    model,
    support_batch,
    device,
    controller: MaskController,
    annealste: AnnealSTEConfigurer | None = None,
    fixed_router: RandomFixedConfigurer | None = None,
    retrieval_router: RetrievalRouter | None = None,
    fixed_block_mask: torch.Tensor | None = None,
    rand_fixed_mask: torch.Tensor | None = None,
    full_mask: torch.Tensor | None = None,
    task_key: str = "",
) -> torch.Tensor:
    """Pick the mask appropriate for `method`. All paths set controller.mask."""
    if method in ("SeqLoRA", "SeqLoRA_ER"):
        # Sequential single-LoRA: always use the SAME fixed k-block across all
        # tasks. No isolation; later tasks overwrite earlier tasks' atoms.
        # Matches the "active parameter budget per update" of FTN.
        assert fixed_block_mask is not None
        controller.set_mask(fixed_block_mask)
        return fixed_block_mask
    if method == "FixedMask":
        assert fixed_block_mask is not None
        controller.set_mask(fixed_block_mask)
        return fixed_block_mask
    if method == "FixedMask_router":
        # Same as FixedMask during training; eval uses retrieval router
        assert fixed_block_mask is not None
        controller.set_mask(fixed_block_mask)
        return fixed_block_mask
    if method == "RandMask_router":
        assert rand_fixed_mask is not None
        controller.set_mask(rand_fixed_mask)
        return rand_fixed_mask
    if method in ("FTN", "FTN_router", "FTN_noreplay"):
        assert annealste is not None
        # Use the support batch to optimize a mask
        loss_fn = _support_loss_factory(model, support_batch, device, controller)
        m = annealste(loss_fn, device=device, model=model)
        return m
    raise ValueError(f"Unknown method: {method}")


def _run_eval_nll(model, examples, tokenizer, device, mask, controller, batch_size=8, max_len=512):
    """Compute mean answer-token NLL and exact-match accuracy on `examples`
    under `mask`."""
    if mask is not None:
        controller.set_mask(mask)
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            if not chunk:
                continue
            batch = collate_for_train(chunk, tokenizer, max_len=max_len)
            ids = batch["input_ids"].to(device)
            mask_a = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask_a, labels=labels)
            losses.append(float(out.loss.detach()) * (labels != -100).sum().item())

            # Per-example accuracy: argmax over predicted next tokens at answer
            # positions; require the FIRST answer token to match.
            logits = out.logits  # (B, L, V)
            preds = logits.argmax(dim=-1)
            # labels[i, j] is the target for position j (i.e. produced from input[j-1])
            # so compare preds[:, j-1] with labels[:, j].
            for b in range(labels.shape[0]):
                ans_pos = (labels[b] != -100).nonzero(as_tuple=True)[0]
                if ans_pos.numel() == 0:
                    continue
                first = ans_pos[0].item()
                # Predicted token at position first comes from input position first-1.
                if first == 0:
                    continue
                pred_tok = preds[b, first - 1].item()
                tgt_tok = labels[b, first].item()
                if pred_tok == tgt_tok:
                    correct += 1
                total += 1

    n_tok = sum((batch["labels"] != -100).sum().item() for batch in [
        collate_for_train(examples[i:i + batch_size], tokenizer, max_len=max_len)
        for i in range(0, len(examples), batch_size)
    ])
    nll = sum(losses) / max(1, n_tok)
    acc = correct / max(1, total)
    return {"nll": nll, "acc": acc, "n_examples": len(examples)}


def block_sequential_train(
    model,
    tokenizer,
    bank_wrappers: list[BankedLinear],
    controller: MaskController,
    tasks: list[dict],
    train_cfg: TrainConfig,
    device,
    seed: int = 0,
    log_fn=None,
) -> dict:
    """Drive the block-sequential CL loop. Returns a dict containing per-task
    head-start metrics, per-(train_task, eval_task) post-block matrix, and a
    per-step loss trace.
    """
    method = train_cfg.method

    # Optimizer over LoRA bank params
    bank_params = [p for w in bank_wrappers for p in w.lora_parameters()]
    optimizer = torch.optim.AdamW(bank_params, lr=train_cfg.lr, weight_decay=0.0)

    annealste = AnnealSTEConfigurer(
        ConfigurerConfig(type="AnnealSTE", inner_steps=train_cfg.cfg_inner_steps,
                         lr=train_cfg.cfg_lr, init="zero"),
        controller,
    )
    fixed_router = RandomFixedConfigurer(controller, seed=seed)
    retrieval_router = RetrievalRouter(controller)

    # Replay buffer
    if method in ("FTN", "FTN_router", "FTN_RandMask"):
        # FTN-family training pipelines all use mask-overlap-filtered replay.
        # FTN_RandMask shares this replay so that the only difference from FTN
        # is the source of the per-batch mask (random vs configurer).
        replay = MaskFilteredReplayBuffer(ReplayBufferConfig(
            capacity=train_cfg.replay_capacity,
            rho_low=train_cfg.replay_rho_low,
            rho_high=train_cfg.replay_rho_high,
            n_per_step=train_cfg.replay_n_per_step,
        ))
    elif method == "SeqLoRA_ER":
        replay = UniformReplayBuffer(ReplayBufferConfig(
            capacity=train_cfg.replay_capacity,
            n_per_step=train_cfg.replay_n_per_step,
        ))
    else:
        replay = None  # FTN_noreplay, NoMask, FixedMask*, RandMask_router

    H, k = controller.H, controller.k
    full_mask = torch.ones(H, device=device)

    # Random per-task masks for RandMask_router: each task gets a fresh random
    # k-of-H mask (overlapping allowed, unlike disjoint FixedMask).
    rand_per_task: dict[int, torch.Tensor] = {}
    rand_gen = torch.Generator(device="cpu").manual_seed(seed * 13 + 7)

    def rand_mask_for(task_idx: int) -> torch.Tensor:
        if task_idx not in rand_per_task:
            m = torch.zeros(H)
            m[torch.randperm(H, generator=rand_gen)[:k]] = 1.0
            rand_per_task[task_idx] = m.to(device)
        return rand_per_task[task_idx]

    # Per-task fixed disjoint blocks for FixedMask-family methods. Slot range
    # [t*k, (t+1)*k). Wrap on overflow if more tasks than H/k.
    def fixed_block_mask_for(task_idx: int) -> torch.Tensor:
        m = torch.zeros(H, device=device)
        start = (task_idx * k) % H
        end = start + k
        if end <= H:
            m[start:end] = 1.0
        else:
            m[start:] = 1.0
            m[: end - H] = 1.0
        return m

    # Sequential LoRA always uses the same fixed block (block 0)
    seq_block_mask = torch.zeros(H, device=device)
    seq_block_mask[:k] = 1.0

    # Stored masks for retrieval-based eval (FixedMask_router, RandMask_router, FTN_router)
    stored_masks: list[tuple[str, torch.Tensor]] = []  # (task_name, mask)

    out = {
        "method": method,
        "head_start": {},          # task -> {nll, acc}
        "post_task_matrix": [],    # list of (train_t, [eval_t -> {...}])
        "task_order": [t["spec"].name for t in tasks],
        "step_log": [],
    }

    for t_idx, task in enumerate(tasks):
        task_name = task["spec"].name
        train_examples = task["train"]
        support_examples = task["support"]
        eval_examples = task["eval"]

        # ---- Head-start: select a mask using support, then evaluate on held-out
        #     before any weight update on this task ----
        # Determine evaluation mask source for head-start
        head_eval_examples = eval_examples[: train_cfg.head_start_eval_size]
        support_batch = collate_for_train(support_examples, tokenizer, max_len=train_cfg.max_len)
        if t_idx == 0:
            # No prior tasks; head-start is undefined (or measured against init).
            # We still report it for symmetry: use the AnnealSTE configurer
            # (or per-method default) to produce a mask, then eval.
            pass

        # For methods that use retrieval at evaluation, the head-start mask is
        # picked via the router, not via the inner-loop optimizer. For
        # FixedMask-family methods, the head-start mask is the would-be NEXT
        # disjoint block (since training hasn't started). For the configurer
        # methods, we run AnnealSTE to mimic what a continual learner would do
        # on encountering a new task with the bank in its current state.
        eval_mask: torch.Tensor
        if method in ("FTN", "FTN_noreplay", "FTN_RandMask"):
            # Run inner-loop optimization on the support batch
            loss_fn = _support_loss_factory(model, support_batch, device, controller)
            eval_mask = annealste(loss_fn, device=device, model=model)
        elif method == "FTN_router":
            # Retrieve from stored FTN-trained masks; if none yet (first task),
            # fall back to AnnealSTE
            if not retrieval_router.store:
                loss_fn = _support_loss_factory(model, support_batch, device, controller)
                eval_mask = annealste(loss_fn, device=device, model=model)
            else:
                loss_fn = _support_loss_factory(model, support_batch, device, controller)
                eval_mask = retrieval_router(loss_fn, device=device)
        elif method == "FixedMask":
            eval_mask = fixed_block_mask_for(t_idx)
        elif method == "FixedMask_router":
            if not retrieval_router.store:
                eval_mask = fixed_block_mask_for(t_idx)
            else:
                loss_fn = _support_loss_factory(model, support_batch, device, controller)
                eval_mask = retrieval_router(loss_fn, device=device)
        elif method == "RandMask_router":
            if not retrieval_router.store:
                eval_mask = rand_mask_for(t_idx)
            else:
                loss_fn = _support_loss_factory(model, support_batch, device, controller)
                eval_mask = retrieval_router(loss_fn, device=device)
        else:  # SeqLoRA, SeqLoRA_ER
            eval_mask = seq_block_mask

        head_metrics = _run_eval_nll(
            model, head_eval_examples, tokenizer, device,
            eval_mask, controller,
            batch_size=train_cfg.eval_batch_size, max_len=train_cfg.max_len,
        )
        out["head_start"][task_name] = head_metrics
        if log_fn:
            log_fn(f"[task {t_idx}: {task_name}] head-start nll={head_metrics['nll']:.4f} acc={head_metrics['acc']:.3f}")

        # ---- Train block ----
        model.train()
        # Independent baselines (not in our list), but FixedMask wants the
        # current task's block always. Configurer methods reconfigure per batch.

        # Set up parameter freezing for FixedMask: only the columns in this
        # block receive gradients. We implement this by zeroing gradients for
        # other slots after backward. Index masking on params in BankedLinear.
        train_block_idx = (t_idx * k) % H
        train_block_end = train_block_idx + k
        train_block_wraparound = train_block_end > H

        step_count = 0
        rng = torch.Generator().manual_seed(seed * 1000 + t_idx)
        for step in range(train_cfg.train_steps_per_task):
            # Build mini-batch
            idx = torch.randint(0, len(train_examples), (train_cfg.batch_size,), generator=rng).tolist()
            chunk = [train_examples[i] for i in idx]
            batch = collate_for_train(chunk, tokenizer, max_len=train_cfg.max_len)

            # Configure mask
            if method in ("FTN", "FTN_router", "FTN_noreplay"):
                # Per-batch reconfiguration on a small support sub-batch from
                # this task. We sample a fresh small support set each step.
                sup_idx = torch.randint(0, len(support_examples), (train_cfg.batch_size,), generator=rng).tolist()
                sup_chunk = [support_examples[i] for i in sup_idx]
                sup_batch = collate_for_train(sup_chunk, tokenizer, max_len=train_cfg.max_len)
                cur_mask = configure_mask_for_method(
                    method, model, sup_batch, device, controller,
                    annealste=annealste, full_mask=full_mask,
                    task_key=task_name,
                )
            elif method == "FTN_RandMask":
                # Random per-batch top-k mask: same training pipeline as FTN
                # (replay, B0 init, schedule) but the configurer is replaced
                # by a fresh random k-of-H mask each step. Tests whether the
                # support-conditioned configurer (rather than the training
                # pipeline) is what produces FTN-LoRA's headline ACC/FM/BWT.
                cur_mask = torch.zeros(H, device=device)
                cur_mask[torch.randperm(H, generator=rng)[:k]] = 1.0
                controller.set_mask(cur_mask)
            elif method == "FixedMask" or method == "FixedMask_router":
                cur_mask = fixed_block_mask_for(t_idx)
                controller.set_mask(cur_mask)
            elif method == "RandMask_router":
                cur_mask = rand_mask_for(t_idx)
                controller.set_mask(cur_mask)
            else:  # SeqLoRA*
                cur_mask = seq_block_mask
                controller.set_mask(cur_mask)

            # Forward + backward on the current batch
            controller.set_mask(cur_mask)
            loss = _model_forward_loss(model, batch, device)
            (loss / train_cfg.grad_accum).backward()

            # Replay step
            if replay is not None:
                replay_examples = replay.sample(cur_mask)
                if replay_examples:
                    rep_batch = collate_for_train(replay_examples, tokenizer, max_len=train_cfg.max_len)
                    # For mask-filtered replay we use the CURRENT mask (the
                    # interfering subset); for uniform ER we also use the
                    # current mask (NoMask_ER has full mask anyway).
                    controller.set_mask(cur_mask)
                    rep_loss = _model_forward_loss(model, rep_batch, device)
                    (rep_loss / train_cfg.grad_accum).backward()

            # Optimizer step on accumulation boundary
            if (step + 1) % train_cfg.grad_accum == 0:
                # For FixedMask-family: zero grads for non-block atoms so the
                # bank remains partitioned. For configurer methods the gradient
                # is naturally zero on inactive atoms (they didn't contribute
                # to the forward), so no extra masking needed.
                if method in ("FixedMask", "FixedMask_router"):
                    with torch.no_grad():
                        for w in bank_wrappers:
                            zero_idx = torch.ones(H, dtype=torch.bool, device=device)
                            if not train_block_wraparound:
                                zero_idx[train_block_idx:train_block_end] = False
                            else:
                                zero_idx[train_block_idx:] = False
                                zero_idx[: train_block_end - H] = False
                            for p in (w.A, w.B):
                                if p.grad is not None:
                                    # zero out grad for non-block atoms
                                    p.grad[zero_idx] = 0.0
                torch.nn.utils.clip_grad_norm_(bank_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            step_count += 1
            # Add example to replay buffer occasionally
            if replay is not None and step_count % 10 == 0:
                # store one example with its current mask (mask-filtered) or
                # without (uniform)
                r_idx = torch.randint(0, len(train_examples), (1,), generator=rng).item()
                replay.add(train_examples[r_idx], cur_mask, task_name)

            if step % 50 == 0 and log_fn:
                log_fn(f"  step {step}/{train_cfg.train_steps_per_task} train_loss={float(loss):.4f}")

        # Stash mask for retrieval at evaluation time
        # We use the configurer-output mask on a fresh support batch as the
        # "training-time mask" stored for retrieval.
        if method in ("FTN", "FTN_router"):
            sup_batch = collate_for_train(support_examples[: train_cfg.batch_size],
                                          tokenizer, max_len=train_cfg.max_len)
            loss_fn = _support_loss_factory(model, sup_batch, device, controller)
            stored = annealste(loss_fn, device=device, model=model)
            retrieval_router.add(task_name, stored)
        elif method in ("FixedMask_router",):
            retrieval_router.add(task_name, fixed_block_mask_for(t_idx))
        elif method == "RandMask_router":
            retrieval_router.add(task_name, rand_mask_for(t_idx))

        # ---- Post-task evaluation matrix row ----
        # Evaluate every task seen so far
        row = {}
        for j_idx, eval_task in enumerate(tasks[: t_idx + 1]):
            j_name = eval_task["spec"].name
            j_eval = eval_task["eval"]
            j_support = eval_task["support"]

            # Pick eval mask per method
            if method in ("FTN", "FTN_RandMask"):
                # Inner-loop optimization at eval. FTN_RandMask uses the
                # configurer at eval to match FTN-LoRA's eval protocol; the
                # only training-time difference is the per-batch mask source.
                sup_batch = collate_for_train(j_support, tokenizer, max_len=train_cfg.max_len)
                loss_fn = _support_loss_factory(model, sup_batch, device, controller)
                m_eval = annealste(loss_fn, device=device, model=model)
            elif method in ("FTN_router", "FixedMask_router", "RandMask_router"):
                # Retrieve from stored
                sup_batch = collate_for_train(j_support, tokenizer, max_len=train_cfg.max_len)
                loss_fn = _support_loss_factory(model, sup_batch, device, controller)
                m_eval = retrieval_router(loss_fn, device=device)
            elif method == "FixedMask":
                # Oracle: use the block this task was trained on
                m_eval = fixed_block_mask_for(j_idx)
            elif method == "FTN_noreplay":
                # Same configurer at eval as FTN
                sup_batch = collate_for_train(j_support, tokenizer, max_len=train_cfg.max_len)
                loss_fn = _support_loss_factory(model, sup_batch, device, controller)
                m_eval = annealste(loss_fn, device=device, model=model)
            else:  # SeqLoRA, SeqLoRA_ER
                m_eval = seq_block_mask

            metrics = _run_eval_nll(
                model, j_eval[: train_cfg.eval_batch_size * 2], tokenizer, device,
                m_eval, controller,
                batch_size=train_cfg.eval_batch_size, max_len=train_cfg.max_len,
            )
            row[j_name] = metrics
        out["post_task_matrix"].append({"train_task": task_name, "evals": row})
        if log_fn:
            avg_acc = sum(row[k]["acc"] for k in row) / len(row)
            log_fn(f"[after task {t_idx}] avg_acc={avg_acc:.3f}")

    return out
