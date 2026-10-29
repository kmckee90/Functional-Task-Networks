"""Bounded component comparison; use only on the m2e pod, never a live agent.

Example: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 uv run python
profile_ftn_comparison.py --output outputs/performance/comparison.json
Timings include host work and synchronization. Concurrent training affects them.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

from agents.agent_ftn import RDDPG_Dual as FTNAgent
from agents.agent_recode import RDDPG_Dual as RECODEAgent
from envs.continuous_maze import ContinuousMaze
from experiment_ftn import Config, Hyperparameters
from ftn_diagnostics import FTNRouteDiagnostic, isolated_rng, SPATIAL_OPTIONS
from agents.ftn_support.spatial_permute import spatial_permutation


def timed(function, repeats=5):
    samples = []
    function()
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return dict(median_s=statistics.median(samples), samples_s=samples)


def original_update(agent, d):
    """The original replay decoder update, with reservoir features held fixed."""
    states = torch.cat((d["o"], d["f"]), -1)
    next_states = torch.cat((d["no"], d["nf"]), -1)
    q_o = agent._q_values_o(d["o"], d["action"])
    next_action = agent.act_target(next_states)
    targets_o = agent._q_targets_o(d["ro"], torch.cat((d["no"], next_action), -1).detach(), d["done"])
    loss_o = F.mse_loss(q_o, targets_o)
    q_f = agent._q_values_f(d["f"], d["action"])
    next_action = agent.act_target(next_states)
    targets_f = agent._q_targets_f(d["rf"], torch.cat((d["nf"], next_action), -1).detach(), d["done"])
    loss_f = F.mse_loss(q_f, targets_f)
    action = agent.act_pred(states)
    actor = -(agent.q_target_o(torch.cat((d["o"], action), -1)).mean()
              + agent.q_target_f(torch.cat((d["f"], action), -1)).mean())
    for key, loss in (("o", loss_o), ("f", loss_f), ("a", actor)):
        optimizer = getattr(agent, "optimizer_" + key)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for target, pred in ((agent.q_target_o, agent.q_pred_o),
                         (agent.q_target_f, agent.q_pred_f), (agent.act_target, agent.act_pred)):
        agent._polyak_update(target.parameters(), pred.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/performance/comparison.json")
    parser.add_argument("--spatial-snapshot", help="Instead, time one bank from a saved diagnostic NPZ")
    parser.add_argument("--spatial-previous", help="Previous block NPZ, to reproduce incremental refinement")
    parser.add_argument("--spatial-bank", default="a", choices=["a", "o", "f"])
    args = parser.parse_args()
    if args.spatial_snapshot:
        bank = args.spatial_bank
        with np.load(args.spatial_snapshot, allow_pickle=False) as data:
            masks = data[f"masks_{bank}"]
            shape = tuple(data[f"grid_shape_{bank}"])
            previous = data[f"permutation_{bank}"]
        if args.spatial_previous:
            with np.load(args.spatial_previous, allow_pickle=False) as data:
                previous = data[f"permutation_{bank}"]
        padded = np.pad(masks, ((0, 0), (0, np.prod(shape) - masks.shape[1])))
        result = dict(snapshot=args.spatial_snapshot, previous=args.spatial_previous,
                      bank=bank, masks=len(masks), neurons=masks.shape[1], sweeps=8)
        print(json.dumps(result), flush=True)
        start = time.perf_counter()
        order = spatial_permutation(padded, shape, previous=previous, sweeps=8, **SPATIAL_OPTIONS)
        result["elapsed_s"] = time.perf_counter() - start
        result["valid_permutation"] = bool(np.array_equal(np.sort(order), np.arange(len(order))))
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return
    torch.set_num_threads(2)
    torch.manual_seed(123)
    cfg = {**asdict(Config()), **asdict(Hyperparameters()), "device": "cuda",
           "ftn_replay_capacity": 0}
    bounds = (-torch.ones(2, device="cuda"), torch.ones(2, device="cuda"))
    base = RECODEAgent(2, 2, bounds, **cfg).cuda()
    ftn = FTNAgent(2, 2, bounds, **cfg).cuda()
    n = cfg["minibatch_size"]
    d = dict(o=torch.randn(n, ftn.rnn_size_o, device="cuda"),
             f=torch.randn(n, ftn.rnn_size_f, device="cuda"),
             no=torch.randn(n, ftn.rnn_size_o, device="cuda"),
             nf=torch.randn(n, ftn.rnn_size_f, device="cuda"),
             action=torch.rand(n, 2, device="cuda") * 2 - 1,
             ro=torch.ones(n, 1, device="cuda"), rf=torch.ones(n, 1, device="cuda"),
             done=torch.zeros(n, 1, dtype=torch.bool, device="cuda"), goal=False)
    result = dict(gpu=torch.cuda.get_device_name(), batch_size=n,
                  threads=torch.get_num_threads(), timing_note="Alongside live training on the same GPU; not an idle-device benchmark",
                  spec=ftn.ftn_spec)
    result["original_decoder_update"] = timed(lambda: original_update(base, d))
    result["ftn_decoder_update_no_archive"] = timed(lambda: ftn._train_support(d))
    result["ftn_search"] = timed(lambda: ftn._search(ftn._trim(d)))
    with torch.no_grad():
        single = torch.randn(1, ftn.rnn_size_o + ftn.rnn_size_f, device="cuda")
        result["original_actor_forward"] = timed(lambda: base.act_pred(single), repeats=30)
        result["ftn_actor_forward"] = timed(lambda: ftn.act_pred(single), repeats=30)
    # A representative full density buffer, isolated from every live run.
    density = ftn.density_estimator
    density.memory.uniform_(-3., 3.)
    density.counts.fill_(1)
    density.size = density.memory_size
    density.d2_ema = torch.tensor(.01, device="cuda")
    density.d2_nearest.fill_(.01)
    env = ContinuousMaze(size=13, old_walls=True, max_steps=100)
    env.reset(seed=123)
    diagnostic = FTNRouteDiagnostic("unused", 4., lambda x: x)
    diagnostic.begin_episode(1, env)
    diagnostic.record_step(ftn)
    diagnostic.end_episode()
    episode = diagnostic.pending[0]
    masks = {k: diagnostic.masks[k][episode["mask_ids"][k]] for k in ("a", "o", "f")}
    policy = (deepcopy(ftn.act_pred).eval(), deepcopy(ftn.state_model_o).eval(),
              deepcopy(ftn.state_model_f).eval(), deepcopy(density))
    for model in policy[1:3]:
        model.rnn.flatten_parameters()
    with isolated_rng():
        result["diagnostic_rollout_100_steps"] = timed(
            lambda: diagnostic._rollout(policy, episode, masks, cfg, 5., 0), repeats=3)
    result["diagnostic_block_rollout_estimate_s"] = 20 * 32 * result["diagnostic_rollout_100_steps"]["median_s"]
    result["ftn_update_vs_original_no_archive"] = (result["ftn_decoder_update_no_archive"]["median_s"]
                                                  / result["original_decoder_update"]["median_s"])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
