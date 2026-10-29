"""Bounded same-GPU overlap benchmark and optional Databricks validation."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from agents.agent_ftn import RDDPG_Dual
from envs.continuous_maze import ContinuousMaze
import experiment_ftn as experiment
from ftn_diagnostic_worker import AsyncFTNRouteDiagnostic
from ftn_diagnostics import FTNRouteDiagnostic, BANKS


def validate_databricks():
    import mlflow
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"ftn_async_diagnostic_validation_{stamp}"
    small = replace(experiment.Hyperparameters(), ftn_k=4, ftn_hidden_size=32,
                    ftn_width=3, ftn_depth=2, hidden_size=8, minibatch_size=8,
                    memory_length=128, goal_memory_length=128, sample_memory_length=32,
                    n_optimizer_steps=1, n_optimizer_epochs=1, ftn_support_size=4)
    args = SimpleNamespace(seed=1900, render=False, log=True, run_name=name,
                           experiment_name="RDDPG Refinements", notes="Bounded asynchronous same-GPU diagnostic upload validation",
                           algo="RDDPG_Dual", task="Fixed", device="cuda", no_FCP=False, no_OCP=False,
                           load=None, gif=None, save=False, no_ftn_diagnostics=False,
                           ftn_diagnostic_dir="outputs/ftn_diagnostics")
    with patch.object(experiment, "NUM_EPISODES", 19), \
         patch.object(experiment, "Hyperparameters", return_value=small), \
         patch.object(experiment, "LGE_Maze", return_value=replace(experiment.LGE_Maze(), max_steps=6)), \
         patch.object(experiment, "Config", return_value=replace(experiment.Config(), gather_interval=0)):
        experiment.main(args)
    client = mlflow.MlflowClient(tracking_uri="databricks")
    exp = client.get_experiment_by_name("/Shared/RDDPG Refinements")
    run = client.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{name}'",
                             max_results=1)[0]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "ftn_diagnostics")]
    result = dict(run_id=run.info.run_id, status=run.info.status, artifacts=artifacts,
                  execution=run.data.params.get("ftn_diagnostic_execution"),
                  rollouts=run.data.params.get("ftn_diagnostic_rollouts"))
    Path("outputs/performance/async_databricks_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def benchmark(args):
    cfg = {**asdict(experiment.Config()), **asdict(experiment.Hyperparameters()),
           "device": "cuda", "ftn_replay_capacity": 0}
    results = {}
    for mode in ("sync", "async"):
        torch.manual_seed(123)
        agent = RDDPG_Dual(2, 2, (-torch.ones(2, device="cuda"), torch.ones(2, device="cuda")), **cfg).cuda()
        density = agent.density_estimator
        density.memory.uniform_(-3., 3.)
        density.counts.fill_(1)
        density.size = density.memory_size
        density.d2_ema = torch.tensor(.01, device="cuda")
        density.d2_nearest.fill_(.01)
        n = cfg["minibatch_size"]
        d = dict(o=torch.randn(n, agent.rnn_size_o, device="cuda"),
                 f=torch.randn(n, agent.rnn_size_f, device="cuda"),
                 no=torch.randn(n, agent.rnn_size_o, device="cuda"),
                 nf=torch.randn(n, agent.rnn_size_f, device="cuda"),
                 action=torch.rand(n, 2, device="cuda") * 2 - 1,
                 ro=torch.ones(n, 1, device="cuda"), rf=torch.ones(n, 1, device="cuda"),
                 done=torch.zeros(n, 1, dtype=torch.bool, device="cuda"), goal=False)
        cls = AsyncFTNRouteDiagnostic if mode == "async" else FTNRouteDiagnostic
        diagnostic = cls(Path(args.output).parent / f"benchmark_{mode}", 4., lambda x: x,
                         block_size=2, rollouts=32)
        env = ContinuousMaze(size=13, old_walls=True, max_steps=100)
        for episode in (1, 2):
            env.reset()
            diagnostic.begin_episode(episode, env)
            diagnostic.record_step(agent)
            diagnostic.end_episode()
        if args.history:
            with np.load(args.history, allow_pickle=False) as data:
                for k in BANKS:
                    history = list(data[f"masks_{k}"])
                    offset = len(history)
                    diagnostic.masks[k] = history + diagnostic.masks[k]
                    for ep in diagnostic.pending:
                        ep["mask_ids"][k] += offset
                        for segment in ep["segments"]:segment["mask_ids"][k] += offset
        agent._train_support(d)  # Warm the training kernels before timing.
        torch.cuda.synchronize()
        start = time.perf_counter()
        if mode == "async":diagnostic.submit(agent, 5.)
        else:diagnostic.evaluate(agent, 5.)
        submission = time.perf_counter() - start
        train_start = time.perf_counter()
        for _ in range(args.updates):agent._train_support(d)
        torch.cuda.synchronize()
        training = time.perf_counter() - train_start
        if mode == "async":diagnostic.close()
        results[mode] = dict(total_s=time.perf_counter() - start,
                             submit_or_evaluate_s=submission, training_s=training)
        print(mode, json.dumps(results[mode]), flush=True)
    results["wall_time_speedup"] = results["sync"]["total_s"] / results["async"]["total_s"]
    results["workload"] = dict(training_updates=args.updates, diagnostic_steps=2 * 32 * 100,
                               history=args.history, concurrent_live_runs=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/performance/async_comparison.json")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--history", help="Diagnostic NPZ with cumulative masks for a realistic spatial workload")
    parser.add_argument("--validate-databricks", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.validate_databricks:validate_databricks()
    else:benchmark(args)


if __name__ == "__main__":main()
