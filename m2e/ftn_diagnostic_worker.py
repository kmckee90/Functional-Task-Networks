"""Spawned same-device diagnostics with frozen snapshots and bounded queuing.

The worker owns its CUDA context, Python/NumPy/PyTorch RNGs and evolving spatial
layouts. It never receives the live agent, optimizer or replay memories.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy
from functools import partial
import multiprocessing
import os
import time

import torch

from ftn_diagnostics import FTNRouteDiagnostic


_worker_diagnostic = None


def _normalize(reward, name):
    if name == "identity":return reward
    if name == "log":return torch.log(reward + 1)
    if name == "exp":return 1 - torch.exp(-reward)
    raise ValueError(f"Unknown density normalization: {name}")


def _move_tree(value, device):
    if isinstance(value, torch.Tensor):return value.detach().to(device)
    if isinstance(value, dict):
        result = copy(value)  # Preserve defaultdict factories used by density metrics.
        result.update({k: _move_tree(v, device) for k, v in value.items()})
        return result
    if isinstance(value, list):return [_move_tree(v, device) for v in value]
    if isinstance(value, tuple):return tuple(_move_tree(v, device) for v in value)
    return value


def _move_density(density, device):
    for name, value in vars(density).items():
        setattr(density, name, _move_tree(value, device))
    density.device = torch.device(device)
    return density


def _worker_init():
    # A single worker per training run; do not multiply BLAS/OpenMP thread pools.
    torch.set_num_threads(1)


def _evaluate_job(job):
    global _worker_diagnostic
    start = time.perf_counter()
    if _worker_diagnostic is None:
        _worker_diagnostic = FTNRouteDiagnostic(
            job["output_dir"], job["obs_scale"], partial(_normalize, name=job["normalization"]),
            block_size=job["block_size"], rollouts=job["rollouts"],
            seed=job["seed"], spatial_sweeps=job["spatial_sweeps"])
    diagnostic = _worker_diagnostic
    first_episode = job["episodes"][0]["episode"]
    # One worker processes blocks in order and retains its previous layouts.
    diagnostic.pending = job["episodes"]
    diagnostic.masks = job["masks"]
    diagnostic.execution = dict(mode="asynchronous", worker_pid=os.getpid(),
                                device=job["device"], snapshot_step=job["training_step"])
    actor, observation_model, feedback_model, density = job["policy"]
    device = torch.device(job["device"])
    policy = (actor.to(device).eval(), observation_model.to(device).eval(),
              feedback_model.to(device).eval(), _move_density(density, device))
    paths = diagnostic.evaluate_policy(policy, job["cfg"], job["max_reward"], job["training_step"])
    if job["run_id"]:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=job["tracking_uri"])
        for path in paths:
            if path.suffix.lower() == ".png":
                client.log_artifact(job["run_id"], str(path), artifact_path="ftn_diagnostics")
    result = dict(paths=paths, first_episode=first_episode,
                  training_step=job["training_step"], device=str(device),
                  elapsed_s=time.perf_counter() - start)
    print(f"FTN diagnostic worker saved: {paths[0]} ({result['elapsed_s']:.1f}s)", flush=True)
    return result


class AsyncFTNRouteDiagnostic(FTNRouteDiagnostic):
    def __init__(self, *args, normalization="identity", run_id=None,
                 tracking_uri=None, max_pending=2, **kwargs):
        super().__init__(*args, **kwargs)
        if max_pending < 1:raise ValueError("max_pending must be positive")
        if normalization not in ("identity", "log", "exp"):
            raise ValueError("Unsupported diagnostic density normalization")
        self.normalization, self.run_id, self.tracking_uri = normalization, run_id, tracking_uri
        self.max_pending = max_pending
        self._executor = None
        self._futures = []
        self._closed = False
        self.completed = []

    def submit(self, agent, max_reward):
        if self._closed:raise RuntimeError("Diagnostic worker is closed")
        if not self.ready:raise RuntimeError("A complete diagnostic block is required")
        self.poll()
        if len(self._futures) >= self.max_pending:
            print("FTN diagnostic queue full; waiting for the oldest block", flush=True)
            self._consume_oldest()
        # Copies are made at this precise block boundary. Only private CPU
        # snapshots cross the process boundary; no live CUDA storage is shared.
        policy = (deepcopy(agent.act_pred).cpu().eval(),
                  deepcopy(agent.state_model_o).cpu().eval(),
                  deepcopy(agent.state_model_f).cpu().eval(),
                  _move_density(deepcopy(agent.density_estimator), "cpu"))
        job = dict(output_dir=self.output_dir, obs_scale=self.obs_scale,
                   normalization=self.normalization, block_size=self.block_size,
                   rollouts=self.rollouts, seed=self.seed, spatial_sweeps=self.spatial_sweeps,
                   episodes=deepcopy(self.pending),
                   masks={k: [m.copy() for m in masks] for k, masks in self.masks.items()},
                   policy=policy, cfg=deepcopy(agent.config), max_reward=max_reward,
                   training_step=agent.cur_step, device=str(agent.act_pred.mask.device),
                   run_id=self.run_id, tracking_uri=self.tracking_uri)
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init)
        self._futures.append(self._executor.submit(_evaluate_job, job))
        self.pending.clear()

    def _consume_oldest(self):
        result = self._futures.pop(0).result()
        # Callbacks stay in the parent for tests/local callers. Production
        # uploads happen in the worker using the explicit run ID and client.
        if self.artifact_logger:
            for path in result["paths"]:
                if path.suffix.lower() == ".png":self.artifact_logger(path)
        self.completed.append(result)
        return result

    def poll(self):
        while self._futures and self._futures[0].done():
            self._consume_oldest()

    def close(self):
        if self._closed:return
        self._closed = True
        try:
            while self._futures:self._consume_oldest()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):return self

    def __exit__(self, *exc):self.close()
