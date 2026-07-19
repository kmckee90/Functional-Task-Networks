"""Checkpoint/resume determinism and leakage checks (design 22.2). CPU-only for
bitwise-stable scatter ops."""

import json
import shutil

import numpy as np
import pytest
import torch

from ftn_nar import algorithms as A
from ftn_nar.config import ExperimentCfg, tiny_model
from ftn_nar.trainer import Trainer, run_name


def _cfg(tmp, steps=6, method="FTN_FULL"):
    cfg = ExperimentCfg()
    cfg.model = tiny_model()
    cfg.method = method
    cfg.held_out = 0
    cfg.seed = 0
    cfg.steps_per_task = steps
    cfg.checkpoint_interval = 3
    cfg.boundary_eval_count = 8
    cfg.data.support_size = 4
    cfg.data.query_batch_size = 6
    cfg.data.support_candidate_pool = 48
    cfg.configurer.inner_steps = 3
    cfg.configurer.eval_restarts = 1
    cfg.n_eval_support_sets = 2
    cfg.device = "cpu"
    cfg.out_dir = str(tmp / "runs")
    cfg.model.hard_attention_warmup_steps = 2
    return cfg


def test_held_out_never_trained(tmp_path):
    cfg = _cfg(tmp_path)
    tr = Trainer(cfg)
    assert cfg.held_out not in tr.tasks
    assert len(tr.tasks) == 7


def test_resume_matches_uninterrupted(tmp_path):
    # run A: 2 blocks straight through (tiny steps)
    cfg = _cfg(tmp_path / "a")
    tr = Trainer(cfg)
    tr.tasks = tr.tasks[:2]
    tr.n_blocks = 2
    tr.train()
    ref = {k: v.clone() for k, v in tr.model.state_dict().items()}

    # run B: identical, but killed after block 1 and resumed
    cfg2 = _cfg(tmp_path / "b")
    tr2 = Trainer(cfg2)
    tr2.tasks = tr2.tasks[:2]
    tr2.n_blocks = 1
    tr2.train()  # stops after block 1 (checkpoint 'final' written at block boundary)
    del tr2
    cfg3 = _cfg(tmp_path / "b")
    tr3 = Trainer(cfg3, resume=True)
    tr3.tasks = tr3.tasks[:2]
    tr3.n_blocks = 2
    tr3.train()

    for k, v in tr3.model.state_dict().items():
        torch.testing.assert_close(v, ref[k], rtol=1e-5, atol=1e-6,
                                   msg=f"mismatch after resume in {k}")


def test_run_dir_no_overwrite(tmp_path):
    cfg = _cfg(tmp_path)
    Trainer(cfg)  # writes init checkpoint
    with pytest.raises(FileExistsError):
        Trainer(cfg)
