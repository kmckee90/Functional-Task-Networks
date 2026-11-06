"""Integrity and small CPU smoke checks; never writes manuscript results."""
from pathlib import Path
import hashlib
import json
import math
import random

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'RESTORATION.json').read_text())
    for rel, record in manifest['files'].items():
        actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert actual == record['sha256'], f'Historical source changed: {rel}'
    print(f"Integrity: {len(manifest['files'])} original files match {manifest['source_commit']}", flush=True)

    import numpy as np
    import torch
    import torchvision
    import plotting  # Check dependencies of the figure generators too.
    from experiments import run_revision as runner
    from tasks import MNISTSharedShuffle, PermutedMNIST, RotatedMNIST, SyntheticTasks

    print(f'Runtime: torch={torch.__version__}, torchvision={torchvision.__version__}', flush=True)
    torch.set_num_threads(1)
    # Exercise original task transforms using identifiable pixels; no MNIST download.
    x = torch.arange(20 * 784, dtype=torch.float32).reshape(20, 1, 28, 28) / (20 * 784)
    y = torch.arange(20) % 10
    for cls in (MNISTSharedShuffle, PermutedMNIST, RotatedMNIST):
        task = cls(n_tasks=2, seed=0)
        task._loaded = True
        task._train_x = task._test_x = x.flatten(1) if cls is MNISTSharedShuffle else x
        task._train_y = task._test_y = y
        for index in range(2):
            xx, yy = task.sample_batch(index, 4, device='cpu')
            assert xx.shape == (4, 784) and yy.shape == (4, 10)
            assert torch.isfinite(xx).all() and torch.all(yy.sum(1) == 1)
        if cls is PermutedMNIST:
            assert torch.equal(task._apply_task(x, 0, 'cpu'), x.flatten(1))
            assert not torch.equal(task._apply_task(x, 1, 'cpu'), x.flatten(1))
        if cls is RotatedMNIST:
            assert task._angles == [0.0, 180.0]
        print(f'Task fixture: {cls.__name__} OK', flush=True)

    # Change only this process's settings; production source/defaults stay intact.
    runner.HIDDEN_SIZE, runner.K = 12, 3
    runner.MODEL_LAYERS, runner.MODEL_INNER_SIZE = 1, 2
    runner.RECONFIG_STEPS, runner.RECONFIG_N = 2, 4
    runner.TASK_REGISTRY['synthetic'] = {
        'build': lambda seed, device: SyntheticTasks(
            input_size=2, output_size=2, n_tasks=2, n_components=3,
            classification=True, seed=seed, device=device, test_size=16),
        'input_size': 2, 'output_size': 2,
        'train_cfg_kwargs': dict(
            n_epochs=1, train_steps=2, batch_size=4, eval_batch_size=8,
            reconfig_batch_size=4, lr=3e-4, replay_n_per_step=1,
            replay_buf_size=2, reconfig_freq='batch'),
    }
    for method in runner.METHOD_SPECS:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        result = runner.run_one(method, 'synthetic', 0, device='cpu')
        matrix = result['perf_matrix_reconfig']
        assert [len(row) for row in matrix] == [1, 2], (method, matrix)
        assert all(math.isfinite(v) and 0 <= v <= 1 for row in matrix for v in row)
        for protocol in ('summary_stored', 'summary_reconfig'):
            assert all(math.isfinite(result[protocol][key]) for key in ('ACC', 'FM', 'BWT'))
        assert math.isfinite(result['head_start'][1]['task_loss_with_mask'])
        print(f'Training/evaluation smoke: {method} OK', flush=True)
    print('PASS: restoration integrity, MNIST transforms, and all 10 method paths', flush=True)


if __name__ == '__main__':
    main()
