"""Linear decoding of input history in WaveTanh; run on the actl base pod.

Matches save_gif: a fixed random mask in u, with independent
Bernoulli(0.5) * N(0,1) * 0.025 inputs added directly to state every 4
simulation steps. Observe immediately after addition, before the recurrent
step, just like the GIF. Delay 0 therefore includes the freshly added input.
Use BOTH state channels (8192 features on a 64x64 sheet) unless --readout u.

Each split contains independently driven, separately initialized trajectories.
No random split of overlapping history windows. Burn-in precedes all samples.
Fit a centered, feature-standardized ridge readout with intercept for each
delay; choose its penalty on validation data, then evaluate once on test data.
Targets are input amplitudes / amplitude, including the zero-valued bits.
Report mean per-input-channel test R² (1: perfect, 0: mean predictor).
Delay -1 is a future IID input, an explicit negative control.

Default: 12 configurations x 3 seeds, 16384/4096/4096 train/val/test samples,
13 past/current delays plus one future control. No learned reservoir weights.
Results are saved after each configuration so an interrupted run is resumable.

Example (in the base pod):
    python linear_tests.py --device cuda:0 --output linear_results
Dependencies: torch, numpy, matplotlib (plotting only).
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from wave_tanh import WaveTanh


@torch.no_grad()
def collect(args, k, rho, seed):
    """Vectorize independent trajectories; hold out whole trajectories."""
    device = torch.device(args.device)
    counts = (args.train_streams, args.val_streams, args.test_streams)
    batch = sum(counts)
    generator = torch.Generator(device=device).manual_seed(seed)
    # Draw a full-size permutation and full Kmax drive for paired comparisons
    # across K and rho: smaller K uses a subset of the same input locations.
    mask = torch.randperm(args.size**2, generator=generator, device=device)[:k]
    start = args.burn_in + max(args.delays)
    events = start + args.samples_per_stream + 1  # include future control
    shape = (events, batch, max(args.ks))
    drive = (torch.rand(shape, generator=generator, device=device) < 0.5)
    drive = (drive * torch.randn(shape, generator=generator, device=device))[..., :k]
    model = WaveTanh(rho=rho, padding=args.padding).to(device).eval()
    state = torch.zeros(batch, 2, args.size, args.size, device=device)
    channels = 2 if args.readout == "both" else 1
    features = torch.empty(batch, args.samples_per_stream,
                           channels*args.size**2, device=device)
    for event in range(start + args.samples_per_stream):
        state[:, 0].reshape(batch, -1)[:, mask] += args.amplitude * drive[event]
        if event >= start:
            features[:, event-start] = state[:, :channels].flatten(1)
        for _ in range(args.interval):
            state = model(state)
    if not torch.isfinite(features).all():
        raise RuntimeError("Nonfinite reservoir states")
    times = torch.arange(start, start+args.samples_per_stream, device=device)
    targets = torch.stack([drive[times-delay].transpose(0, 1)
                           for delay in args.delays], dim=2)
    splits = []
    offset = 0
    for count in counts:
        splits.append((features[offset:offset+count].reshape(-1, features.shape[-1]),
                       targets[offset:offset+count].reshape(-1, len(args.delays), k)))
        offset += count
    return splits, mask.cpu().tolist(), features.square().mean().sqrt().item()


def channel_r2(prediction, target):
    """Mean channel R², computed without clipping negative scores."""
    variance = (target-target.mean(0)).square().mean(0)
    return (1 - (prediction-target).square().mean(0) / variance.clamp_min(1e-12)).mean()


@torch.no_grad()
def decode(args, splits):
    (train, ytrain), (val, yval), (test, ytest) = splits
    mean = train.mean(0)
    scale = train.std(0, correction=0).clamp_min(1e-8)
    train = (train-mean)/scale
    val = (val-mean)/scale
    test = (test-mean)/scale
    n = train.shape[0]
    # Objective: mean squared error + alpha * ||weights||². One symmetric
    # eigendecomposition supports all lags and penalties; no matrix inverse.
    gram = train.T @ train / n
    eigenvalues, vectors = torch.linalg.eigh(gram)
    if eigenvalues.min().item() < -1e-3:
        raise RuntimeError("Unexpectedly indefinite feature covariance")
    eigenvalues = eigenvalues.clamp_min(0)
    val_modes = val @ vectors
    test_modes = test @ vectors
    rows = []
    for j, delay in enumerate(args.delays):
        target_mean = ytrain[:, j].mean(0)
        cross = train.T @ (ytrain[:, j]-target_mean) / n
        coefficients = vectors.T @ cross
        best_score = -math.inf
        best_alpha = None
        best_weights = None
        for alpha in args.alphas:
            weights = coefficients / (eigenvalues[:, None] + alpha)
            prediction = val_modes @ weights + target_mean
            score = channel_r2(prediction, yval[:, j]).item()
            if score > best_score:
                best_score, best_alpha, best_weights = score, alpha, weights
        prediction = test_modes @ best_weights + target_mean
        truth = ytest[:, j]
        rows.append(dict(delay_events=delay, delay_steps=delay*args.interval,
                         alpha=best_alpha, val_r2=best_score,
                         test_r2=channel_r2(prediction, truth).item(),
                         test_rmse=(prediction-truth).square().mean().sqrt().item()
                         * args.amplitude,
                         mean_baseline_r2=channel_r2(target_mean.expand_as(truth), truth).item()))
    return rows


def report(output, records, args):
    """Write raw scores, mean/SD summary, and standalone memory curves."""
    rows = [dict(k=r['k'], rho=r['rho'], seed=r['seed'], **row)
            for r in records for row in r['scores']]
    with (output/'scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for k in args.ks:
        for rho in args.rhos:
            for delay in args.delays:
                selected = [r for r in rows if (r['k'], r['rho'], r['delay_events'])
                            == (k, rho, delay)]
                if not selected:
                    continue
                values = [r['test_r2'] for r in selected]
                summary.append(dict(k=k, rho=rho, delay_events=delay,
                                    delay_steps=delay*args.interval, seeds=len(values),
                                    mean_r2=float(np.mean(values)),
                                    sd_r2=float(np.std(values, ddof=1)) if len(values)>1 else 0.))
    with (output/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.ks), figsize=(5*len(args.ks), 4),
                             sharey=True, squeeze=False)
    for ax, k in zip(axes[0], args.ks):
        for rho in args.rhos:
            values = [r for r in summary if r['k']==k and r['rho']==rho
                      and r['delay_events']>=0]
            if not values:
                continue
            x = np.array([r['delay_steps'] for r in values])
            y = np.array([r['mean_r2'] for r in values])
            sd = np.array([r['sd_r2'] for r in values])
            line, = ax.plot(x, y, '.-', label=f'rho={rho:g}')
            ax.fill_between(x, y-sd, y+sd, color=line.get_color(), alpha=.12)
        ax.axhline(0, color='gray', linewidth=.7)
        ax.axhline(.5, color='gray', linewidth=.7, linestyle=':')
        ax.set(title=f'K={k}', xlabel='Delay (simulation steps)', ylim=(-.1, 1.03))
        ax.legend()
    axes[0, 0].set_ylabel('Held-out R² (mean ± SD across seeds)')
    fig.tight_layout()
    fig.savefig(output/'memory_curves.png', dpi=160)
    plt.close(fig)
    chosen = [d for d in (0, 1, 4, 8, 16, 32, 64) if d in args.delays]
    lines = ['# Linear history decoding', '',
             f'Sheet: {args.size}×{args.size}; padding: {args.padding}; readout: {args.readout}.',
             f'Inputs: Bernoulli(0.5) × N(0,1) × {args.amplitude}, every {args.interval} steps.',
             f'Samples: {args.train_streams*args.samples_per_stream} train / '
             f'{args.val_streams*args.samples_per_stream} validation / '
             f'{args.test_streams*args.samples_per_stream} test per configuration and seed.',
             'Independent trajectories per split. Ridge chosen on validation; test R² averaged across input channels.',
             'Delay 0 is measured immediately after injection; delay -1 is the future-input control.',
             'Tables show mean across seeds. Full scores and standard deviations are in CSV files.', '',
             '| K | rho | ' + ' | '.join(f'lag {d}' for d in chosen) + ' | future |',
             '|---|---|' + '---|'*(len(chosen)+1)]
    for k in args.ks:
        for rho in args.rhos:
            lookup = {r['delay_events']:r['mean_r2'] for r in summary
                      if r['k']==k and r['rho']==rho}
            if lookup:
                lines.append(f'| {k} | {rho:g} | ' + ' | '.join(
                    f'{lookup[d]:.3f}' if d in lookup else '—' for d in chosen+[-1]) + ' |')
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--ks', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--rhos', nargs='+', type=float, default=[.975, .99, .995, 1.])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--delays', nargs='+', type=int,
                        default=[-1, 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    parser.add_argument('--alphas', nargs='+', type=float,
                        default=[1e-4, 1e-3, 1e-2, .1, 1., 10.])
    parser.add_argument('--amplitude', type=float, default=.025)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--burn-in', type=int, default=256, help='injection events')
    parser.add_argument('--samples-per-stream', type=int, default=1024)
    parser.add_argument('--train-streams', type=int, default=16)
    parser.add_argument('--val-streams', type=int, default=4)
    parser.add_argument('--test-streams', type=int, default=4)
    parser.add_argument('--padding', choices=['zeros', 'circular', 'reflect'], default='circular')
    parser.add_argument('--readout', choices=['both', 'u'], default='both')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=Path('linear_results'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (args.size < 2 or not all(0 < k <= args.size**2 for k in args.ks)
            or not all(math.isfinite(r) and 0 < r <= 1 for r in args.rhos)
            or not all(math.isfinite(a) and a > 0 for a in args.alphas)
            or min(args.delays) < -1 or max(args.delays) < 0
            or min(args.interval, args.samples_per_stream, args.train_streams,
                   args.val_streams, args.test_streams) < 1
            or args.burn_in < 0 or not 0 < args.amplitude < math.inf):
        parser.error('invalid sizes, delays, rho, alpha, amplitude, or sample counts')
    args.delays = sorted(set(args.delays))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = {k:v for k, v in vars(args).items() if k not in ('output', 'resume', 'device')}
    config['wave_tanh_sha256'] = hashlib.sha256(Path(__file__).with_name('wave_tanh.py').read_bytes()).hexdigest()
    config['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_path = output/'config.json'
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            parser.error('output already exists; use --resume with identical settings or a new --output')
    else:
        config_path.write_text(json.dumps(config, indent=2))
    metadata = dict(torch=torch.__version__, device=args.device,
                    gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None)
    (output/'runtime.json').write_text(json.dumps(metadata, indent=2))
    records = []
    for seed in args.seeds:
        for k in args.ks:
            for rho in args.rhos:
                path = output/f'k{k}_rho{rho:g}_seed{seed}.json'
                if args.resume and path.exists():
                    record = json.loads(path.read_text())
                else:
                    started = time.monotonic()
                    print(f'START seed={seed} K={k} rho={rho:g}', flush=True)
                    splits, mask, rms = collect(args, k, rho, seed)
                    print(f'  collected in {time.monotonic()-started:.1f}s; state RMS={rms:.5f}', flush=True)
                    scores = decode(args, splits)
                    record = dict(k=k, rho=rho, seed=seed, mask=mask, state_rms=rms,
                                  seconds=time.monotonic()-started, scores=scores)
                    path.write_text(json.dumps(record, indent=2))
                    del splits
                    print(f'  DONE {record["seconds"]:.1f}s: ' + ' '.join(
                        f'lag{r["delay_events"]}={r["test_r2"]:.3f}' for r in scores), flush=True)
                records.append(record)
    report(output, records, args)
    print(f'Results: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
