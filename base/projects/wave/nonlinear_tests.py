"""Matched nonlinear history decoding. Run this script in the actl base pod.

Uses linear_results/config.json and linear_tests.collect verbatim, preserving
the same masks, drives, samples, and targets. Fits the same validation-tuned
ridge baseline, then trains a two-hidden-layer GELU MLP to predict its residual.
The ridge prediction is a fixed skip connection, not additional input data.
MLP inputs are both standardized state channels only. Residual targets are
scaled using training statistics to balance losses across lags/channels.
All lag heads share hidden layers. Select width/checkpoint per lag using only
validation R². Test predictions are computed after selection. Also report the
validation-selected choice between the linear and nonlinear readouts.

Example: python nonlinear_tests.py --output nonlinear_results
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from linear_tests import collect, channel_r2


class ResidualMLP(nn.Module):
    def __init__(self, inputs, width, outputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, outputs))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def prepare(splits, alphas):
    """Fit ridge and return standardized features, targets, and predictions."""
    xs, ys = map(list, zip(*splits))
    mean = xs[0].mean(0)
    scale = xs[0].std(0, correction=0).clamp_min(1e-8)
    xs = [(x-mean)/scale for x in xs]
    train, val, test = xs
    vals, vectors = torch.linalg.eigh(train.T @ train / len(train))
    vals = vals.clamp_min(0)
    modes = [x @ vectors for x in xs]
    predictions = [torch.empty_like(y) for y in ys]
    selected = []
    for lag in range(ys[0].shape[1]):
        target_mean = ys[0][:, lag].mean(0)
        cross = vectors.T @ (train.T @ (ys[0][:, lag]-target_mean) / len(train))
        best = -float('inf')
        for alpha in alphas:
            weights = cross / (vals[:, None]+alpha)
            score = channel_r2(modes[1] @ weights + target_mean, ys[1][:, lag]).item()
            if score > best:
                best, chosen, best_weights = score, alpha, weights
        selected.append(dict(alpha=chosen, val_r2=best))
        for p, m in zip(predictions, modes):
            p[:, lag] = m @ best_weights + target_mean
    return xs, ys, predictions, selected


@torch.no_grad()
def predict(model, x, batch=1024):
    return torch.cat([model(part) for part in x.split(batch)])


def train_mlp(xs, ys, base, args, seed):
    """Test targets/features are not accessed while choosing checkpoints."""
    xtrain, xval = xs[:2]
    ytrain, yval = ys[:2]
    n, lags, k = ytrain.shape
    residual = ytrain-base[0]
    scale = residual.square().mean(0).sqrt().clamp_min(1e-5)
    target = (residual/scale).flatten(1)
    best = [None]*lags
    history = []
    for width in args.widths:
        torch.manual_seed(10000+seed)
        model = ResidualMLP(xtrain.shape[1], width, lags*k).to(xtrain.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        stale = 0
        best_average = -float('inf')
        for epoch in range(1, args.epochs+1):
            model.train()
            order = torch.randperm(n, device=xtrain.device)
            for indices in order.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = (model(xtrain[indices])-target[indices]).square().mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            if epoch % args.eval_every and epoch != args.epochs:
                continue
            model.eval()
            with torch.no_grad():
                correction = predict(model, xval).reshape(-1, lags, k)*scale
                scores = [channel_r2(base[1][:, j]+correction[:, j], yval[:, j]).item()
                          for j in range(lags)]
                for j, score in enumerate(scores):
                    if best[j] is None or score > best[j]['val_r2']:
                        best[j] = dict(val_r2=score, width=width, epoch=epoch,
                                       state=copy.deepcopy(model.state_dict()))
                train_loss = (predict(model, xtrain)-target).square().mean().item()
            average = float(np.mean(scores))
            history.append(dict(width=width, epoch=epoch,
                                train_normalized_residual_mse=train_loss, val_r2=scores))
            print(f'    width={width} epoch={epoch} train residual MSE={train_loss:.4f}'
                  f' mean val R2={average:.4f}', flush=True)
            if average > best_average + 1e-5:
                best_average, stale = average, 0
            else:
                stale += 1
            if stale >= args.patience:
                break
    return best, scale, history


def experiment(config, args, k, rho, seed):
    started = time.monotonic()
    splits, mask, rms = collect(config, k, rho, seed)
    xs, ys, base, ridge = prepare(splits, config.alphas)
    del splits
    best, scale, history = train_mlp(xs, ys, base, args, seed)
    rows = []
    for j, delay in enumerate(config.delays):
        chosen = best[j]
        model = ResidualMLP(xs[0].shape[1], chosen['width'], len(config.delays)*k).to(xs[0].device)
        model.load_state_dict(chosen['state'])
        model.eval()
        with torch.no_grad():
            correction = predict(model, xs[2]).reshape(-1, len(config.delays), k)[:, j]*scale[j]
            nonlinear = channel_r2(base[2][:, j]+correction, ys[2][:, j]).item()
            linear = channel_r2(base[2][:, j], ys[2][:, j]).item()
        selected = 'nonlinear' if chosen['val_r2'] > ridge[j]['val_r2'] else 'linear'
        rows.append(dict(delay_events=delay, delay_steps=delay*config.interval,
                         linear_r2=linear, nonlinear_r2=nonlinear,
                         delta_r2=nonlinear-linear, linear_val_r2=ridge[j]['val_r2'],
                         nonlinear_val_r2=chosen['val_r2'], width=chosen['width'],
                         epoch=chosen['epoch'], selected=selected,
                         selected_r2=nonlinear if selected=='nonlinear' else linear))
    return dict(k=k, rho=rho, seed=seed, scores=rows, mask=mask,
                state_rms=rms, history=history, seconds=time.monotonic()-started)


def report(output, records, config):
    rows = [dict(k=r['k'], rho=r['rho'], seed=r['seed'], **s) for r in records for s in r['scores']]
    with (output/'scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for k in config.ks:
        for rho in config.rhos:
            for delay in config.delays:
                subset = [r for r in rows if (r['k'],r['rho'],r['delay_events'])==(k,rho,delay)]
                if not subset:
                    continue
                row = dict(k=k, rho=rho, delay_events=delay, delay_steps=delay*config.interval)
                for key in ('linear_r2', 'nonlinear_r2', 'delta_r2', 'selected_r2'):
                    values = [r[key] for r in subset]
                    row[key] = float(np.mean(values))
                    row[key+'_sd'] = float(np.std(values, ddof=1)) if len(values)>1 else 0.
                row['nonlinear_selected_seeds'] = sum(r['selected']=='nonlinear' for r in subset)
                row['seeds'] = len(subset)
                summary.append(row)
    with (output/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(config.ks), len(config.rhos), figsize=(16, 10),
                             sharex=True, sharey=True, squeeze=False)
    for i, k in enumerate(config.ks):
        for j, rho in enumerate(config.rhos):
            values = [r for r in summary if r['k']==k and r['rho']==rho and r['delay_events']>=0]
            ax = axes[i,j]
            for key, label in [('linear_r2','Linear ridge'), ('nonlinear_r2','Ridge + MLP')]:
                ax.plot([r['delay_steps'] for r in values], [r[key] for r in values], '.-', label=label)
            ax.set_title(f'K={k}, rho={rho:g}')
            ax.axhline(0, color='gray', lw=.5)
            ax.set_ylim(-.12, 1.03)
    axes[0,0].legend()
    fig.supxlabel('Delay (simulation steps)')
    fig.supylabel('Held-out R², mean across seeds')
    fig.tight_layout()
    fig.savefig(output/'comparison.png', dpi=150)
    plt.close(fig)
    lines = ['# Nonlinear history decoding', '',
             'Matched data and ridge baseline from linear_tests.py; two-hidden-layer GELU residual MLP.',
             'Widths/checkpoints selected per lag on validation only; independent test trajectories.',
             'All delays share hidden layers. Values below are nonlinear R² (change from linear), averaged across seeds.',
             'The selected_r2 CSV column additionally allows validation to choose the original linear baseline.', '',
             '| K | rho | 16 steps | 64 steps | 128 steps | 256 steps |',
             '|---|---|---|---|---|---|']
    for k in config.ks:
        for rho in config.rhos:
            lookup = {r['delay_steps']:r for r in summary if r['k']==k and r['rho']==rho}
            if lookup:
                lines.append(f'| {k} | {rho:g} | ' + ' | '.join(
                    f'{lookup[d]["nonlinear_r2"]:.3f} ({lookup[d]["delta_r2"]:+.3f})'
                    if d in lookup else '—' for d in [16,64,128,256]) + ' |')
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--linear-config', type=Path, default=Path('linear_results/config.json'))
    parser.add_argument('--output', type=Path, default=Path('nonlinear_results'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--widths', type=int, nargs='+', default=[256,512])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight-decay', type=float, default=.001)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--only-k', type=int)
    parser.add_argument('--only-rho', type=float)
    parser.add_argument('--only-seed', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.widths+[args.epochs,args.batch_size,args.eval_every,args.patience]) <= 0:
        parser.error('widths, epochs, batch size, evaluation interval and patience must be positive')
    original = json.loads(args.linear_config.read_text())
    root = Path(__file__).parent
    for file, key in [('wave_tanh.py','wave_tanh_sha256'), ('linear_tests.py','script_sha256')]:
        if hashlib.sha256((root/file).read_bytes()).hexdigest() != original[key]:
            parser.error(f'{file} changed since the linear run; cannot claim matched data')
    config = SimpleNamespace(**original, device=args.device)
    provenance = dict(data=original, training={k:v for k,v in vars(args).items()
                     if k not in ('output','linear_config','resume','device','only_k','only_rho','only_seed')},
                     script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output/'config.json'
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != provenance:
            parser.error('use --resume with identical settings or a new output directory')
    else:
        config_path.write_text(json.dumps(provenance, indent=2))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    for seed in config.seeds:
        for k in config.ks:
            for rho in config.rhos:
                path = args.output/f'k{k}_rho{rho:g}_seed{seed}.json'
                if path.exists() and args.resume:
                    records.append(json.loads(path.read_text()))
                    continue
                if ((args.only_k is not None and k!=args.only_k)
                        or (args.only_rho is not None and rho!=args.only_rho)
                        or (args.only_seed is not None and seed!=args.only_seed)):
                    continue
                print(f'START seed={seed} K={k} rho={rho:g}', flush=True)
                record = experiment(config,args,k,rho,seed)
                path.write_text(json.dumps(record, indent=2))
                records.append(record)
                print(f'DONE {record["seconds"]:.1f}s ' + ' '.join(
                    f'lag{s["delay_events"]}: {s["nonlinear_r2"]:.3f} ({s["delta_r2"]:+.3f})'
                    for s in record['scores']), flush=True)
    if records:
        report(args.output,records,config)


if __name__ == '__main__':
    main()
