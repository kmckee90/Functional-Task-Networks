"""Block-sequential SiT lifecycle; independent GPU processes, resumable stages.

No test images enter training, replay, or mask search. Evaluation restores both
routing state and RNG state. No task identity is supplied to cold inference.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torchvision.utils import save_image
from diffusers import AutoencoderKL

from .model import BayesSiT, InputPrior, transport_batch
from .prepare import DOMAINS, CLASSES
from .bayesian_adv_config_snapshot import BayesLoRAReactiveLow


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


class Replay:
    def __init__(self, seed, low=.2, high=.8):
        self.entries = []
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device='cuda').manual_seed(seed)
        self.low, self.high = low, high
        self.attempts = self.hits = 0

    def add(self, z, y, mask):
        self.entries.append((z[:4].clone(), y[:4].clone(), mask.clone()))

    def backward(self, model):
        self.attempts += 1
        overlap = [(m * model.m).sum().item() / model.k for _, _, m in self.entries]
        eligible = [i for i, v in enumerate(overlap) if self.low < v <= self.high]
        if not eligible:
            return None
        z, y, mask = self.entries[int(self.rng.choice(eligible))]
        x, target = transport_batch(z, y, self.generator, dropout=.1)
        loss = model.loss(model(x, m=mask), target)
        loss.backward()
        self.hits += 1
        return loss.item()


def balanced_indices(labels, count):
    unique = labels.unique(sorted=True)
    # Interleave classes, so any prefix is deterministic and roughly balanced.
    indices = torch.stack([torch.where(labels == y)[0] for y in unique]).T.flatten()
    return indices[:count]


@torch.no_grad()
def query_loss(model, packed, targets, mask, batch=20):
    losses = []
    for start in range(0, len(packed), batch):
        pred = model(packed[start:start+batch], m=mask)
        losses.extend((pred - targets[start:start+batch]).square().flatten(1).mean(1).cpu().tolist())
    return {'loss': float(np.mean(losses)), 'se': float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
            'per_example': losses}


@torch.no_grad()
def generate(model, vae, mask, count=20, steps=50, seed=908, batch=20):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    zs = torch.randn(count, 4, 32, 32, device='cuda', generator=gen)
    labels = torch.tensor(list(CLASSES.values()), device='cuda').repeat(math.ceil(count/10))[:count]
    model.controller.mask = mask
    images = []
    for start in range(0, count, batch):
        x = zs[start:start+batch]
        y = labels[start:start+batch]
        ys = torch.cat([y, torch.full_like(y, 1000)])
        def velocity(x, t):
            xx = torch.cat([x, x])
            tt = torch.full((len(xx),), t, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return model.backbone.forward_with_cfg(xx, tt, ys, cfg_scale=4.)[:len(x)].float()
        # Fixed-grid Heun ODE, same solver and seeds in every condition.
        dt = 1. / steps
        for step in range(steps):
            v = velocity(x, step * dt)
            vp = velocity(x + dt * v, (step + 1) * dt)
            x = x + dt * .5 * (v + vp)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            decoded = vae.decode(x / .18215).sample
        images.append(((decoded.float() + 1) / 2).clamp(0, 1).cpu())
    return torch.cat(images)


def evaluate(model, data, saved_masks, router, stage, args, out, vae=None):
    training_state = model.get_reconfig_state()
    row = {'stage': stage, 'domains': {}, 'seed': args.seed}
    masks = {}
    start_time = time.time()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for domain in DOMAINS:
            torch.manual_seed(88001 + DOMAINS.index(domain))
            support, test = data[domain]['support'], data[domain]['test']
            si = balanced_indices(support['y'], args.support)
            qi = balanced_indices(test['y'], args.queries)
            sg = torch.Generator(device='cuda').manual_seed(189)
            qg = torch.Generator(device='cuda').manual_seed(901)
            sx, sy = transport_batch(support['z'][si], support['y'][si], sg)
            qx, qy = transport_batch(test['z'][qi], test['y'][qi], qg)
            candidates = {'base': torch.zeros_like(model.m), 'current': training_state['m']}
            random_gen = torch.Generator(device='cuda').manual_seed(65)
            candidates['random'] = model._topk_mask(torch.randn(model.hidden_size, device='cuda', generator=random_gen))
            infer_stats = {}
            for budget in sorted(set([4, args.support])):
                tic = time.time()
                if args.method == 'plain':
                    mask = torch.ones_like(model.m)
                else:
                    # Reset seed at every lifecycle stage for paired search noise.
                    torch.manual_seed(7801 + budget + DOMAINS.index(domain))
                    model.cold_reconfigure(sx[:budget], sy[:budget], steps=args.eval_steps,
                                           take_best=False, restarts=1)
                    mask = model.m.clone()
                candidates[f'cold{budget}'] = mask
                infer_stats[f'cold{budget}'] = {'seconds': time.time()-tic,
                    'support_loss': model.loss(model(sx[:budget], m=mask), sy[:budget]).item()}
            if domain in saved_masks:
                candidates['stored'] = saved_masks[domain]
            if args.ic:
                for refine in [False, True]:
                    name = 'ic_refine' if refine else 'ic_direct'
                    torch.manual_seed(7801 + 4 + DOMAINS.index(domain))
                    infer_stats[name] = router.infer(model, support['z'][si[:4]], sx[:4], sy[:4], refine)
                    candidates[name] = model.m.clone()
            results = {}
            for name, mask in candidates.items():
                results[name] = query_loss(model, qx, qy, mask)
                results[name]['mask'] = mask.cpu().tolist()
                if domain in saved_masks:
                    results[name]['stored_overlap'] = float((mask * saved_masks[domain]).sum().item() / model.k)
            results['inference'] = infer_stats
            row['domains'][domain] = results
            masks[domain] = candidates
            print(json.dumps({'eval': stage, 'domain': domain,
                  'losses': {n: round(v['loss'], 5) for n, v in results.items() if 'loss' in v}}), flush=True)
            if vae is not None and args.samples:
                image_dir = out / 'samples' / stage
                image_dir.mkdir(parents=True, exist_ok=True)
                im = generate(model, vae, candidates[f'cold{args.support}'], args.samples,
                              args.sample_steps)
                save_image(im, image_dir / f'{domain}_cold.png', nrow=10)
                if stage == 'initial' and domain == 'photo':
                    im = generate(model, vae, candidates['base'], args.samples, args.sample_steps)
                    save_image(im, image_dir / 'photo_base.png', nrow=10)
                if domain in saved_masks:
                    im = generate(model, vae, candidates['stored'], args.samples, args.sample_steps)
                    save_image(im, image_dir / f'{domain}_stored.png', nrow=10)
    model.set_reconfig_state(training_state)
    row['seconds'] = time.time()-start_time
    atomic_json(out / f'eval_{stage}.json', row)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/dev/sit_state')
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--support', type=int, default=40)
    p.add_argument('--queries', type=int, default=200)
    p.add_argument('--eval-steps', type=int, default=30)
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--sample-steps', type=int, default=50)
    p.add_argument('--H', type=int, default=64)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--rank', type=int, default=1)
    p.add_argument('--alpha', type=float, default=4.)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--bayes-preset', choices=['baseline','reactive_low'], default='baseline')
    p.add_argument('--reactive-sd', type=float, default=None, help='Explicit detector-threshold ablation; defaults to source preset')
    p.add_argument('--replay', type=int, choices=[0, 1], default=1)
    p.add_argument('--method', choices=['bayes', 'plain'], default='bayes')
    p.add_argument('--order', default='monochrome,poster,sketch')
    p.add_argument('--ic', action='store_true')
    p.add_argument('--ic-repair', action='store_true', help='Repair old input priors from training replay after each block')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    if args.lr is None:
        args.lr = BayesLoRAReactiveLow.TRAIN_LR if args.bayes_preset == 'reactive_low' else 1e-4
    if args.bayes_preset == 'reactive_low':
        args.replay = 0
        if args.ic_repair:
            raise ValueError('ReactiveLow is replay-free; IC replay repair cannot be enabled')
    if args.ic_repair:
        args.ic = True
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    root = Path(args.root)
    out = root / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'config.json', vars(args))
    data = {d: {s: {k: v.cuda() for k, v in torch.load(root / f'{d}_{s}.pt', weights_only=True).items()}
                for s in ['train', 'support', 'test']} for d in DOMAINS}
    H, k, rank, alpha = args.H, args.k, args.rank, args.alpha
    if args.method == 'plain':
        # Ordinary rank-k*r LoRA: same active rank and scaling, one atom.
        H, k, rank, alpha = 1, 1, args.k * args.rank, args.alpha / args.k
    model = BayesSiT(root / 'SiT-XL-2-256.pt', H, k, rank, alpha,
                     reactive_low=args.bayes_preset == 'reactive_low').cuda().eval()
    if args.method == 'plain':
        model.m.fill_(1)
    params = [p for p in model.parameters() if p.requires_grad]
    print(json.dumps({'trainable': sum(p.numel() for p in params),
                      'adapted_layers': len(model.adapted_names), 'torch': torch.__version__,
                      'mask_source_sha256': hashlib.sha256(Path(__file__).with_name('bayes_reconfig.py').read_bytes()).hexdigest()}), flush=True)
    vae = AutoencoderKL.from_pretrained(root / 'vae').cuda().eval().requires_grad_(False) if args.samples else None
    opt = torch.optim.Adam(params, lr=args.lr)
    replay = Replay(args.seed + 1000)
    router = InputPrior()
    saved_masks, history = {}, []
    tg = torch.Generator(device='cuda').manual_seed(args.seed + 10)
    dg = torch.Generator(device='cuda').manual_seed(args.seed + 20)
    mu, variance, since = None, 0., 0
    order = args.order.split(',')
    start_block = 0
    if args.resume and (out / 'latest.pt').exists():
        ck = torch.load(out / 'latest.pt', map_location='cuda', weights_only=False)
        model.load_adapter(ck['adapter'])
        saved_masks = ck['saved_masks']
        history = ck['history']
        replay.entries = ck['replay']
        replay.rng.bit_generator.state = ck['replay_rng']
        replay.generator.set_state(ck['replay_generator'].cpu())
        router.entries = ck['router']
        torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        torch.set_rng_state(ck['cpu_rng'].cpu())
        tg.set_state(ck['tg'].cpu())
        dg.set_state(ck['dg'].cpu())
        mu, variance, since = ck['detector']
        start_block = ck['completed_blocks']
    elif not args.pilot:
        history.append(evaluate(model, data, saved_masks, router, 'initial', args, out, vae))
        atomic_json(out / 'lifecycle.json', history)

    log = (out / 'train.jsonl').open('a', buffering=1)
    for block, domain in enumerate(order):
        if block < start_block:
            continue
        # Match original experiment's per-block Adam reset. We do not force a
        # mask reset at known boundaries: the original reactive detector decides.
        opt.state = defaultdict(dict)
        train = data[domain]['train']
        cp_count = 0
        tic = time.time()
        for step in range(args.steps):
            idx = torch.randint(len(train['z']), (args.batch,), device='cuda', generator=dg)
            z, y = train['z'][idx], train['y'][idx]
            x, target = transport_batch(z, y, tg, dropout=.1)
            cp = False
            detector_loss, detector_threshold = None, None
            if args.method == 'bayes':
                with torch.no_grad():
                    detector_loss = model.loss(model(x), target).item()
                r_alpha = BayesLoRAReactiveLow.RECONFIG_REACTIVE_EMA if args.bayes_preset == 'reactive_low' else .9
                r_sd = args.reactive_sd if args.reactive_sd is not None else (BayesLoRAReactiveLow.RECONFIG_REACTIVE_SD if args.bayes_preset == 'reactive_low' else 5.)
                detector_threshold = None if mu is None else mu + r_sd * math.sqrt(max(variance, 0.))
                # Preserve the source runner's exact ceil expression for this preset.
                warmup = math.ceil(1. / max(1. - r_alpha, 1e-8)) if args.bayes_preset == 'reactive_low' else 10
                cp = mu is None or (since >= warmup and detector_loss > detector_threshold)
                if cp:
                    model.reconfigure_changepoint(x, target)
                    cp_count += 1
                    since = 0
                else:
                    model.reconfigure(x, target)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(model(x), target)
            value = loss.item()
            if cp or mu is None:
                mu, variance = value, 0.
            else:
                diff = value - mu
                r_alpha = BayesLoRAReactiveLow.RECONFIG_REACTIVE_EMA if args.bayes_preset == 'reactive_low' else .9
                mu, variance = r_alpha * mu + (1-r_alpha) * value, r_alpha * variance + (1-r_alpha) * diff * diff
            since += 1
            if step < 32 and args.bayes_preset != 'reactive_low':
                replay.add(z, y, model.m)
            loss.backward()
            rep_loss = replay.backward(model) if args.replay else None
            opt.step()
            entry = {'block': block, 'domain': domain, 'step': step,
                     'loss': value, 'replay_loss': rep_loss, 'changepoint': cp,
                     'detector_loss': detector_loss, 'detector_threshold': detector_threshold,
                     'elapsed': time.time()-tic, 'mask': model.m.cpu().tolist(),
                     'mask_kl': model.last_reconfig_stats.get('kl')}
            log.write(json.dumps(entry)+'\n')
            if step % 20 == 0 or step == args.steps-1:
                print(json.dumps({k: v for k, v in entry.items() if k != 'mask'}), flush=True)
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite training loss')
        saved_masks[domain] = model.m.clone()
        router.archive(train['z'][balanced_indices(train['y'], 256)], model.mask_logits)
        if args.ic_repair:
            repairs = router.repair_from_replay(model, replay.entries,
                live_projector=router.entries[-1]['projector'], seed=args.seed+7401+block)
            atomic_json(out / f'repair_after_{block+1}.json', repairs)
        print('BLOCK_FINISHED', domain, 'seconds', time.time()-tic, 'changepoints', cp_count,
              'replay_hits', replay.hits, 'attempts', replay.attempts, flush=True)
        stage = f'after_{block+1}_{domain}'
        history.append(evaluate(model, data, saved_masks, router, stage, args, out, vae))
        atomic_json(out / 'lifecycle.json', history)
        checkpoint = {'adapter': model.adapter_state(), 'saved_masks': saved_masks,
                      'history': history, 'replay': replay.entries, 'router': router.entries,
                      'replay_rng': replay.rng.bit_generator.state,
                      'replay_generator': replay.generator.get_state(),
                      'cuda_rng': torch.cuda.get_rng_state(), 'cpu_rng': torch.get_rng_state(),
                      'tg': tg.get_state(), 'dg': dg.get_state(),
                      'detector': (mu, variance, since), 'completed_blocks': block+1}
        torch.save(checkpoint, out / f'{stage}.pt')
        torch.save(checkpoint, out / 'latest.tmp')
        (out / 'latest.tmp').replace(out / 'latest.pt')
    log.close()
    atomic_json(out / 'completed.json', {'completed': True, 'blocks': len(order),
        'replay_hits': replay.hits, 'replay_attempts': replay.attempts})
    print('COMPLETE', args.name, flush=True)


if __name__ == '__main__':
    main()
