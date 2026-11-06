"""Final-checkpoint evaluation with disjoint, domain-specific labeled support."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ftn_models import VARIANTS, build, load_family
from run_ftn import Stream
from storage import bulk, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--dataset', default='tinyimagenet')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=['features', 'vit'], default='features')
    args = parser.parse_args()
    torch.set_default_device('cuda')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = f'{args.dataset}/{args.mode}/{args.variant}/seed{args.seed}'
    checkpoint = bulk('checkpoints', 'ftn', run, 'latest.pt')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg, runner = load_family(args.variant)
    for name, value in saved['config'].items():
        # Derived configuration properties are reconstructed by the native class.
        if not isinstance(getattr(type(cfg), name, None), property):
            setattr(cfg, name, value)
    runner.Config = cfg
    with torch.device('cpu'):
        stream = Stream(args.dataset, args.seed, cfg, args.mode)
    split = json.loads(bulk('runs', 'ftn', run, 'split.json').read_text())
    assert stream.groups == split['groups']
    assert [v.tolist() for v in stream.support_indices] == split['support']
    assert [v.tolist() for v in stream.train_indices] == split['train']
    assert saved['completed_task'] == len(stream.groups) - 1
    model = build(cfg, stream.input_size if args.mode != 'vit' else 1,
                  stream.n_classes, vit=args.mode == 'vit', pretrained=False)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    model.requires_grad_(False)
    output = bulk('reevaluations', 'domain_support', run)
    output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'method': args.variant, 'dataset': args.dataset,
              'seed': args.seed, 'mode': args.mode, 'checkpoint': str(checkpoint),
              'checkpoint_sha256': hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
              'protocol': 'one cold mask search per learned block using held-out training support; separate test queries; all output classes available',
              'support_rows_per_block': cfg.T, 'blocks': [],
              'eval_reconfig_steps': cfg.EVAL_RECONFIG_STEPS,
              'support_pool_rows': sum(map(len, stream.support_indices)),
              'support_used_for_weight_training': False, 'query_labels_used_for_search': False}
    start = time.perf_counter()
    initial = model.get_reconfig_state() if hasattr(model, 'get_reconfig_state') else {'m': model.m.clone()}
    for block, group in enumerate(stream.groups):
        if hasattr(model, 'set_reconfig_state'):
            model.set_reconfig_state(initial)
        else:
            model.m.copy_(initial['m'])
        # Keep the same deterministic class-balanced sampler as the original run,
        # but draw ONLY from this block's held-out training examples.
        candidates = stream.support_indices[block]
        labels = np.asarray(stream.train.targets)[candidates]
        rng = np.random.default_rng(stream.seed + 9103 + block * 17)
        buckets = [rng.permutation(candidates[labels == c]) for c in np.unique(labels)]
        selected = []
        for row in range(max(map(len, buckets))):
            for bucket in buckets:
                if row < len(bucket) and len(selected) < cfg.T:
                    selected.append(int(bucket[row]))
            if len(selected) >= cfg.T:
                break
        assert not set(selected).intersection(stream.train_indices[block].tolist())
        assert set(selected).issubset(set(split['support'][block]))
        support = stream.fetch(selected)
        if args.variant.startswith('Bayes'):
            runner.eval_reconfigure(model, *support)
        else:
            model.reconfig_steps = cfg.EVAL_RECONFIG_STEPS
            if getattr(model, 'spec1', None) is not None:
                model.spec1.steps = cfg.EVAL_RECONFIG_STEPS
            runner.eval_reconfigure(model, *support)
        mask = model.m.detach().cpu().clone()
        assert int(torch.count_nonzero(mask)) == cfg.K
        indexes = np.flatnonzero(np.isin(stream.test.targets, group))
        with torch.device('cpu'):
            loader = DataLoader(Subset(stream.test, indexes.tolist()), batch_size=cfg.T,
                                shuffle=False, num_workers=0, generator=torch.Generator(device='cpu'))
            iterator = iter(loader)
        correct = total = 0
        with torch.no_grad():
            for x, y in iterator:
                predictions = model(x.cuda()).argmax(-1)
                correct += int((predictions == y.cuda()).sum())
                total += len(y)
        record = {'block': block, 'classes': group, 'support_indexes': selected,
                  'support_count': len(selected), 'correct': correct, 'queries': total,
                  'accuracy': correct / total, 'active_atoms': mask.nonzero().flatten().tolist()}
        state = model.get_reconfig_state() if hasattr(model, 'get_reconfig_state') else {'m': model.m.clone()}
        torch.save({'mask': mask, 'reconfiguration_state': state, 'support_indexes': selected},
                   output / f'block{block}_mask.pt')
        result['blocks'].append(record)
        write_json(output / 'progress.json', result)
        print(json.dumps({'variant': args.variant, 'block': block, 'accuracy': record['accuracy']}), flush=True)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach().cpu(), saved['model'][name]), f'Weight changed: {name}'
    result.update(status='complete', final_accuracy=float(np.mean([b['accuracy'] for b in result['blocks']])),
                  micro_accuracy=sum(b['correct'] for b in result['blocks']) / sum(b['queries'] for b in result['blocks']),
                  weights_unchanged=True, wall_seconds=time.perf_counter() - start)
    write_json(output / 'result.json', result)
    print(json.dumps({'variant': args.variant, 'final_accuracy': result['final_accuracy']}), flush=True)


if __name__ == '__main__':
    main()
