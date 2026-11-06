"""Standard vision streams driven by the unchanged native FTN training loops.

The harness supplies images, validation-only support, and full test queries.
It does not replace mask search, the reactive detector, optimizer resets, or
overlap-filtered replay. Replay stores sample indexes instead of image copies.
"""
import argparse
import copy
import hashlib
import json
import time
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ftn_models import VARIANTS, build, config_dict, load_family
from isolation import IsolationAdam
from storage import bulk, write_json
from vision_data import N_CLASSES, raw_datasets, split_indices,feature_datasets


class Stream:
    def __init__(self, name, seed, cfg, mode):
        self.name, self.seed, self.cfg, self.mode = name, seed, cfg, mode
        size = 224 if mode == 'vit' else {'cifar10':32, 'cifar100':32, 'tinyimagenet':64, 'core50':128}[name]
        self.encoder_parameters=0
        self.encoder=None
        if mode=='features':
            self.train,self.test,self.groups,self.encoder_parameters,self.encoder=feature_datasets(name,seed)
        else:
            self.train, self.test, self.groups = raw_datasets(name, seed, size, prompt=mode=='vit')
        self.train_indices, self.support_indices = split_indices(self.train.targets, self.groups, seed)
        self.rng = np.random.default_rng(seed + 300)
        self.last_indices = None
        self.current_task = -1
        self.input_size = self.train.data.shape[1] if mode=='features' else 3*size*size
        self.n_classes = N_CLASSES[name]

    def fetch(self, indexes):
        samples = [self.train[int(i)] for i in indexes]
        x = torch.stack([x for x, _ in samples]).cuda()
        labels = torch.tensor([int(y) for _, y in samples], device='cuda')
        if self.mode == 'pixels':
            x = x.flatten(1)
        y = torch.nn.functional.one_hot(labels, self.n_classes).float()
        return x, y

    def __call__(self, task):
        self.current_task = task
        self.last_indices = self.rng.choice(self.train_indices[task], self.cfg.T, replace=True)
        return self.fetch(self.last_indices)

    def support(self, tasks):
        # Class-balanced labeled calibration from held-out TRAIN examples.
        # These labels never come from test queries. Limit total support to T.
        candidates = np.concatenate([self.support_indices[t] for t in tasks])
        rng = np.random.default_rng(self.seed + 9103 + sum(tasks) * 17)
        labels = np.asarray(self.train.targets)[candidates]
        buckets = [rng.permutation(candidates[labels == c]) for c in np.unique(labels)]
        chosen = []
        while len(chosen) < min(self.cfg.T, len(candidates)):
            progress = False
            for bucket in buckets:
                j = len(chosen) // len(buckets)
                if j < len(bucket) and len(chosen) < self.cfg.T:
                    chosen.append(bucket[j])
                    progress = True
            if not progress:
                break
        return self.fetch(chosen)


class IndexReplay:
    """Same native replay contents, loss and sampling; no duplicated images.

    Images have deterministic transforms. A stored sample index regenerates
    the exact same float32 input. Native masks stay on GPU. Label and data
    capacity remain those of the requested configuration.
    """
    def __init__(self, stream, maxlen, batch_size):
        self.stream = stream
        self.data = deque(maxlen=maxlen)
        self.masks = deque(maxlen=maxlen)
        self.batch_size = batch_size
        self.max_rows = 0

    def add(self, dat, mask):
        indexes = self.stream.last_indices.copy()
        if self.batch_size is not None and self.batch_size < len(indexes):
            chosen = torch.randperm(len(indexes))[:self.batch_size].cpu().numpy()
            indexes = indexes[chosen]
        self.data.append(indexes)
        self.masks.append(mask.clone())
        self.max_rows = max(self.max_rows, sum(map(len, self.data)))

    def get(self, i):
        return self.stream.fetch(self.data[i])

    def __len__(self):
        return len(self.data)


def summarize(matrix):
    a = np.asarray(matrix, dtype=float)
    n = len(a)
    acc = float(a[-1].mean())
    forgetting = float(np.mean([max(a[t, j] for t in range(j, n)) - a[-1, j] for j in range(n-1)])) if n > 1 else 0.
    bwt = float(np.mean([a[-1,j] - a[j,j] for j in range(n-1)])) if n > 1 else 0.
    return {'final_accuracy': acc, 'forgetting': forgetting, 'bwt': bwt,
            'average_incremental_accuracy': float(np.mean([a[i,:i+1].mean() for i in range(n)]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--dataset', choices=N_CLASSES, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--mode', choices=['features','pixels', 'vit'], default='features')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--timing', action='store_true', help='Two blocks, 50 updates each, full model dimensions; excluded from comparisons')
    a = p.parse_args()
    assert not (a.smoke and a.timing)
    # Native FTNs create masks/noise with torch default-device constructors.
    torch.set_default_device('cuda')
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg, runner = load_family(a.variant)
    cfg.SEED = a.seed
    cfg.RENDER_STEPS = False
    cfg.EVAL_PROSPECTIVE = False  # novel-class calibration is not available in CIL
    cfg.EVAL_BACKWARDS = True
    if a.smoke:
        cfg.TRAIN_DURATION, cfg.T = 2, 4
        cfg.RECONFIG_SAMPLE_SIZE = 4
        cfg.SQUARE_DIM,cfg.K=8,8
        cfg.LAYERS,cfg.INNER_DIM=2,4
        cfg.LORA_WIDTH,cfg.LORA_LAYERS=16,2
        cfg.RECONFIG_STEPS,cfg.EVAL_RECONFIG_STEPS=3,3
        cfg.BAYES_CHANGEPOINT_RECONFIG_STEPS=3
    # torchvision data remains on CPU even though native model defaults to CUDA.
    with torch.device('cpu'):
        stream = Stream(a.dataset, a.seed, cfg, a.mode)
    if a.timing:
        cfg.TRAIN_DURATION = 50
    cfg.N_TASK = min(2,len(stream.groups)) if a.smoke or a.timing else len(stream.groups)
    if a.smoke or a.timing:
        stream.groups=stream.groups[:cfg.N_TASK]
    model = build(cfg, stream.input_size if a.mode != 'vit' else 1, stream.n_classes, vit=a.mode == 'vit')
    optimizer = IsolationAdam(model,lr=cfg.TRAIN_LR)
    replay = IndexReplay(stream, cfg.REPLAY_MAXLEN, cfg.REPLAY_BATCH_SIZE)
    if cfg.RECONFIG_MATCH_STORED and hasattr(model, 'set_stored_masks_ref'):
        model.set_stored_masks_ref(replay.masks)
    runner.Config = cfg
    name = f'{a.dataset}/{a.mode}/{a.variant}/seed{a.seed}' + ('_smoke' if a.smoke else '_timing' if a.timing else '')
    out = bulk('runs', 'ftn', name)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = bulk('checkpoints', 'ftn', name, 'latest.pt')
    config = config_dict(cfg)
    write_json(out/'config.json', config)
    write_json(out/'split.json', {'groups':stream.groups, 'train':[v.tolist() for v in stream.train_indices], 'support':[v.tolist() for v in stream.support_indices]})
    results = {'method':a.variant, 'dataset':a.dataset, 'mode':a.mode, 'seed':a.seed,
               'smoke':a.smoke or a.timing, 'timing':a.timing, 'status':'running', 'n_tasks':cfg.N_TASK,
               'support_scope':'evaluated block only',
               'test_queries_per_block':[int(np.isin(stream.test.targets,g).sum()) for g in stream.groups],
               'config':config, 'class_il_matrix':[], 'training_active_atoms':[], 'eval_seconds':[],
               'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
               'total_parameters':sum(p.numel() for p in model.parameters())+stream.encoder_parameters,
               'shared_frozen_encoder_parameters':stream.encoder_parameters,'encoder':stream.encoder,
               'support_rows_per_search':cfg.T,
               'calibration_pool_rows':sum(len(v) for v in stream.support_indices[:cfg.N_TASK]),
               'weight_training_presentations':cfg.N_TASK*cfg.N_EPOCHS*cfg.TRAIN_DURATION*cfg.T,
               'insertion_points':getattr(model, 'insertion_points', []),
               'frozen_disabled_modules':getattr(model,'isolation_frozen_dormant',[]),
               'checkpoint':str(checkpoint)}
    results['optimizer_policy']=optimizer.policy()
    start = time.perf_counter()

    def search(x, y):
        if a.variant.startswith('Bayes'):
            return runner.eval_reconfigure(model, x, y)
        old = model.reconfig_steps
        spec = getattr(model, 'spec1', None)
        old_spec = spec.steps if spec is not None else None
        model.reconfig_steps = cfg.EVAL_RECONFIG_STEPS or old
        if spec is not None:
            spec.steps = model.reconfig_steps
        try:
            return runner.eval_reconfigure(model, x, y)
        finally:
            model.reconfig_steps = old
            if spec is not None:
                spec.steps = old_spec

    def query(task_ids, allowed_classes=None):
        rows = {}
        targets = np.asarray(stream.test.targets)
        for task in task_ids:
            indexes = np.flatnonzero(np.isin(targets, stream.groups[task]))
            dataset = Subset(stream.test, indexes.tolist())
            # CPU loader generator avoids default CUDA-generator mismatch.
            with torch.device('cpu'):
                loader = DataLoader(dataset, batch_size=cfg.T, shuffle=False, num_workers=0, generator=torch.Generator(device='cpu'))
                iterator = iter(loader)
            correct, total, loss_sum = 0, 0, 0.
            for x, y in iterator:
                x, y = x.cuda(), y.cuda()
                if a.mode == 'pixels':
                    x = x.flatten(1)
                with torch.no_grad():
                    pred = model(x)
                    if allowed_classes is not None:
                        invalid = torch.ones(stream.n_classes, dtype=torch.bool)
                        invalid[allowed_classes] = False
                        pred = pred.masked_fill(invalid[None,:], -torch.inf)
                    correct += int((pred.argmax(-1) == y).sum())
                    loss_sum += float(torch.nn.functional.cross_entropy(pred, y, reduction='sum'))
                    total += len(y)
                if a.smoke:
                    break
            rows[task] = {'acc':correct/total, 'loss':loss_sum/total}
        return rows

    def evaluate(model_arg, data_gen, task_indices):
        state = model.get_reconfig_state() if hasattr(model, 'get_reconfig_state') else {'m':model.m.clone()}
        results['training_active_atoms'].append(state['m'].nonzero().flatten().cpu().tolist())
        was_training = model.training
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state()
        model.eval()
        t0 = time.perf_counter()
        seen = list(range(stream.current_task+1))
        classes = [c for task in seen for c in stream.groups[task]]
        try:
            # Labeled holdout support identifies the evaluated context; no task
            # identifier or query labels are passed to the model or search.
            cil = {}
            for task in seen:
                if hasattr(model, 'set_reconfig_state'):
                    model.set_reconfig_state(state)
                else:
                    model.m.copy_(state['m'])
                search(*stream.support([task]))
                cil.update(query([task], classes))
            results['class_il_matrix'].append([cil[j]['acc'] if j in cil else None for j in range(cfg.N_TASK)])
            results['eval_seconds'].append(time.perf_counter()-t0)
            results['max_replay_rows'] = replay.max_rows
            if len(seen)==cfg.N_TASK:
                # Passive analysis of masks used during TRAINING. No task-
                # conditioned evaluation search or oracle query class group.
                masks=torch.zeros((len(seen),model.hidden_size),dtype=torch.bool)
                for t,indices in enumerate(results['training_active_atoms']):
                    masks[t,indices]=True
                intersection=(masks[:,None,:]&masks[None,:,:]).sum(-1).float()
                union=(masks[:,None,:]|masks[None,:,:]).sum(-1).clamp_min(1)
                results['task_mask_jaccard']=(intersection/union).cpu().tolist()
                if a.dataset=='cifar100':
                    import pickle
                    from scipy.stats import spearmanr
                    with bulk('data','cifar100','cifar-100-python','train').open('rb') as f:
                        hierarchy=pickle.load(f,encoding='latin1')
                    coarse=dict(zip(hierarchy['fine_labels'],hierarchy['coarse_labels']))
                    categories=[set(coarse[c] for c in g) for g in stream.groups]
                    expected=[]; observed=[]
                    for i in range(len(seen)):
                        for j in range(i):
                            expected.append(len(categories[i]&categories[j])/len(categories[i]|categories[j]))
                            observed.append(results['task_mask_jaccard'][i][j])
                    rho=float(spearmanr(expected,observed).statistic)
                    results['semantic_relatedness_mask_spearman']=rho if np.isfinite(rho) else None
            results['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
            results['optimizer_policy']=optimizer.policy()
            write_json(out/'progress.json', results)
            # Evaluation must not influence later training, including RNG.
        finally:
            if hasattr(model, 'set_reconfig_state'):
                model.set_reconfig_state(state)
            else:
                model.m.copy_(state['m'])
            model.train(was_training)
            torch.set_rng_state(torch_state)
            torch.cuda.set_rng_state(cuda_state)
        # Keep the final inference checkpoint; block metrics are persisted
        # above. Rewriting multi-GB model+Adam states every block would dominate
        # wall time on the shared filesystem. These are not resume checkpoints.
        if stream.current_task==cfg.N_TASK-1:
            temp = checkpoint.with_suffix('.partial')
            torch.save({'model':model.state_dict(),'config':config,
                        'completed_task':stream.current_task,'results':results,
                        'checkpoint_kind':'final inference weights'},temp)
            temp.replace(checkpoint)
        return cil

    runner.evaluate_backwards = evaluate
    collector = runner.metrics_mod.MetricsCollector()
    runner.train(model, stream, list(range(cfg.N_TASK)), cfg.N_EPOCHS, a.variant,
                 optimizer, replay, collector=collector, save_viz=False)
    results['wall_seconds'] = time.perf_counter()-start
    results['training_metrics'] = collector.summary()
    results['optimizer_policy']=optimizer.policy()
    results['status'] = 'complete'
    for key in ['class_il_matrix']:
        # Fill unused upper triangle with zero only for final metric reduction.
        matrix = [[0. if x is None else x for x in row] for row in results[key]]
        results[key.replace('_matrix','_metrics')] = summarize(matrix)
        counts = np.asarray(results['test_queries_per_block'][:len(matrix)], dtype=float)
        results[key.replace('_matrix','_metrics')]['final_micro_accuracy'] = float(np.average(matrix[-1], weights=counts))
    write_json(out/'result.json', results)
    print(json.dumps({'completed':name, 'class_il':results['class_il_metrics']}), flush=True)


if __name__ == '__main__':
    main()
