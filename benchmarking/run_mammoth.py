"""Native DER++, L2P and DualPrompt via Mammoth's complete training pipeline.

Dataset adapters share the FTN class order and held-out calibration split.
MammothDatasetWrapper, store_masked_loaders, meta_* hooks, training recipes,
and the method implementations remain active.
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from storage import WORKSPACE, bulk, write_json
from vision_data import N_CLASSES, N_TASKS, raw_datasets, split_indices,feature_datasets


class IndexImages(Dataset):
    def __init__(self, base, indexes, transform, training):
        self.base = base
        self.data = np.array(indexes, dtype=np.int64)
        self.targets = np.asarray(base.targets)[self.data]
        self.transform, self.training = transform, training
        self.not_aug_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, _ = self.base[int(self.data[i])]
        target = int(self.targets[i])
        if self.training:
            return self.transform(image), target, self.not_aug_transform(image.copy())
        return self.transform(image), target


class IndexFeatures(IndexImages):
    def __getitem__(self,i):
        x=self.base.data[int(self.data[i])]
        y=int(self.targets[i])
        return (x,y,x.clone()) if self.training else (x,y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['derpp','l2p','dualprompt'], required=True)
    p.add_argument('--dataset', choices=N_CLASSES, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--buffer-size', type=int, default=500)
    p.add_argument('--matched-memory', action='store_true')
    p.add_argument('--features',action='store_true')
    p.add_argument('--head-atoms', type=int, default=4096,
                   help='Feature-head bank size; 256 matches FTN Base active parameters')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--timing', action='store_true', help='Two full-data blocks, one epoch each; excluded from comparisons')
    a = p.parse_args()
    assert not (a.smoke and a.timing)
    assert a.head_atoms > 0
    assert a.head_atoms == 4096 or (a.features and a.method == 'derpp')
    label = a.method
    if a.method == 'derpp':
        label='DER++-matched' if a.matched_memory else 'DER++-500'
        if a.matched_memory:
            a.buffer_size=100*256*N_TASKS[a.dataset]
        if a.features: label+='-features'
        if a.head_atoms != 4096: label += f'-h{a.head_atoms}'
    # All paths must be absolute: Mammoth changes cwd when importing packages.
    sys.path.insert(0, str(WORKSPACE/'third_party/mammoth'))
    import main as mammoth
    from datasets import register_dataset
    from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
    from datasets.utils import set_default_from_args
    from datasets.transforms.denormalization import DeNormalize
    from utils.training import train
    from backbone import MammothBackbone,register_backbone

    prompt = a.method != 'derpp'
    size = 224 if prompt else {'cifar10':32,'cifar100':32,'tinyimagenet':64,'core50':128}[a.dataset]
    encoder_parameters=0
    encoder=None
    if a.features:
        assert a.method=='derpp'
        base_train,base_test,groups,encoder_parameters,encoder=feature_datasets(a.dataset,a.seed)
        source=WORKSPACE/'sources/ftn'
        sys.path.insert(0,str(source))
        from ftn import ParallelMLP
        class MatchedBank(MammothBackbone):
            def __init__(self,num_classes):
                super().__init__()
                self.parallel=ParallelMLP(768,1,a.head_atoms,layers=8,inner_size=8)
                self.classifier=torch.nn.Linear(a.head_atoms,num_classes,bias=False)
            def forward(self,x,returnt='out'):
                features=self.parallel(x)
                out=self.classifier(features)
                if returnt=='features': return features
                if returnt in ('both','full','all'): return out,features
                return out
        @register_backbone('bm-ftn-bank')
        def matched_bank(num_classes:int):
            return MatchedBank(num_classes)
    else:
        base_train, base_test, groups = raw_datasets(a.dataset, a.seed, size)
    # The shared dataset supplies original PIL images to native augmentation.
    base_train.transform = transforms.Lambda(lambda image:image)
    base_test.transform = transforms.Lambda(lambda image:image)
    train_indices, _ = split_indices(base_train.targets, groups, a.seed)
    train_indices = np.concatenate(train_indices)
    # Default native CIFAR/prompt preprocessing and augmentation recipes.
    mean, std = ((0.,0.,0.),(1.,1.,1.)) if prompt else ((.485,.456,.406),(.229,.224,.225))
    if a.features:
        augmentation=test_transform=torch.nn.Identity()
    elif prompt:
        augmentation = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean,std)])
        test_transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean,std)])
    else:
        augmentation = transforms.Compose([transforms.RandomCrop(size,padding=4), transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(), transforms.Normalize(mean,std)])
        test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])

    class SharedVision(ContinualDataset):
        NAME = 'bm-' + a.dataset
        SETTING = 'class-il'
        N_CLASSES = N_CLASSES[a.dataset]
        N_TASKS = N_TASKS[a.dataset]
        N_CLASSES_PER_TASK = [len(g) for g in groups] if a.dataset == 'core50' else len(groups[0])
        SIZE = (size,size)
        MEAN, STD = mean, std
        TRANSFORM, TEST_TRANSFORM = augmentation, test_transform
        def __init__(self,args):
            super().__init__(args)
            # The upstream custom-order parser produces a list; its standard
            # splitter indexes that mapping with NumPy arrays.
            self.args.class_order=np.asarray(self.args.class_order,dtype=np.int64)
        def get_data_loaders(self):
            wrapper=IndexFeatures if a.features else IndexImages
            return store_masked_loaders(wrapper(base_train,train_indices,augmentation,True),
                    wrapper(base_test,np.arange(len(base_test)),test_transform,False),self)
        @staticmethod
        @set_default_from_args('backbone')
        def get_backbone():
            return 'resnet18'
        @staticmethod
        def get_transform():
            if a.features: return torch.nn.Identity()
            return transforms.Compose([transforms.ToPILImage(),augmentation])
        @staticmethod
        def get_loss():
            return torch.nn.functional.cross_entropy
        @staticmethod
        def get_normalization_transform():
            if a.features: return torch.nn.Identity()
            return transforms.Normalize(mean,std)
        @staticmethod
        def get_denormalization_transform():
            if a.features: return torch.nn.Identity()
            return DeNormalize(mean,std)
        def get_class_names(self):
            return [str(i) for i in range(self.N_CLASSES)]

    register_dataset(SharedVision.NAME)(SharedVision)
    run_id = f'{a.dataset}/{label}/seed{a.seed}' + ('_smoke' if a.smoke else '_timing' if a.timing else '')
    output = bulk('runs','mammoth',run_id)
    output.mkdir(parents=True,exist_ok=True)
    command = ['--model',a.method,'--dataset',SharedVision.NAME,
               '--seed',str(a.seed),'--base_path',str(bulk('data','mammoth'))+'/',
               '--results_path',str(output/'native_results'),
               '--checkpoint_path',str(bulk('checkpoints','mammoth',run_id)),
               '--num_workers','4','--non_verbose','1',
               '--backbone','bm-ftn-bank' if a.features else 'resnet18','--device','0', '--code_optimization','0',
               '--custom_class_order',','.join(map(str,np.argsort([c for g in groups for c in g]))),
               '--permute_classes','1',
               '--batch_size','128' if prompt else '32',
               '--lr','0.03','--n_epochs','5' if prompt else '50',
               '--enable_other_metrics','1', '--savecheck','last']
    if a.method == 'derpp':
        command += ['--buffer_size',str(a.buffer_size),'--minibatch_size','32','--alpha','0.3','--beta','0.5']
    if prompt:
        command += ['--batchwise_prompt','0']
    if a.smoke:
        command[command.index('--n_epochs')+1] = '1'
        command += ['--debug_mode','1','--stop_after','2']
    if a.timing:
        command[command.index('--n_epochs')+1] = '1'
        command += ['--stop_after','2']
    sys.argv = ['main.py'] + command
    args = mammoth.parse_args(verbose=False)
    model, dataset, args = mammoth.initialize(args)
    results = {'method':label,'dataset':a.dataset,'seed':a.seed,'smoke':a.smoke or a.timing,
               'timing':a.timing,'epoch_timings':[],
               'test_queries_per_block':[int(np.isin(base_test.targets,g).sum()) for g in groups],
               'class_il_matrix':{},'task_il_matrix':{},'status':'running',
               'recipe':command,'groups':groups,'eval_seconds':0.,
               'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
               'total_parameters':sum(p.numel() for p in model.parameters())+encoder_parameters,
               'encoder':encoder,'shared_frozen_encoder_parameters':encoder_parameters}
    if a.timing:
        epoch_clock = {}
        original_begin = model.meta_begin_epoch
        original_end = model.meta_end_epoch
        @functools.wraps(original_begin)
        def timed_begin(epoch, dataset):
            original_begin(epoch, dataset)
            torch.cuda.synchronize()
            epoch_clock['start'] = time.perf_counter()
        @functools.wraps(original_end)
        def timed_end(epoch, dataset):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - epoch_clock['start']
            results['epoch_timings'].append({'epoch':epoch,'seconds':elapsed,
                'training_rows':len(dataset.train_loader.dataset),
                'training_batches':len(dataset.train_loader)})
            original_end(epoch, dataset)
            write_json(output/'timing.json', results['epoch_timings'])
        model.meta_begin_epoch = timed_begin
        model.meta_end_epoch = timed_end
    original_evaluate = dataset.eval_fn
    @functools.wraps(original_evaluate)
    def record_eval(*args, **kwargs):
        start = time.perf_counter()
        value = original_evaluate(*args,**kwargs)
        results['eval_seconds'] += time.perf_counter()-start
        if len(dataset.test_loaders):
            k = len(dataset.test_loaders)-1
            results['class_il_matrix'][k] = list(map(lambda x:float(x)/100.,value[0]))
            results['task_il_matrix'][k] = list(map(lambda x:float(x)/100.,value[1]))
            write_json(output/'progress.json',results)
        return value
    dataset.eval_fn = record_eval
    start = time.perf_counter()
    train(model,dataset,args)
    results['wall_seconds'] = time.perf_counter()-start
    results['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
    results['status'] = 'complete'
    # Importing run_ftn only here avoids loading any FTN absolute imports.
    from run_ftn import summarize
    for name in ['class_il','task_il']:
        by_row = results[name+'_matrix']
        rows = [by_row[i]+[0.]*(len(by_row)-len(by_row[i])) for i in sorted(by_row)]
        results[name+'_matrix'] = rows
        results[name+'_metrics'] = summarize(rows)
        counts = np.asarray(results['test_queries_per_block'][:len(rows)], dtype=float)
        results[name+'_metrics']['final_micro_accuracy'] = float(np.average(rows[-1], weights=counts))
    write_json(output/'result.json',results)
    print(json.dumps({'complete':run_id,'metrics':results['class_il_metrics']}),flush=True)


if __name__ == '__main__':
    main()
