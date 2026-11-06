"""Run the authors' SPG under its native task-incremental protocol."""
import argparse
import inspect
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from storage import WORKSPACE, bulk, write_json
from vision_data import N_CLASSES, raw_datasets, split_indices


class TaskLabels(Dataset):
    def __init__(self, base, indexes, classes):
        self.base, self.indexes = base, list(map(int,indexes))
        self.mapping = {int(c):i for i,c in enumerate(classes)}
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self,i):
        x,y = self.base[self.indexes[i]]
        return x,self.mapping[int(y)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=N_CLASSES,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)
    sys.path.insert(0,str(WORKSPACE/'third_party/SPG'))
    from approaches.spg.appr_spg import Appr
    # PyTorch removed only the scheduler's verbose presentation argument.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    if 'verbose' not in inspect.signature(scheduler).parameters:
        def compatible_scheduler(*args,verbose=None,**kwargs):
            return scheduler(*args,**kwargs)
        torch.optim.lr_scheduler.ReduceLROnPlateau = compatible_scheduler
    size=32  # authors' AlexNet recipes resize CIFAR and TinyImageNet to 32px
    train,test,groups=raw_datasets(a.dataset,a.seed,size)
    training,validation=split_indices(train.targets,groups,a.seed)
    config=yaml.safe_load((WORKSPACE/'third_party/SPG/conf/appr/spg.yaml').read_text())
    recipe = 'tinyimagenet_10' if a.dataset in ('tinyimagenet','core50') else 'cifar100_10'
    pars={'device':'cuda','list__ncls':[len(g) for g in groups], 'inputsize':(3,size,size),
          'lr':.25,'lr_factor':3,'lr_min':1e-3,'epochs_max':1 if a.smoke else 500,
          'patience_max':5,'backbone':'alexnet','nhid':2048,'lamb':0,
          **config['tuned'][recipe]}
    appr=Appr(pars)
    run_id=f'{a.dataset}/SPG/seed{a.seed}'+('_smoke' if a.smoke else '')
    out=bulk('runs','spg',run_id)
    out.mkdir(parents=True,exist_ok=True)
    results={'method':'SPG','dataset':a.dataset,'seed':a.seed,'smoke':a.smoke,
             'protocol':'native task-incremental (oracle task ID)', 'recipe':pars,
             'status':'running','task_il_matrix':[],'task_training':[],
             'trainable_parameters':sum(p.numel() for p in appr.model.parameters() if p.requires_grad),
             'total_parameters':sum(p.numel() for p in appr.model.parameters())}
    test_loaders=[]
    t0=time.perf_counter()
    for task,group in enumerate(groups[:2] if a.smoke else groups):
        ti=training[task][:128] if a.smoke else training[task]
        vi=validation[task][:64] if a.smoke else validation[task]
        train_loader=DataLoader(TaskLabels(train,ti,group),batch_size=64,shuffle=True,num_workers=2)
        val_loader=DataLoader(TaskLabels(train,vi,group),batch_size=64,num_workers=2)
        indexes=np.flatnonzero(np.isin(test.targets,group))
        if a.smoke: indexes=indexes[:64]
        test_loaders.append(DataLoader(TaskLabels(test,indexes,group),batch_size=64,num_workers=2))
        results['task_training'].append(appr.train(task,train_loader,val_loader))
        appr.complete_learning(task,dl_train=train_loader)
        rows=[appr.test(j,loader)['acc_test'] for j,loader in enumerate(test_loaders)]
        results['task_il_matrix'].append(list(map(float,rows)))
        write_json(out/'progress.json',results)
        if task==len(groups)-1 or a.smoke:
            path=bulk('checkpoints','spg',run_id,'latest.pt')
            temp=path.with_suffix('.partial')
            torch.save({'model':appr.model.state_dict(),'completed_task':task},temp)
            temp.replace(path)
    from run_ftn import summarize
    rows=results['task_il_matrix']
    matrix=[r+[0.]*(len(rows)-len(r)) for r in rows]
    results['task_il_metrics']=summarize(matrix)
    results['wall_seconds']=time.perf_counter()-t0
    results['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
    results['status']='complete'
    write_json(out/'result.json',results)
    print(results['task_il_metrics'],flush=True)


if __name__=='__main__': main()
