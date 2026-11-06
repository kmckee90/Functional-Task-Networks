"""Authors' MNTDP-D: native task-driven prior, path search, blocks, and freezing.

Candidate paths are trained serially with the original Ignite trainer rather
than with the authors' Ray/multiprocessing launcher. Each candidate starts from
an independent copy, and validation accuracy selects the path as in the source.
"""
import argparse
import copy
import time
import shutil

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from mntdp_compat import load_mntdp
from run_spg import TaskLabels
from storage import WORKSPACE, bulk, write_json
from vision_data import N_CLASSES, raw_datasets, split_indices


class NoDashboard:
    def __bool__(self): return False
    def text(self,*args,**kwargs): pass


def loss_fn(pred,target):
    return torch.nn.functional.cross_entropy(pred[0] if isinstance(pred,(tuple,list)) else pred,target.flatten())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=N_CLASSES,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)
    if shutil.which('dot') is None:
        raise RuntimeError('Native MNTDP finish_task requires Graphviz; schedule this on bm1.')
    MNTDP,native_train=load_mntdp()
    config=yaml.safe_load((WORKSPACE/'third_party/MNTDP/configs/default/modules.yaml').read_text())
    pars=dict(config['pssn-resnet'])
    pars.update(strat='search_all',n_source_models=1,arch_loss_coef=0.,entropy_coef=0.,
                split_training=True,grid_params={},ray_resources={'cpu':1,'gpu':1})
    learner=MNTDP(**pars)
    # Authors' native residual backbone is a 32px model.
    train,test,groups=raw_datasets(a.dataset,a.seed,32)
    training,validation=split_indices(train.targets,groups,a.seed)
    run_id=f'{a.dataset}/MNTDP-D/seed{a.seed}'+('_smoke' if a.smoke else '')
    out=bulk('runs','mntdp',run_id)
    out.mkdir(parents=True,exist_ok=True)
    results={'method':'MNTDP-D','dataset':a.dataset,'seed':a.seed,'smoke':a.smoke,
             'protocol':'native task-incremental (oracle task ID)',
             'status':'running','recipe':pars,'task_il_matrix':[],'candidate_search':[]}
    test_loaders=[]
    t0=time.perf_counter()
    for task,group in enumerate(groups[:2] if a.smoke else groups):
        ti=training[task][:128] if a.smoke else training[task]
        vi=validation[task][:64] if a.smoke else validation[task]
        train_loader=DataLoader(TaskLabels(train,ti,group),batch_size=128,shuffle=True,num_workers=0)
        val_loader=DataLoader(TaskLabels(train,vi,group),batch_size=256,num_workers=0)
        test_idx=np.flatnonzero(np.isin(test.targets,group))
        if a.smoke: test_idx=test_idx[:64]
        test_loaders.append(DataLoader(TaskLabels(test,test_idx,group),batch_size=256,num_workers=0))
        search=learner.get_model(task,x_dim=(3,32,32),n_classes=[len(group)],
                    descriptor=None,dataset=(train_loader,val_loader))
        search.init_models()
        best_score=-1.
        best_weights=None
        best_index=None
        task_search=[]
        for path,index in search.models_idx.items():
            candidate=copy.deepcopy(search.models[index]).cuda()
            optimizer=torch.optim.Adam([p for p in candidate.parameters() if p.requires_grad],lr=1e-3)
            steps,metrics,best=native_train(candidate,train_loader,[train_loader,val_loader],optimizer,loss_fn,
                n_it_max=2 if a.smoke else 10000,patience=100,split_names=['Train','Val'],
                device='cuda',log_steps=[1] if a.smoke else [1,2,3,4,5],log_epoch=False)
            search.res[path]=(steps,metrics,best)
            task_search.append({'path':str(path),'steps':steps,'validation_accuracy':float(best['value'])})
            if best['value']>best_score:
                best_score=float(best['value'])
                best_weights=copy.deepcopy(best['state_dict'])
                best_index=index
            del candidate,optimizer
            torch.cuda.empty_cache()
        search.models[best_index].load_state_dict(best_weights)
        results['candidate_search'].append(task_search)
        figures=out/f'graph_task{task}'
        figures.mkdir(exist_ok=True)
        learner.finish_task(TaskLabels(train,ti,group),task,viz=NoDashboard(),path=str(figures))
        accuracies=[]
        for j,loader in enumerate(test_loaders):
            network=learner.get_model(j).get_frozen_model().cuda().eval()
            correct,total=0,0
            with torch.no_grad():
                for x,y in loader:
                    pred=network(x.cuda())
                    if isinstance(pred,(tuple,list)): pred=pred[0]
                    correct+=int((pred.argmax(-1)==y.cuda()).sum())
                    total+=len(y)
            accuracies.append(correct/total)
        results['task_il_matrix'].append(accuracies)
        results['total_parameters']=int(learner.n_params(task))
        write_json(out/'progress.json',results)
        if task==len(groups)-1 or a.smoke:
            path=bulk('checkpoints','mntdp',run_id,'latest.pt')
            temp=path.with_suffix('.partial')
            torch.save({'learner':learner,'completed_task':task},temp)
            temp.replace(path)
    from run_ftn import summarize
    rows=results['task_il_matrix']
    results['task_il_metrics']=summarize([r+[0.]*(len(rows)-len(r)) for r in rows])
    results['wall_seconds']=time.perf_counter()-t0
    results['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
    results['status']='complete'
    write_json(out/'result.json',results)
    print(results['task_il_metrics'],flush=True)


if __name__=='__main__': main()
