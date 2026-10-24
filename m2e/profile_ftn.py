"""Bounded, isolated decoder benchmark. Run on a pod, never in a live agent."""
import argparse
import cProfile
from dataclasses import asdict
import io
import json
import pstats
import time

import torch
from agents.agent_ftn import RDDPG_Dual
from experiment_ftn import Config, Hyperparameters


def timed(fn, repeats=3):
    torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(repeats):fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)/repeats


def main():
    p=argparse.ArgumentParser();p.add_argument('--k',type=int,default=0)
    args=p.parse_args()
    original_threads=torch.get_num_threads();torch.set_num_threads(2)
    cfg={**asdict(Config()),**asdict(Hyperparameters()),'device':'cuda',
         'ftn_k':args.k or None,'ftn_replay_every':1000000}
    agent=RDDPG_Dual(2,2,(torch.full((2,),-1.,device='cuda'),torch.ones(2,device='cuda')),**cfg).cuda()
    n=cfg['minibatch_size']
    d=dict(o=torch.randn(n,agent.rnn_size_o,device='cuda'),f=torch.randn(n,agent.rnn_size_f,device='cuda'),
           no=torch.randn(n,agent.rnn_size_o,device='cuda'),nf=torch.randn(n,agent.rnn_size_f,device='cuda'),
           action=torch.rand(n,2,device='cuda')*2-1,ro=torch.ones(n,1,device='cuda'),rf=torch.ones(n,1,device='cuda'),
           done=torch.zeros(n,1,dtype=torch.bool,device='cuda'),goal=False)
    print(json.dumps(dict(spec=agent.ftn_spec,default_threads=original_threads)),flush=True)
    agent._train_support(d)
    result=dict(empty_archive_update_s=timed(lambda:agent._train_support(d)))
    masks=agent._masks();stored=agent._trim(d,cpu=True)
    entries=[]
    for _ in range(256):
        old={k:torch.zeros_like(m,device='cpu').scatter_(0,torch.randperm(len(m))[:agent.posteriors[k].k],1.) for k,m in masks.items()}
        entries.append(dict(masks=old,data=stored))

    def scan():
        eligible=[]
        for entry in agent.ftn_archive:
            overlaps=[float((entry['masks'][k].to(m.device)*m).sum())/agent.posteriors[k].k for k,m in masks.items()]
            if any(.2<=v<=.8 for v in overlaps):eligible.append(entry)
        return len(eligible)

    result['archive_scan_s']={}
    for count in (16,64,256):
        agent.ftn_archive=entries[:count]
        result['archive_scan_s'][count]=timed(scan)
    # Equivalent eligibility computation entirely on CPU: three mask transfers,
    # not three transfers and synchronizations per archive entry.
    cpu_masks={k:m.cpu() for k,m in masks.items()}
    def cpu_scan():
        overlaps=torch.stack([(torch.stack([e['masks'][k] for e in entries])*cpu_masks[k]).sum(-1)/agent.posteriors[k].k for k in masks],-1)
        return int(((overlaps>=.2)&(overlaps<=.8)).any(-1).sum())
    assert cpu_scan()==scan()
    result['batched_cpu_scan_s']=timed(cpu_scan)
    # Guarantee eligible retained supports so this measures actual extra replay.
    for entry in entries:
        for k,m in masks.items():
            ids=m.nonzero().flatten().cpu();off=(~m.bool()).nonzero().flatten().cpu()
            chosen=torch.cat((ids[:len(ids)//2],off[:len(ids)-len(ids)//2]))
            entry['masks'][k]=torch.zeros_like(m,device='cpu').scatter_(0,chosen,1.)
    result['full_archive_update_s']=timed(lambda:agent._train_support(d))
    profiler=cProfile.Profile();profiler.enable();agent._train_support(d);torch.cuda.synchronize();profiler.disable()
    out=io.StringIO();pstats.Stats(profiler,stream=out).sort_stats('cumtime').print_stats(18)
    print(out.getvalue(),flush=True)
    torch.set_num_threads(original_threads)
    result['full_archive_default_threads_s']=timed(lambda:agent._train_support(d))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
