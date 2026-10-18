"""Read-only, reproducible cached-Q&A retention probe; execute on the pod."""
import argparse
import dataclasses
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from config import CFG
from model import build, load_bank
from data import format_distill, load_qa
from input_routing import InputRoutingArchive
from bayes_configurer import BayesianPerplexityConfigurer, ReactiveState
from spatial import kwta
from train import _gather_answer_logits


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--samples',type=int,default=64)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    CFG.ckpt_dir=Path(args.checkpoint)
    state=json.loads((CFG.ckpt_dir/'state.json').read_text())
    CFG.grid_dim=math.isqrt(state['H']);CFG.k=state['k'];CFG.rank=state['rank']
    CFG.lora_layer_frac=state['layer_frac']
    for k,v in state['lateral'].items():setattr(CFG,k,v)
    torch.set_default_device('cuda');torch.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model,tok,ctrl,wrappers,_=build(CFG)
    assert len(wrappers)==state['n_wrappers']
    load_bank(wrappers,CFG.ckpt_dir/'bank.pt')
    model.eval();model.requires_grad_(False);model.gradient_checkpointing_disable()
    model.config.use_cache=False
    router=InputRoutingArchive(CFG,tok,model)
    router.load_state_dict(torch.load(CFG.ckpt_dir/'routing.pt',map_location='cpu',weights_only=True))
    posterior=torch.load(CFG.ckpt_dir/'posterior.pt',map_location='cpu',weights_only=True)
    cfgr=BayesianPerplexityConfigurer(CFG,ctrl,state['reactive'])
    zero=torch.zeros(ctrl.H,device=ctrl.mask.device)
    protocol={'checkpoint':str(CFG.ckpt_dir),'checkpoint_state':state,
        'config':dataclasses.asdict(CFG),'config_provenance':'Saved state overrides geometry and lateral settings; other settings use current project config, consistent with serving. No complete historical config snapshot available.',
        'routing':'Direct input-conditioned retrieval; on a miss use saved posterior/detector and question-only Bayes fallback. Reset state before each independent question.',
        'sampling_seed':20261015,'samples_per_topic':args.samples,
        'split':'Deterministic cached training-Q&A sample excluding stored routing exemplar questions. NOT held-out from bank training or lexical routing statistics.',
        'aggregation':'Answer-token-weighted CE, perplexity, token accuracy, and clipped forward KL*T^2 + ce_weight*CE. No generation accuracy is claimed.',
        'source_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['config.py','model.py','data.py','input_routing.py','bayes_configurer.py','eval_retention_ic.py']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2,default=str))
    records=[];start=time.time()
    @torch.no_grad()
    def logits(ex,mask,teacher=False):
        ctrl.set_mask(mask)
        ids=ex['t_ids' if teacher else 'ids'].unsqueeze(0).to(ctrl.mask.device)
        labels=ex['t_lbl' if teacher else 'lbl'].unsqueeze(0).to(ctrl.mask.device)
        return _gather_answer_logits(model(input_ids=ids).logits,labels)
    def metrics(student,targets,teacher):
        # Chunk the vocabulary-distribution calculations to bound GPU memory.
        sums={'ce':0.,'kl':0.,'objective':0.,'accuracy':0.}
        for s,t,y in zip(student.split(32),teacher.split(32),targets.split(32)):
            ce=F.cross_entropy(s.float(),y,reduction='none')
            sl=F.log_softmax(s.float()/CFG.kl_temp,-1)
            tl=F.log_softmax(t.float()/CFG.kl_temp,-1)
            kl=(tl.exp()*(tl-sl)).sum(-1)
            clipped=kl.clamp(max=CFG.kl_clip) if CFG.kl_clip>0 else kl
            sums['ce']+=ce.sum().item();sums['kl']+=kl.sum().item()
            sums['objective']+=(clipped*CFG.kl_temp**2+CFG.ce_weight*ce).sum().item()
            sums['accuracy']+=(s.argmax(-1)==y).sum().item()
        return {k:v/len(targets) for k,v in sums.items()}
    with (out/'rows.jsonl').open('w',buffering=1) as log:
        for task in state['trained']:
            data=load_qa(CFG,task)
            exemplars={q for r in router.routes.values() for q in r['questions']}
            pool=[i for i,t in enumerate(data) if t['question'] not in exemplars]
            selected=random.Random(20261015+sum(map(ord,task))).sample(pool,min(args.samples,len(pool)))
            oracle=kwta(router.routes[task]['logits'],CFG.k).to(ctrl.mask.device)
            for j,i in enumerate(selected):
                t=data[i];ex=format_distill(t['question'],t['answer'],t.get('context'),tok,CFG)
                torch.manual_seed(CFG.seed+i)
                cfgr.load_state_dict(posterior);cfgr.detector=ReactiveState(state['reactive'])
                mask,info=router.match(t['question'])
                if mask is None:
                    mask,_,_=cfgr(model,ex['recfg'].unsqueeze(0).to(ctrl.mask.device))
                mask=mask.to(ctrl.mask.device)
                teacher,y=logits(ex,zero,True)
                row={'task':task,'index':i,'question':t['question'],'question_sha256':hashlib.sha256(t['question'].encode()).hexdigest(),
                    'tokens':len(y),'route':info,'metrics':{}}
                row['teacher_ce']=metrics(teacher,y,teacher)['ce']
                for name,m in [('base',zero),('ic',mask),('oracle',oracle)]:
                    if name=='oracle' and torch.equal(mask,oracle):
                        row['metrics'][name]=row['metrics']['ic'].copy();continue
                    s,sy=logits(ex,m);assert torch.equal(y,sy)
                    row['metrics'][name]=metrics(s,sy,teacher);del s
                del teacher
                records.append(row);log.write(json.dumps(row)+'\n')
                print(task,j+1,'/',len(selected),'route',info.get('task'),info['reason'],'CE',round(row['metrics']['ic']['ce'],4),'seconds',round(time.time()-start),flush=True)
    summary={}
    for task in state['trained']:
        rows=[r for r in records if r['task']==task];n=sum(r['tokens'] for r in rows)
        summary[task]={'questions':len(rows),'tokens':n,'route_correct':sum(r['route']['reason']=='matched' and r['route']['task']==task for r in rows),
            'route_misses':sum(r['route']['reason']!='matched' for r in rows),'teacher_ce':sum(r['tokens']*r['teacher_ce'] for r in rows)/n,'methods':{}}
        for method in ['base','ic','oracle']:
            m={k:sum(r['tokens']*r['metrics'][method][k] for r in rows)/n for k in ['ce','kl','objective','accuracy']}
            m['perplexity']=math.exp(m['ce']);summary[task]['methods'][method]=m
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
