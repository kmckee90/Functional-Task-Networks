"""Generate closed-book answers for a fixed subset of the retention probe."""
import dataclasses
import json
import math
import os
import time
from pathlib import Path
import torch
from config import CFG
from model import build,load_bank
from data import load_qa
from input_routing import InputRoutingArchive
from bayes_configurer import BayesianPerplexityConfigurer,ReactiveState
from spatial import kwta

ROOT=Path('/home/dev/ftn_chat_state/evaluations/qa_ic_20261015')
OUT=ROOT/os.environ.get('FTN_GEN_OUTPUT','generation')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    CFG.ckpt_dir=Path('/home/dev/ftn_chat_state/checkpoints_bayes_qa_ic')
    state=json.loads((CFG.ckpt_dir/'state.json').read_text())
    CFG.grid_dim=math.isqrt(state['H']);CFG.k=state['k'];CFG.rank=state['rank'];CFG.lora_layer_frac=state['layer_frac']
    for k,v in state['lateral'].items():setattr(CFG,k,v)
    torch.set_default_device('cuda');torch.manual_seed(CFG.seed)
    model,tok,ctrl,wrappers,_=build(CFG);load_bank(wrappers,CFG.ckpt_dir/'bank.pt')
    model.eval();model.requires_grad_(False);model.gradient_checkpointing_disable();model.config.use_cache=True
    router=InputRoutingArchive(CFG,tok,model)
    router.load_state_dict(torch.load(CFG.ckpt_dir/'routing.pt',map_location='cpu',weights_only=True))
    posterior=torch.load(CFG.ckpt_dir/'posterior.pt',map_location='cpu',weights_only=True)
    cfgr=BayesianPerplexityConfigurer(CFG,ctrl,state['reactive'])
    old=[json.loads(x) for x in (ROOT/'rows.jsonl').read_text().splitlines()]
    selected=[r for task in state['trained'] for r in [x for x in old if x['task']==task][:10]]
    selected=selected[int(os.environ.get('FTN_GEN_START','0')):int(os.environ.get('FTN_GEN_END','30'))]
    if os.environ.get('FTN_GEN_INDICES'):
        keep={int(x) for x in os.environ['FTN_GEN_INDICES'].split(',')}
        selected=[r for r in selected if r['index'] in keep]
    thinking=os.environ.get('FTN_GEN_THINKING','0')=='1'
    methods=os.environ.get('FTN_GEN_METHODS','base,ic,oracle').split(',')
    (OUT/'protocol.json').write_text(json.dumps(dict(config=dataclasses.asdict(CFG),
        thinking=thinking, evaluated_methods=methods, samples='First ten questions per topic in the preselected random retention sample; no selection by scores.',
        generation='Greedy, thinking flag as recorded, max_new_tokens=384. Only system instruction and question in prompt. No source or reference answer.',
        reset='Saved posterior and detector restored independently for each question; direct IC with question-only fallback.',
        methods='Frozen base, retrieved IC, oracle topic mask only when different from retrieved mask.',
        split='Training-cache recall, not held-out generalization.'),default=str,indent=2))
    @torch.no_grad()
    def generate(q,mask):
        ctrl.set_mask(mask)
        enc=tok.apply_chat_template([{'role':'system','content':CFG.qa_sys},{'role':'user','content':q}],
            tokenize=True,add_generation_prompt=True,enable_thinking=thinking,return_tensors='pt',return_dict=True).to(ctrl.mask.device)
        result=model.generate(**enc,max_new_tokens=384,do_sample=False,pad_token_id=tok.eos_token_id)
        answer=result[0,enc['input_ids'].shape[1]:]
        return dict(text=tok.decode(answer,skip_special_tokens=True),tokens=len(answer),hit_token_limit=len(answer)>=384)
    start=time.time()
    with (OUT/'answers.jsonl').open('w',buffering=1) as f:
        for n,r in enumerate(selected):
            task,i=r['task'],r['index'];triple=load_qa(CFG,task)[i]
            assert triple['question']==r['question']
            torch.manual_seed(CFG.seed+i);cfgr.load_state_dict(posterior);cfgr.detector=ReactiveState(state['reactive'])
            mask,info=router.match(triple['question'])
            if mask is None:
                ids=tok(triple['question'],add_special_tokens=False,return_tensors='pt')['input_ids'][:,:CFG.qa_max_len].to(ctrl.mask.device)
                mask,_,_=cfgr(model,ids)
            mask=mask.to(ctrl.mask.device);oracle=kwta(router.routes[task]['logits'],CFG.k).to(ctrl.mask.device)
            row=dict(task=task,index=i,question=triple['question'],reference=triple['answer'],context=triple.get('context'),route=info,answers={})
            for name,m in [('base',torch.zeros_like(mask)),('ic',mask),('oracle',oracle)]:
                if name not in methods:continue
                if name=='oracle' and torch.equal(mask,oracle):row['answers'][name]=row['answers']['ic'];continue
                row['answers'][name]=generate(triple['question'],m)
            f.write(json.dumps(row)+'\n');print(n+1,'/30',task,i,'route',info['reason'],info.get('task'),'seconds',round(time.time()-start),flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
