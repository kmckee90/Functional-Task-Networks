"""The only training-data gateway: active block or retained memory."""
import torch
from pathlib import Path
from ..union11 import common as base
from ..union11.common import ASSETS,NAMES,FILTERS,FACES,SOURCE,DATA,atomic,model,generate,images
ROOT=ASSETS/'memory_budget_v1'
ART=Path(__file__).parent/'artifacts'
BUDGETS=[50,25,10,5]
MEMORY={}
ACTIVE=None
ACCESSES=[]

def features(n,split,branch):
    if split=='train':
        assert n in MEMORY, f'Forbidden full historical train-feature read: {n}'
        ACCESSES.append(dict(task=n,kind='retained_features',count=len(MEMORY[n]['ids'])))
        return MEMORY[n]['features'][branch]
    assert split=='support'
    return base.features(n,split,branch)

def unified(n,split):
    return torch.cat([torch.nn.functional.normalize(features(n,split,b),dim=1)
                      for b in ['dino','style','face']],dim=1)/(3**.5)

def open_block(n):
    global ACTIVE
    assert ACTIVE is None and n not in MEMORY
    assert n==NAMES[len(MEMORY)]
    ACTIVE=n
    data=torch.load(DATA[SOURCE[n]]/f'{n}_train.pt',weights_only=True,map_location='cpu')
    fs={b:base.features(n,'train',b).cpu() for b in ['dino','style','face']}
    ACCESSES.append(dict(task=n,kind='active_full_train',count=len(data['z'])))
    return {k:v.cuda() for k,v in data.items()},fs

def close_block(n,entry):
    global ACTIVE
    assert ACTIVE==n and n not in MEMORY
    MEMORY[n]=entry
    ACTIVE=None

def data():
    # Test/support are evaluation-only; future training sets are never loaded.
    return {n:{s:{k:v.cuda() for k,v in torch.load(DATA[SOURCE[n]]/f'{n}_{s}.pt',weights_only=True).items()}
               for s in ['test']} for n in NAMES}

def attach_retained(ds):
    assert ACTIVE is None
    for n,e in MEMORY.items():ds[n]['train']={k:e[k].cuda() for k in ['z','y']}
    assert {n for n in ds if 'train' in ds[n]}==set(MEMORY)
