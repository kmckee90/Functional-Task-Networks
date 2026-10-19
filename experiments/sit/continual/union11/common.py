import json
from pathlib import Path
import torch
from ..model import BayesSiT
from ..ic4_recall.common import atomic

ASSETS=Path('/home/dev/sit_state')
ROOT=ASSETS/'union11_v2'
ART=Path(__file__).parent/'artifacts_v2'
NAMES=['monochrome','poster','sketch','George_W_Bush','Colin_Powell','Tony_Blair',
       'Whoopi_Goldberg','Bruce_Willis','Nicole_Kidman','Giant_anteater','BMW_i8']
FILTERS=NAMES[:3]
FACES=NAMES[3:9]
SOURCE={n:('filters' if i<3 else 'lfw' if i<6 else 'distinct' if i<9 else 'anteater') for i,n in enumerate(NAMES)}
DATA={'filters':ASSETS,'lfw':ASSETS/'identity_lfw','distinct':ASSETS/'identity_distinct','anteater':ASSETS/'mixed_anteater'}
DATA_ROOT=ROOT

def names():return list(NAMES)

def data():
    return {n:{s:{k:v.cuda() for k,v in torch.load(DATA[SOURCE[n]]/f'{n}_{s}.pt',weights_only=True).items()}
               for s in ['train','support','test']} for n in NAMES}

def model(seed=125,plain=False):
    assert not plain
    torch.set_num_threads(4);torch.manual_seed(seed);torch.backends.cuda.matmul.allow_tf32=True
    return BayesSiT(ASSETS/'SiT-XL-2-256.pt',H=1024,k=256,rank=1,alpha=64.,reactive_low=True).cuda().eval()

def features(n,split,branch):
    suite=SOURCE[n]
    root=ASSETS/'mixed_anteater/ic4' if suite=='anteater' else ASSETS/'ic4_recall'
    return torch.load(root/'features'/suite/f'{n}_{split}_{branch}.pt',weights_only=True)

def unified(n,split):
    return torch.cat([torch.nn.functional.normalize(features(n,split,b),dim=1)
                      for b in ['dino','style','face']],dim=1)/(3**.5)

def images(n,split):
    if SOURCE[n]=='anteater':
        from PIL import Image
        manifest=json.loads((DATA['anteater']/'manifest.json').read_text())
        return [Image.open(DATA['anteater']/r['image']).convert('RGB') for r in manifest['splits'][n][split]]
    from ..ic4_recall.common import image_rows
    return image_rows(SOURCE[n],n,split)

def generate(m,vae,mask,count=64,seed=99233,domain=None):
    from ..ic4_recall.search import generate as sample
    return sample('filters' if domain in FILTERS else 'union',m,vae,mask,count,seed)

# The copied lifecycle's old evaluation functions are superseded by stage_eval.
def evaluate(*args,**kwargs):raise RuntimeError('Use unified stage evaluator')
