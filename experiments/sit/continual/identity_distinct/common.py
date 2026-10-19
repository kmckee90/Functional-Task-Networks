"""Shared configuration and sampling for the isolated identity study."""
import hashlib
import json
from pathlib import Path
import torch
from diffusers import AutoencoderKL
from ..model import BayesSiT
from ..ic4_recall.common import ASSETS, atomic
ROOT = ASSETS/'identity_distinct'
ART = Path(__file__).parent/'artifacts'


def names(): return json.loads((ROOT/'prepared.json').read_text())['names']


def model(seed=123, plain=False):
    torch.set_num_threads(4); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=True
    return BayesSiT(ASSETS/'SiT-XL-2-256.pt',H=1 if plain else 64,k=1 if plain else 16,
                   rank=16 if plain else 1,alpha=.25 if plain else 4.,reactive_low=True).cuda().eval()


def data():
    return {n:{s:{k:v.cuda() for k,v in torch.load(ROOT/f'{n}_{s}.pt',weights_only=True).items()}
               for s in ['train','support','test']} for n in names()}


@torch.no_grad()
def generate(m, vae, mask, count=64, seed=99233, label=1000):
    g=torch.Generator(device='cuda').manual_seed(seed)
    noise=torch.randn(count,4,32,32,device='cuda',generator=g)
    m.controller.mask=mask
    images=[]
    for start in range(0,count,16):
        x=noise[start:start+16]; y=torch.full((len(x),),label,device='cuda')
        def v(x,t):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return m.backbone(x,torch.full((len(x),),t,device='cuda'),y).float()
        for step in range(50):
            velocity=v(x,step/50)
            x=x+.01*(velocity+v(x+.02*velocity,(step+1)/50))
        with torch.autocast('cuda',dtype=torch.bfloat16): out=vae.decode(x/.18215).sample
        images.append(((out.float()+1)/2).clamp(0,1).cpu())
    return torch.cat(images)


def protocol():
    sources=[Path(__file__).parent.parent/p for p in ['model.py','bayes_reconfig.py','bayesian_adv_config_snapshot.py','run.py']]
    p={'question':'Can sequentially post-trained shared LoRA atoms retain three specific people and be recalled from two/four reference photos?',
       'names':names(),'bank':{'H':64,'k':16,'rank':1,'alpha':4,'layers':112},
       'training':{'updates_per_person':1500,'batch':16,'lr':3e-4,'seeds':[123,124,125],
                   'reactive':'source BayesLoRAReactiveLow, detector SD3, EMA0.9, exact warmup11',
                   'optimizer':'Adam reset each known block, matching source; no forced mask resets',
                   'conditioning':'same frozen null embedding1000 for every person; no identity labels supplied',
                   'arms':['reactive3_noreplay','reactive3_replay'],
                   'replay':'Matched reactive3 behavior with mask-filtered replay toggled; replay arm is an explicit ablation, not literal source ReactiveLow preset.',
                   'controls':['plain active-rank16 sequential seed123','single-person FTN seed123, independently initialized for each identity']},
       'evaluation':{'support_budgets':[2,4,16],'support_draws':3,'mask_steps':30,
                     'query':'24 heldout images per person, four paired corruptions per image; report image-level means',
                     'IC':'pooled clean latent PCA rank8/threshold.5; block-end archive; no training prior or novelty; no repair in this study',
                     'generation':'Heun50, no classifier-free guidance, same null label, paired noise;64 final images/method/person;32 intermediate cold4 images',
                     'identity_metric':'Frozen MTCNN + VGGFace2 FaceNet; target accept rate includes non-detections as failures; calibrated on other identities, never generated images',
                     'novelty':'Pretraining membership cannot be certified. Test frozen zero-adapter and untrained bank support search before post-training; also probe human-related ImageNet labels.',
                     'limitations':['Only3 identities; public figures; pretraining membership unknown','24 query images; corruptions are not independent images','Perceptual duplicate filtering cannot ensure capture-event separation','Face evaluator may have seen these people; used for measurement only, never optimization']},
       'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    atomic(ROOT/'protocol.json',p); ART.mkdir(exist_ok=True); atomic(ART/'protocol.json',p)
    return p
