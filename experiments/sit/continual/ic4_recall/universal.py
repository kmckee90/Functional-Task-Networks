"""Reusable frozen image encoder and checkpoint-compatible distributed atom mapper.

No dataset name is supplied to encoding or prediction. Legacy experiment CLIs
remain in the recovered modules; this is the direct API for new callers.
"""
import os
import torch
from torchvision import transforms
from .common import ROOT
from .recognition import style_features
from .search import dino_model
from .atom_conditioner import AtomReadout

BRANCHES=('dino','style','face')

def combine(branches):
    return torch.cat([torch.nn.functional.normalize(branches[b],dim=-1) for b in BRANCHES],dim=-1)/(3**.5)

class FrozenEncoder:
    def __init__(self):
        os.environ['TORCH_HOME']=str(ROOT/'torch')
        from facenet_pytorch import MTCNN,InceptionResnetV1
        self.dino=dino_model()
        self.detector=MTCNN(image_size=160,margin=14,keep_all=False,device='cuda')
        self.face=InceptionResnetV1(pretrained='casia-webface').cuda().eval().requires_grad_(False)
        self.prep=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                     transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

    @torch.no_grad()
    def branches(self,images):
        images=[im.convert('RGB') for im in images]
        semantic=[];faces=[]
        for start in range(0,len(images),32):
            x=torch.stack([self.prep(im) for im in images[start:start+32]]).cuda()
            semantic.append(torch.nn.functional.normalize(self.dino(x).float(),dim=1).cpu())
        for im in images:
            face=self.detector(im)
            if face is None:face=(transforms.ToTensor()(im.resize((160,160))).cuda()*255-127.5)/128
            faces.append(face.cpu())
        values=torch.cat([self.face(torch.stack(faces[i:i+32]).cuda()).cpu() for i in range(0,len(faces),32)])
        return dict(dino=torch.cat(semantic),style=style_features(images),face=values)

    def __call__(self,images):return combine(self.branches(images))


def load_readout(path,device='cuda'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    return AtomReadout(ck['projection'].to(device),ck['weight'].to(device)).eval()


def fit_readout(groups,teachers,seed=93811,episodes=512,regularization=.01,device='cuda'):
    """Same retained-memory fit as memory_budget.evaluate.fit, with explicit data inputs.

    groups maps association keys to already-combined image features. Keys are
    used only to pair features with supervision; prediction takes images alone.
    The caller must supply only data allowed by its retained-memory policy.
    """
    if not groups:raise ValueError('At least one supervised association is required')
    if set(groups)!=set(teachers):raise ValueError('Feature groups and teacher masks must match')
    dim=next(iter(groups.values())).shape[1]
    h=len(next(iter(teachers.values())))
    gen=torch.Generator().manual_seed(seed)
    projection=torch.randn(dim,128,generator=gen).to(device)
    model=AtomReadout(projection,torch.zeros(dim+129,h,device=device))
    gram=torch.eye(dim+129,device=device,dtype=torch.float64)*regularization
    rhs=torch.zeros(dim+129,h,device=device,dtype=torch.float64)
    for key,x in groups.items():
        if len(x)<4:raise ValueError('Each group must contain at least four images')
        ids=torch.stack([torch.randperm(len(x),generator=gen)[:4] for _ in range(episodes)])
        design=model.features(x[ids].to(device)).double()
        target=(torch.as_tensor(teachers[key],device=device)*2-1).double().expand(episodes,-1)
        gram+=design.T@design;rhs+=design.T@target
    model.weight.copy_(torch.linalg.solve(gram,rhs).float())
    return model.eval()
