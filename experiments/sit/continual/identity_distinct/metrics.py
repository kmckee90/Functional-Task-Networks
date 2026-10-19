"""Identity evaluation only; never used for training, masks, or sample selection."""
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from .common import ROOT


class IdentityMetric:
    def __init__(self):
        self.detector=MTCNN(image_size=160,margin=14,keep_all=False,device='cuda')
        self.net=InceptionResnetV1(pretrained='vggface2').cuda().eval().requires_grad_(False)
        self.ref=torch.load(ROOT/'identity_metric.pt',weights_only=False)

    @torch.no_grad()
    def __call__(self, images):
        faces=[]; indices=[]
        for i,x in enumerate(images):
            im=Image.fromarray((x.permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8'))
            face=self.detector(im)
            if face is not None: faces.append(face); indices.append(i)
        embeddings=torch.zeros(len(images),512)
        for start in range(0,len(faces),32):
            e=self.net(torch.stack(faces[start:start+32]).cuda()).cpu()
            embeddings[indices[start:start+len(e)]]=e
        scores=embeddings@self.ref['gallery'].T
        detected=embeddings.norm(dim=1)>0
        pred=scores.argmax(1); accepted=scores.max(1).values>=self.ref['threshold']
        result={'count':len(images),'face_detection_rate':float(detected.float().mean()),'targets':{},
                'per_image_cosines':scores.tolist(),'detected':detected.tolist()}
        for j,n in enumerate(self.ref['names']):
            result['targets'][n]={'accepted_identity_rate':float((detected&accepted&(pred==j)).float().mean()),
                'mean_cosine_all':float(scores[:,j].mean()),
                'mean_cosine_detected':float(scores[detected,j].mean()) if detected.any() else None}
        return result
