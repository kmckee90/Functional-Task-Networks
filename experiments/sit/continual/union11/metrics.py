"""Independent frozen evaluators; never passed to mask search or readout fitting."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.models import ResNet50_Weights
from .common import ASSETS,FACES,FILTERS

class Metrics:
    def __init__(self):
        from facenet_pytorch import MTCNN,InceptionResnetV1
        from ..mixed_anteater.metrics import CLIPModel,CLIPProcessor,LABELS
        from cleanfid import fid
        self.detector=MTCNN(image_size=160,margin=14,keep_all=False,device='cuda')
        self.face=InceptionResnetV1(pretrained='vggface2').cuda().eval().requires_grad_(False)
        refs=[torch.load(ASSETS/n/'identity_metric.pt',weights_only=False) for n in ['identity_lfw','identity_distinct']]
        assert sum((r['names'] for r in refs),[])==FACES
        self.gallery=torch.cat([r['gallery'] for r in refs]).cpu()
        self.threshold=torch.tensor([r['threshold'] for r in refs for _ in range(3)])
        path=ASSETS/'ic4_recall/original_assets/clip'
        self.clip=CLIPModel.from_pretrained(path).cuda().eval().requires_grad_(False)
        self.proc=CLIPProcessor.from_pretrained(path)
        self.labels=LABELS+['a '+n for n in ResNet50_Weights.DEFAULT.meta['categories']]
        with torch.no_grad():
            self.text=torch.cat([F.normalize(self.clip.get_text_features(**self.proc(
                text=['a photo of '+n for n in self.labels[i:i+128]],padding=True,return_tensors='pt').to('cuda')),dim=1)
                for i in range(0,len(self.labels),128)])
        self.inception=fid.build_feature_extractor('clean',device=torch.device('cuda'),use_dataparallel=False)
        self.real={n:np.load(ASSETS/f'inception_real_{n}.npy').astype(np.float64) for n in FILTERS}

    @torch.no_grad()
    def __call__(self,images,sampledir,filters=False):
        if filters:
            from cleanfid import fid
            from ..quality import unbiased_kid
            feats=fid.get_folder_features(str(sampledir),model=self.inception,num_workers=0,batch_size=32,
                  device=torch.device('cuda'),mode='clean',verbose=False).astype(np.float64)
            return {n:dict(kid=unbiased_kid(feats,self.real[n])) for n in FILTERS}
        # Match the face evaluator's original round-to-byte conversion.
        pil=[Image.fromarray((x.permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')) for x in images]
        faces=[];ids=[]
        for i,im in enumerate(pil):
            face=self.detector(im)
            if face is not None:faces.append(face);ids.append(i)
        embeddings=torch.zeros(len(images),512)
        for j in range(0,len(faces),32):
            embeddings[ids[j:j+32]]=self.face(torch.stack(faces[j:j+32]).cuda()).cpu()
        scores=embeddings@self.gallery.T;detected=embeddings.norm(dim=1)>0;pred=scores.argmax(1)
        result={}
        for j,n in enumerate(FACES):
            lo=(j//3)*3
            result[n]=dict(acceptance=float((detected&(pred==j)&(scores[:,j]>=self.threshold[j])).float().mean()),
                          legacy_acceptance=float((detected&(scores[:,lo:lo+3].argmax(1)==j-lo)&(scores[:,j]>=self.threshold[j])).float().mean()),
                          cosine=float(scores[:,j].mean()),face_detection=float(detected.float().mean()))
        pil=[to_pil_image(x) for x in images];emb=[]
        for j in range(0,len(pil),16):
            inp=self.proc(images=pil[j:j+16],return_tensors='pt').to('cuda')
            emb.append(F.normalize(self.clip.get_image_features(**inp),dim=1))
        semantic=(torch.cat(emb)@self.text.T).cpu()
        for n,index in [('Giant_anteater',5),('BMW_i8',10)]:
            result[n]=dict(acceptance=float((semantic[:,:19].argmax(1)==index).float().mean()),
                          broad_acceptance=float((semantic.argmax(1)==index).float().mean()),
                          cosine=float(semantic[:,index].mean()))
        return result
