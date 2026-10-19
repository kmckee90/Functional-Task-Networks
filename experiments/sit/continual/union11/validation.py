"""Calibrate independent union evaluators on existing real test references."""
import torch
from torchvision.transforms.functional import to_tensor
from .common import ART,NAMES,images,atomic
from .metrics import Metrics

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    metric=Metrics();results={}
    for n in NAMES[3:]:
        pil=images(n,'test');excluded=[]
        if n=='Giant_anteater':
            excluded=[23]
            pil=[im for i,im in enumerate(pil) if i not in excluded]
        ims=torch.stack([to_tensor(im) for im in pil])
        r=metric(ims,None,False)
        results[n]=dict(count=len(ims),excluded_indices=excluded,metrics=r[n],cross_target_metrics=r)
        atomic(ART/'evaluator_validation.json',dict(results=results,complete=False))
        print('REAL_REFERENCE',n,len(ims),r[n],flush=True)
    atomic(ART/'evaluator_validation.json',dict(complete=True,results=results,
      note='No threshold fitting. Uses original galleries and thresholds. Anteater index23 excluded from strict held-out calibration due earlier recognition-screen overlap; original24 retained for velocity diagnostics.'))

if __name__=='__main__':main()
