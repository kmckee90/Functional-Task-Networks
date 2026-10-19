"""Matched generation/KID for replay-free ReactiveLow, cold versus IC."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from cleanfid import fid
from .model import BayesSiT
from .run import generate,atomic_json
from .quality import unbiased_kid


def main():
    p=argparse.ArgumentParser(); p.add_argument('--name',required=True)
    p.add_argument('--domains',default=None)
    p.add_argument('--tag',default='')
    p.add_argument('--single-gpu',action='store_true')
    args=p.parse_args(); torch.set_num_threads(4)
    root=Path('/home/dev/sit_state'); run=root/'runs'/args.name; out=run/'reactive_quality'
    out.mkdir(exist_ok=True)
    if args.domains is None:
        # Each seed owns two of the available GPUs. The supervisor itself is
        # CPU-only; split independent domain-generation work across the pair.
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        workers=[]
        partitions=[(gpu,'photo,monochrome,poster,sketch','all')] if args.single_gpu else [(gpu,'photo,monochrome','a'),(gpu+3,'poster,sketch','b')]
        for device,domains,tag in partitions:
            env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(device)
            workers.append(subprocess.Popen([sys.executable,'-u','-m','continual.reactive_quality',
                '--name',args.name,'--domains',domains,'--tag',tag],env=env))
        codes=[w.wait() for w in workers]
        if any(codes):
            raise RuntimeError(f'Generation shards failed: {codes}')
        combined=[]
        for _,_,tag in partitions:
            combined.extend(json.loads((out/f'quality_{tag}.json').read_text()))
        atomic_json(out/'quality.json',combined)
        atomic_json(out/'quality_completed.json',{'complete':True,'records':len(combined)})
        return
    cfg=json.loads((run/'config.json').read_text())
    final=json.loads((run/'lifecycle.json').read_text())[-1]
    model=BayesSiT(root/'SiT-XL-2-256.pt',cfg['H'],cfg['k'],cfg['rank'],cfg['alpha']).cuda().eval()
    ck=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
    model.load_adapter(ck['adapter'])
    vae=AutoencoderKL.from_pretrained(root/'vae').cuda().eval().requires_grad_(False)
    extractor=fid.build_feature_extractor('clean',device=torch.device('cuda'),use_dataparallel=False)
    records=[]
    result_path=out/f'quality_{args.tag}.json'
    if result_path.exists():
        records=json.loads(result_path.read_text())
    done={(r['domain'],r['mode']) for r in records}
    for domain in args.domains.split(','):
        modes=['cold4','cold40','ic_direct','ic_refine']
        if 'stored' in final['domains'][domain]:
            modes.append('stored')
        cache={}
        for mode in modes:
            signature=tuple(final['domains'][domain][mode]['mask'])
            folder=out/'samples'/f'{domain}_{mode}'
            if (domain,mode) in done:
                cache[signature]=next(r for r in records if r['domain']==domain and r['mode']==mode)
                continue
            if signature in cache:
                record=dict(cache[signature],mode=mode,identical_mask_to=cache[signature]['mode'])
            else:
                folder.mkdir(parents=True,exist_ok=True)
                mask=torch.tensor(signature,device='cuda')
                if len(list(folder.glob('[0-9]*.png')))!=200:
                    images=generate(model,vae,mask,count=200,steps=cfg['sample_steps'],seed=99233)
                    for i,im in enumerate(images):
                        save_image(im,folder/f'{i:05d}.png')
                    save_image(images[:20],folder/'grid.png',nrow=10)
                # grid is kept outside feature input folder to avoid contamination.
                grid=out/f'{domain}_{mode}.png'
                if (folder/'grid.png').exists():
                    (folder/'grid.png').replace(grid)
                features=fid.get_folder_features(str(folder),model=extractor,num_workers=2,
                    batch_size=32,device=torch.device('cuda'),mode='clean',verbose=False).astype(np.float64)
                assert len(features)==200
                real=np.load(root/f'inception_real_{domain}.npy').astype(np.float64)
                record={'domain':domain,'mode':mode,'kid':unbiased_kid(features,real),
                        'generated_count':200,'reference_count':len(real),
                        'grid':str(grid),'mask':list(signature)}
                cache[signature]=record
            records.append(record); atomic_json(result_path,records)
            print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
