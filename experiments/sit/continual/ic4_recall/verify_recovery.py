"""Read-only checks against historical checkpoints; run on the SiT pod."""
import importlib
import json
from pathlib import Path
import torch
from .universal import FrozenEncoder, combine, load_readout, fit_readout
from ..union11 import common as C

def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    report={'imports':[], 'runs':{}, 'features':{}}
    for name in ['union11.training','union11.evaluate','union11.metrics',
                 'memory_budget.training','memory_budget.evaluate',
                 'memory_refinement.run','memory_refinement.conservative']:
        importlib.import_module('continual.'+name)
        report['imports'].append(name)
    support={n:C.unified(n,'support') for n in C.NAMES}
    for package,artifact in [('union11','artifacts_v2'),('memory_budget','artifacts')]:
        root=Path(__file__).parents[1]/package/artifact
        for path in sorted(root.glob('*/after_11_BMW_i8/conditioner.pt')):
            model=load_readout(path)
            rows=json.loads(path.with_name('results.json').read_text())
            matched=0
            for row in rows:
                x=support[row['domain']][row['support_indices']].cuda()[None]
                actual=model.mask(x,256)[0].cpu()
                assert torch.equal(actual,torch.tensor(row['mask'])), (path,row['domain'],row['draw'])
                matched+=1
            expected=torch.randn(1384,128,generator=torch.Generator().manual_seed(93811))
            assert torch.equal(model.projection.cpu(),expected)
            report['runs'][path.parent.parent.name]={'exact_masks':matched,'weight_shape':list(model.weight.shape)}
            print('MASKS',path.parent.parent.name,matched,flush=True)
    assert len(report['runs'])==8
    ck=torch.load(C.ASSETS/'memory_budget_v1/runs/replay_p05_s125/latest.pt',map_location='cpu',weights_only=False,mmap=True)
    path=Path(__file__).parents[1]/'memory_budget/artifacts/replay_p05_s125/after_11_BMW_i8/conditioner.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    groups={n:combine(ck['memory'][n]['features']) for n in saved['teachers']}
    fitted=fit_readout(groups,{n:v['mask'] for n,v in saved['teachers'].items()})
    error=float((fitted.weight.cpu()-saved['weight']).abs().max())
    assert torch.allclose(fitted.weight.cpu(),saved['weight'],atol=1e-6,rtol=1e-5),error
    report['refit']={'run':'replay_p05_s125','max_weight_error':error}
    print('REFIT',error,flush=True)
    encoder=FrozenEncoder()
    for n in [C.NAMES[0],C.NAMES[3],C.NAMES[6],C.NAMES[9],C.NAMES[10]]:
        actual=encoder.branches(C.images(n,'support')[:4])
        checks={}
        for branch,value in actual.items():
            expected=C.features(n,'support',branch)[:4]
            cosine=torch.nn.functional.cosine_similarity(value,expected).min().item()
            error=(value-expected).abs().max().item()
            assert cosine>.9999,(n,branch,cosine,error)
            checks[branch]={'min_cosine':cosine,'max_absolute_error':error}
        report['features'][n]=checks
        print('FEATURES',n,checks,flush=True)
    report['passed']=True
    out=Path(__file__).parent/'recovery_verification.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('PASSED',out,flush=True)

if __name__=='__main__':main()
