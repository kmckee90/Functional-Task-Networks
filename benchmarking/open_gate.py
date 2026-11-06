"""Release production jobs only after concrete successful validation artifacts."""
import argparse
import hashlib
import json

from ftn_models import VARIANTS
from storage import WORKSPACE,bulk,write_json


def require(path,status='complete'):
    result=json.loads(path.read_text())
    assert result.get('status')==status,(path,result.get('status'))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('gate',choices=['ftn_pixels','ftn_features','ftn_vit','derpp','derpp_features','l2p','dualprompt','spg','mntdp'])
    a=p.parse_args()
    evidence=[]
    if a.gate.startswith('ftn'):
        for variant in VARIANTS:
            path=bulk('validation','isolation',variant+'.json')
            require(path,'pass'); evidence.append(str(path))
        if a.gate=='ftn_vit':
            for variant in ['LoRA','BayesLoRA']:
                path=bulk('validation','isolation',variant+'_vit.json')
                require(path,'pass'); evidence.append(str(path))
            variant='LoRA'
            mode='vit'
        else:
            variant='Base'
            mode='features' if a.gate=='ftn_features' else 'pixels'
        path=bulk('runs','ftn','tinyimagenet',mode,variant,'seed0_smoke','result.json')
        run=require(path)
        assert run['smoke'] and run['optimizer_policy']['weight_decay']==0.
        assert run['optimizer_policy']['reconfiguration_resets']>0
        evidence.append(str(path))
    else:
        if a.gate in ['derpp','derpp_features','l2p','dualprompt']:
            method='DER++-500-features' if a.gate=='derpp_features' else ('DER++-500' if a.gate=='derpp' else a.gate)
            path=bulk('runs','mammoth','tinyimagenet',method,'seed0_smoke','result.json')
        elif a.gate=='spg':
            path=bulk('runs','spg','tinyimagenet','SPG','seed0_smoke','result.json')
        else:
            path=bulk('runs','mntdp','tinyimagenet','MNTDP-D','seed0_smoke','result.json')
        run=require(path)
        assert run['smoke']
        assert len(run.get('class_il_matrix',run.get('task_il_matrix',[])))==2
        evidence.append(str(path))
    # Confirm the native model source has not drifted from the audited snapshot.
    sources=json.loads((WORKSPACE/'sources/manifest.json').read_text())
    for rel,entry in sources.items():
        actual=hashlib.sha256((WORKSPACE/'sources'/rel).read_bytes()).hexdigest()
        assert actual==entry['sha256'],rel
    code={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in WORKSPACE.glob('*.py')}
    write_json(bulk('validation',a.gate+'.json'),{'gate':a.gate,'status':'pass','evidence':evidence,'harness_sha256':code})
    print('Released:',a.gate,flush=True)


if __name__=='__main__': main()
