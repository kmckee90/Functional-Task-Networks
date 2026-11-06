"""The declared experiment matrix. No smoke runs count toward final results."""
from ftn_models import VARIANTS
from vision_data import N_CLASSES

SEEDS=list(range(5))


def jobs():
    result=[]
    # Round-robin seeds and datasets give the report broad coverage early.
    for seed in SEEDS:
        for dataset in ['cifar10','cifar100','tinyimagenet','core50']:
            for variant in VARIANTS:
                result.append({'id':f'ftn/{dataset}/features/{variant}/seed{seed}',
                    'dataset':dataset,'gate':'ftn_features','needs_features':True,
                    'command':['run_ftn.py','--variant',variant,'--mode','features','--dataset',dataset,'--seed',str(seed)]})
            for method in ['derpp','l2p']:
                label='DER++-500-features' if method=='derpp' else method
                result.append({'id':f'mammoth/{dataset}/{label}/seed{seed}',
                    'dataset':dataset,'gate':'derpp_features' if method=='derpp' else method,'needs_features':method=='derpp',
                    'command':['run_mammoth.py','--method',method,'--dataset',dataset,'--seed',str(seed)]+(['--features'] if method=='derpp' else [])})
            result.append({'id':f'mammoth/{dataset}/DER++-matched-features/seed{seed}',
                    'dataset':dataset,'gate':'derpp_features','needs_features':True,
                    'command':['run_mammoth.py','--method','derpp','--features','--matched-memory','--dataset',dataset,'--seed',str(seed)]})
            for variant in ['LoRA','LoRA_Reactive_Low_NoReplay','BayesLoRA']:
                result.append({'id':f'ftn/{dataset}/vit/{variant}/seed{seed}',
                    'dataset':dataset,'gate':'ftn_vit','command':['run_ftn.py','--variant',variant,'--mode','vit','--dataset',dataset,'--seed',str(seed)]})
    return result


if __name__=='__main__':
    import json
    from storage import bulk,write_json
    plan={'seeds':SEEDS,'jobs':jobs(),'count':len(jobs())}
    write_json(bulk('manifests','plan.json'),plan)
    print(json.dumps({'jobs':len(jobs()),'seeds':SEEDS}))
