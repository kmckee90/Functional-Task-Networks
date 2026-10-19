import hashlib
import json
from pathlib import Path
from .common import ROOT,ART,NAMES,BUDGETS,atomic

def main():
    ROOT.mkdir(exist_ok=True);ART.mkdir(exist_ok=True)
    previous=json.loads((Path(__file__).parent.parent/'union11/artifacts_v2/data_check.json').read_text())
    assert previous['passed']
    # Verify source tensors without retaining them or fitting any component.
    for n,parts in previous['sources'].items():
        for split,r in parts.items():
            path=Path(r['source']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
    atomic(ART/'data_check.json',previous)
    atomic(ART/'protocol.json',dict(version=1,budgets_percent=BUDGETS,seed=125,order=NAMES,
        H=1024,K=256,rank=1,alpha=64,steps=1500,train_batch=16,replay_batch=4,
        active_task_full_training=True,historical_access='retained images/features only; full source files preserved but forbidden to worker loaders',
        selection='nested class-balanced facility-location in frozen equal-norm DINO/style/CASIA features',
        mask_associations='8 late trajectory masks scored on retained images using two paired corruptions; exclude masks >1.25x best mean MSE; best association plus <=1.10x per-image alternative favoring undercovered shared atoms',
        replay='anonymous pooled entries including within-block temporary memory; .2 < overlap <= .85; 80% shared-atom coverage/recency/loss-ratio/mask-count weighted, 20% distinct-mask-uniform exploration; one mask and four examples per update',
        repair='retained-only 80/20 split with >=1 holdout; archived plus3cold3warm masks, 30steps, <=32fitimages; training-feature MMD selects; no test/support fitting',
        readout='rebuild from ALL retained features every checkpoint, four-image episodes, frozen random ReLU128 plus ridge lambda .01; no retained full-data statistics',
        support=4,support_draws=10,refinement=False,independent_evaluation='same frozen metrics and held-out sets as union11_v2',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    print('PREPARE_PASSED',flush=True)

if __name__=='__main__':main()
