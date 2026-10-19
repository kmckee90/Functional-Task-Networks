"""Validate source tensors/features without modifying or copying source datasets."""
import hashlib
import json
import torch
from .common import *

def main():
    ROOT.mkdir(parents=True,exist_ok=True);ART.mkdir(exist_ok=True)
    records={}
    for n in NAMES:
        records[n]={}
        for split in ['train','support','test']:
            path=DATA[SOURCE[n]]/f'{n}_{split}.pt'
            x=torch.load(path,weights_only=True)
            assert len(x['z'])==len(x['y']) and torch.isfinite(x['z']).all()
            records[n][split]=dict(source=str(path),count=len(x['z']),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if split!='test':
                for b in ['dino','style','face']:
                    f=features(n,split,b);assert len(f)==len(x['z']) and torch.isfinite(f).all()
    atomic(ART/'data_check.json',dict(passed=True,sources=records))
    atomic(ART/'protocol.json',dict(order=NAMES,seeds=[123,125],arms=['noreplay','replay'],
      H=1024,k=256,rank=1,alpha=64,steps_per_block=1500,batch_size=16,learning_rate=.0003,
      optimizer='Adam reset each block',detector='original reactive Bayesian SD3 EMA0.9',
      support=4,support_draws=10,refinement=False,feature_encoder='frozen DINOv2/style/CASIA equal-norm',
      conditioner='shared frozen random ReLU128 plus scalar ridge readout per atom; lambda .01',
      teacher_search='seen tasks only; archived mask plus 3 cold and 3 warm 30-step candidates; training-fit32; generated20; internal training validation',
      conditioner_update='Rebuild accumulated ridge sufficient statistics for seen tasks when repaired teacher masks change; no future-task fitting',
      future_evaluation='same direct4 readout extrapolated to unseen support; initial fixed seeded top-k prior; no future repair',
      mask_selection='style+.25 DINO for filters; DINO+.1 style otherwise; no independent evaluator selection',
      generation_counts=dict(filters=200,other=64),generation_seed=99233,
      filter_labels='original ten ImageNet labels; CFG4',other_labels='null1000; CFG1',
      class_dropout=dict(filters=.1,faces_and_objects=0.),
      supersedes='union11_v1 pilot: missing filter class dropout, stopped after two completed blocks; pilot artifacts preserved separately',
      replay='original representative schedule 32 batches x4 per block; overlap-filtered replay; no-replay has no generator replay gradients',
      offline_memory='both arms revisit earlier training images for mask repair and conditioner fitting',
      primary_metrics=['filter KID','six-way face acceptance','legacy three-way face acceptance','closed-label object acceptance','1019-label object acceptance','fixed-noise query MSE'],
      initial_and_after_every_block=True,immutable_old_experiments=True,
      source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    print('DATA_CHECK_PASSED',len(records),flush=True)

if __name__=='__main__':main()
