"""Recovered CLIP compatibility exports and exact 19-label evaluator vocabulary.

The current evaluator is continual.union11.metrics.Metrics. The old mixed-object
training/evaluation programs are not required and are not recreated here.
"""
import sys
import torch
sys.path.insert(0,'/home/dev/sit_state/ic4_recall/deps5')
from transformers import CLIPModel as BaseCLIP,CLIPProcessor as BaseProcessor

class CLIPModel(BaseCLIP):
    def get_image_features(self,*args,**kwargs):
        out=super().get_image_features(*args,**kwargs)
        return out if isinstance(out,torch.Tensor) else out.pooler_output
    def get_text_features(self,*args,**kwargs):
        out=super().get_text_features(*args,**kwargs)
        return out if isinstance(out,torch.Tensor) else out.pooler_output

class CLIPProcessor(BaseProcessor):
    @classmethod
    def from_pretrained(cls,*args,**kwargs):
        return super().from_pretrained(*args,use_fast=False,**kwargs)

LABELS=['Whoopi Goldberg','Oprah Winfrey','Wanda Sykes','Sherri Shepherd','Alfre Woodard',
        'a giant anteater','a southern tamandua','an aardvark','a badger','a sloth bear',
        'a BMW i8 coupe','a BMW i3 car','an Audi R8 car','a McLaren sports car','a generic sports car',
        'an armadillo','a three-toed sloth','a pangolin','a generic mammal']
TARGETS=[0,5,10]
