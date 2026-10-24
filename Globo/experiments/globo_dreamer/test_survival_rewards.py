"""Run on the pod: reward relabeling and death-to-respawn transition regression."""
from types import SimpleNamespace as NS
import numpy as np
import torch
from .m2e_ftn import M2EFTN
from .model_free import ModelFreeBehavior

def main():
    a=NS(_novelty=lambda emb:emb,online_coef=1.,offline_coef=1.,
         goal_sparse_reward_scale=5.,main_sparse_reward_scale=5.,
         env_reward_coef=0.,death_penalty=0.,survival=False)
    d=dict(emb=torch.tensor([1.,2.,3.]),online_nov=torch.ones(3),action=torch.zeros(3,1),
           env_r=torch.tensor([0.,-1.,0.]),terminal=torch.zeros(3,dtype=torch.bool),is_goal=True,frontier_endpoint=True)
    f=NS(agent=a);old=M2EFTN.labels(f,d);a.survival=True;new=M2EFTN.labels(f,d)
    assert torch.allclose(new[:-1,1],old[:-1,1]*.25)
    assert new[-1,1]==old[-1,1] and new[1,2]==-old[-1,1]
    assert new[:,3].eq(1).all(), 'Death must not disable bootstrapping'
    assert new[[0,2],2].eq(0).all()
    b=ModelFreeBehavior.__new__(ModelFreeBehavior)
    b.continuing_lives=True;b.survival=True;b.pending_death=False
    b.pending=(torch.zeros(1),2,0.,0.);b.image=torch.zeros(3,2,2)
    b.probability=0.;b.rng=np.random.default_rng(1);b.model_feedback=False;b.updates=0
    b.novelty_scale=NS(transform=lambda raw:(0.,0.),reward_scale=1.,second=1.)
    saved=[]
    b.agent=NS(device='cpu',note_action=lambda *args:None,encode=lambda *args:torch.zeros(1),
        embed=lambda img:torch.zeros(1),remember=lambda *args,**kw:saved.append(args),
        step_schedule=lambda:None,act=lambda z:[0],model_reward_weight=0.,cur_step=2,
        memory=NS(size=1),epsilon=.01,_last_metrics={})
    def obs(first,last,terminal):return dict(image=np.zeros((1,2,2,3)),density=np.zeros((1,1)),
        body=np.zeros((1,43)),is_first=[first],is_last=[last],is_terminal=[terminal],reward=[0.])
    b.policy(obs(False,True,True),{})
    assert not saved and b.pending_death
    b.policy(obs(True,False,False),{})
    assert len(saved)==1 and saved[0][2]==-1. and saved[0][7] is False
    assert not b.pending_death
    b.policy(obs(False,False,False),{})
    assert saved[-1][2]==0., 'One death must yield exactly one penalty label'
    print('PASS: quarter novelty, retained goal, matched death cost, cross-life bootstrap, one death label')

if __name__=='__main__':main()
