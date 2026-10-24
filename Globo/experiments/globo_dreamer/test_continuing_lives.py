import unittest
from types import SimpleNamespace
import numpy as np
import torch
from .model_free import ModelFreeBehavior
from .hud_feedback import encode, NAMES
from .scalable_replay import trajectory

class ContinuingTest(unittest.TestCase):
    def test_hud_bounds(self):
        self.assertTrue(np.isfinite(encode({'effects':None})).all())
        x=encode(dict(hp=-2,hp_max=1000,energy=1e6,energy_max=1000,
                      slots=[4],slot_max=[4],wield=1,effects=[['Drunk',1]]))
        self.assertEqual(len(x),len(NAMES))
        self.assertTrue(np.isfinite(x).all() and (x>=0).all() and (x<=1).all())
        self.assertEqual(x[NAMES.index('slot_1_quantity')],1)
        self.assertEqual(x[NAMES.index('Drunk')],1)

    def test_death_respawn_keeps_history_and_bridges_replay(self):
        class Agent:
            device='cpu';model_reward_weight=0.;cur_step=0;epsilon=0.
            memory=SimpleNamespace(size=0);_last_metrics={}
            def __init__(self):self.hidden=0;self.rows=[];self.feedback=[]
            def reset_hidden(self):raise AssertionError('Reservoir reset across lives')
            def note_action(self,a,n):self.feedback.append((a,n))
            def encode(self,*args):self.hidden+=1;return torch.tensor(float(self.hidden))
            def embed(self,img):return img
            def act(self,z):return [12]
            def remember(self,*args,**kw):self.rows.append(args)
            def step_schedule(self):self.cur_step+=1
        b=ModelFreeBehavior.__new__(ModelFreeBehavior);b.agent=Agent()
        b.continuing_lives=True;b.pending=None;b.image=torch.zeros(1)
        b.probability=0.;b.model_feedback=False;b.updates=0
        b.rng=np.random.default_rng(1)
        b.novelty_scale=SimpleNamespace(transform=lambda x:(0.,0.),reward_scale=1.,second=1.)
        def obs(first=False,last=False,nov=.2):
            return dict(image=np.zeros((1,1)),is_first=[first],is_last=[last],
                        is_terminal=[last],density=np.array([[nov]]),reward=[-float(last)],
                        body=np.zeros((1,len(NAMES))))
        b.policy(obs(first=True),{})
        b.policy(obs(last=True,nov=.9),{})
        self.assertEqual(len(b.agent.rows),0)
        b.policy(obs(first=True,nov=.1),{})
        row=b.agent.rows[0]
        self.assertEqual((row[0].item(),row[1],row[5].item()),(1.,12,3.))
        self.assertEqual(row[2],0.)
        self.assertEqual(row[3],.1)
        self.assertEqual(row[6:8],(False,False))
        self.assertEqual(len(b.agent.feedback),2)
        b.policy(obs(),{})
        self.assertEqual(b.agent.rows[1][0].item(),3.)
        buf=SimpleNamespace(ptr=3,size=3,capacity=8,device='cpu',boundary=torch.zeros(8,dtype=torch.bool))
        self.assertEqual(trajectory(buf,2,64).tolist(),[0,1,2])
        # Update volume changes without changing the 50-interaction cadence.
        from unittest.mock import Mock
        for count in (50,200):
            b.offline_updates=count;b.updates=0
            b.agent.memory=SimpleNamespace(size=500);b.agent.cur_step=49
            b.agent.train_round=Mock(return_value=.1)
            b.policy(obs(),{})
            b.agent.train_round.assert_called_once_with(count)
            self.assertEqual(b.updates,count)
            b.policy(obs(),{})
            b.agent.train_round.assert_called_once_with(count)

if __name__=='__main__':unittest.main()
