import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .novelty_scale import NoveltyScale
from .model_free import ModelFreeBehavior
from .continuous_monitor import ContinuousMonitor


class NoveltyTests(unittest.TestCase):
    def test_scale_limits_decay_outlier_and_resume(self):
        scale = NoveltyScale()
        self.assertEqual(scale.transform(0), (0., 0.))
        before = scale.reward_scale
        inp, reward = scale.transform(1e30)
        self.assertLessEqual(inp, 1)
        self.assertAlmostEqual(reward, .3)
        self.assertLess(scale.reward_scale, before * 1.01)
        for _ in range(20000): scale.transform(.0001)
        self.assertGreaterEqual(scale.reward_scale, before)
        self.assertLess(scale.transform(.0001)[1], .001)
        restored = NoveltyScale(); restored.load(scale.save())
        self.assertEqual(restored.transform(.123), scale.transform(.123))
        for bad in (float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError): scale.transform(bad)

    def test_executed_action_feedback_replay_reset(self):
        b = ModelFreeBehavior(device='cpu',dreamer_probability=1.,
            replay_capacity=16,goal_capacity=16,recode_capacity=32,
            model_feedback=True,model_reward_weight=1.)
        image=np.full((64,64,3),100,np.uint8)
        def obs(first=False):
            nov=b.observe_frame(image)
            return dict(image=image[None],density=np.array([[nov]],np.float32),
                is_first=np.array([first]),is_last=np.array([False]),
                is_terminal=np.array([False]),reward=np.array([0.],np.float32))
        values=np.zeros((1,25));values[0,11]=.2
        b.policy(obs(True),dict(action=np.array([11])),values)
        expected=b.pending[3];bonus=b.pending[2]
        b.policy(obs(),dict(action=np.array([2])),values)
        self.assertAlmostEqual(b.agent._model_feedback,expected)
        self.assertAlmostEqual(float(b.agent.memory.model_nov[0]),bonus,places=6)
        self.assertEqual(int(b.agent.memory.action[0,0]),11)
        vec=b.agent._feedback_vector(b.agent._bin_novelty(torch.tensor(.1)))
        self.assertAlmostEqual(float(vec[0,-1]),expected,places=6)
        b.agent.goal.insert(b.agent.memory.gather(torch.tensor([0])))
        self.assertAlmostEqual(float(b.agent.goal.model_nov[0]),bonus,places=6)
        b.policy(obs(True),dict(action=np.array([2])),values)
        self.assertEqual(b.agent.memory.size,1)
        self.assertEqual(b.agent._model_feedback,0.)
        state=b.save();b.novelty_scale.transform(10);b.load(state)
        self.assertEqual(b.novelty_scale.save(),state['novelty_scale'])

    def test_stagnation_recovery_and_frame_reset(self):
        with tempfile.TemporaryDirectory() as root:
            monitor=ContinuousMonitor(root,frames=2,stall_steps=3)
            record=dict(cells_lifetime=1,terminal=False,low_detail=False,sky_fraction=0,density=.1,action=0)
            for step in range(1,5):
                monitor.observe(dict(is_first=step==1),record,step,{})
            self.assertTrue(monitor.state['stalled'])
            record['cells_lifetime']=2
            monitor.observe(dict(is_first=True),record,5,{})
            self.assertFalse(monitor.state['stalled'])
            self.assertEqual(len(monitor.buffer),1)
            events=[json.loads(x) for x in (Path(root)/'events.jsonl').read_text().splitlines()]
            self.assertEqual([x['kind'] for x in events],['stagnation_confirmed','recovered'])

    def test_bounded_dreamer_replay_still_resumes(self):
        from .train import embodied
        from .continuous_monitor import bounded_cleanup
        with tempfile.TemporaryDirectory() as root:
            replay=embodied.replay.Replay(length=4,capacity=8,
                directory=str(Path(root)/'replay'),chunksize=4,save_wait=True)
            for i in range(40):
                replay.add(dict(value=np.int32(i)))
                if i%4==3:replay.save()
            replay.save()
            bounded_cleanup(root,replay)
            self.assertLessEqual(len(list((Path(root)/'replay').glob('*.npz'))),len(replay.chunks))
            restored=embodied.replay.Replay(length=4,capacity=8,
                directory=str(Path(root)/'replay'),chunksize=4,save_wait=True)
            restored.load()
            batch=restored.sample(4)['value']
            self.assertEqual(batch.shape,(4,4))
            self.assertTrue(np.all(np.diff(batch,axis=1)==1))
            self.assertGreaterEqual(int(batch.min()),28)


if __name__=='__main__':unittest.main()
