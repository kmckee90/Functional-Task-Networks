"""Pod-only checks of novelty reward and executed-action bookkeeping."""
import unittest
import numpy as np
from .model_free import ModelFreeBehavior


class ModelFreeTests(unittest.TestCase):
    def test_repeated_view_loses_novelty_without_world_model(self):
        behavior=ModelFreeBehavior(device='cpu')
        sky=np.full((64,64,3),[80,130,200],np.uint8)
        for _ in range(40):
            score=behavior.observe_frame(sky)
        self.assertLess(score,1e-4)
        ground=np.full((64,64,3),[130,70,20],np.uint8)
        self.assertGreater(behavior.observe_frame(ground),.05)

    def test_executed_model_action_enters_model_free_replay(self):
        behavior=ModelFreeBehavior(device='cpu',dreamer_probability=1.)
        image=np.full((64,64,3),100,np.uint8)
        def observe(first=False,last=False):
            novelty=behavior.observe_frame(image)
            return dict(image=image[None],density=np.array([[novelty]],np.float32),
                is_first=np.array([first]),is_last=np.array([last]),
                is_terminal=np.array([last]),reward=np.array([0.],np.float32))
        result=behavior.policy(observe(first=True),dict(action=np.array([11],np.int32)))
        self.assertEqual(int(result['action'][0]),11)
        behavior.policy(observe(last=True),dict(action=np.array([3],np.int32)))
        self.assertEqual(int(behavior.agent.memory.action[0,0]),11)
        self.assertTrue(bool(behavior.agent.memory.terminal[0]))
        behavior.policy(observe(first=True),dict(action=np.array([2],np.int32)))
        self.assertEqual(behavior.agent.memory.size,1)  # no transition across reset


if __name__=='__main__': unittest.main()
