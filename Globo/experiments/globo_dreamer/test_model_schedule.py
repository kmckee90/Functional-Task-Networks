import unittest
from .model_progress import target_updates


class ScheduleTests(unittest.TestCase):
    def test_no_delayed_catchup(self):
        for ratio in (256,512,1024,2048):
            self.assertEqual(target_updates(20000,20000,256,ratio),0)
            self.assertEqual(target_updates(20001,20000,256,ratio),ratio//256)
            self.assertEqual(target_updates(60000,20000,256,ratio),40000*ratio//256)

    def test_replay_warmup_and_budget(self):
        self.assertEqual(target_updates(255,0,256,256),0)
        self.assertEqual(target_updates(256,0,256,256),1)
        self.assertEqual(target_updates(60000,0,256,2048),59745*8)
        # At a checkpoint, the cumulative budget yields no duplicate updates.
        completed=target_updates(1250,0,256,512)
        self.assertEqual(target_updates(1251,0,256,512)-completed,2)


if __name__=='__main__':unittest.main()
