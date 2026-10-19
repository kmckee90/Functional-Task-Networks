"""Evaluation bookkeeping invariants; CPU-only, run on the pod."""
import tempfile
import unittest
from pathlib import Path
from .visitation import Visitation
from .terrain_maps import bounds
from .phase2 import experiments
from .achievement_charts import checked_achievements
from .achievement_catalog import LABELS

class Phase2Tests(unittest.TestCase):
    def test_visit_resume_and_negative_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            v=Visitation(d)
            record=dict(position=[-0.1,100,4.1],life=1,look_yaw=1.2)
            v.observe(record,100);v.observe(record,100)
            self.assertEqual(v.state['counts'],{'-1,1':1})
            v.save();v=Visitation(d);v.observe(record,101)
            self.assertEqual(v.state['counts'],{'-1,1':2})
            self.assertEqual(v.state['yaw'],1.2)
    def test_common_extent_and_no_shrinking(self):
        states=[{'counts':{'-2,1':10}},{'counts':{'50,-40':1}}]
        b=bounds(states);self.assertLessEqual(b[0],-8);self.assertGreaterEqual(b[1],204)
        self.assertLessEqual(b[2],-160);self.assertGreaterEqual(b[3],8)
        self.assertEqual(bounds(states[:1],b),b)
    def test_unique_not_repeated(self):
        key=next(iter(LABELS));r={'achievement_counts':{key:1},'achievement_discoveries':{'one':1}}
        self.assertEqual(checked_achievements(r),2)
        r['achievement_counts'][key]=500;self.assertEqual(checked_achievements(r),2)
    def test_isolated_assignments(self):
        ts=experiments();self.assertEqual(len(ts),8)
        self.assertEqual(len({(t['pod'],t['gpu']) for t in ts}),8)
        self.assertEqual(len({t['name'] for t in ts}),8)
        self.assertFalse(any(t.get('mf_init_weights') for t in ts))
        for seed in (19,101):self.assertEqual(len([t for t in ts if t['seed']==seed]),4)

if __name__=='__main__':unittest.main()
