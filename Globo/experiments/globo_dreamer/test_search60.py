"""Pod-only checks for fixed explorer, matched seeds and stage isolation."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .search60 import BASE, FIXED, BOUNDS, DreamerSearch, propose, score60


class Search60Tests(unittest.TestCase):
    def test_mutations_never_change_explorer(self):
        rng=np.random.default_rng(7)
        parent=BASE
        for i in range(1000):
            child=propose(rng,None if i%5==0 else parent,.4)
            self.assertEqual({k:child[k] for k in FIXED},FIXED)
            for key,(lo,hi) in BOUNDS.items():
                if key in ('disagreement','prediction_error') and child[key]==0: continue
                self.assertTrue(lo-1e-8<=child[key]<=hi+1e-8)
            parent=child

    def test_isolation_and_matching(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,GLOBO_STATE=tmp):
            root=Path(tmp);seeds=root/'seeds.json'
            seeds.write_text(json.dumps([dict(seed=s,spawn='1,2,3') for s in (7,19,101)]))
            args=argparse.Namespace(state=str(root/'search60k'),seeds=str(seeds),steps=60000,
                                    gpus=0,search_seed=11,poll_seconds=.01)
            s=DreamerSearch(args)
            self.assertEqual(len(s.db['configs']),8)
            self.assertEqual(len(s.db['trials']),24)
            self.assertTrue(all(t['name'].startswith('s60_') for t in s.db['trials']))
            old=root/'runs/s40_c000024_s7/ckpt/sentinel';old.parent.mkdir(parents=True);old.touch()
            s.prune();self.assertTrue(old.exists())
            with self.assertRaises(ValueError):s.add_config(dict(BASE,mf_gamma=.5),None,.18,'invalid')
            s.lock.close()

    def test_late_progress_summary(self):
        windows=[dict(new_4m_cells=i,new_visual_views=2*i) for i in range(60)]
        with patch('globo_dreamer.search60.score_run',return_value=dict(windows=windows)) as scorer:
            result=score60(Path('/tmp/unused'))
        self.assertEqual(scorer.call_args.kwargs['tail_windows'],20)
        self.assertEqual(result['post40k_new_spatial_cells'],sum(range(40,60)))
        self.assertEqual(result['final10k_new_spatial_cells'],sum(range(50,60)))


if __name__=='__main__': unittest.main()
