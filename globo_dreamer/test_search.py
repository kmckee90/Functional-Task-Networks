"""Pod-only supervisor tests: bounds, seed matching, restart, and safe retention."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .search import BASE, BOUNDS, Search, propose, uninterruptible_kill_pending


class SearchTests(unittest.TestCase):
    def test_quarantine_requires_kernel_block_and_pending_kill(self):
        for state,mask,expected in [('D','00000100',True),('D','00000000',False),('S','00000100',False)]:
            with patch('pathlib.Path.read_text',return_value=f'State:\t{state}\nSigPnd:\t{mask}\nShdPnd:\t0\n'):
                self.assertEqual(uninterruptible_kill_pending(123),expected)

    def test_mutation_bounds_and_changes(self):
        rng=np.random.default_rng(3)
        parent=BASE
        for i in range(500):
            candidate=propose(rng,parent,.4)
            self.assertNotEqual(candidate,parent)
            for k,(lo,hi) in BOUNDS.items():
                value=1/(1-candidate['mf_gamma']) if k=='horizon' else candidate[k]
                self.assertGreaterEqual(value,lo-1e-8); self.assertLessEqual(value,hi+1e-8)
            parent=candidate

    def test_seed_matching_restart_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,GLOBO_STATE=tmp):
            root=Path(tmp); seeds=root/'seeds.json'
            seeds.write_text(json.dumps([dict(seed=i,spawn='1,2,3') for i in (7,19,42)]))
            args=argparse.Namespace(state=str(root/'search'),seeds=str(seeds),steps=40000,
                                    gpus=0,search_seed=11,poll_seconds=.01)
            s=Search(args)
            # Two lucky seeds must never be enough to become a ranked parent.
            for t in s.db['trials'][:2]: t.update(status='done',result=dict(score=1.))
            self.assertEqual(s.rankings(),[])
            s.db['trials'][2].update(status='done',result=dict(score=.5))
            self.assertEqual(len(s.rankings()),1)
            s.evolve(); self.assertEqual(len(s.db['configs']),6)
            active=s.db['trials'][-1]; active['status']='running'
            path=root/'runs'/active['name']/'environment'; path.mkdir(parents=True)
            (path/'sentinel').write_text('keep')
            s.prune(); self.assertTrue((path/'sentinel').exists())
            state=s.db.copy(); s.lock.close()
            resumed=Search(args)
            self.assertEqual(len(resumed.db['trials']),18)
            self.assertEqual(resumed.db['configs'],state['configs'])
            resumed.lock.close()


if __name__=='__main__': unittest.main()
