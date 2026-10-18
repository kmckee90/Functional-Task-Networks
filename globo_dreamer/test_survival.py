import json
from pathlib import Path
import tempfile
import unittest
from .survival import summary,chart,panel

class SurvivalTests(unittest.TestCase):
    def test_deaths_only_and_partial_append(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'environment/episodes.jsonl';p.parent.mkdir()
            rows=[dict(step=100,life=1,life_step=99,game_seconds=60,terminal=True),dict(step=200,life=2,life_step=99,game_seconds=90,terminal=False,truncated=True),dict(step=250,life=3,life_step=49,game_seconds=120,terminal=True)]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows)+json.dumps(rows[-1])+'\n'+ '{"step":')
            s=summary(d,dict(step=300,environment=dict(life=4,life_step=50,game_seconds=900,terminal=False)))
            self.assertEqual(s['recorded_deaths'],2);self.assertEqual(s['recorded_timeouts'],1)
            self.assertEqual(s['all']['seconds']['mean'],90);self.assertEqual(s['all']['seconds']['best'],120)
            self.assertEqual(s['current']['seconds'],900);self.assertEqual(s['trend'][-1]['mean_seconds'],90)
            self.assertIn('Current life',panel(s));self.assertIn('Environment observations',chart([('x','x','#fff',s)]))
    def test_recent_window(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'environment/episodes.jsonl';p.parent.mkdir()
            p.write_text(''.join(json.dumps(dict(step=i*100,life=i,life_step=i,game_seconds=i,terminal=True))+'\n' for i in range(1,26)))
            s=summary(d,{})
            self.assertEqual(s['recent_n'],20);self.assertEqual(s['recent']['seconds']['mean'],15.5)
            self.assertEqual(s['all']['seconds']['mean'],13)
    def test_unfinished_is_not_zero_lifespan(self):
        with tempfile.TemporaryDirectory() as d:
            s=summary(d,dict(environment=dict(life_step=100,game_seconds=50)))
            self.assertIsNone(s['all']['seconds']);self.assertFalse(s['episode_log_available'])
            self.assertEqual(s['trend'],[])

if __name__=='__main__':unittest.main()
