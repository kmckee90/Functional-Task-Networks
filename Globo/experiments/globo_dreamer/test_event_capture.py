import tempfile,unittest,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .event_capture import Triggers,EventCapture,sample_frames

class CaptureTest(unittest.TestCase):
    def record(self,t,x,new=True,**kw):
        return dict(game_seconds=t,life=1,position=[x,0,0],new_spatial_cell=new,**kw)
    def test_run_cells(self):
        t=Triggers();t.observe(self.record(0,0))
        for x in range(1,6):
            self.assertEqual(t.observe(self.record(x,x*4)),[])
            self.assertEqual(t.observe(self.record(x+.1,x*4)),[])
        self.assertEqual(t.observe(self.record(6,24))[0]['kind'],'long_run')
        t.observe(self.record(7,0,False));self.assertEqual(t.streak,0)

    def test_events_not_repeated(self):
        t=Triggers();t.observe(self.record(0,0))
        events=[dict(id=1,kind='climb_tree',detail='tree')]
        self.assertEqual(len(t.observe(self.record(1,0,video_events=events))),1)
        self.assertEqual(t.observe(self.record(2,0,video_events=events)),[])
        out=t.observe(self.record(3,0,achievement_delta={'kill_other':1},achievement_details={'kill_other/animals:puma':1}))
        self.assertEqual(out[0]['details'],['animals:puma'])

    def test_video_center_and_duration(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            cap=EventCapture(directory,SimpleNamespace(layout=np.arange(4),grid=(2,2)))
            row={'log/human':np.zeros((256,256,3),np.uint8),'log/m2e_ftn_ema':np.zeros((4,3),np.uint8),
                 'log/offline_goal':np.zeros((16,32),np.uint8),'log/offline_goal_step':np.int64(1)}
            for i in range(201):
                record=self.record(i/10,0,achievement_delta={'place_block':1} if i==100 else {})
                cap.observe(row,record,i)
            cap.close()
            files=list(Path(directory).glob('*.mp4'));self.assertEqual(len(files),1)
            with av.open(str(files[0])) as f:self.assertEqual(sum(1 for _ in f.decode(video=0)),161)
            meta=json.loads(files[0].with_suffix('.json').read_text())
            self.assertEqual(meta['event_at_seconds'],80/6);self.assertEqual(meta['padding_after'],0)
            rows=[(i*i/10,i,{}) for i in range(201)]
            selected=sample_frames(rows,100)
            self.assertEqual([r[1] for r in selected],list(range(20,181)))
            self.assertEqual(meta['observations_before'],80)
            self.assertEqual(meta['observations_after'],80)

    def test_short_context_is_not_frozen(self):
        rows=[(i,i,{}) for i in range(10)]
        self.assertEqual(sample_frames(rows,5),rows)

if __name__=='__main__':unittest.main()
