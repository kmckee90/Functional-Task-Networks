"""Run monitor/control checks on the pod, with isolated runtime settings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import numpy as np
sys.path.insert(0,'/home/dev/globo_dreamer_state/vendor/dreamerv3')
from .continuous_monitor import ContinuousMonitor,report_batch
from .monitor_settings import CLIP_CHOICES,MAX_CLIP_FRAMES


def main():
    for frames in CLIP_CHOICES:
        rows=[(i,dict(image=np.full((2,2,3),i,np.uint8),action=np.int32(i),is_first=np.bool_(False))) for i in range(frames)]
        batch,offset=report_batch(rows)
        assert batch['image'].shape==(8,MAX_CLIP_FRAMES,2,2,3)
        np.testing.assert_array_equal(batch['image'][0,offset:offset+frames,0,0,0],np.arange(frames))
        assert batch['is_first'][0,offset]
        midpoint=MAX_CLIP_FRAMES//2
        np.testing.assert_array_equal(batch['action'][0,midpoint-1:midpoint-1+frames//2],np.arange(frames//2-1,frames-1))
    with tempfile.TemporaryDirectory(dir='/home/dev/globo_dreamer_state/diagnostics') as td:
        root=Path(td);settings=root/'continuous_runs/clip_settings.json'
        proc=subprocess.Popen([sys.executable,'-m','globo_dreamer.gallery_server','--port','8766','--directory',td],
            env=dict(os.environ,GLOBO_STATE=td),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            url='http://127.0.0.1:8766/api/clip-settings'
            for _ in range(50):
                try:
                    with urllib.request.urlopen(url,timeout=1) as r:assert json.load(r)['frames']==192
                    break
                except urllib.error.URLError:time.sleep(.1)
            else:raise RuntimeError('Test server did not start')
            def post(data,origin=None):
                headers={'Content-Type':'application/json'}
                if origin:headers['Origin']=origin
                return urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(data).encode(),headers=headers),timeout=2)
            monitor=ContinuousMonitor(root/'live',settings_path=settings)
            for frames in (64,256,128,192):
                with post(dict(frames=frames)) as r:assert json.load(r)['frames']==frames
                monitor.settings_checked=0;monitor.poll_settings()
                assert monitor.frames==frames and monitor.buffer.maxlen==frames
            for value in (0,65,512,True,'192'):
                try:post(dict(frames=value))
                except urllib.error.HTTPError as e:assert e.code==400
                else:raise AssertionError('Invalid setting accepted')
            try:post(dict(frames=64),origin='http://other.example')
            except urllib.error.HTTPError as e:assert e.code==403
            else:raise AssertionError('Foreign origin accepted')
            assert json.loads(settings.read_text())['frames']==192
        finally:proc.terminate();proc.wait(timeout=5)
    print('PASS: fixed-shape temporal alignment, clip resizing, API persistence/validation, origin checks')


if __name__=='__main__':main()
