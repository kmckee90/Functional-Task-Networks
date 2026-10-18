"""Runtime-only controls shared by gallery server and live monitors."""
import json
import os
from pathlib import Path

CLIP_CHOICES=(64,128,192,256)
DEFAULT_CLIP_FRAMES=192
MAX_CLIP_FRAMES=max(CLIP_CHOICES)
SETTINGS=Path(os.environ.get('GLOBO_CLIP_SETTINGS',str(Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))/'continuous_runs/clip_settings.json')))


def read_settings(path=SETTINGS):
    try:
        data=json.loads(Path(path).read_text())
        frames=data['frames']
        if type(frames) is not int or frames not in CLIP_CHOICES:raise ValueError('Invalid clip length')
    except (OSError,ValueError,KeyError,TypeError):frames=DEFAULT_CLIP_FRAMES
    return dict(frames=frames,seconds=frames/6,choices=list(CLIP_CHOICES))
