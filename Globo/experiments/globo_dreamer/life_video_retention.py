"""Runtime-configurable life-video retention; zero keeps every life."""
import json
import os
from pathlib import Path

MIN_GAME_SECONDS=600

def threshold(path=None):
    path=path or os.environ.get('GLOBO_LIFE_SETTINGS')
    if path:
        try:
            minutes=json.loads(Path(path).read_text())['minimum_minutes']
            if type(minutes) is int and minutes>=0:return minutes*60
        except (OSError,ValueError,KeyError,TypeError):pass
    return MIN_GAME_SECONDS

def prune(folder,minimum_seconds=None):
    minimum_seconds=threshold() if minimum_seconds is None else minimum_seconds
    if minimum_seconds==0:return dict(removed=0,bytes_freed=0)
    removed=0;freed=0
    for metadata in Path(folder).glob('*.json'):
        try:
            info=json.loads(metadata.read_text())
            seconds=float(info['game_seconds'])
            # Only remove the finalized video belonging to this sidecar.
            # Active .partial.mp4 recordings have no finalized sidecar.
            if not 0<=seconds<=minimum_seconds:continue
            if info.get('file')!=metadata.stem+'.mp4':continue
            video=metadata.with_suffix('.mp4')
            if video.exists():
                freed+=video.stat().st_size
                video.unlink()
            metadata.unlink();removed+=1
        except (OSError,ValueError,KeyError,TypeError):continue
    return dict(removed=removed,bytes_freed=freed)
