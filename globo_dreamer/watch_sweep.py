"""Incremental, read-only sweep observer. Artifacts must stay outside source sync.

Flags candidates for review; never stops or changes a training process.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import time
from .report import window


class Watch:
    def __init__(self, run, width):
        self.run, self.width = run, width
        self.offset = 0
        self.rows = []
        self.seen = set()
        self.windows = deque(maxlen=5)
        self.observations = 0

    def poll(self):
        trace = self.run / 'environment/trace.jsonl'
        if trace.exists():
            with trace.open() as stream:
                stream.seek(self.offset)
                while True:
                    line = stream.readline()
                    if not line.endswith('\n'):
                        break
                    row = json.loads(line)
                    self.offset = stream.tell()
                    self.rows.append(row)
                    self.observations += 1
                    if len(self.rows) == self.width:
                        stats = window(self.rows)
                        stats.pop('action_counts', None)
                        cells = {(math.floor(r['position'][0]/4),
                                  math.floor(r['position'][2]/4)) for r in self.rows}
                        stats['new_4m_cells'] = len(cells - self.seen)
                        self.seen.update(cells)
                        self.windows.append(stats)
                        self.rows.clear()
        status_path = self.run/'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        recent = list(self.windows)[-3:]
        # This is an alert, not an objective or an automatic stopping rule.
        low_coverage = (self.observations >= 20000 and len(recent) == 3 and
                        all(w['new_4m_cells'] <= 1 for w in recent))
        return dict(observations=self.observations,
            completed_windows=list(self.windows),
            updates=status.get('updates'), running=status.get('running'),
            seconds_since_status=time.time()-status['time'] if 'time' in status else None,
            review_low_coverage=low_coverage,
            note='Spatial alerts can miss useful stationary interaction. Check frames, '
                 'density, model quality and simulator health before retiring a run.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--interval', type=float, default=60)
    p.add_argument('--window', type=int, default=1000)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    if args.interval <= 0 or args.window < 1:
        p.error('interval and window must be positive')
    source = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(source):
        p.error('Runtime output must be outside source sync')
    watchers = [Watch(run, args.window) for run in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        result = dict(time=time.time(), runs={})
        for watch in watchers:
            try:
                result['runs'][watch.run.name] = watch.poll()
            except Exception as exc:
                result['runs'][watch.run.name] = dict(observer_error=str(exc))
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2))
        temp.replace(args.output)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__': main()
