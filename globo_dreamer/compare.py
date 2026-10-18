"""Pod-only matched-age comparison of exploration runs; writes no training state."""
import argparse
import json
from pathlib import Path
from .report import records, progress_windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--window', type=int, default=1000)
    args = p.parse_args()
    traces = {str(run): list(records(run/'environment/trace.jsonl')) for run in args.runs}
    shared = min(map(len, traces.values())) // args.window * args.window
    result = dict(matched_observations=shared, runs={})
    for name, rows in traces.items():
        metrics = list(records(Path(name)/'metrics.jsonl'))
        result['runs'][name] = dict(observations=len(rows),
            matched_windows=progress_windows(rows[:shared], args.window),
            recent_windows=progress_windows(rows, args.window)[-4:],
            latest_metrics=metrics[-1] if metrics else {})
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
