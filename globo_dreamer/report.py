"""Read-only behavioral summary. Run on the pod, including during training."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def records(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A concurrent writer may not have finished its final line.
                    continue


def window(rows):
    if not rows:
        return {}
    actions = Counter(x['action'] for x in rows if x['life_step'] > 0)
    count = max(1, sum(actions.values()))
    cells = {(math.floor(x['position'][0] / 4),
              math.floor(x['position'][2] / 4)) for x in rows}
    result = dict(
        steps=[rows[0]['step'], rows[-1]['step']], observations=len(rows),
        occupied_4m_cells=len(cells), deaths=sum(x['terminal'] for x in rows),
        health_range=[min(x['hp'] for x in rows), max(x['hp'] for x in rows)],
        mean_novelty=sum(x['density'] for x in rows) / len(rows),
        action_entropy_nats=-sum((v/count)*math.log(v/count) for v in actions.values()),
        action_counts=dict(actions),
    )
    measured = [r for r in rows if 'low_detail' in r]
    if measured:
        result.update(low_detail_fraction=sum(r['low_detail'] for r in measured)/len(measured),
            sky_dominated_fraction=sum(r['sky_fraction']>.85 for r in measured)/len(measured),
            looking_up_fraction=sum(r.get('look_pitch',0)<-.8 for r in measured)/len(measured),
            looking_down_fraction=sum(r.get('look_pitch',0)>.8 for r in measured)/len(measured))
    return result


def progress_windows(rows, width):
    """Nonoverlapping windows; first visits are measured against all prior rows.

    Spatial telemetry is diagnostic only and never supplied to either policy.
    First visits do not count revisiting spawn after a death as new territory.
    """
    seen = set()
    result = []
    for start in range(0, len(rows), width):
        block = rows[start:start + width]
        cells = {(math.floor(x['position'][0] / 4),
                  math.floor(x['position'][2] / 4)) for x in block}
        summary = window(block)
        summary.pop('action_counts', None)
        summary.update(new_4m_cells=len(cells - seen),
                       complete_window=len(block) == width)
        seen.update(cells)
        summary['cumulative_4m_cells'] = len(seen)
        result.append(summary)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--window', type=int, default=1000)
    args = parser.parse_args()
    if args.window < 1:
        parser.error('window must be positive')
    env = args.run / 'environment'
    trace = list(records(env / 'trace.jsonl'))
    episodes = list(records(env / 'episodes.jsonl'))
    metrics = list(records(args.run / 'metrics.jsonl'))
    feedback = list(records(args.run / 'agent_feedback.jsonl'))
    print(json.dumps(dict(
        run=str(args.run), observations=len(trace),
        first=window(trace[:args.window]), last=window(trace[-args.window:]),
        progress_windows=progress_windows(trace, args.window),
        latest_state=trace[-1] if trace else {},
        completed_lives=[{k: x[k] for k in ('life', 'life_step', 'game_seconds',
            'terminal', 'truncated', 'death_reason', 'cells_life')} for x in episodes],
        latest_metrics=metrics[-1] if metrics else {},
        latest_feedback=feedback[-1] if feedback else {},
        videos=[str(x) for x in sorted(env.glob('behavior_*.mp4'))],
        interpretation='Descriptive only; improvement requires matched comparisons. '
            'Early and late windows overlap when the run is short.'), indent=2))


if __name__ == '__main__':
    main()
