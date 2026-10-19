"""Compact read-only progress for the detached detector sweep."""
import json
from pathlib import Path


def main():
    root = Path('/home/dev/sit_state')
    status = json.loads((root / 'reactive_sweep_status.json').read_text())
    for name, state in status.items():
        folder = root / 'runs' / name
        rows = []
        if (folder / 'train.jsonl').exists():
            for line in (folder / 'train.jsonl').read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # A writer may be in the middle of its newest line.
        history = json.loads((folder / 'lifecycle.json').read_text()) if (folder / 'lifecycle.json').exists() else []
        quality = folder / 'reactive_quality' / 'quality_all.json'
        n_quality = len(json.loads(quality.read_text())) if quality.exists() else 0
        counts = [sum(r['changepoint'] for r in rows if r['block'] == b) for b in range(3)]
        near = [[r['step'] for r in rows if r['block'] == b and r['step'] < 50 and r['changepoint']] for b in range(3)]
        print(name, state, 'updates', len(rows), 'evals', len(history), 'quality', n_quality,
              'CP/block', counts, 'CP/first50', near, flush=True)


if __name__ == '__main__':
    main()
