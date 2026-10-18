"""Pod-side diagnostics of completed lives and stored novelty targets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .memory import Recode, embedding
from .report import records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--after-step', type=int, default=0)
    parser.add_argument('--before-step', type=int)
    args = parser.parse_args()
    lives = list(records(args.run / 'environment/episodes.jsonl'))
    lives = [r for r in lives if r['step'] > args.after_step and
             (args.before_step is None or r['step'] <= args.before_step)]
    result = {'completed_lives': len(lives)}
    for name, rows in [('first_5', lives[:5]), ('last_5', lives[-5:])]:
        if rows:
            result[name] = dict(
                mean_seconds=float(np.mean([r['game_seconds'] for r in rows])),
                median_seconds=float(np.median([r['game_seconds'] for r in rows])),
                mean_cells=float(np.mean([r['cells_life'] for r in rows])),
                deaths=dict(Counter(r['death_reason'] for r in rows)))
    chunks = sorted((args.run / 'replay').glob('*.npz'))
    if chunks:
        memory = Recode()
        memory.load(args.run / 'environment/memory.npz')
        past, current = [], []
        for path in chunks[::max(1, len(chunks)//12)]:
            with np.load(path) as data:
                for i in range(0, len(data['image']), 32):
                    past.append(float(data['density'][i, 0]))
                    current.append(memory.query(embedding(data['image'][i])))
        past, current = np.array(past), np.array(current)
        result['density_relabel_audit'] = dict(samples=len(past),
            collection_mean=float(past.mean()), current_mean=float(current.mean()),
            mean_absolute_change=float(np.abs(current-past).mean()),
            note='Current memory is the latest saved snapshot; values are diagnostic.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
