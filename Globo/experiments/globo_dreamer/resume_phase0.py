"""Resume one retained phase-0 run on an explicitly chosen unused GPU (pod only)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from .continuous_runs import STATE, ROOT
from .search import atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', nargs='?')
    parser.add_argument('--gpu', type=int, choices=range(8))
    args = parser.parse_args()
    archived = json.loads((STATE/'phases/phase0/state.json').read_text())
    if args.name is None:
        for trial in archived:
            print(trial['name'], 'seed', trial.get('seed', 7))
        return
    if args.gpu is None:
        parser.error('--gpu is required when resuming a run')
    trial = next((dict(t) for t in archived if t['name'] == args.name), None)
    if trial is None:
        parser.error('Unknown phase-0 run')
    if not (STATE/'runs'/args.name/'ckpt/latest').exists():
        parser.error('No retained latest checkpoint')
    path = ROOT/'state.json'
    current = json.loads(path.read_text())
    def check_available(trials):
        if any(t['status'] in ('running', 'pending') and
               (t['gpu'] == args.gpu or t['name'] == args.name) for t in trials):
            raise RuntimeError('GPU or run already assigned to an active trial')
    check_available(current)
    # The launcher remains alive and adopts existing workers after reloading.
    pid = int((ROOT/'supervisor.pid').read_text())
    if b'globo_dreamer.continuous_runs' not in Path(f'/proc/{pid}/cmdline').read_bytes():
        raise RuntimeError('Unexpected supervisor PID')
    os.kill(pid, signal.SIGKILL)
    with (ROOT/'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads(path.read_text())
        check_available(current)
        atomic(ROOT/f'state.before-resume-{time.time_ns()}.json', current)
        trial.update(gpu=args.gpu, status='pending', phase='phase0-resumed')
        for key in ('pid', 'retry_after', 'pause_reason', 'archive_display'):
            trial.pop(key, None)
        current = [t for t in current if t['name'] != args.name] + [trial]
        atomic(path, current)
    print('Queued checkpoint resume:', args.name, 'on GPU', args.gpu)
    print('Launcher reloads within 30 seconds; phase-0 frozen monitor stays unchanged.')


if __name__ == '__main__':
    main()
