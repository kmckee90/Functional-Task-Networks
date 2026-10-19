"""Detached pod-side supervisor. Does not depend on laptop SSH or sync lifetime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=600)
    args = parser.parse_args()
    root = Path('/home/dev/sit_state')
    jobs = []
    for i, seed in enumerate([123, 124, 125]):
        jobs.append((2*i, f'bayes_replay_s{seed}', ['--seed', str(seed), '--replay', '1']))
        jobs.append((2*i+1, f'bayes_noreplay_s{seed}', ['--seed', str(seed), '--replay', '0']))
    jobs.append((6, 'plain_lora_s123', ['--seed', '123', '--replay', '0', '--method', 'plain']))
    jobs.append((7, 'bayes_replay_reverse_s123', ['--seed', '123', '--replay', '1', '--order', 'sketch,poster,monochrome']))
    running = []
    for gpu, name, extra in jobs:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', HF_HOME=str(root/'hf'))
        command = [sys.executable, '-u', '-m', 'continual.worker', '--name', name,
                   '--steps', str(args.steps), '--resume'] + extra
        log = (root/f'{name}.log').open('a', buffering=1)
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        running.append((name, proc, log))
    while True:
        status = {name: {'pid': p.pid, 'exit_code': p.poll()} for name,p,_ in running}
        (root/'supervisor_status.json').write_text(json.dumps(status, indent=2))
        with (root/'artifacts.log').open('a') as output:
            subprocess.run([sys.executable, '-m', 'continual.artifacts'], stdout=output,
                           stderr=subprocess.STDOUT)
        if all(p.poll() is not None for _,p,_ in running):
            break
        time.sleep(60)
    for _,_,log in running:
        log.close()
    if any(p.returncode for _,p,_ in running):
        raise RuntimeError('One or more experiments failed; see supervisor_status.json and individual logs.')


if __name__ == '__main__':
    main()
