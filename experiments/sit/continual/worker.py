"""One GPU's detached lifecycle followed by generated-distribution evaluation."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args, rest = parser.parse_known_args()
    subprocess.run([sys.executable, '-u', '-m', 'continual.run', '--name', args.name] + rest, check=True)
    command = [sys.executable, '-u', '-m', 'continual.quality', '--name', args.name]
    if args.name in ['bayes_replay_s123', 'bayes_noreplay_s123']:
        command.append('--all-stages')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
