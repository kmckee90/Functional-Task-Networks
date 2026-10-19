"""Finish audits and reports after the detached threshold workers complete."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/home/dev/sit_state')
    while True:
        status = json.loads((root / 'reactive_sweep_status.json').read_text())
        failed = {k: v for k, v in status.items() if v['exit_code'] not in (None, 0)}
        if failed:
            raise RuntimeError(f'Threshold workers failed: {failed}')
        if len(status) == 6 and all(v['phase'] == 'quality' and v['exit_code'] == 0 for v in status.values()):
            break
        time.sleep(30)
    for module, args in [
        ('reactive_verify', ['--sweep']),
        ('reactive_report', []),
        ('reactive_report', ['--sd', '3']),
        ('reactive_report', ['--sd', '1']),
        ('reactive_sweep_report', []),
    ]:
        subprocess.run([sys.executable, '-m', f'continual.{module}', *args], check=True)
    (root / 'reactive_finished.json').write_text(json.dumps({'audits_and_reports_complete': True}))
    print('AUDITS_AND_REPORTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
