"""All bulk output must live on the remote shared PVC."""
import json
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent


def root():
    value = os.environ.get('BM_ROOT')
    if not value:
        raise RuntimeError('Source remote_env.sh on a benchmark pod first.')
    path = Path(value).resolve()
    if not path.is_relative_to(Path('/mnt/obelisk-shared')):
        raise RuntimeError(f'Bulk storage must be on the shared PVC: {path}')
    if path.is_relative_to(WORKSPACE):
        raise RuntimeError('Bulk storage must be outside the sync tree.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def bulk(*parts):
    path = root().joinpath(*parts)
    if not path.resolve().is_relative_to(root()):
        raise ValueError(f'Path escapes benchmark storage: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)
