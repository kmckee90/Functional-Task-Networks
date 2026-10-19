"""Run on a pod to check a migrated source tree without starting an agent."""
import ast
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parent
files = list((root / 'experiments/globo_dreamer').rglob('*.py'))
for path in files:
    ast.parse(path.read_text(), filename=str(path))
for name in ('connect.sh', 'activate_source.sh'):
    subprocess.run(['bash', '-n', str(root / name)], check=True)

from globo_dreamer.legacy.env import DEFAULT_GLOBO_SRC
from globo_dreamer.legacy.agent import GloboExplorerDQN

assert pathlib.Path(DEFAULT_GLOBO_SRC).resolve() == root / 'craftium/game'
assert (pathlib.Path(DEFAULT_GLOBO_SRC) / 'game.conf').is_file()
for name in ('runs', 'artifacts', 'craftium_globo_env', 'craftium_play_env'):
    assert not (root / 'experiments/original_explorer' / name).exists(), name
print(f'PASS: {len(files)} Python files, shell syntax, explorer imports, game path, bulk exclusions.')
print(f'Game: {DEFAULT_GLOBO_SRC}')
