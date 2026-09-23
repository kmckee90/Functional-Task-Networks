"""_paths.py - make the parent base/ package importable from composition/.

Every composition entry script does `import _paths` (first, before importing any
base module) so that the shared model / config / task modules that live in
../ (base/) resolve by their real names: `import config`, `import ftn`,
`import stack_ftn`, `import lora_rnn_ftn`, `import metrics`, `import tasks`, ...

Single source of truth: composition never copies base model code, it imports it.
The consolidated experiment+model config for this project is `compose_config`
(deliberately NOT named `config`, so it can't shadow base/config.py on sys.path).
"""

import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.append(_BASE)
