from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config file."""
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML configs require PyYAML. Install with `uv pip install -r requirements.txt`."
        ) from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {p}")
    return data


def get_path(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def run_dir(cfg: dict[str, Any], model_name: str | None = None) -> Path:
    root = artifact_path(cfg, get_path(cfg, "run", "output_dir", default="runs"))
    name = get_path(cfg, "run", "name", default="long_context_run")
    return root / name / model_name if model_name else root / name


def processed_dir(cfg: dict[str, Any]) -> Path:
    return artifact_path(cfg, get_path(cfg, "data", "processed_dir", default="data/processed/default"))


def artifact_path(cfg: dict[str, Any], path: str | Path) -> Path:
    """Resolve generated-artifact paths under the configured artifact root.

    Configs can keep ergonomic relative paths such as `runs` and
    `data/processed/foo`; when `run.artifact_root` or
    `LONG_CONTEXT_ARTIFACT_ROOT` is set, those paths are rooted outside the
    synced workspace. Absolute paths remain absolute.
    """
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if p.is_absolute():
        return p
    root = _artifact_root(cfg)
    return root / p if root is not None else p


def _artifact_root(cfg: dict[str, Any]) -> Path | None:
    root = get_path(cfg, "run", "artifact_root", default=None)
    if root is None:
        root = os.environ.get("LONG_CONTEXT_ARTIFACT_ROOT")
    if not root:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(root))))


def seed(cfg: dict[str, Any]) -> int:
    return int(get_path(cfg, "run", "seed", default=0))
