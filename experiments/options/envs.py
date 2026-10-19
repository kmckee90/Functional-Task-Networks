"""Tiny env registry / factory shared by the trainer, the collection workers, and
the demo, so the active environment is selected in ONE place (a config string).

Deliberately torch-free (only the env modules, which are numpy + gymnasium): the
multiprocessing 'spawn' collection workers in ``_par_collect.py`` import this and
must never touch CUDA. Both worlds expose the same interface the pipeline uses
(``obs_dim``, ``action_space``, ``reset``/``step`` with pos/heading/speed info,
``config()`` for faithful reconstruction, and ``render_world``).
"""

from __future__ import annotations

from color_world import ColorWorld
from continuous_world import ContinuousWorld

# Canonical selector keys -> env class.
ENVS = {"color": ColorWorld, "continuous": ContinuousWorld}

# Accept the class names too (older graph.pt files stored "ColorWorld").
_ALIASES = {"colorworld": "color", "continuousworld": "continuous"}


def env_key(name: str | None) -> str:
    """Normalize a (possibly None / class-name) selector to a canonical key."""
    if name is None:
        return "color"
    k = str(name).lower()
    return _ALIASES.get(k, k)


def make_env(name: str | None, **kwargs):
    """Build the selected env. ``kwargs`` are passed straight to its constructor
    (use ``env.config()`` to get the exact kwargs that rebuild a given world)."""
    key = env_key(name)
    if key not in ENVS:
        raise ValueError(f"unknown env {name!r}; choose from {sorted(ENVS)}")
    return ENVS[key](**kwargs)
