# FTN-M2E Doom experiment

`ftn_m2e/` is parent-owned experiment code extracted from the ViZDoom checkout.
It uses the installed ViZDoom Python package; it does not modify the engine.
The clean dependency checkout is retained alongside it in `vizdoom/`.
The package name remains `ftn_m2e` to preserve imports and checkpoint module names.

On the pod, with the existing environment and user-provided WAD:

```bash
PYTHON=/home/dev/vizdoom_state/venv/bin/python bash experiments/doom/run.sh -m ftn_m2e.train --help
```

Use `--wad /absolute/path/to/doom.wad` if the game assets are stored elsewhere.
The local `wad/` directory moved here intact and is ignored. It is not included
in Git or copied to a pod by this change. Outputs still belong in the external
`/home/dev/vizdoom_state` runtime directory. No runs are started by this refactor.

The existing `ftn_m2e/supervise.sh` now resolves this experiment directory itself,
so its imports and default relative WAD path do not depend on the caller's cwd.
The previous notes in `ftn_m2e/README.md` describe the historical experiment setup.
The `vizdoom` actl workspace is mapped here and syncs to `/home/dev/experiment`.
From that remote directory use `bash run.sh ...`. Reconnect from this local
directory with:

```bash
actl pod up vizdoom -n obelisk --sync --remote-path /home/dev/experiment
```

The old remote `/home/dev/workspace` and external runtime directories remain
intact. Since WADs are excluded from source sync, pass `--wad` with the existing
remote asset path when launching from the new directory.
