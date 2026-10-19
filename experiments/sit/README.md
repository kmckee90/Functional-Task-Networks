# FTN continual learning with SiT

This directory belongs to the parent Functional Task Networks repository.
`continual/` was moved intact from the upstream SiT checkout, including existing
local results. Results remain on disk but are ignored by Git.

The backbone remains the unmodified `willisma/SiT` checkout at `sit/` beside `continual/`.
The original experiment package name is still `continual`, so serialized module
names, internal imports, and subprocess module names remain unchanged.

On the pod, using the existing experiment environment:

**Environment note:** the retained venv currently has an incomplete PyTorch
installation. See `../../docs/refactoring/POD_MIGRATION.md` for the verified
import-check workaround; training has not been restarted or validated.

```bash
PYTHON=/home/dev/sit_state/venv/bin/python bash experiments/sit/run.sh -m continual.run --help
```

Set `PYTHON` to the actual environment's interpreter. Set `SIT_UPSTREAM` to an
absolute SiT checkout path when it is not at the default location. The launcher
adds both the experiment directory and upstream SiT to `PYTHONPATH`, including
for child processes. It does not install dependencies or launch jobs by itself.
The pinned upstream revision is recorded in `../upstreams.json`.

For a fresh parent checkout, obtain the upstream dependency without a fork:

```bash
git clone https://github.com/willisma/SiT.git experiments/sit/sit
git -C experiments/sit/sit checkout --detach cbde832
```

The `sit` actl workspace is mapped to this experiment directory and syncs to
`/home/dev/experiment`. From that remote directory use `bash run.sh ...`.
Reconnect from this local directory with:

```bash
actl pod up sit -n obelisk --sync --remote-path /home/dev/experiment
```

Keep bulk runtime output outside synchronized directories. The old remote
`/home/dev/workspace` is preserved but is no longer this session's source root.
