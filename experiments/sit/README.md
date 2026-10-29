# FTN-SiT

Functional Task Networks continual-learning experiments built on
[SiT](https://github.com/willisma/SiT).

This standalone repository was extracted from `experiments/sit` in
`kmckee90/functional_task_networks`. The experiment package remains `continual`
to preserve imports and serialized module names. Local datasets, model weights,
and generated results are excluded from Git.

## Setup

Clone this repository, then obtain the pinned upstream dependency:

```bash
git clone https://github.com/kmckee90/FTN-SiT.git
cd FTN-SiT
git clone https://github.com/willisma/SiT.git sit
git -C sit checkout --detach cbde832a40b153ccc79603412409da9c9b0c568c
```

The upstream environment specification is in `environment.yml`. Set `PYTHON`
to your environment's interpreter and run from the repository root:

```bash
PYTHON=/path/to/environment/bin/python bash run.sh -m continual.run --help
```

Set `SIT_UPSTREAM` if the dependency is stored somewhere other than `./sit`.
The launcher adds this repository and upstream SiT to `PYTHONPATH`, including
for child processes. It does not install dependencies or start jobs itself.

## Existing cluster workspace

`connect.sh` reconnects the `sit` actl workspace in namespace `obelisk` and
syncs this directory to `/home/dev/experiment`. On that workspace, run
`bash run.sh ...` from `/home/dev/experiment`. Keep bulk runtime output outside
synchronized directories.

The previously retained environment at `/home/dev/sit_state/venv` was recorded
as having an incomplete PyTorch installation; its current health and training
have not been verified as part of this repository extraction.

## Attribution

SiT source retains its original MIT license in `LICENSE.txt`.
