# Globo

One source tree for the Craftium game, the inactive WarpCraft alternative,
and the exploration experiments.

- `craftium/game/`: authoritative Lua game. Consolidates the original game
  copies and preserves the current animal-attack missing-height fix.
- `experiments/globo_dreamer/`: current M2E, FTN-M2E, Dreamer and P2E experiments,
  monitoring and reports. Its `legacy/` package remains actively used; the name
  and Python package paths are retained for checkpoint compatibility.
- `experiments/original_explorer/`: original standalone PyTorch explorer,
  retained for reference. It now uses the shared game as well.
- `warpcraft/explorer/`: inactive JAX explorer. WarpCraft itself is an external
  dependency pinned in its uv.lock, not a bundled copy of the Craftium game.

## Pod workflow

Run `bash connect.sh` locally to sync this entire folder to
`/home/dev/globo_source` on the existing `globo1` workspace. Set `GLOBO_PROJECT`
to target another existing Globo workspace explicitly. Do not sync just the
experiment subfolder: it requires `craftium/game/` beside it.

All five workspace volumes have the migrated source and package alias:
`globo-dreamer`, `globo1`, `globo2`, `globo3`, `globo4`.
For example, `GLOBO_PROJECT=globo3 bash connect.sh` reconnects globo3.
The helper selects the original GPU sizes explicitly: 8 for globo-dreamer,
4 for globo3, and 2 for globo1/globo2/globo4. Pass `--profile cpu-small` for
maintenance. The single actl folder-default record cannot retain five different
profiles, so use this helper rather than relying on its most recent default.

After sync, run `bash /home/dev/globo_source/activate_source.sh` **on the pod**.
This updates the existing package alias without reinstalling dependencies.
Use the existing runtime venv and `PYTHONPATH=/home/dev/globo_dreamer_state/packages`.
Fresh installations use `experiments/globo_dreamer/setup_pod.sh` on the pod.

Checkpoints, replay, engine worlds, results and media remain in the existing
external runtime/shared storage. The reorganization does not migrate or delete
those files. Old remote source trees can remain until continuation is verified.

Generated worlds under the original explorer are historical runtime artifacts,
not additional authoritative game sources. They are excluded from sync/Git.
Historical experiment notes may mention old paths; this document describes the
current layout. The proposal source link and rehost helper use the new paths.
