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

## Watching a fresh FTN-M2E agent live

Connect locally with `GLOBO_PROJECT=globo1 bash connect.sh --port 8000:8000`.
On the pod, run:

```bash
bash /home/dev/globo_source/experiments/original_explorer/setup_pod.sh --system-only
bash /home/dev/globo_source/experiments/globo_dreamer/start_live.sh my_fresh_run
```

Open `http://localhost:8000`. The Phase-4-style conv FTN-M2E run starts with
fresh weights and buffers on world 19, uses GPU 0, and has no step limit.
The three panels show current 256px observations, the action mask/EMA, and
the frontier observation. This is a spectator applet; browser input does not
control the agent. Frames are streamed directly from the training trajectory,
with a visible age indicator during training/checkpoint pauses. No additional
frame files are created. Checkpoints and logs remain under
`/home/dev/globo_dreamer_state/runs/<name>`; SIGINT/SIGTERM saves before exit.
The launcher refuses existing run names to prevent accidental checkpoint reuse.
For an explicit continuation, use `globo_dreamer.train` with the same launch
arguments recorded in the run's `launch.json`, including `--live-port 8000`.

### Live terrain and coverage view

The read-only sidecar adds the Phase 4/5 terrain map without restarting the
agent. Observation and map have equal size; the half-size goal and FTN mask
are stacked between them. On the pod:

```bash
PYTHONPATH=/home/dev/globo_source/experiments \
  /home/dev/globo_dreamer_state/venv/bin/python -m globo_dreamer.live_map_viewer \
  --run /home/dev/globo_dreamer_state/runs/YOUR_RUN --port 8001
```

Forward port 8001 as well (`connect.sh --port 8000:8000 --port 8001:8001`)
and open `http://localhost:8001`. The browser polls position/trail every 1.5s;
the source diagnostics update every 50/100 agent steps respectively. Coverage
is redrawn every 10s. Terrain is cached in the viewer and refreshed every 10s
from the existing seed-19 tile archive, independently of training. Uncached
areas appear gray; this viewer does not generate extra simulation worlds.
