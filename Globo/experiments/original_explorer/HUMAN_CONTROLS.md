# Human play controls

Run `play.py` on the pod. Restart the process after source updates, then refresh
the browser. W/S/A/D and arrows work as before; mouse capture adds continuous
look. Esc releases the mouse.

- Left click or F: dig / attack / primary use.
- Right click or E: place / interact.
- 1–4: select Globo's four default inventory slots. The browser readout shows
  the selected slot, item counts, and dig/place counters. Slots 3 and 4 normally
  contain the egg and sleep items.

Number-key taps are retained across in-flight browser requests and until the
game tick consumes them. A later selection supersedes an unprocessed selection.
Inventory and crosshair displays are human-only.

## Exact agent parity

From `Globo/experiments/original_explorer` on the pod:

```bash
/home/dev/globo_dreamer_state/venv/bin/python play.py --exact-parity
```

This mode calls `globo_dreamer.env.Globo` directly, including its shared native
input fix. It does **not** apply `HumanFrameRepeat`. Select an action in the dropdown
or with a shortcut, then press Enter or Step to perform one agent interaction:
eight native engine frames. Holding Enter does not repeat. Actions stay selected
until changed; selecting alone does not step. Reset is explicit after death.
Both the original render and the downsampled agent observation are displayed.
The underlying asynchronous game server continues simulating between Enter
presses, just as it can while an agent computes. This mode pauses action
submission, not the entire world. Images and telemetry are last-step snapshots;
slot telemetry can lag the selection action.

Defaults match the current comparison world: seed 19, spawn `12.0,39.1,120.0`,
256×256 render downsampled to 64×64, full 25-action space. Override with
`--world-seed`, `--spawn`, `--render-size`, or `--reduced-actions` (14 actions).
`--port` still works. `--config`, `--width`, `--height`, `--look` and `--fps`
do not alter this mode; its engine settings come from Globo-Dreamer.
This tests the environment and motor actions, not the policy network.

Runtime files are isolated under `$GLOBO_STATE/human_exact` (default
`/home/dev/globo_dreamer_state/human_exact`) and removed on normal shutdown.
No training checkpoint is loaded and no training run is modified.
Existing `--parity` remains continuous human play.

## Placement workaround (2026-10-20)

The installed Craftium engine drops virtual PLACE between skipped frames.
Its virtual inputs also do not supply the normal `wasKeyPressed` event.
An isolated test with a selected stack of sand and an in-range sandstone target
produced zero placements using native frame skipping.

Human play now uses native frameskip 1, repeats the configured number of frames
in `HumanFrameRepeat`, and holds movement/dig/place through those frames.
Mouse deltas and inventory selections are sent only on the first frame.
Human-only settings enable repeat placement with a 0.01-second threshold.
This uses the normal engine placement checks, inventory consumption, and
callbacks. The probe verified two placements and a stack decrease from 10 to 8.

This human-only workaround does not repair every missing engine press event.
Globo-Dreamer and exact-parity now instead use a patched native engine which
provides dig/place press events and retains place/sneak across skipped frames.
See `../globo_dreamer/INPUT_PROTOCOL.md`. Existing processes need restarting
to adopt the corrected engine; archived placement results used the old engine.

`probe_placement.py` runs an isolated, temporary test world on the pod, asserts
successful placement, and verifies the human inventory telemetry. It does not
touch the active human world. Browser input was separately checked with a quick
slot tap during a simulated 200 ms request delay.

### Single-placement human test

`--exact-parity` now uses the separate `input-v2-single-place` native binary:
PLACE is pressed for the first frame, then released for the rest of the eight
frames. Consecutive place actions each produce a new press. Movement remains
held. This variant has not been enabled for the training agents.

Each new human world starts with four `nodes_nature:silt_wet` (dirt-like) blocks
in slot 1; this item's stack limit is four in the human test only. Press E to
select place, then Enter once per attempt. The control test verified one
placement from one command and three from three commands, including two
consecutive commands, with the target kept clear to expose hold-to-repeat bugs.

Build on the pod with `python -m globo_dreamer.build_input_engine --single-place`.
