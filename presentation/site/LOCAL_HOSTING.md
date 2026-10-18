# Local proposal and monitor hosting

## Current deployment (October 16, 2026)

The proposal now lives at https://kmckee90.github.io/nascent/. Build output goes
in `/proposal_githubio/` at the repository root, which is the independent
`kmckee90/nascent` Git repository. Preserve its `.git` directory and `.nojekyll`.
Use this directory for site updates, rather than `proposal_offline/`.
The local launcher starts the monitor and a small gateway on port 8080 that
keeps existing Tailscale `/globo/` links working and redirects proposal visits
to GitHub. It does not serve proposal content. Older proposal-hosting
instructions below are historical.
Render, build, and validate on the pod, then copy completed output into the
local deployment directory. Keep GitHub Pages paths relative.

The user authorized local web hosting on October 14, 2026. Training and model
execution remain on the pods. No local training dependencies are needed.

From the repository root:

    bash presentation/site/rehost_local.sh

Or from globo_dreamer, the existing `bash rehost.sh` now uses local hosting after
migration. `GLOBO_REMOTE_HTTP=1 bash rehost.sh` retains the old pod HTTP forwarding
fallback.

- Proposal: http://127.0.0.1:8080/
- Monitor: http://127.0.0.1:8765/continuous/
- Phase 0: http://127.0.0.1:8765/phase0/

The proposal serves local source/assets plus 38.3 MiB of generated media copied
into `~/.cache/globo-local-hosting/media`. Base is no longer required to view it.
Refresh that copy only when new generated assets are published, while base is up:

    bash presentation/site/rehost_local.sh --fetch-media

The monitor's HTTP process is local. It reads current pages and streams clips
from globo-dreamer using Kubernetes exec, with HTTP byte ranges for seeking.
No video history, checkpoint, or training dataset is downloaded to disk. The
existing training supervisor continues producing the HTML and artifacts on the
pod. Clip-length controls still update the remote runtime setting. Kubernetes
access/network is required for live viewing; training continues while the laptop
is asleep or offline. No remote HTTP server is required for this monitor route.

ChatFTN's UI is proxied through a local port forward when its pod is available.
LLM inference still needs the remote app/model running; the hosting launcher does
not start model inference or GPU jobs. A stopped backend does not affect the
proposal or other results pages.

Both local servers bind only to 127.0.0.1. They continue after the launch command
returns; rerun it to reconnect/restart after sleep if needed. PID files and logs
are under ~/.cache/globo-local-hosting. The launcher only replaces its own web
processes and obsolete kubectl forwards on their ports. It never stops a pod or
training worker. This document supersedes the older pod-hosting README.
