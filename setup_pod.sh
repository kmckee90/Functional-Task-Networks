#!/usr/bin/env bash
#
# setup_pod.sh — one-shot provisioning for a fresh actl pod.
#
# Reconciles Python-version selection across this repo's two uv projects and
# installs globo_explorer's craftium system libraries. Run once from the
# workspace root after every pod recreate (`actl pod up`):
#
#   bash setup_pod.sh
#
# After this, bare `uv run python <script>` picks the correct interpreter in
# every project — no `-p` / `--python` flag needed:
#
#   uv run python base/run_experiments.py            # 3.12  (root project)
#   uv run python rl/robotics/train_mt10_expert.py   # 3.12  (root project)
#   cd globo_explorer && uv run python train.py      # 3.11  (craftium)
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- 1. Neutralize the image's UV_PYTHON override ---------------------------
# The astera base image exports UV_PYTHON=python3.12 into every shell. In uv's
# interpreter-selection order that env var OUTRANKS each project's
# .python-version and requires-python (only an explicit `-p`/`--python` flag
# beats it), so bare `uv run`/`uv sync` always resolved to 3.12 — which breaks
# globo_explorer (craftium is a cp311 wheel, needs 3.11). Unset it so the
# per-project .python-version files decide instead.
#
# Both files are on the container ROOT FS (not the home PVC), so `actl pod
# commit` bakes this fix into the image — every future pod, even one on a fresh
# PVC, selects the right interpreter with zero per-shell setup:
#   - /etc/profile.d/... : login shells
#   - /etc/bash.bashrc   : interactive shells (the `actl pod up` terminal)
printf 'unset UV_PYTHON\n' > /etc/profile.d/00-unset-uv-python.sh 2>/dev/null || true
if ! grep -qsF 'unset UV_PYTHON' /etc/bash.bashrc 2>/dev/null; then
  echo 'unset UV_PYTHON  # actl: let per-project .python-version win (setup_pod.sh)' >> /etc/bash.bashrc
fi
unset UV_PYTHON   # take effect for the rest of this script too

# --- 2. Root project (Python 3.12) ------------------------------------------
# base/run_experiments.py, rl/robotics/train_mt10_expert.py, explorer/*, etc.
( cd "$HERE" && uv sync )

# --- 3. globo_explorer (Python 3.11 + craftium system libs) -----------------
bash "$HERE/globo_explorer/setup_pod.sh"

cat <<'EOF'

[setup_pod] done. Open a fresh shell (or `unset UV_PYTHON` in this one), then:
    uv run python base/run_experiments.py            # 3.12
    uv run python rl/robotics/train_mt10_expert.py   # 3.12
    cd globo_explorer && uv run python train.py      # 3.11

After `actl pod commit`, the image carries the system libs + the UV_PYTHON fix,
so:
  - same workspace (PVC kept on pod down/up): nothing to run — venvs persist.
  - fresh PVC (teammate / --delete-data): just re-run this script; it only needs
    to rebuild the venvs (`uv sync` auto-installs the 3.11 interpreter). The ~12G
    of CUDA venvs are too large to bake into the image, so that step can't be
    eliminated.
EOF
