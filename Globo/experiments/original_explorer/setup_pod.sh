#!/usr/bin/env bash
#
# setup_pod.sh — provision a fresh actl pod to run globo_explorer (train.py).
#
# WHY THIS EXISTS
#   The prebuilt craftium/Luanti engine is a cp311 manylinux wheel that renders
#   headlessly via EGL (no X server / display needed). To run it the pod needs:
#     (a) CPython 3.11  — the wheel is cp311-only (see pyproject: requires-python)
#     (b) X11 client libs — the Luanti binary is dynamically linked against them
#     (c) EGL / GLES / GBM — for offscreen OpenGL rendering
#   The base image (astera-pytorch-cuda-n-jax) ships none of (b)/(c), and these
#   apt packages live on the container root fs, so they are LOST on every pod
#   recreate (`actl pod down` -> `pod up`). Re-run this once after each recreate.
#   (The uv .venv lives under the home PVC and survives a plain `pod down`.)
#
# USAGE (on the pod, from the workspace root):
#   bash globo_explorer/setup_pod.sh
#
set -euo pipefail

# --- 1. System libraries (root fs; not persisted across pod recreate) -------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  libx11-6 libxext6 libxxf86vm1 libxi6 libxcursor1 libxrandr2 libxinerama1 \
  libgl1 libglu1-mesa \
  libegl1 libegl-mesa0 libgles2 libgbm1

# Existing Globo workspaces retain their Python venv on the home volume.
# Repair only the container's system libraries without reinstalling Python.
if [[ ${1:-} == --system-only ]]; then
  echo '[setup_pod] Craftium system libraries ready; existing Python environment unchanged.'
  exit 0
fi

# --- 2. Python 3.11 + project dependencies ----------------------------------
# craftium is cp311-only; force uv to a managed 3.11 regardless of the system
# interpreter (the base image's default python3 is 3.12, which uv picks first).
cd "$(dirname "$0")"   # -> globo_explorer/
uv python install 3.11
uv sync -p 3.11

cat <<'EOF'

[setup_pod] done. System libs + Python 3.11 environment are ready.
            Once UV_PYTHON is neutralized (repo-root setup_pod.sh handles this),
            no -p flag is needed:
              cd globo_explorer && uv run python train.py
EOF
