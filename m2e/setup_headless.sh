#!/usr/bin/env bash
# Run on the pod when CPU-only MuJoCo rendering is needed.
set -euo pipefail
apt-get update -qq
apt-get install -y -qq libosmesa6
printf 'Use MUJOCO_GL=osmesa for headless CPU rendering.\n'
