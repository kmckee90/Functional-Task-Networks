#!/usr/bin/env bash
# Execute ON THE POD. Runtime and packages live outside the synced source.
set -euo pipefail
STATE=${GLOBO_STATE:-/home/dev/globo_dreamer_state}
SOURCE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$STATE/cache" "$STATE/vendor" "$STATE/packages"
export UV_CACHE_DIR="$STATE/cache/uv" UV_PYTHON_INSTALL_DIR="$STATE/python"
export PYTHONDONTWRITEBYTECODE=1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libx11-6 libxext6 libxxf86vm1 libxi6 libxcursor1 libxrandr2 libxinerama1 libgl1 libglu1-mesa libegl1 libegl-mesa0 libgles2 libgbm1
uv python install 3.11
if [ ! -x "$STATE/venv/bin/python" ]; then uv venv --python 3.11 "$STATE/venv"; fi
uv pip install --python "$STATE/venv/bin/python" -r "$SOURCE/requirements-pod.txt"
if [ ! -d "$STATE/vendor/dreamerv3/.git" ]; then
  git clone https://github.com/danijar/dreamerv3 "$STATE/vendor/dreamerv3"
fi
git -c safe.directory="$STATE/vendor/dreamerv3" -C "$STATE/vendor/dreamerv3" checkout e01491fad6434b2245a3b8ca201dd7faedcc458c
# Package alias supports a pod synced directly from globo_dreamer/.
ln -sfn "$SOURCE" "$STATE/packages/globo_dreamer"
printf '%s\n' "Use PYTHONPATH=$STATE/packages and $STATE/venv/bin/python"
