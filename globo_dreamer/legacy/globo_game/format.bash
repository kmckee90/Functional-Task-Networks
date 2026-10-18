#!/usr/bin/env bash

set -euo pipefail

cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"

LUA_FORMAT_BIN="${LUA_FORMAT_BIN:-tools/lua-format/linux-x64/lua-format}" 

if [ "${GITHUB_ACTIONS+x}" ]; then
    find mods -name '*.lua' -print0 | xargs -0 "$LUA_FORMAT_BIN" --check
else
    find mods -name '*.lua' -print0 | xargs -0 "$LUA_FORMAT_BIN" -i
fi
