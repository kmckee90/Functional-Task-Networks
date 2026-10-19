#!/usr/bin/env bash

set -euo pipefail

cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"

version=3.13.2
dir="tools/lua-language-server/linux-x64"

if [ -d "$dir" ]; then
    installed_version=""
    if [ -f "$dir/VERSION" ]; then
        installed_version=$(<"$dir/VERSION")
    fi
    if [ "$version" != "$installed_version" ]; then
        rm -r "$dir"
    fi
fi

if [ ! -d "$dir" ]; then
    echo "Downloading lua-language-server $version..."
    mkdir "$dir" >/dev/null
    wget -qO- https://github.com/LuaLS/lua-language-server/releases/download/3.13.2/lua-language-server-3.13.2-linux-x64.tar.gz | tar xz -C "$dir"
    echo "$version" > "$dir/VERSION"
fi

"$dir/bin/lua-language-server" --checklevel Information --check . >/dev/null

# See https://github.com/LuaLS/lua-language-server/issues/2830#issuecomment-2315627616
if [ "${GITHUB_ACTIONS+x}" ]; then
    jq -r '
        to_entries[] |
        (.key | sub("^.*?\\./"; "")) as $file | 
        .value[] |
        .code as $title |
        (.range.start.line + 1) as $line |
        (.range.start.character + 1) as $col |
        .message as $message |
        "\($file):\($line):\($col): \($message)"
        + "\n::error file=\($file),line=\($line),col=\($col),title=\($title)::\($message)"
    ' "$dir/log/check.json"
else
    jq -r '
        to_entries[] |
        (.key | sub("^.*?\\./"; "")) as $file | 
        .value[] |
        .code as $title |
        (.range.start.line + 1) as $line |
        (.range.start.character + 1) as $col |
        .message as $message |
        "\($file):\($line):\($col): \($message)"
    ' "$dir/log/check.json"
fi

if [ $(jq -r 'length' "$dir/log/check.json") -gt 0 ]; then
    exit 1
fi
