#!/usr/bin/env bash
# Install the standalone CLI on the pod's persistent home volume.
set -euo pipefail
cli_version=1.21.0
case "$(uname -m)" in
    x86_64) cli_arch=amd64 ;;
    aarch64|arm64) cli_arch=arm64 ;;
    *) printf 'Unsupported architecture\n' >&2; exit 1 ;;
esac
cli_bin="$HOME/.local/bin/databricks"
mkdir -p "$(dirname -- "$cli_bin")"
if [[ ! -x "$cli_bin" ]] || [[ "$("$cli_bin" --version)" != "Databricks CLI v$cli_version" ]]; then
    cli_tmp=$(mktemp -d)
    trap 'rm -rf -- "$cli_tmp"' EXIT
    curl -fsSL "https://github.com/databricks/cli/releases/download/v${cli_version}/databricks_cli_${cli_version}_linux_${cli_arch}.zip" -o "$cli_tmp/cli.zip"
    python - "$cli_tmp/cli.zip" "$cli_bin" <<'PY'
import shutil
import sys
import zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    with archive.open("databricks") as src, open(sys.argv[2], "wb") as dst:
        shutil.copyfileobj(src, dst)
PY
    chmod +x "$cli_bin"
fi
if [[ -w /usr/local/bin && ! -e /usr/local/bin/databricks ]]; then
    ln -s "$cli_bin" /usr/local/bin/databricks
fi
# Shell configuration lives on the home PVC, so PATH survives pod recreation.
path_entry='export PATH="$HOME/.local/bin:$PATH"'
for startup in "$HOME/.bashrc" "$HOME/.profile"; do
    if ! grep -Fqx "$path_entry" "$startup" 2>/dev/null; then
        printf '\n# Persistent user-installed command-line tools.\n%s\n' "$path_entry" >> "$startup"
    fi
done
export PATH="$HOME/.local/bin:$PATH"
"$cli_bin" --version
printf 'Login: databricks auth login --host https://YOUR-WORKSPACE --profile DEFAULT\n'
