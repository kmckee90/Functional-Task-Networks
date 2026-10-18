#!/usr/bin/env bash
# Laptop-side helper. Prefer local hosting after its one-time media migration.
# Optional: GLOBO_PORT=8876 GLOBO_NAMESPACE=obelisk GLOBO_OWNER=kmckee90 ./rehost.sh
set -euo pipefail
local_host="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../presentation/site" && pwd)/rehost_local.sh"
if [[ ${GLOBO_REMOTE_HTTP:-0} != 1 && -f ${XDG_CACHE_HOME:-$HOME/.cache}/globo-local-hosting/media/globo-m2e.mp4 ]]; then
  exec bash "$local_host"
fi
die() { printf 'rehost: %s\n' "$*" >&2; exit 1; }
for tool in kubectl curl flock setsid; do
  command -v "$tool" >/dev/null || die "Missing $tool; install it and rerun."
done
namespace=${GLOBO_NAMESPACE:-obelisk}
owner=${GLOBO_OWNER:-kmckee90}
project=globo-dreamer
port=${GLOBO_PORT:-8765}
[[ $port =~ ^[0-9]+$ ]] && ((port >= 1024 && port <= 65515)) || die 'GLOBO_PORT must be 1024–65515.'
cache=${XDG_CACHE_HOME:-$HOME/.cache}/globo-dreamer-rehost
mkdir -p "$cache"
exec 9>"$cache/lock"
flock -w 60 9 || die 'Another rehost is still working. Try again shortly.'
kube=(kubectl)
if [[ -z ${KUBECONFIG:-} && -f $HOME/.kube/astera-sh-oidc.yaml ]]; then
  kube+=(--kubeconfig "$HOME/.kube/astera-sh-oidc.yaml")
fi
kube+=(-n "$namespace")
printf 'Finding the current %s pod…\n' "$project"
pods=$("${kube[@]}" get pods --request-timeout=30s \
  -l "astera.org/owner=$owner,astera.org/project=$project" --sort-by=.metadata.creationTimestamp \
  -o 'jsonpath={range .items[*]}{.metadata.name}{" "}{.status.phase}{" "}{.metadata.deletionTimestamp}{"\n"}{end}') || die 'Cluster access failed. Check network/VPN and Kubernetes login, then rerun.'
pod=
while read -r name phase deleting; do
  if [[ $phase == Running && -z $deleting ]]; then pod=$name; fi
done <<< "$pods"
[[ -n $pod ]] || die 'No running Globo pod. Start/resume it with actl pod up globo-dreamer -n obelisk, then rerun. No training or pod was restarted.'
"${kube[@]}" wait --for=condition=Ready "pod/$pod" --timeout=60s >/dev/null || die 'Pod is not ready yet; rerun once it is ready.'
printf 'Checking the monitor server on %s…\n' "$pod"
"${kube[@]}" exec -i "$pod" -- /home/dev/globo_dreamer_state/venv/bin/python - <<'PY' || exit 1
import fcntl, os, signal, subprocess, time, urllib.request
from pathlib import Path
state = Path('/home/dev/globo_dreamer_state')
def healthy():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765/api/clip-settings', timeout=3) as r:
            return r.status == 200
    except Exception:
        return False
with (state / 'gallery-rehost.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not healthy():
        # Only replace this project's gallery process, never training workers.
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                args = (proc / 'cmdline').read_bytes().split(b'\0')
                if b'globo_dreamer.gallery_server' in args:
                    if b'--port' in args and args[args.index(b'--port')+1] != b'8765':
                        continue
                    os.kill(int(proc.name), signal.SIGTERM)
            except (OSError, IndexError):
                pass
        time.sleep(1)
        env = dict(os.environ, PYTHONPATH=str(state / 'packages'),
                   GLOBO_STATE=str(state), PYTHONDONTWRITEBYTECODE='1')
        with (state / 'gallery-server.log').open('ab') as log:
            subprocess.Popen([str(state / 'venv/bin/python'), '-u', '-m',
                'globo_dreamer.gallery_server', '--port', '8765'], cwd=state,
                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, close_fds=True)
        for _ in range(20):
            if healthy(): break
            time.sleep(1)
        else:
            raise SystemExit('Monitor server could not start. See /home/dev/globo_dreamer_state/gallery-server.log on the pod.')
    if not (state / 'reviews/config24/continuous/index.html').exists():
        raise SystemExit('Server is ready, but the monitor page is missing. Training has not been changed.')
PY

# Remove only this workspace's forwards to the gallery port, including stale
# tunnels from earlier manual rehosting. Leave unrelated port users alone.
for proc in /proc/[0-9]*; do
  args=()
  [[ -r $proc/cmdline && -O $proc ]] || continue
  mapfile -d '' -t args < "$proc/cmdline" 2>/dev/null || continue
  ((${#args[@]})) || continue
  [[ ${args[0]##*/} == kubectl ]] || continue
  is_forward=false; is_project=false; is_port=false; is_namespace=false
  for ((i=0; i<${#args[@]}; i++)); do
    [[ ${args[i]} == port-forward ]] && is_forward=true
    [[ ${args[i]} == "pod/actl-ws-$owner-$project-"* ]] && is_project=true
    [[ ${args[i]} =~ ^[0-9]+:8765$ ]] && is_port=true
    if [[ ${args[i]} == -n || ${args[i]} == --namespace ]]; then
      [[ ${args[i+1]:-} == "$namespace" ]] && is_namespace=true
    fi
  done
  if $is_forward && $is_project && $is_port && $is_namespace; then
    kill "${proc##*/}" 2>/dev/null || true
  fi
done
sleep 1
for ((candidate=port; candidate<=port+20; candidate++)); do
  log="$cache/tunnel.log"
  nohup setsid "${kube[@]}" port-forward --address 127.0.0.1 "pod/$pod" "$candidate:8765" \
    >"$log" 2>&1 < /dev/null 9>&- &
  tunnel=$!
  for ((attempt=0; attempt<20; attempt++)); do
    if ! kill -0 "$tunnel" 2>/dev/null; then break; fi
    # Require this tunnel to have bound the port, not just some other HTTP server.
    if grep -q "Forwarding from 127.0.0.1:$candidate" "$log" &&
       curl -fsS --max-time 3 "http://127.0.0.1:$candidate/continuous/" -o /dev/null; then
      printf '\nMonitor ready: http://127.0.0.1:%s/continuous/\n' "$candidate"
      printf 'Phase 0 archive: http://127.0.0.1:%s/phase0/\n' "$candidate"
      printf 'The tunnel stays open after this script exits. Run this script again after laptop sleep.\n'
      exit 0
    fi
    sleep 1
  done
  kill "$tunnel" 2>/dev/null || true
  if ! grep -qiE 'address already in use|unable to listen' "$log"; then
    cat "$log" >&2
    die 'Tunnel failed. Check network/VPN and rerun.'
  fi
done
die "Ports $port through $((port+20)) are occupied. Set GLOBO_PORT to another port and rerun."
