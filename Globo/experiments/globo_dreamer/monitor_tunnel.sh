#!/usr/bin/env bash
# Local hosting infrastructure only: reconnect a stalled Kubernetes tunnel.
set -u
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/astera-sh-oidc.yaml}"
monitor_port="${GLOBO_MONITOR_PORT:-8016}"
monitor_pod="${GLOBO_MONITOR_POD:-globo-actions}"
exec 9>"/tmp/globo-monitor-${monitor_port}-tunnel.lock"
flock -n 9 || exit 0
tunnel_pid=''
cleanup() {
  if [[ -n "$tunnel_pid" ]]; then
    kill "$tunnel_pid" 2>/dev/null || true
    wait "$tunnel_pid" 2>/dev/null || true
  fi
}
trap 'cleanup; exit' TERM INT EXIT
while true; do
  printf '%s Starting monitor tunnel\n' "$(date -Is)"
  kubectl -n obelisk port-forward "deployment/actl-ws-kmckee90-${monitor_pod}-devspace" \
    "${monitor_port}:${monitor_port}" >"/tmp/globo-monitor-${monitor_port}-forward.log" 2>&1 &
  tunnel_pid=$!
  sleep 3
  while kill -0 "$tunnel_pid" 2>/dev/null; do
    if ! curl --fail --silent --max-time 8 "http://localhost:${monitor_port}/api/clip-settings" >/dev/null; then
      printf '%s Tunnel health check failed; reconnecting\n' "$(date -Is)"
      break
    fi
    sleep 15
  done
  cleanup
  tunnel_pid=''
  sleep 2
done
