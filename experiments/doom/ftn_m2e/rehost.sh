#!/usr/bin/env bash
# Run locally. Supervise the connection; training stays on the pod.
set -euo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/astera-sh-oidc.yaml}"
exec 9>"${XDG_RUNTIME_DIR:-/tmp}/doom-monitor-${UID}.lock"
if ! flock -n 9; then
    echo 'Monitor connection supervisor is already running: http://127.0.0.1:8766/'
    exit 0
fi
forward_pid=''
cleanup() {
    if [[ -n "$forward_pid" ]]; then
        kill "$forward_pid" 2>/dev/null || true
        wait "$forward_pid" 2>/dev/null || true
        forward_pid=''
    fi
}
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT
echo 'Monitor: http://127.0.0.1:8766/ (keep this terminal open; reconnects automatically)'
while true; do
    kube=(kubectl -n obelisk --request-timeout=15s)
    # Resolve once per reconnect, avoiding DNS lookups during tunnel setup.
    api_ip=$(timeout 5 getent ahostsv4 kctl.astera.sh | awk 'NR==1 {print $1}' || true)
    if [[ -n "$api_ip" ]]; then
        kube+=(--server="https://${api_ip}:6443" --tls-server-name=kctl.astera.sh)
    fi
    pod=$("${kube[@]}" get pods -l 'astera.org/owner=kmckee90,astera.org/project=vizdoom' \
        --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [[ -z "$pod" ]]; then
        echo 'Cannot reach the Doom pod; retrying in 10 seconds.'
        sleep 10
        continue
    fi
    "${kube[@]}" port-forward "pod/$pod" 8766:8766 9>&- &
    forward_pid=$!
    sleep 3
    failures=0
    while kill -0 "$forward_pid" 2>/dev/null; do
        if curl -fsS --max-time 5 http://127.0.0.1:8766/ -o /dev/null; then
            failures=0
        else
            failures=$((failures + 1))
            if (( failures >= 2 )); then
                echo 'Monitor connection stalled; replacing tunnel.'
                break
            fi
        fi
        sleep 10
    done
    cleanup
    echo 'Reconnecting in 3 seconds. Ctrl-C stops forwarding.'
    sleep 3
done
