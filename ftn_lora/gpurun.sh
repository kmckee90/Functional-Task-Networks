#!/usr/bin/env bash
#
# gpurun.sh - launch a run_experiments.py run on an automatically chosen free
# GPU, backgrounded with nohup and logged. Replaces the hand-written
#
#   CUDA_VISIBLE_DEVICES=2 nohup uv run python run_experiments.py \
#       --config BayesLoRA Smoke NoReplay ... &
#
# with
#
#   ./gpurun.sh BayesLoRA Smoke NoReplay DomainShift
#
# Usage:
#   ./gpurun.sh [options] [CONFIG ...] [-- EXTRA run_experiments.py ARGS]
#   With no CONFIG, runs BayesLoRA. Paths are relative to this script's folder.
#
# Options:
#   -g, --gpu N       Force GPU index N instead of auto-picking a free one.
#   -n, --name NAME   Run name (forwarded as --name). Default: let
#                     run_experiments.py derive one via its own --name auto.
#   -f, --foreground  Run in the foreground (no nohup/&); stream output here.
#   --noidx           Forward --noidx to run_experiments.py: with --name auto,
#                     use <base>_0 and overwrite instead of incrementing.
#   --dry-run         Print the command that would run and exit.
#   -h, --help        Show this help.
#
# Everything after a literal `--` is forwarded verbatim to run_experiments.py,
# e.g.  ./gpurun.sh BayesLoRA Smoke -- --reconfig-every 5 --render-steps 0
#
# GPU auto-selection: a GPU counts as free when its used memory is below
# $FTN_GPU_MEM_THRESHOLD MB (default 1000). Among free GPUs the one with the
# least used memory wins. A short-lived lock (under $FTN_GPU_LOCK_DIR) keeps two
# launches issued a few seconds apart from both grabbing the same card before
# its CUDA process shows up in nvidia-smi.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# nohup redirects stdout to a file; keep progress visible to tail immediately.
export PYTHONUNBUFFERED=1

MEM_THRESHOLD="${FTN_GPU_MEM_THRESHOLD:-1000}"   # MB; below this = "free"
LOCK_DIR="${FTN_GPU_LOCK_DIR:-${TMPDIR:-/tmp}/ftn_gpu_locks}"
LOCK_TTL_MIN=2                                   # locks older than this are stale

GPU=""
NAME=""
FOREGROUND=0
DRYRUN=0
NOIDX=0
CONFIGS=()
PASSTHRU=()

usage() { sed -n '2,/^set -euo pipefail/{ /^#/s/^# \{0,1\}//p; }' "$SCRIPT_DIR/gpurun.sh"; exit "${1:-0}"; }

require_value() {
    if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "gpurun: $1 requires a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpu)   require_value "$@"; GPU="$2"; shift 2 ;;
        -n|--name)  require_value "$@"; NAME="$2"; shift 2 ;;
        -f|--foreground) FOREGROUND=1; shift ;;
        --noidx)    NOIDX=1; shift ;;
        --dry-run)  DRYRUN=1; shift ;;
        -h|--help)  usage 0 ;;
        --)         shift; PASSTHRU+=("$@"); break ;;
        -*)         echo "gpurun: unknown option '$1' (put run_experiments.py flags after '--')" >&2; exit 2 ;;
        *)          CONFIGS+=("$1"); shift ;;
    esac
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    CONFIGS=(BayesLoRA)
fi

# --- pick a GPU ---------------------------------------------------------------
pick_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "gpurun: nvidia-smi not found; pass -g N to choose a GPU manually" >&2
        exit 3
    }
    mkdir -p "$LOCK_DIR"
    local best="" best_mem=999999999
    # index, used-memory(MB)
    while IFS=',' read -r idx used; do
        idx="${idx// /}"; used="${used// /}"
        [[ -z "$idx" ]] && continue
        # honor a fresh lock from a sibling launch that hasn't allocated yet
        local lock="$LOCK_DIR/gpu_${idx}.lock"
        if [[ -e "$lock" ]] && [[ -n "$(find "$lock" -mmin "-${LOCK_TTL_MIN}" 2>/dev/null)" ]]; then
            continue
        fi
        if (( used < MEM_THRESHOLD )) && (( used < best_mem )); then
            best="$idx"; best_mem="$used"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    if [[ -z "$best" ]]; then
        echo "gpurun: no free GPU (all >= ${MEM_THRESHOLD}MB used or locked). Current usage:" >&2
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv >&2
        echo "Pass -g N to force one anyway." >&2
        exit 4
    fi
    if [[ "$DRYRUN" -eq 0 ]]; then
        touch "$LOCK_DIR/gpu_${best}.lock"
    fi
    echo "$best"
}

if [[ -z "$GPU" ]]; then
    GPU="$(pick_gpu)"
fi

# --- assemble the command -----------------------------------------------------
CMD=(uv run python -u run_experiments.py --config "${CONFIGS[@]}")
[[ -n "$NAME" ]] && CMD+=(--name "$NAME")
[[ "$NOIDX" -eq 1 ]] && CMD+=(--noidx)
[[ ${#PASSTHRU[@]} -gt 0 ]] && CMD+=("${PASSTHRU[@]}")

# Log name: configs joined with '_' (+ name) + gpu, timestamped. The resolved
# --name auto value is printed by run_experiments.py into this log's first lines.
tag="$(IFS=_; echo "${CONFIGS[*]}")"
[[ -n "$NAME" ]] && tag="${tag}__${NAME}"
log="$SCRIPT_DIR/logs/${tag}_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"

echo "GPU           : $GPU"
echo "configs       : ${CONFIGS[*]}"
[[ -n "$NAME" ]] && echo "name          : $NAME" || echo "name          : (auto)"
[[ ${#PASSTHRU[@]} -gt 0 ]] && echo "extra args    : ${PASSTHRU[*]}"
echo "command       : CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"

if [[ "$DRYRUN" -eq 1 ]]; then
    echo "(dry run — not launched)"
    exit 0
fi

if [[ "$FOREGROUND" -eq 1 ]]; then
    echo "log           : (foreground)"
    CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
    exit $?
fi

echo "log           : $log"
mkdir -p logs
CUDA_VISIBLE_DEVICES="$GPU" nohup "${CMD[@]}" >"$log" 2>&1 &
pid=$!
echo "pid           : $pid"
echo "follow with   : tail -f $log"

# Give it a beat and confirm it didn't die on import/config error.
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
    echo "gpurun: process exited immediately — last lines of log:" >&2
    tail -n 20 "$log" >&2
    exit 5
fi
