#!/bin/bash
# Encode ActionBench O-Voxel 4D view tars into shape+SS latents.
#
# Usage:
#   bash shs/trellis.2/encode_actionbench_1view_120f_latents.sh [world_size] [rank]
#
# Env (optional):
#   MAX_ITEMS=1
#   RESOLUTION=1024
#   SS_RESOLUTION=32
#   FRAME_CHUNK_SIZE=8
#   PREFETCH=2
#   TASK_SHARD_MODE=filter_then_shard
#   GLOBAL_PROGRESS_SNAPSHOT=/path/to/snapshot.json
#   WRITE_GLOBAL_PROGRESS_SNAPSHOT=/path/to/snapshot.json
#   S3_DATA_ROOT=s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data
#   SCRATCH_ROOT=/tmp

set -uo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUNBUFFERED=1

if [[ -f /etc/profile.d/koala_env.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/koala_env.sh
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export HF_HOME="${HF_HOME:-/local-ssd/hf_cache}"

S3_DATA_ROOT="${S3_DATA_ROOT:-s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data}"
INPUT_ROOT="${INPUT_ROOT:-${S3_DATA_ROOT}/actionbench/dual_grid_v1_1view_120f_uid_ovoxel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${S3_DATA_ROOT}/actionbench/dual_grid_v1_1view_120f_uid_ovoxel_latent}"

RESOLUTION="${RESOLUTION:-1024}"
LOG_SUFFIX="_${RESOLUTION//,/_}"
VOXEL_LOGS_DIR="${VOXEL_LOGS_DIR:-${INPUT_ROOT}/logs${LOG_SUFFIX}}"
SS_RESOLUTION="${SS_RESOLUTION:-32}"
FRAME_CHUNK_SIZE="${FRAME_CHUNK_SIZE:-8}"
PREFETCH="${PREFETCH:-2}"
TASK_SHARD_MODE="${TASK_SHARD_MODE:-filter_then_shard}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/tmp}"

MAX_ITEMS_ARG=()
if [[ -n "${MAX_ITEMS:-}" ]]; then
    MAX_ITEMS_ARG=(--max_items "$MAX_ITEMS")
fi
SNAPSHOT_ARG=()
if [[ -n "${GLOBAL_PROGRESS_SNAPSHOT:-}" ]]; then
    SNAPSHOT_ARG=(--global_progress_snapshot "$GLOBAL_PROGRESS_SNAPSHOT")
fi
WRITE_SNAPSHOT_ARG=()
if [[ -n "${WRITE_GLOBAL_PROGRESS_SNAPSHOT:-}" ]]; then
    WRITE_SNAPSHOT_ARG=(--write_global_progress_snapshot "$WRITE_GLOBAL_PROGRESS_SNAPSHOT")
fi

WS=${1:-1}
RANK=${2:-0}

LOCAL_STDOUT_DIR="${SCRATCH_ROOT}/encode_actionbench_stdouts${LOG_SUFFIX}"
LOCAL_LOG="${LOCAL_STDOUT_DIR}/rank_${RANK}.log"
REMOTE_STDOUT_DIR="${OUTPUT_ROOT}/_logs${LOG_SUFFIX}/std_outs"
REMOTE_LOG="${REMOTE_STDOUT_DIR}/rank_${RANK}.log"
STDOUT_SYNC_INTERVAL="${STDOUT_SYNC_INTERVAL:-60}"

mkdir -p "$LOCAL_STDOUT_DIR"

sync_stdout_quiet() {
    if [[ -f "$LOCAL_LOG" ]]; then
        aws s3 cp --only-show-errors "$LOCAL_LOG" "$REMOTE_LOG" || true
    fi
}

sync_stdout_loud() {
    echo ""
    echo "=============================================="
    echo "Syncing stdout log..."
    if [[ -f "$LOCAL_LOG" ]]; then
        sync_stdout_quiet
        echo "stdout synced to: $REMOTE_LOG"
    else
        echo "stdout sync skipped: $LOCAL_LOG not found"
    fi
    echo "=============================================="
}

cleanup() {
    kill "${SYNC_PID:-}" 2>/dev/null || true
    sync_stdout_loud
}
trap cleanup EXIT INT TERM

exec > >(tee "$LOCAL_LOG") 2>&1

(
    while sleep "$STDOUT_SYNC_INTERVAL"; do
        sync_stdout_quiet
    done
) &
SYNC_PID=$!

echo "================================================================"
echo "[encode:actionbench] host=$(hostname) rank=$RANK/$WS res=$RESOLUTION ss_res=$SS_RESOLUTION"
echo "  voxel_logs_dir  = $VOXEL_LOGS_DIR"
echo "  input_root      = $INPUT_ROOT"
echo "  output_root     = $OUTPUT_ROOT"
echo "  max_items       = ${MAX_ITEMS:-<unset>}"
echo "  frame_chunk_size= $FRAME_CHUNK_SIZE"
echo "  prefetch        = $PREFETCH"
echo "  task_shard_mode = $TASK_SHARD_MODE"
echo "  global_snapshot = ${GLOBAL_PROGRESS_SNAPSHOT:-<unset>}"
echo "  write_snapshot  = ${WRITE_GLOBAL_PROGRESS_SNAPSHOT:-<unset>}"
echo "  scratch_root    = $SCRATCH_ROOT"
echo "  local_log       = $LOCAL_LOG"
echo "  remote_log      = $REMOTE_LOG  (sync every ${STDOUT_SYNC_INTERVAL}s)"
echo "================================================================"

nvidia-smi || true

python -u tools/encode_ovoxel_to_latents_actionbench.py \
    --voxel_logs_dir   "$VOXEL_LOGS_DIR" \
    --input_root       "$INPUT_ROOT" \
    --output_root      "$OUTPUT_ROOT" \
    --resolution       "$RESOLUTION" \
    --log_suffix       "$LOG_SUFFIX" \
    --ss_resolution    "$SS_RESOLUTION" \
    --frame_chunk_size "$FRAME_CHUNK_SIZE" \
    --prefetch         "$PREFETCH" \
    --task_shard_mode  "$TASK_SHARD_MODE" \
    --state_dir "${SCRATCH_ROOT}/encode_actionbench_state" \
    --tmp_dir   "${SCRATCH_ROOT}/encode_actionbench_tmp" \
    --world_size "$WS" --rank "$RANK" \
    "${SNAPSHOT_ARG[@]}" \
    "${WRITE_SNAPSHOT_ARG[@]}" \
    "${MAX_ITEMS_ARG[@]}"
