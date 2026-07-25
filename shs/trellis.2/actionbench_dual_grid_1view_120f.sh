#!/bin/bash
# Voxelize ActionBench rendered_v1_1view_120f_uid meshes into o-voxel tars.
#
# Usage:
#   bash shs/trellis.2/actionbench_dual_grid_1view_120f.sh [world_size] [rank]
#
# Env (optional):
#   RESOLUTION=1024
#   MAX_FRAMES=0              # 0 means all frames
#   FRAME_SAMPLING=all
#   MAX_ITEMS=1               # debug only
#   DATA_ROOT=/threed-code/yanruibin/efs/4D_video_data_process/data
#   SCRATCH_ROOT=/tmp

set -uo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

DATA_ROOT="${DATA_ROOT:-/threed-code/yanruibin/efs/4D_video_data_process/data}"
INPUT_ROOT="${INPUT_ROOT:-${DATA_ROOT}/actionbench/rendered_v1_1view_120f_uid}"
ANN_FILE="${ANN_FILE:-${INPUT_ROOT}/anns_min61_sub20.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/actionbench/dual_grid_v1_1view_120f_uid_ovoxel}"

RESOLUTION="${RESOLUTION:-1024}"
MAX_FRAMES="${MAX_FRAMES:-0}"
FRAME_SAMPLING="${FRAME_SAMPLING:-all}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/tmp}"

WS=${1:-1}
RANK=${2:-0}

MAX_ITEMS_ARG=()
if [[ -n "${MAX_ITEMS:-}" ]]; then
    MAX_ITEMS_ARG=(--max_items "$MAX_ITEMS")
fi

LOG_SUFFIX="_${RESOLUTION//,/_}"
LOCAL_STDOUT_DIR="${SCRATCH_ROOT}/actionbench_dual_grid_stdouts${LOG_SUFFIX}"
LOCAL_LOG="${LOCAL_STDOUT_DIR}/rank_${RANK}.log"
REMOTE_STDOUT_DIR="${OUTPUT_ROOT}/_logs${LOG_SUFFIX}/std_outs"
REMOTE_LOG="${REMOTE_STDOUT_DIR}/rank_${RANK}.log"
STDOUT_SYNC_INTERVAL="${STDOUT_SYNC_INTERVAL:-60}"

mkdir -p "$LOCAL_STDOUT_DIR" "$REMOTE_STDOUT_DIR"

sync_stdout_quiet() {
    if [[ -f "$LOCAL_LOG" ]]; then
        cp "$LOCAL_LOG" "$REMOTE_LOG" || true
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
echo "[actionbench-voxel] host=$(hostname) rank=$RANK/$WS"
echo "  ann_file       = $ANN_FILE"
echo "  input_root     = $INPUT_ROOT"
echo "  output_root    = $OUTPUT_ROOT"
echo "  resolution     = $RESOLUTION"
echo "  max_frames     = $MAX_FRAMES"
echo "  frame_sampling = $FRAME_SAMPLING"
echo "  max_items      = ${MAX_ITEMS:-<unset>}"
echo "  scratch_root   = $SCRATCH_ROOT"
echo "  local_log      = $LOCAL_LOG"
echo "  remote_log     = $REMOTE_LOG"
echo "================================================================"

python -u tools/voxel/dual_grid_actionbench_uid.py \
    --ann_file "$ANN_FILE" \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --resolution "$RESOLUTION" \
    --max_frames "$MAX_FRAMES" \
    --frame_sampling "$FRAME_SAMPLING" \
    --state_dir "${SCRATCH_ROOT}/actionbench_dual_grid_state" \
    --tmp_dir "${SCRATCH_ROOT}/actionbench_tmp_dual_grid" \
    --world_size "$WS" --rank "$RANK" \
    "${MAX_ITEMS_ARG[@]}"
