#!/bin/bash
# Voxelize dynamic_obj renders into per-view dual-grid .vxz tars.
# Single-process per rank; spawn multiple ranks across pods via koala.
#
# Usage: bash shs/trellis.2/objxl_dual_grid_4d.sh [world_size] [rank]
# Env (optional): MAX_ITEMS, RESOLUTION, MAX_FRAMES, FRAME_SAMPLING,
#                 STDOUT_SYNC_INTERVAL (seconds, default 60)

set -uo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUNBUFFERED=1

# Pull koala-injected AWS credentials (non-interactive ssh shells skip /etc/profile.d).
if [[ -f /etc/profile.d/koala_env.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/koala_env.sh
fi

FINISHED_VIEWS="s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/objxl/render_finished_view.json"
S3_INPUT="s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/objxl/dynamic_obj_rendered"
S3_OUTPUT="s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/objxl/dynamic_obj_voxel_32f"

RESOLUTION="${RESOLUTION:-512}"
LOG_SUFFIX=""
if [[ "$RESOLUTION" != "512" ]]; then
    LOG_SUFFIX="_${RESOLUTION//,/_}"
fi
MAX_FRAMES="${MAX_FRAMES:-32}"
FRAME_SAMPLING="${FRAME_SAMPLING:-center}"
MAX_ITEMS_ARG=()
if [[ -n "${MAX_ITEMS:-}" ]]; then
    MAX_ITEMS_ARG=(--max_items "$MAX_ITEMS")
fi

WS=${1:-1}
RANK=${2:-0}

# --- Full stdout: tee to local file + periodic upload to S3 ---
LOCAL_STDOUT_DIR="/tmp/objxl_dual_grid_stdouts${LOG_SUFFIX}"
LOCAL_LOG="$LOCAL_STDOUT_DIR/rank_${RANK}.log"
S3_STDOUT_LOG="$S3_OUTPUT/_logs${LOG_SUFFIX}/std_outs/rank_${RANK}.log"
STDOUT_SYNC_INTERVAL="${STDOUT_SYNC_INTERVAL:-60}"

mkdir -p "$LOCAL_STDOUT_DIR"

sync_stdout_quiet() {
    if [[ -f "$LOCAL_LOG" ]]; then
        aws s3 cp --only-show-errors "$LOCAL_LOG" "$S3_STDOUT_LOG" || true
    fi
}

sync_stdout_loud() {
    echo ""
    echo "=============================================="
    echo "Syncing stdout log to S3..."
    if [[ -f "$LOCAL_LOG" ]]; then
        sync_stdout_quiet
        echo "stdout synced to: $S3_STDOUT_LOG"
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

# After this line everything goes to BOTH terminal and $LOCAL_LOG.
exec > >(tee "$LOCAL_LOG") 2>&1

(
    while sleep "$STDOUT_SYNC_INTERVAL"; do
        sync_stdout_quiet
    done
) &
SYNC_PID=$!

echo "================================================================"
echo "[voxel] host=$(hostname) rank=$RANK/$WS resolution=$RESOLUTION"
echo "  finished_views = $FINISHED_VIEWS"
echo "  s3_input_root  = $S3_INPUT"
echo "  s3_output_root = $S3_OUTPUT"
echo "  max_frames     = $MAX_FRAMES"
echo "  frame_sampling = $FRAME_SAMPLING"
echo "  max_items      = ${MAX_ITEMS:-<unset>}"
echo "  local_log      = $LOCAL_LOG"
echo "  s3_log         = $S3_STDOUT_LOG  (sync every ${STDOUT_SYNC_INTERVAL}s)"
echo "================================================================"

python -u tools/voxel/dual_grid_dynamic_obj.py \
    --finished_views "$FINISHED_VIEWS" \
    --s3_input_root  "$S3_INPUT" \
    --s3_output_root "$S3_OUTPUT" \
    --resolution "$RESOLUTION" \
    --max_frames "$MAX_FRAMES" \
    --frame_sampling "$FRAME_SAMPLING" \
    --state_dir /tmp/objxl_dual_grid_state \
    --tmp_dir   /tmp/objxl_tmp_dual_grid \
    --world_size "$WS" --rank "$RANK" \
    "${MAX_ITEMS_ARG[@]}"
