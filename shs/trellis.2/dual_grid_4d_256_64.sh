#!/usr/bin/env bash
# Voxelize the original 4D source into 256 and 64 geometry O-Voxels.
#
# Usage:
#   bash shs/trellis.2/dual_grid_4d_256_64.sh [world_size] [rank]
#
# The two resolutions are produced from the same loaded mesh/view.  No 512 or
# 1024 finished-view list is used: those lists do not prove that 256/64 outputs
# exist.  Resume state and stdout live under log_256_64.
set -euo pipefail

export PYTHONUNBUFFERED=1

WORLD_SIZE="${1:-1}"
RANK="${2:-0}"
RESOLUTIONS="${RESOLUTIONS:-256,64}"
RES_TAG="${RESOLUTIONS//,/_}"

DATA_ROOT="${DATA_ROOT:-/threed-code/yanruibin/efs/4D_video_data_process/data}"
RENDERED_ROOT="$DATA_ROOT/objverse_minghao_4d_mine_40075/rendering_v5"
OUTPUT_ROOT="$DATA_ROOT/trellis.2/dual_grid_4d_v3"
ANN_FILE="$DATA_ROOT/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json"
PRIORITY_LIST="${PRIORITY_LIST:-claude_tmp/objv1_sketchfab_intersection.txt}"
MAX_WORKERS="${MAX_WORKERS:-1}"
TMP_DIR="${TMP_DIR:-/local-ssd/tmp_dual_grid_4d_${RES_TAG}}"

LOCAL_STDOUT_DIR="/local-ssd/dual_grid_4d_${RES_TAG}_stdouts"
LOCAL_LOG="$LOCAL_STDOUT_DIR/rank_${RANK}.log"
REMOTE_STDOUT_DIR="$OUTPUT_ROOT/log_${RES_TAG}/std_outs"
REMOTE_LOG="$REMOTE_STDOUT_DIR/rank_${RANK}.log"
STDOUT_SYNC_INTERVAL="${STDOUT_SYNC_INTERVAL:-60}"

# Optional only: set this to a finished-view list built specifically for the
# requested resolutions.  It intentionally has no 512/1024 default.
FINISHED_VIEWS_ARG=()
if [[ -n "${FINISHED_VIEWS:-}" ]]; then
    FINISHED_VIEWS_ARG=(--finished_views "$FINISHED_VIEWS")
fi

mkdir -p "$LOCAL_STDOUT_DIR" "$TMP_DIR"

sync_stdout() {
    if [[ -f "$LOCAL_LOG" ]]; then
        mkdir -p "$REMOTE_STDOUT_DIR"
        cp "$LOCAL_LOG" "$REMOTE_LOG"
    fi
}

cleanup() {
    kill "${SYNC_PID:-}" 2>/dev/null || true
    sync_stdout || true
}
trap cleanup EXIT INT TERM

exec > >(tee "$LOCAL_LOG") 2>&1

(
    while sleep "$STDOUT_SYNC_INTERVAL"; do
        sync_stdout || true
    done
) &
SYNC_PID=$!

echo "==============================================================="
echo "Voxelize original 4D shape O-Voxels"
echo "  rank:           $RANK/$WORLD_SIZE"
echo "  resolutions:    $RESOLUTIONS"
echo "  rendered_root:  $RENDERED_ROOT"
echo "  output_root:    $OUTPUT_ROOT"
echo "  finished_views: ${FINISHED_VIEWS:-<unset>}"
echo "  max_workers:    $MAX_WORKERS"
echo "  tmp_dir:        $TMP_DIR"
echo "  local_log:      $LOCAL_LOG"
echo "  remote_log:     $REMOTE_LOG"
echo "==============================================================="

python -u data_toolkit/dual_grid_v2.py \
    --ann_file "$ANN_FILE" \
    --rendered_root "$RENDERED_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --resolution "$RESOLUTIONS" \
    --max_workers "$MAX_WORKERS" \
    --max_frames 32 \
    --frame_sampling center \
    --write_frame_meta \
    --priority_list "$PRIORITY_LIST" \
    --tmp_dir "$TMP_DIR" \
    --world_size "$WORLD_SIZE" \
    --rank "$RANK" \
    "${FINISHED_VIEWS_ARG[@]}"
