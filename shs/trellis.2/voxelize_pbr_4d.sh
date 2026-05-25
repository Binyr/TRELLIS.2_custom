#!/usr/bin/env bash
set -uo pipefail

export PYTHONUNBUFFERED=1

WORLD_SIZE=${1:-1}
RANK=${2:-0}
DATA_ROOT=/threed-code/yanruibin/efs/4D_video_data_process/data
RESOLUTION=1024
OUTPUT_ROOT=$DATA_ROOT/trellis.2/pbr_voxels_4d
LOCAL_STDOUT_DIR=/local-ssd/voxelize_pbr_stdouts
LOCAL_LOG=$LOCAL_STDOUT_DIR/rank_${RANK}.log
REMOTE_STDOUT_DIR=$OUTPUT_ROOT/log_${RESOLUTION}/std_outs
REMOTE_LOG=$REMOTE_STDOUT_DIR/rank_${RANK}.log
STDOUT_SYNC_INTERVAL=${STDOUT_SYNC_INTERVAL:-60}

mkdir -p "$LOCAL_STDOUT_DIR"

sync_logs_quiet() {
    if [[ -f "$LOCAL_LOG" ]]; then
        mkdir -p "$REMOTE_STDOUT_DIR"
        cp "$LOCAL_LOG" "$REMOTE_LOG"
    fi
}

sync_logs() {
    echo ""
    echo "=============================================="
    echo "Syncing stdout log to EFS..."
    if [[ -f "$LOCAL_LOG" ]]; then
        sync_logs_quiet
        echo "stdout synced to: $REMOTE_LOG"
    else
        echo "stdout sync skipped: $LOCAL_LOG not found"
    fi
    echo "=============================================="
}

cleanup() {
    kill "${SYNC_PID:-}" 2>/dev/null || true
    sync_logs
}
trap cleanup EXIT INT TERM

exec > >(tee "$LOCAL_LOG") 2>&1

export VOXELIZE_PBR_STDOUT_LOCAL="$LOCAL_LOG"
export VOXELIZE_PBR_STDOUT_REMOTE="$REMOTE_LOG"

(
    while sleep "$STDOUT_SYNC_INTERVAL"; do
        sync_logs_quiet
    done
) &
SYNC_PID=$!

echo "=============================================="
echo "Voxelize PBR 4D"
echo "  rank:       $RANK"
echo "  world_size: $WORLD_SIZE"
echo "  local_log:  $LOCAL_LOG"
echo "  remote_log: $REMOTE_LOG"
echo "  sync_every: ${STDOUT_SYNC_INTERVAL}s"
echo "=============================================="

python data_toolkit/voxelize_pbr_v2.py \
  --ann_file $DATA_ROOT/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
  --pbr_shared_root $DATA_ROOT/trellis.2/pbr_shared \
  --rendered_root $DATA_ROOT/objverse_minghao_4d_mine_40075/rendering_v5 \
  --output_root $OUTPUT_ROOT \
  --resolution $RESOLUTION \
  --max_workers 1 \
  --priority_list claude_tmp/objv1_sketchfab_intersection.txt \
  --finished_views $OUTPUT_ROOT/finished_views_512.json \
  --vxz_compression zstd \
  --vxz_compression_level 5 \
  --world_size "$WORLD_SIZE" --rank "$RANK"

exit 0
