#!/usr/bin/env bash
set -uo pipefail

WORLD_SIZE=${1:-1}
RANK=${2:-0}
DATA_ROOT=/threed-code/yanruibin/efs/4D_video_data_process/data
RESOLUTION=1024
OUTPUT_ROOT=$DATA_ROOT/trellis.2/pbr_voxels_4d
LOCAL_STDOUT_DIR=/local-ssd/voxelize_pbr_stdouts
LOCAL_LOG=$LOCAL_STDOUT_DIR/rank_${RANK}.log
REMOTE_STDOUT_DIR=$OUTPUT_ROOT/log_${RESOLUTION}/std_outs
REMOTE_LOG=$REMOTE_STDOUT_DIR/rank_${RANK}.log

upload_stdout_log() {
    if [[ -f "$LOCAL_LOG" ]]; then
        mkdir -p "$REMOTE_STDOUT_DIR"
        cp "$LOCAL_LOG" "$REMOTE_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] stdout uploaded: $REMOTE_LOG"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] stdout upload skipped: $LOCAL_LOG not found"
    fi
}
trap upload_stdout_log EXIT

mkdir -p "$LOCAL_STDOUT_DIR"
: > "$LOCAL_LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] rank=$RANK world_size=$WORLD_SIZE local_log=$LOCAL_LOG"

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
  --world_size "$WORLD_SIZE" --rank "$RANK" \
  2>&1 | tee "$LOCAL_LOG"

exit "${PIPESTATUS[0]:-0}"
