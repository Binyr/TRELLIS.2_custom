#!/usr/bin/env bash
set -uo pipefail

export PYTHONUNBUFFERED=1

WS=${1:-1}
RANK=${2:-0}

DATA_ROOT=/threed-code/yanruibin/efs/4D_video_data_process/data
ANN_FILE="${ANN_FILE:-$DATA_ROOT/texverse_1k_animate/texverse_B_voxel_latent/1024/anns_texverse_B.json}"
S3_RENDER_ROOT="${S3_RENDER_ROOT:-s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/texverse_1k_animate/rendered_v1_stageB}"
S3_PBR_ROOT="${S3_PBR_ROOT:-s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/texverse_1k_animate/pbr_B_voxel}"
S3_OUTPUT_ROOT="${S3_OUTPUT_ROOT:-s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/texverse_1k_animate/pbr_B_voxelized}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/local-ssd/voxelize_pbr_texb_cache}"
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-/local-ssd/tmp_voxelize_pbr_texb}"
LOCAL_STATE_ROOT="${LOCAL_STATE_ROOT:-/local-ssd/voxelize_pbr_texb_state}"
LOCAL_STDOUT_DIR="${LOCAL_STDOUT_DIR:-/local-ssd/voxelize_pbr_texb_stdouts}"
RESOLUTION="${RESOLUTION:-1024}"
STDOUT_SYNC_INTERVAL="${STDOUT_SYNC_INTERVAL:-60}"
VXZ_COMPRESSION="${VXZ_COMPRESSION:-zstd}"
VXZ_COMPRESSION_LEVEL="${VXZ_COMPRESSION_LEVEL:-5}"

MAX_ITEMS_ARG=()
if [[ -n "${MAX_ITEMS:-}" ]]; then
    MAX_ITEMS_ARG=(--max_items "$MAX_ITEMS")
fi

DEBUG_ARG=()
if [[ -n "${DEBUG:-}" ]]; then
    DEBUG_ARG=(--debug)
fi

LOCAL_LOG="$LOCAL_STDOUT_DIR/rank_${RANK}.log"
REMOTE_LOG="$S3_OUTPUT_ROOT/logs/std_outs/rank_${RANK}.log"
mkdir -p "$LOCAL_STDOUT_DIR"

sync_logs_quiet() {
    if [[ -f "$LOCAL_LOG" ]]; then
        aws s3 cp --only-show-errors "$LOCAL_LOG" "$REMOTE_LOG" >/dev/null 2>&1 || true
    fi
}

cleanup() {
    kill "${SYNC_PID:-}" 2>/dev/null || true
    sync_logs_quiet
}
trap cleanup EXIT INT TERM

exec > >(tee "$LOCAL_LOG") 2>&1

(
    while sleep "$STDOUT_SYNC_INTERVAL"; do
        sync_logs_quiet
    done
) &
SYNC_PID=$!

echo "================================================================"
echo "[voxelize_pbr_texb] host=$(hostname) rank=$RANK/$WS"
echo "  ann_file       = $ANN_FILE"
echo "  render_root    = $S3_RENDER_ROOT"
echo "  pbr_root       = $S3_PBR_ROOT"
echo "  output_root    = $S3_OUTPUT_ROOT"
echo "  local_cache    = $LOCAL_CACHE_ROOT"
echo "  local_tmp      = $LOCAL_TMP_ROOT"
echo "  local_state    = $LOCAL_STATE_ROOT"
echo "  resolution     = $RESOLUTION"
echo "  compression    = $VXZ_COMPRESSION:$VXZ_COMPRESSION_LEVEL"
echo "  max_items      = ${MAX_ITEMS:-}"
echo "================================================================"

python tools/bl_rendering/batch_voxelize_pbr_dynamic.py \
    --ann_file "$ANN_FILE" \
    --s3_render_root "$S3_RENDER_ROOT" \
    --s3_pbr_root "$S3_PBR_ROOT" \
    --s3_output_root "$S3_OUTPUT_ROOT" \
    --local_cache_root "$LOCAL_CACHE_ROOT" \
    --tmp_dir "$LOCAL_TMP_ROOT" \
    --state_dir "$LOCAL_STATE_ROOT" \
    --resolution "$RESOLUTION" \
    --vxz_compression "$VXZ_COMPRESSION" \
    --vxz_compression_level "$VXZ_COMPRESSION_LEVEL" \
    --world_size "$WS" \
    --rank "$RANK" \
    "${MAX_ITEMS_ARG[@]}" \
    "${DEBUG_ARG[@]}"
