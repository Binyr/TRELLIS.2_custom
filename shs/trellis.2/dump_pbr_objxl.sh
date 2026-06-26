#!/usr/bin/env bash
set -uo pipefail

export PYTHONUNBUFFERED=1

WS=${1:-1}
RANK=${2:-0}

BLENDER_PATH="${BLENDER_PATH:-/tmp/blender-4.5.1-linux-x64/blender}"
MANIFEST="${MANIFEST:-/threed-code/yanruibin/efs/4D_video_data_process/data/objxl/dynamic_obj_manifest.json}"
ANN_FILE="${ANN_FILE:-/threed-code/yanruibin/efs/4D_video_data_process/data/objxl/dynamic_obj_voxel_32f_latent/1024/anns_objxl.json}"
S3_RENDER_ROOT="${S3_RENDER_ROOT:-s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/objxl/dynamic_obj_rendered}"
S3_OUTPUT_ROOT="${S3_OUTPUT_ROOT:-s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/objxl/dynamic_obj_pbr_shared}"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-/local-ssd/dynamic_obj_pbr_shared}"
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-/local-ssd/tmp_dump_pbr_objxl}"
LOCAL_STATE_ROOT="${LOCAL_STATE_ROOT:-/local-ssd/dump_pbr_objxl_state}"
MAX_FRAMES="${MAX_FRAMES:-121}"
VERTEX_SET_EPS="${VERTEX_SET_EPS:-1e-4}"
BLENDER_TIMEOUT_S="${BLENDER_TIMEOUT_S:-3600}"
MAX_ITEMS_ARG=()
if [[ -n "${MAX_ITEMS:-}" ]]; then
    MAX_ITEMS_ARG=(--max_items "$MAX_ITEMS")
fi

echo "================================================================"
echo "[dump_pbr_objxl] host=$(hostname) rank=$RANK/$WS"
echo "  manifest       = $MANIFEST"
echo "  ann_file       = $ANN_FILE"
echo "  render_root    = $S3_RENDER_ROOT"
echo "  output_root    = $S3_OUTPUT_ROOT"
echo "  local_output   = $LOCAL_OUTPUT_ROOT"
echo "  local_tmp      = $LOCAL_TMP_ROOT"
echo "  local_state    = $LOCAL_STATE_ROOT"
echo "  blender        = $BLENDER_PATH"
echo "  max_frames     = $MAX_FRAMES"
echo "  vertex_set_eps = $VERTEX_SET_EPS"
echo "  max_items      = ${MAX_ITEMS:-}"
echo "================================================================"

python tools/bl_rendering/batch_dump_pbr_dynamic_obj.py \
    --manifest "$MANIFEST" \
    --ann_file "$ANN_FILE" \
    --s3_render_root "$S3_RENDER_ROOT" \
    --s3_output_root "$S3_OUTPUT_ROOT" \
    --local_output_root "$LOCAL_OUTPUT_ROOT" \
    --tmp_dir "$LOCAL_TMP_ROOT" \
    --state_dir "$LOCAL_STATE_ROOT" \
    --blender_path "$BLENDER_PATH" \
    --max_frames "$MAX_FRAMES" \
    --vertex_set_eps "$VERTEX_SET_EPS" \
    --blender_timeout_s "$BLENDER_TIMEOUT_S" \
    --world_size "$WS" \
    --rank "$RANK" \
    "${MAX_ITEMS_ARG[@]}"
