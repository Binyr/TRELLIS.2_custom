#!/bin/bash
# Render dynamic OBJ files (v2: frame-major, obj-level resume, no normals).
# Usage: bash shs/render_dynamic_obj_v2.sh [world_size] [rank]

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

BLENDER_PATH="/tmp/blender-4.5.1-linux-x64/blender"
MANIFEST="/threed-code/yanruibin/efs/4D_video_data_process/data/objxl/dynamic_obj_manifest.json"
S3_OUTPUT_ROOT="s3://arcwm-code-us-west-2/yanruibin/4D_video_data_process/data/objxl/dynamic_obj_rendered_v2"
LOCAL_OUTPUT_ROOT="/local-ssd/dynamic_obj_rendered_v2"
RENDER_SCRIPT="tools/bl_rendering/dynamic_obj_rendering_v2.py"

python tools/bl_rendering/batch_render_dynamic_obj_v2.py \
  --manifest "$MANIFEST" \
  --s3_output_root "$S3_OUTPUT_ROOT" \
  --local_output_root "$LOCAL_OUTPUT_ROOT" \
  --blender_path "$BLENDER_PATH" \
  --render_script "$RENDER_SCRIPT" \
  --resolution 1024 \
  --num_cameras 16 \
  --camera_stride 2 \
  --cycles_samples 256 \
  --render_engine CYCLES \
  --cycles_device GPU \
  --cycles_backend OPTIX \
  --world_size ${1:-1} --rank ${2:-0}
