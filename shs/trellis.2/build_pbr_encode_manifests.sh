#!/usr/bin/env bash
# Build one immutable PBR/shape-success intersection manifest per source.
# Usage: bash shs/trellis.2/build_pbr_encode_manifests.sh <manifest_tag>
set -euo pipefail

TAG="${1:?usage: $0 <manifest_tag>}"
RESOLUTION="${RESOLUTION:-1024}"
S3_BASE="s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data"

build_manifest() {
    local source="$1" pbr_root="$2" pbr_logs="$3" shape_root="$4"
    local shape_mode="$5" shape_logs="$6" output_root="$7"
    local output_uri="$output_root/manifests/eligible_${RESOLUTION}_${TAG}.json"
    local shape_progress_arg=()
    [[ -n "$shape_logs" ]] && shape_progress_arg=(--shape_progress_prefix "$shape_logs")

    echo "[build-manifest] source=$source output=$output_uri"
    python3 tools/build_pbr_shape_manifest.py \
        --source "$source" \
        --resolution "$RESOLUTION" \
        --pbr_root "$pbr_root" \
        --pbr_progress_prefix "$pbr_logs" \
        --shape_root "$shape_root" \
        --shape_success_mode "$shape_mode" \
        "${shape_progress_arg[@]}" \
        --output_uri "$output_uri"
}

build_manifest \
    4d \
    "$S3_BASE/trellis.2/pbr_voxels_4d" \
    "$S3_BASE/trellis.2/pbr_voxels_4d/log_1024" \
    "$S3_BASE/trellis.2/dual_grid_4d_v3_latent" \
    objects "" \
    "$S3_BASE/trellis.2/pbr_voxels_4d_latent"

build_manifest \
    objxl \
    "$S3_BASE/objxl/dynamic_obj_pbr_voxelized" \
    "$S3_BASE/objxl/dynamic_obj_pbr_voxelized/logs" \
    "$S3_BASE/objxl/dynamic_obj_voxel_32f_latent" \
    progress "$S3_BASE/objxl/dynamic_obj_voxel_32f_latent/logs_1024" \
    "$S3_BASE/objxl/dynamic_obj_pbr_voxelized_latent"

build_manifest \
    texa \
    "$S3_BASE/texverse_1k_animate/pbr_A_voxelized" \
    "$S3_BASE/texverse_1k_animate/pbr_A_voxelized/logs" \
    "$S3_BASE/texverse_1k_animate/texverse_A_voxel_latent" \
    progress "$S3_BASE/texverse_1k_animate/texverse_A_voxel_latent/logs_1024" \
    "$S3_BASE/texverse_1k_animate/pbr_A_voxelized_latent"

build_manifest \
    texb \
    "$S3_BASE/texverse_1k_animate/pbr_B_voxelized" \
    "$S3_BASE/texverse_1k_animate/pbr_B_voxelized/logs" \
    "$S3_BASE/texverse_1k_animate/texverse_B_voxel_latent" \
    progress "$S3_BASE/texverse_1k_animate/texverse_B_voxel_latent/logs_1024" \
    "$S3_BASE/texverse_1k_animate/pbr_B_voxelized_latent"

echo "[build-manifest] complete tag=$TAG resolution=$RESOLUTION"
