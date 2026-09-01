#!/usr/bin/env bash
# Resumable official material-VAE encoding worker.
#
# Usage:
#   SOURCE=all bash shs/trellis.2/encode_pbr_latents.sh <world_size> <rank>
#
# SOURCE: 4d, objxl, texa, texb, or all (default).
# Required: PBR_MANIFEST_TAG, shared by all ranks in one launch.
# Optional: MAX_ITEMS=10 for a small smoke run; STDOUT_SYNC_INTERVAL=60.
# SHAPE_COORD_AUDIT_EVERY=1000 checks a stable ~0.1% sample (0 disables).
set -euo pipefail

WORLD_SIZE="${1:?usage: $0 <world_size> <rank>}"
RANK="${2:?usage: $0 <world_size> <rank>}"
SOURCE="${SOURCE:-all}"
RESOLUTION="${RESOLUTION:-1024}"
PBR_MANIFEST_TAG="${PBR_MANIFEST_TAG:?set PBR_MANIFEST_TAG to the frozen manifest tag}"
MAX_ITEMS_ARG=()
[[ -n "${MAX_ITEMS:-}" ]] && MAX_ITEMS_ARG=(--max_items "$MAX_ITEMS")

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/local-ssd/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The worker never installs or mutates this environment.
source /local-ssd/trellis.2-venv/bin/activate

S3_BASE="s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data"
LOCAL_LOG_DIR="/local-ssd/encode_pbr_latents_stdouts/${SOURCE}_${RESOLUTION}"
mkdir -p "$LOCAL_LOG_DIR"
LOCAL_LOG="$LOCAL_LOG_DIR/rank_${RANK}.log"
STDOUT_SYNC_INTERVAL="${STDOUT_SYNC_INTERVAL:-60}"
SHAPE_COORD_AUDIT_EVERY="${SHAPE_COORD_AUDIT_EVERY:-1000}"
PBR_PIPELINE_MODE="${PBR_PIPELINE_MODE:-three_stage}"
PBR_PREFETCH_TASKS="${PBR_PREFETCH_TASKS:-2}"
PBR_SAVE_QUEUE_DEPTH="${PBR_SAVE_QUEUE_DEPTH:-2}"
PBR_UPLOAD_WORKERS="${PBR_UPLOAD_WORKERS:-1}"
PBR_PROGRESS_SYNC_EVERY="${PBR_PROGRESS_SYNC_EVERY:-10}"

run_source() {
    local name="$1" input_root="$2" shape_root="$3" output_root="$4" frame_mapping="$5" max_frames="$6"
    local remote_log="$output_root/logs/std_outs/rank_${RANK}.log"
    local manifest="$output_root/manifests/eligible_${RESOLUTION}_${PBR_MANIFEST_TAG}.json"
    local max_frames_arg=()
    [[ "$max_frames" -gt 0 ]] && max_frames_arg=(--max_frames "$max_frames")
    echo "[encode-pbr] source=$name input=$input_root shape=$shape_root output=$output_root manifest=$manifest max_frames=$max_frames shape_audit_every=$SHAPE_COORD_AUDIT_EVERY pipeline=$PBR_PIPELINE_MODE prefetch=$PBR_PREFETCH_TASKS save_queue=$PBR_SAVE_QUEUE_DEPTH upload_workers=$PBR_UPLOAD_WORKERS"
    python -u tools/encode_pbr_voxel_latents.py \
        --s3_input_root "$input_root" \
        --s3_shape_root "$shape_root" \
        --s3_output_root "$output_root" \
        --task_manifest "$manifest" \
        --frame_mapping "$frame_mapping" \
        --shape_audit_every "$SHAPE_COORD_AUDIT_EVERY" \
        --pipeline_mode "$PBR_PIPELINE_MODE" \
        --prefetch_tasks "$PBR_PREFETCH_TASKS" \
        --save_queue_depth "$PBR_SAVE_QUEUE_DEPTH" \
        --upload_workers "$PBR_UPLOAD_WORKERS" \
        --progress_sync_every "$PBR_PROGRESS_SYNC_EVERY" \
        --resolution "$RESOLUTION" \
        --world_size "$WORLD_SIZE" --rank "$RANK" \
        --state_dir "/local-ssd/encode_pbr_${name}_state" \
        --tmp_dir "/local-ssd/encode_pbr_${name}_tmp" \
        "${max_frames_arg[@]}" \
        "${MAX_ITEMS_ARG[@]}"
    aws s3 cp --only-show-errors "$LOCAL_LOG" "$remote_log" || true
}

sync_logs() {
    [[ -f "$LOCAL_LOG" ]] || return 0
    for root in "$@"; do
        aws s3 cp --only-show-errors "$LOCAL_LOG" "$root/logs/std_outs/rank_${RANK}.log" || true
    done
}

case "$SOURCE" in
  4d) ROOTS=("$S3_BASE/trellis.2/pbr_voxels_4d_latent") ;;
  objxl) ROOTS=("$S3_BASE/objxl/dynamic_obj_pbr_voxelized_latent") ;;
  texa) ROOTS=("$S3_BASE/texverse_1k_animate/pbr_A_voxelized_latent") ;;
  texb) ROOTS=("$S3_BASE/texverse_1k_animate/pbr_B_voxelized_latent") ;;
  all) ROOTS=("$S3_BASE/trellis.2/pbr_voxels_4d_latent" "$S3_BASE/objxl/dynamic_obj_pbr_voxelized_latent" "$S3_BASE/texverse_1k_animate/pbr_A_voxelized_latent" "$S3_BASE/texverse_1k_animate/pbr_B_voxelized_latent") ;;
  *) echo "unknown SOURCE=$SOURCE (use 4d|objxl|texa|texb|all)" >&2; exit 2 ;;
esac

trap 'sync_logs "${ROOTS[@]}"' EXIT INT TERM
exec > >(tee "$LOCAL_LOG") 2>&1
(
  while sleep "$STDOUT_SYNC_INTERVAL"; do sync_logs "${ROOTS[@]}"; done
) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; sync_logs "${ROOTS[@]}"' EXIT INT TERM

case "$SOURCE" in
  4d) run_source 4d "$S3_BASE/trellis.2/pbr_voxels_4d" "$S3_BASE/trellis.2/dual_grid_4d_v3_latent" "$S3_BASE/trellis.2/pbr_voxels_4d_latent" identity 32 ;;
  objxl) run_source objxl "$S3_BASE/objxl/dynamic_obj_pbr_voxelized" "$S3_BASE/objxl/dynamic_obj_voxel_32f_latent" "$S3_BASE/objxl/dynamic_obj_pbr_voxelized_latent" meta 0 ;;
  texa) run_source texa "$S3_BASE/texverse_1k_animate/pbr_A_voxelized" "$S3_BASE/texverse_1k_animate/texverse_A_voxel_latent" "$S3_BASE/texverse_1k_animate/pbr_A_voxelized_latent" meta 0 ;;
  texb) run_source texb "$S3_BASE/texverse_1k_animate/pbr_B_voxelized" "$S3_BASE/texverse_1k_animate/texverse_B_voxel_latent" "$S3_BASE/texverse_1k_animate/pbr_B_voxelized_latent" meta 0 ;;
  all)
    run_source 4d "$S3_BASE/trellis.2/pbr_voxels_4d" "$S3_BASE/trellis.2/dual_grid_4d_v3_latent" "$S3_BASE/trellis.2/pbr_voxels_4d_latent" identity 32
    run_source objxl "$S3_BASE/objxl/dynamic_obj_pbr_voxelized" "$S3_BASE/objxl/dynamic_obj_voxel_32f_latent" "$S3_BASE/objxl/dynamic_obj_pbr_voxelized_latent" meta 0
    run_source texa "$S3_BASE/texverse_1k_animate/pbr_A_voxelized" "$S3_BASE/texverse_1k_animate/texverse_A_voxel_latent" "$S3_BASE/texverse_1k_animate/pbr_A_voxelized_latent" meta 0
    run_source texb "$S3_BASE/texverse_1k_animate/pbr_B_voxelized" "$S3_BASE/texverse_1k_animate/texverse_B_voxel_latent" "$S3_BASE/texverse_1k_animate/pbr_B_voxelized_latent" meta 0
    ;;
esac
