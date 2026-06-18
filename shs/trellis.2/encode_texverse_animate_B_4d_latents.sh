#!/bin/bash
# Encode O-Voxel 4D view tars (from texverse_animate_stageB_dual_grid_4d.sh)
# into shape (+ optional SS) latents. TexVerse-Animation Stage B flavor.
#
# Reads success views from the voxel-stage progress logs at startup, so the
# input list always tracks the latest voxelization completion state.
#
# Usage: bash shs/trellis.2/encode_texverse_animate_B_4d_latents.sh [world_size] [rank]
# Env (optional): MAX_ITEMS, RESOLUTION, SS_RESOLUTION, FRAME_CHUNK_SIZE,
#                 PREFETCH, STDOUT_SYNC_INTERVAL (seconds, default 60)
# Set FRAME_CHUNK_SIZE=1 to fall back to one-frame-per-forward (bit-exact
# vs. the original single-frame implementation).
# PREFETCH=N keeps N views in background CPU/IO pipeline so GPU stays busy.

set -uo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUNBUFFERED=1

if [[ -f /etc/profile.d/koala_env.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/koala_env.sh
fi

export HF_HOME=/local-ssd/hf_cache

S3_INPUT="s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/texverse_1k_animate/texverse_B_voxel"
S3_OUTPUT="s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/texverse_1k_animate/texverse_B_voxel_latent"

RESOLUTION="${RESOLUTION:-512}"
LOG_SUFFIX=""
if [[ "$RESOLUTION" != "512" ]]; then
    LOG_SUFFIX="_${RESOLUTION//,/_}"
fi
VOXEL_LOGS_PREFIX="s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/texverse_1k_animate/texverse_B_voxel/logs${LOG_SUFFIX}/"
SS_RESOLUTION="${SS_RESOLUTION:-32}"
FRAME_CHUNK_SIZE="${FRAME_CHUNK_SIZE:-8}"
PREFETCH="${PREFETCH:-2}"
MAX_ITEMS_ARG=()
if [[ -n "${MAX_ITEMS:-}" ]]; then
    MAX_ITEMS_ARG=(--max_items "$MAX_ITEMS")
fi

WS=${1:-1}
RANK=${2:-0}

# --- Full stdout: tee + periodic upload to S3 ---
LOCAL_STDOUT_DIR="/local-ssd/encode_texverse_B_stdouts${LOG_SUFFIX}"
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

exec > >(tee "$LOCAL_LOG") 2>&1

(
    while sleep "$STDOUT_SYNC_INTERVAL"; do
        sync_stdout_quiet
    done
) &
SYNC_PID=$!

echo "================================================================"
echo "[encode:texverseB] host=$(hostname) rank=$RANK/$WS res=$RESOLUTION ss_res=$SS_RESOLUTION"
echo "  voxel_logs_prefix = $VOXEL_LOGS_PREFIX"
echo "  s3_input_root     = $S3_INPUT"
echo "  s3_output_root    = $S3_OUTPUT"
echo "  max_items         = ${MAX_ITEMS:-<unset>}"
echo "  frame_chunk_size  = $FRAME_CHUNK_SIZE"
echo "  prefetch          = $PREFETCH"
echo "  local_log         = $LOCAL_LOG"
echo "  s3_log            = $S3_STDOUT_LOG  (sync every ${STDOUT_SYNC_INTERVAL}s)"
echo "================================================================"

nvidia-smi || true

python -u tools/encode_ovoxel_to_latents_objxl.py \
    --voxel_logs_prefix "$VOXEL_LOGS_PREFIX" \
    --s3_input_root     "$S3_INPUT" \
    --s3_output_root    "$S3_OUTPUT" \
    --resolution        "$RESOLUTION" \
    --log_suffix        "$LOG_SUFFIX" \
    --ss_resolution     "$SS_RESOLUTION" \
    --frame_chunk_size  "$FRAME_CHUNK_SIZE" \
    --prefetch          "$PREFETCH" \
    --state_dir /local-ssd/encode_texverse_B_state \
    --tmp_dir   /local-ssd/encode_texverse_B_tmp \
    --world_size "$WS" --rank "$RANK" \
    "${MAX_ITEMS_ARG[@]}"
