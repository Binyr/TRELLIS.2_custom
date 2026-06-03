#!/bin/bash
# Multi-process launcher for TexVerse-Animation STAGE B rendering.
#   B-1-nested  : outer zip wraps inner zip with fbx/glb/gltf
#   B-2-blend   : top-level .blend (uses wm.open_mainfile importer)
#   B-3-archive : top-level .7z / .rar (requires p7zip-full + unrar in image)
#
# Outputs go to a DIFFERENT S3 root from stage A so the two never collide:
#   stage A -> .../texverse_1k_animate/rendered_v1/
#   stage B -> .../texverse_1k_animate/rendered_v1_stageB/
#
# Pick which stage(s) to run via STAGE_FILTER env (default = all B-*):
#   STAGE_FILTER=B-1-nested            bash shs/render_texverse_animate_stageB_mp.sh
#   STAGE_FILTER=B-2-blend             bash shs/render_texverse_animate_stageB_mp.sh
#   STAGE_FILTER=B-3-archive           bash shs/render_texverse_animate_stageB_mp.sh
#   STAGE_FILTER="B-1-nested,B-2-blend" ...
#
# (Copied from render_texverse_animate_mp.sh; only paths, manifest, S3 root,
# and the --stage_filter argument differ.)
#
# Per pod:
#   - auto-detect GPU count G (override with NUM_GPUS env)
#   - launch BLENDER_PER_GPU workers per GPU (default 1)
#   - each worker is a `batch_render_dynamic_obj.py` invocation with its own
#     manifest shard, scratch dir, progress.json, and stdout log
#   - failed workers are auto-respawned after RESPAWN_SLEEP seconds
#   - launcher streams a "death log" so koala logs surface crashes immediately
#
# Outer (cross-pod) sharding is unchanged: pass GLOBAL_WS / GLOBAL_RANK as
# $1 / $2. Inner sharding multiplies world size by per-pod worker count, so
# every pod x worker gets a unique global rank.
#
# Usage:
#   bash shs/render_dynamic_obj_mp.sh [global_world_size] [global_rank]
#
# Env knobs:
#   BLENDER_PER_GPU   (default 1)        - workers per GPU
#       Benchmark on H200 + 20-core cgroup (4 obj x 4 view x 10 frame, 1024 res):
#         1p: wall=86.8s  per-view mean=5.50s
#         2p: wall=79.3s  per-view mean=9.90s  (+9% wall, but 1.8x slower per view)
#       2p only helps when there is spare CPU to overlap setup/encode with the
#       neighbor's GPU work. Stick to 1 unless CPU quota >> 20 cores.
#   NUM_GPUS          (auto)             - override GPU count
#   RESPAWN_SLEEP     (default 30)       - sleep before restarting a dead worker
#   STDOUT_SYNC_SEC   (default 60)       - sync each worker stdout to S3 every N sec
#   MAX_RESPAWN       (default 1000000)  - safety cap per worker (effectively unlimited)
#   FAIL_FAST         (default 0)        - if 1, kill all on first death instead of respawn

set -uo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUNBUFFERED=1

# Detect cgroup CPU quota so multi-threaded libs (x265, OpenMP, MKL, ...)
# do not spawn `nproc` (host-side, 192) threads inside a 24-core slice and
# thrash on CFS throttling.
CPU_QUOTA=0
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    CPU_QUOTA=$(awk '{ if ($1 == "max") print 0; else printf "%d\n", ($1+$2-1)/$2 }' /sys/fs/cgroup/cpu.max)
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [[ "$q" -gt 0 && "$p" -gt 0 ]] && CPU_QUOTA=$(( (q + p - 1) / p ))
fi
if (( CPU_QUOTA > 0 )); then
    export OMP_NUM_THREADS=$CPU_QUOTA
    export MKL_NUM_THREADS=$CPU_QUOTA
    export OPENBLAS_NUM_THREADS=$CPU_QUOTA
    export NUMEXPR_NUM_THREADS=$CPU_QUOTA
    export BLENDER_NUM_THREADS=$CPU_QUOTA
fi

GLOBAL_WS=${1:-1}
GLOBAL_RANK=${2:-0}

BLENDER_PER_GPU=${BLENDER_PER_GPU:-1}
RESPAWN_SLEEP=${RESPAWN_SLEEP:-30}
STDOUT_SYNC_SEC=${STDOUT_SYNC_SEC:-60}
MAX_RESPAWN=${MAX_RESPAWN:-1000000}
MAX_CONSEC_FAILURES=${MAX_CONSEC_FAILURES:-10}  # give up a worker after this many back-to-back crashes
FAIL_FAST=${FAIL_FAST:-0}
MAX_ITEMS=${MAX_ITEMS:-}   # if non-empty, passed through to each worker for smoke tests

# Config (texverse_animate stage B; v2 manifest with `stage` field).
BLENDER_PATH="/tmp/blender-4.5.1-linux-x64/blender"
MANIFEST="/threed-code/yanruibin/efs/4D_video_data_process/data/texverse_1k_animate/texverse_animate_manifest_v2.json"
S3_OUTPUT_ROOT="s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/texverse_1k_animate/rendered_v1_stageB"
LOCAL_OUTPUT_ROOT="/local-ssd/texverse_animate_stageB_rendered"
LOCAL_TMP_ROOT="/local-ssd/texverse_animate_stageB_tmp"
LOCAL_STATE_ROOT="/local-ssd/texverse_animate_stageB_state"
LOCAL_LOG_ROOT="/local-ssd/texverse_animate_stageB_logs"
RENDER_SCRIPT="tools/bl_rendering/dynamic_obj_rendering.py"
STAGE_FILTER="${STAGE_FILTER:-B-1-nested,B-2-blend,B-3-archive}"

mkdir -p "$LOCAL_LOG_ROOT"

# GPU detection
if [[ -n "${NUM_GPUS:-}" ]]; then
    G=$NUM_GPUS
else
    G=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
if (( G == 0 )); then
    echo "[launcher] FATAL: no GPUs detected"
    exit 1
fi
TOTAL_PER_POD=$((G * BLENDER_PER_GPU))
SUB_WS=$((GLOBAL_WS * TOTAL_PER_POD))
SUB_RANK_BASE=$((GLOBAL_RANK * TOTAL_PER_POD))

echo "================================================================"
echo "[launcher] dynamic_obj_rendering multi-process launcher"
echo "  host              = $(hostname)"
echo "  GLOBAL_WS         = $GLOBAL_WS"
echo "  GLOBAL_RANK       = $GLOBAL_RANK"
echo "  detected GPUs     = $G"
echo "  BLENDER_PER_GPU   = $BLENDER_PER_GPU"
echo "  workers this pod  = $TOTAL_PER_POD"
echo "  cgroup CPU quota  = $CPU_QUOTA cores  (nproc=$(nproc))"
echo "  effective WS      = $SUB_WS"
echo "  rank range        = [$SUB_RANK_BASE, $((SUB_RANK_BASE + TOTAL_PER_POD - 1))]"
echo "  FAIL_FAST         = $FAIL_FAST"
echo "  RESPAWN_SLEEP     = $RESPAWN_SLEEP"
echo "  log dir           = $LOCAL_LOG_ROOT"
echo "  s3 logs prefix    = $S3_OUTPUT_ROOT/logs/"
echo "  STAGE_FILTER      = $STAGE_FILTER"
echo "  MANIFEST          = $MANIFEST"
echo "================================================================"

# Ensure all children die when the launcher exits.
declare -a WORKER_PIDS=()
declare -a SYNC_PIDS=()
cleanup() {
    echo "[launcher] cleanup: killing children"
    for p in "${WORKER_PIDS[@]}" "${SYNC_PIDS[@]}"; do
        kill -TERM "$p" 2>/dev/null || true
    done
    sleep 2
    for p in "${WORKER_PIDS[@]}" "${SYNC_PIDS[@]}"; do
        kill -KILL "$p" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# Background: tail-uploads one worker's stdout to S3 periodically.
sync_stdout_loop() {
    local rank=$1
    local local_log=$2
    local s3_log="$S3_OUTPUT_ROOT/logs/worker_${rank}.stdout"
    while true; do
        sleep "$STDOUT_SYNC_SEC"
        [[ -f "$local_log" ]] && aws s3 cp --only-show-errors "$local_log" "$s3_log" 2>/dev/null || true
    done
}

# Supervisor: keep one worker (rank) running, respawn on crash, log deaths.
worker_supervisor() {
    local rank=$1
    local gpu=$2
    local worker_tag=$3
    local local_log="$LOCAL_LOG_ROOT/worker_${rank}.stdout"
    local death_log="$LOCAL_LOG_ROOT/deaths.log"
    local attempts=0

    # Tail-sync this worker stdout in background.
    sync_stdout_loop "$rank" "$local_log" &
    local sync_pid=$!
    SYNC_PIDS+=("$sync_pid")

    local max_items_arg=()
    if [[ -n "$MAX_ITEMS" ]]; then
        max_items_arg=(--max_items "$MAX_ITEMS")
    fi

    local consec_failures=0
    local last_items_done=0
    local hb_file="$LOCAL_STATE_ROOT/rank_${rank}/heartbeat_${rank}.json"
    while (( attempts < MAX_RESPAWN )); do
        attempts=$((attempts + 1))
        echo "[supervisor:r${rank}] attempt #$attempts gpu=$gpu tag=$worker_tag pid_parent=$$" \
            | tee -a "$death_log"
        CUDA_VISIBLE_DEVICES=$gpu python tools/bl_rendering/batch_render_texverse_animate.py \
            --manifest "$MANIFEST" \
            --s3_output_root "$S3_OUTPUT_ROOT" \
            --local_output_root "$LOCAL_OUTPUT_ROOT" \
            --tmp_dir "$LOCAL_TMP_ROOT" \
            --state_dir "$LOCAL_STATE_ROOT" \
            --blender_path "$BLENDER_PATH" \
            --render_script "$RENDER_SCRIPT" \
            --resolution 1024 \
            --num_cameras 16 \
            --camera_stride 4 \
            --max_frames 32 \
            --cycles_samples 256 \
            --render_engine CYCLES \
            --cycles_device GPU \
            --cycles_backend OPTIX \
            --blender_timeout_s 3600 \
            --no_render_normal_map \
            --stage_filter "$STAGE_FILTER" \
            --world_size "$SUB_WS" --rank "$rank" \
            --worker_tag "$worker_tag" \
            "${max_items_arg[@]}" \
            > "$local_log" 2>&1
        rc=$?
        ts=$(date -Is)
        msg="[$ts] [DEATH] rank=$rank attempt=$attempts gpu=$gpu rc=$rc tag=$worker_tag tail=$(tail -n 3 "$local_log" | tr '\n' ' | ')"
        echo "$msg" | tee -a "$death_log"
        # Push the final stdout + death log to S3 immediately.
        aws s3 cp --only-show-errors "$local_log" "$S3_OUTPUT_ROOT/logs/worker_${rank}.stdout" 2>/dev/null || true
        aws s3 cp --only-show-errors "$death_log" "$S3_OUTPUT_ROOT/logs/deaths.log" 2>/dev/null || true

        if (( rc == 0 )); then
            echo "[supervisor:r${rank}] worker exited cleanly, stopping respawn"
            break
        fi
        if (( FAIL_FAST == 1 )); then
            echo "[supervisor:r${rank}] FAIL_FAST=1, signalling launcher to die"
            kill -TERM $$ 2>/dev/null || true
            break
        fi
        # If the worker made forward progress this attempt, reset the
        # consecutive-failure counter so transient errors do not exhaust
        # the budget.
        local cur_items_done=0
        if [[ -r "$hb_file" ]]; then
            cur_items_done=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('items_done',0))" "$hb_file" 2>/dev/null || echo 0)
        fi
        if (( cur_items_done > last_items_done )); then
            consec_failures=0
        else
            consec_failures=$((consec_failures + 1))
        fi
        last_items_done=$cur_items_done

        if (( consec_failures >= MAX_CONSEC_FAILURES )); then
            echo "[supervisor:r${rank}] giving up after $consec_failures consecutive failures (items_done=$cur_items_done)" \
                | tee -a "$death_log"
            aws s3 cp --only-show-errors "$death_log" "$S3_OUTPUT_ROOT/logs/deaths.log" 2>/dev/null || true
            break
        fi
        echo "[supervisor:r${rank}] consec_failures=$consec_failures items_done=$cur_items_done, sleeping ${RESPAWN_SLEEP}s before respawn"
        sleep "$RESPAWN_SLEEP"
    done

    kill "$sync_pid" 2>/dev/null || true
}

# Per-worker thread cap so that BLENDER_PER_GPU workers don't oversubscribe.
if (( CPU_QUOTA > 0 )); then
    PER_WORKER_CPU=$(( CPU_QUOTA / TOTAL_PER_POD ))
    (( PER_WORKER_CPU < 1 )) && PER_WORKER_CPU=1
else
    PER_WORKER_CPU=0
fi
echo "  per-worker CPU cap = $PER_WORKER_CPU (OMP/MKL/x265 threads inside each worker)"
echo "================================================================"

# Launch one supervisor per worker.
for i in $(seq 0 $((TOTAL_PER_POD - 1))); do
    GPU=$((i / BLENDER_PER_GPU))
    LOCAL_PROC=$((i % BLENDER_PER_GPU))
    RANK=$((SUB_RANK_BASE + i))
    TAG="g${GPU}_p${LOCAL_PROC}"
    if (( PER_WORKER_CPU > 0 )); then
        OMP_NUM_THREADS=$PER_WORKER_CPU \
        MKL_NUM_THREADS=$PER_WORKER_CPU \
        OPENBLAS_NUM_THREADS=$PER_WORKER_CPU \
        NUMEXPR_NUM_THREADS=$PER_WORKER_CPU \
        BLENDER_NUM_THREADS=$PER_WORKER_CPU \
        worker_supervisor "$RANK" "$GPU" "$TAG" &
    else
        worker_supervisor "$RANK" "$GPU" "$TAG" &
    fi
    WORKER_PIDS+=("$!")
    sleep 1
done

echo "[launcher] all $TOTAL_PER_POD supervisors started: ${WORKER_PIDS[*]}"

# Wait for all supervisors. Exit code = 0 if everyone exited cleanly,
# nonzero otherwise (so koala can mark the pod failed if needed).
EXIT_RC=0
for p in "${WORKER_PIDS[@]}"; do
    if ! wait "$p"; then
        EXIT_RC=1
    fi
done

echo "[launcher] all supervisors done, exit_rc=$EXIT_RC"
exit $EXIT_RC
