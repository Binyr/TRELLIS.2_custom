"""
Batch render TexVerse-Animation zips using Blender, with view-level S3 resume.

This is the TexVerse counterpart of `batch_render_dynamic_obj.py`. Manifest
entries use `id` (32-char TexVerse uuid) instead of `sha256`, but otherwise
the schema and per-view resume / S3 layout are identical. The same
`dynamic_obj_rendering.py` blender script is reused unchanged (we just feed
our `id` into its `--sha256` arg, which it treats purely as a case label).

Per-rank state lives in S3:
    {s3_output_root}/logs/progress_{rank}.json
    {s3_output_root}/logs/status_{rank}.log

Usage:
    python tools/bl_rendering/batch_render_texverse_animate.py \
        --manifest /local-ssd/data/texverse_1k_animate/texverse_animate_manifest.json \
        --s3_output_root s3://.../texverse_1k_animate/rendered_v1 \
        --local_output_root /local-ssd/texverse_animate_rendered \
        --blender_path /tmp/blender-4.5.1-linux-x64/blender \
        --world_size 1 --rank 0
"""

import argparse
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone


RENDERABLE_EXTS = (".fbx", ".glb", ".gltf")  # what dynamic_obj_rendering.py natively renders without .blend importer
ARCHIVE_BLEND_EXT = ".blend"


VIEW_DONE_PREFIX = "[VIEW_DONE] "
TERMINAL_SKIP_STATUSES = {"success", "missing_resource"}


# =====================================================================================
# S3 helpers
# =====================================================================================


def _is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def run_aws(args, retries: int = 2, sleep_s: float = 2.0, check: bool = False) -> subprocess.CompletedProcess:
    """Run `aws ...` with simple retry."""
    last = None
    for attempt in range(retries + 1):
        proc = subprocess.run(["aws"] + args, capture_output=True, text=True)
        if proc.returncode == 0:
            return proc
        last = proc
        if attempt < retries:
            time.sleep(sleep_s * (attempt + 1))
    if check:
        raise RuntimeError(
            f"aws {' '.join(args)} failed rc={last.returncode}\nstdout={last.stdout[-500:]}\nstderr={last.stderr[-500:]}"
        )
    return last


def s3_cp_dir(local_dir: str, s3_uri: str, retries: int = 2) -> bool:
    proc = run_aws(["s3", "cp", "--recursive", "--only-show-errors", local_dir, s3_uri], retries=retries)
    if proc.returncode != 0:
        print(f"[s3] cp --recursive FAILED {local_dir} -> {s3_uri}\n{proc.stderr.strip()[:500]}")
        return False
    return True


def s3_cp_file(local_path: str, s3_uri: str, retries: int = 2) -> bool:
    proc = run_aws(["s3", "cp", "--only-show-errors", local_path, s3_uri], retries=retries)
    if proc.returncode != 0:
        print(f"[s3] cp FAILED {local_path} -> {s3_uri}\n{proc.stderr.strip()[:500]}")
        return False
    return True


def s3_get_file_to_local(s3_uri: str, local_path: str, retries: int = 2) -> bool:
    proc = run_aws(["s3", "cp", "--only-show-errors", s3_uri, local_path], retries=retries)
    return proc.returncode == 0


def s3_exists_file(s3_uri: str) -> bool:
    """Check whether an S3 object exists via `aws s3 ls`."""
    proc = run_aws(["s3", "ls", s3_uri], retries=1)
    if proc.returncode != 0:
        return False
    return s3_uri.rsplit("/", 1)[-1] in proc.stdout


# =====================================================================================
# Progress / status log (per-rank, stored on S3)
# =====================================================================================


class ProgressStore:
    """Per-rank progress JSON + append-only status log, mirrored to S3."""

    def __init__(self, s3_output_root: str, rank: int, local_state_dir: str,
                 push_every_updates: int = 8, push_min_interval_s: float = 30.0):
        os.makedirs(local_state_dir, exist_ok=True)
        self.rank = int(rank)
        self.s3_logs_root = s3_output_root.rstrip("/") + "/logs"
        self.s3_progress_uri = f"{self.s3_logs_root}/progress_{rank}.json"
        self.s3_status_uri = f"{self.s3_logs_root}/status_{rank}.log"
        self.local_progress = os.path.join(local_state_dir, f"progress_{rank}.json")
        self.local_status = os.path.join(local_state_dir, f"status_{rank}.log")
        self.progress: dict = {}
        # S3 push throttling: avoid 1 put per view (8 PUTs per obj * 1000s obj).
        self.push_every_updates = max(1, int(push_every_updates))
        self.push_min_interval_s = float(push_min_interval_s)
        self._updates_since_push = 0
        self._last_push_ts = 0.0

    def load(self):
        # Try to pull progress + status from S3 (best-effort).
        if s3_get_file_to_local(self.s3_progress_uri, self.local_progress, retries=1):
            try:
                with open(self.local_progress) as f:
                    self.progress = json.load(f)
                print(f"[progress] Loaded {len(self.progress)} entries from {self.s3_progress_uri}")
            except Exception as e:
                print(f"[progress] Failed to parse {self.local_progress}: {e}. Starting empty.")
                self.progress = {}
        else:
            print(f"[progress] No remote progress at {self.s3_progress_uri}; starting empty.")
            self.progress = {}
        # Pull status log so appends are non-destructive.
        if not s3_get_file_to_local(self.s3_status_uri, self.local_status, retries=1):
            open(self.local_status, "w").close()

    def view_key(self, sha256: str, view_idx: int) -> str:
        return f"{sha256}/view_{int(view_idx):02d}"

    def view_status(self, sha256: str, view_idx: int) -> str:
        return self.progress.get(self.view_key(sha256, view_idx), {}).get("status", "")

    def is_view_success(self, sha256: str, view_idx: int) -> bool:
        return self.view_status(sha256, view_idx) == "success"

    def is_view_terminal(self, sha256: str, view_idx: int) -> bool:
        return self.view_status(sha256, view_idx) in TERMINAL_SKIP_STATUSES

    def pending_views(self, sha256: str, all_view_indices) -> list:
        return [v for v in all_view_indices if not self.is_view_terminal(sha256, v)]

    def _save_progress_local(self):
        tmp = self.local_progress + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.progress, f)
        os.replace(tmp, self.local_progress)

    def _append_status_local(self, line: str):
        with open(self.local_status, "a") as f:
            f.write(line.rstrip("\n") + "\n")

    def update_view(self, sha256: str, view_idx: int, entry: dict, status_line: str,
                    force_push: bool = False):
        key = self.view_key(sha256, view_idx)
        self.progress[key] = entry
        self._save_progress_local()
        self._append_status_local(status_line)
        # Push to S3 with throttling: status log is append-only and cheap to
        # re-upload, but progress.json grows linearly. Push when either
        # (a) force_push (e.g. obj-level boundary), (b) we've accumulated
        # `push_every_updates` updates, or (c) `push_min_interval_s` has
        # elapsed since the last push.
        self._updates_since_push += 1
        now = time.time()
        should_push = (
            force_push
            or self._updates_since_push >= self.push_every_updates
            or (now - self._last_push_ts) >= self.push_min_interval_s
        )
        if should_push:
            s3_cp_file(self.local_progress, self.s3_progress_uri, retries=1)
            s3_cp_file(self.local_status, self.s3_status_uri, retries=1)
            self._updates_since_push = 0
            self._last_push_ts = now

    def flush(self):
        """Force-push current progress + status to S3 (e.g. on shutdown)."""
        s3_cp_file(self.local_progress, self.s3_progress_uri, retries=1)
        s3_cp_file(self.local_status, self.s3_status_uri, retries=1)
        self._updates_since_push = 0
        self._last_push_ts = time.time()


# =====================================================================================
# Manifest loading (supports S3 paths)
# =====================================================================================


def _scan_for_main_model(root_dir: str) -> str:
    """Walk root_dir, prefer the shallowest fbx/glb/gltf. Returns absolute path or ""."""
    best = None
    best_key = None
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in RENDERABLE_EXTS:
                p = os.path.join(dp, fn)
                # Lower priority = better: depth, then path length, then lex.
                rel = os.path.relpath(p, root_dir)
                key = (rel.count(os.sep), len(rel), rel.lower(),
                       RENDERABLE_EXTS.index(ext))
                if best_key is None or key < best_key:
                    best = p
                    best_key = key
    return best or ""


def _scan_blend_only(root_dir: str) -> bool:
    """True if root_dir contains a .blend but no fbx/glb/gltf (B-3 skip rule)."""
    has_blend = False
    has_renderable = False
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext == ARCHIVE_BLEND_EXT:
                has_blend = True
            elif ext in RENDERABLE_EXTS:
                has_renderable = True
    return has_blend and not has_renderable


def _extract_inner_zip(outer_root: str, inner_zip_rel: str, dest_dir: str) -> bool:
    """B-1: extract inner_zip_rel (inside the already-extracted outer zip) into dest_dir."""
    inner_zip_abs = os.path.join(outer_root, inner_zip_rel)
    if not os.path.isfile(inner_zip_abs):
        print(f"  [B-1] inner zip not found: {inner_zip_rel}")
        return False
    try:
        with zipfile.ZipFile(inner_zip_abs, "r") as izf:
            izf.extractall(dest_dir)
        return True
    except Exception as e:
        print(f"  [B-1] inner zip extract failed: {type(e).__name__}: {e}")
        return False


def _extract_archive(archive_path: str, kind: str, dest_dir: str, timeout_s: int = 300) -> bool:
    """B-3: shell out to 7z / unrar to expand a .7z/.rar."""
    os.makedirs(dest_dir, exist_ok=True)
    if kind == "7z":
        cmd = ["7z", "x", "-y", f"-o{dest_dir}", archive_path]
    elif kind == "rar":
        cmd = ["unrar", "x", "-y", "-o+", archive_path, dest_dir + os.sep]
    else:
        print(f"  [B-3] unknown archive kind: {kind}")
        return False
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            print(f"  [B-3] {cmd[0]} rc={proc.returncode}: "
                  f"{proc.stderr.strip()[:300] or proc.stdout.strip()[-300:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [B-3] {cmd[0]} timeout after {timeout_s}s on {archive_path}")
        return False
    except FileNotFoundError:
        print(f"  [B-3] {cmd[0]} not installed on this machine")
        return False


def resolve_object_path(item: dict, tmp_extract_dir: str) -> tuple:
    """Given an already-extracted outer zip dir, return (object_path, skip_reason).
    object_path is the absolute path of the file to feed to Blender's --object_path;
    skip_reason is "" on success, non-empty string when the item should be skipped."""
    stage = item.get("stage", "A")
    if stage == "A":
        p = os.path.join(tmp_extract_dir, item["file_path_in_zip"])
        if not os.path.exists(p):
            return "", f"extract_failed:{p}"
        return p, ""

    if stage == "B-2-blend":
        p = os.path.join(tmp_extract_dir, item["file_path_in_zip"])
        if not os.path.exists(p):
            return "", f"extract_failed:{p}"
        return p, ""

    if stage == "B-1-nested":
        # Outer zip already extracted; now unwrap the inner zip.
        inner_dest = os.path.join(tmp_extract_dir, "_b1_inner")
        if not _extract_inner_zip(tmp_extract_dir, item["file_path_in_zip"], inner_dest):
            return "", "b1_inner_extract_failed"
        # Prefer the path the manifest already pinpointed, fallback to a re-scan.
        hint = item.get("nested_inner_path", "")
        if hint:
            p = os.path.join(inner_dest, hint)
            if os.path.exists(p):
                return p, ""
        p = _scan_for_main_model(inner_dest)
        if not p:
            return "", "b1_no_renderable_after_extract"
        return p, ""

    if stage == "B-3-archive":
        kind = item.get("archive_kind", "")
        archive_member = os.path.join(tmp_extract_dir, item["file_path_in_zip"])
        if not os.path.isfile(archive_member):
            return "", f"b3_archive_member_missing:{archive_member}"
        dest = os.path.join(tmp_extract_dir, "_b3_extract")
        if not _extract_archive(archive_member, kind, dest):
            return "", f"b3_archive_extract_failed:{kind}"
        if _scan_blend_only(dest):
            return "", "b3_archive_only_blend_skipped"
        p = _scan_for_main_model(dest)
        if not p:
            return "", "b3_no_renderable_in_archive"
        return p, ""

    return "", f"unknown_stage:{stage}"


def load_manifest(manifest_path: str, local_state_dir: str) -> list:
    if _is_s3_uri(manifest_path):
        local = os.path.join(local_state_dir, "manifest.json")
        if not s3_get_file_to_local(manifest_path, local, retries=3):
            raise RuntimeError(f"Failed to download manifest from {manifest_path}")
        manifest_path = local
    with open(manifest_path) as f:
        data = json.load(f)
    return data["manifest"]


# =====================================================================================
# Blender driver
# =====================================================================================


def _traj_seed_from_id(obj_id: str) -> int:
    # Deterministic per-obj seed so each obj gets a different azim0/elevation
    # set, while resumes for the same obj reproduce identical cameras.
    # TexVerse ids are 32-char hex; fall back to a string hash otherwise.
    try:
        v = int(obj_id, 16)
    except ValueError:
        v = abs(hash(obj_id))
    return v & 0x7FFFFFFF


def build_blender_cmd(args, item, extracted_path: str, local_obj_dir: str, views_to_render):
    traj_seed = _traj_seed_from_id(item["id"])
    cmd = [
        args.blender_path,
        "--background",
        "--python", args.render_script,
        "--",
        "--object_path", extracted_path,
        "--obj_root", local_obj_dir,
        "--sha256", item["id"],
        "--resolution", str(args.resolution),
        "--render_engine", args.render_engine,
        "--num_cameras", str(args.num_cameras),
        "--cycles_samples", str(args.cycles_samples),
        "--cycles_device", args.cycles_device,
        "--cycles_backend", args.cycles_backend,
        "--video_fps", str(args.video_fps),
        "--max_frames", str(args.max_frames),
        "--traj_seed", str(traj_seed),
        "--traj_id", "0",
        "--transparent_bg",
    ]
    if args.render_normal_map:
        cmd.append("--render_normal_map")
    else:
        cmd.append("--no_render_normal_map")
    if views_to_render:
        cmd += ["--render_view_indices"] + [str(int(v)) for v in views_to_render]
    return cmd


def run_blender_streaming(cmd, timeout_s: float, on_view_done):
    """Run blender, stream combined stdout/stderr, call on_view_done for VIEW_DONE markers.
    Returns (returncode, last_stdout_tail, last_stderr_tail).
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines = []
    started_at = time.time()
    timed_out = False

    def handle_line(line: str):
        line = line.rstrip("\n")
        stdout_lines.append(line)
        if len(stdout_lines) > 4000:
            del stdout_lines[:-3000]
        if line.startswith(VIEW_DONE_PREFIX):
            payload_str = line[len(VIEW_DONE_PREFIX):].strip()
            try:
                payload = json.loads(payload_str)
                on_view_done(payload)
            except Exception as e:
                print(f"[batch] Failed to parse VIEW_DONE: {e}; line={line!r}")
        elif line:
            print(line)

    try:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while proc.poll() is None:
            if timeout_s and (time.time() - started_at) > timeout_s:
                timed_out = True
                proc.kill()
                break
            ready, _, _ = select.select([fd], [], [], 1.0)
            if not ready:
                continue
            line = proc.stdout.readline()
            if line == "":
                break
            handle_line(line)

        for line in proc.stdout:
            handle_line(line)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    stdout_tail = "\n".join(stdout_lines[-200:])
    if timed_out:
        return -1, stdout_tail, ""
    return proc.returncode, stdout_tail, ""


# =====================================================================================
# Main
# =====================================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path or s3 URI of dynamic_obj_manifest.json.")
    parser.add_argument("--s3_output_root", type=str, required=True,
                        help="Final S3 destination prefix, e.g. s3://bucket/path/dynamic_obj_rendered")
    parser.add_argument("--local_output_root", type=str, default="/local-ssd/dynamic_obj_rendered",
                        help="Local scratch dir where Blender writes outputs before s3 cp.")
    parser.add_argument("--blender_path", type=str, default="/tmp/blender-4.5.1-linux-x64/blender")
    parser.add_argument("--render_script", type=str,
                        default=os.path.join(os.path.dirname(__file__), "dynamic_obj_rendering.py"))
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_cameras", type=int, default=16)
    parser.add_argument("--camera_stride", type=int, default=1)
    parser.add_argument("--cycles_samples", type=int, default=256)
    parser.add_argument("--render_engine", type=str, default="CYCLES")
    parser.add_argument("--cycles_device", type=str, default="GPU")
    parser.add_argument("--cycles_backend", type=str, default="OPTIX")
    parser.add_argument("--video_fps", type=int, default=24)
    parser.add_argument("--render_normal_map", action="store_true", default=True)
    parser.add_argument("--no_render_normal_map", dest="render_normal_map", action="store_false")
    parser.add_argument("--tmp_dir", type=str, default="/local-ssd/tmp_extract",
                        help="Where to extract zip contents.")
    parser.add_argument("--state_dir", type=str, default="/local-ssd/render_dynamic_obj_state",
                        help="Local dir for progress/status mirror and manifest cache.")
    parser.add_argument("--blender_timeout_s", type=float, default=60 * 60 * 12,
                        help="Per-obj Blender timeout (default 12h).")
    parser.add_argument("--max_items", type=int, default=None, help="Limit obj count (debug).")
    parser.add_argument("--stage_filter", type=str, default="",
                        help="If non-empty, only render manifest entries whose stage matches "
                             "(supports comma list e.g. 'B-1-nested,B-2-blend'). Empty = all.")
    parser.add_argument("--max_frames", type=int, default=121,
                        help="Per-obj animation frame cap, passed through to the Blender script.")
    parser.add_argument("--worker_tag", type=str, default="",
                        help="Free-form tag (e.g. 'g0_p1') prefixed on logs to disambiguate parallel workers.")
    parser.add_argument("--heartbeat_every_objs", type=int, default=1,
                        help="Push heartbeat JSON to S3 every N completed objs (success+skip+fail).")
    args = parser.parse_args()

    if not _is_s3_uri(args.s3_output_root):
        raise SystemExit(f"--s3_output_root must be an s3:// URI, got: {args.s3_output_root}")

    # When multiple workers share one pod, isolate their scratch / state dirs.
    args.local_output_root = os.path.join(args.local_output_root, f"rank_{args.rank}")
    args.tmp_dir = os.path.join(args.tmp_dir, f"rank_{args.rank}")
    args.state_dir = os.path.join(args.state_dir, f"rank_{args.rank}")
    os.makedirs(args.local_output_root, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)

    tag = f"[{args.worker_tag}]" if args.worker_tag else "[batch]"
    print(f"{tag} rank={args.rank}/{args.world_size} pid={os.getpid()} "
          f"local_out={args.local_output_root} state={args.state_dir}")

    all_view_indices = list(range(0, args.num_cameras, args.camera_stride))
    if not all_view_indices:
        raise SystemExit("No views to render (check num_cameras / camera_stride).")
    print(f"[batch] all view indices = {all_view_indices}")

    manifest = load_manifest(args.manifest, args.state_dir)
    print(f"[batch] Loaded manifest with {len(manifest)} items")

    if args.stage_filter:
        wanted = {s.strip() for s in args.stage_filter.split(",") if s.strip()}
        before = len(manifest)
        manifest = [it for it in manifest if it.get("stage", "A") in wanted]
        print(f"[batch] stage_filter={sorted(wanted)} kept {len(manifest)}/{before}")

    start = len(manifest) * args.rank // args.world_size
    end = len(manifest) * (args.rank + 1) // args.world_size
    manifest = manifest[start:end]
    print(f"[batch] Rank {args.rank}/{args.world_size}: {len(manifest)} items")

    if args.max_items is not None:
        manifest = manifest[:args.max_items]
        print(f"[batch] Limited to {len(manifest)} items (--max_items)")

    progress = ProgressStore(args.s3_output_root, args.rank, args.state_dir)
    progress.load()

    s3_heartbeat_uri = f"{args.s3_output_root.rstrip('/')}/logs/heartbeat_{args.rank}.json"
    local_heartbeat = os.path.join(args.state_dir, f"heartbeat_{args.rank}.json")
    started_at = time.time()

    def push_heartbeat(i_done: int, n_total: int, status: str = "running",
                       success: int = 0, skip: int = 0, fail: int = 0,
                       extra: dict = None):
        payload = {
            "rank": args.rank,
            "world_size": args.world_size,
            "worker_tag": args.worker_tag,
            "pid": os.getpid(),
            "host": os.uname().nodename,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "started_at": started_at,
            "ts": time.time(),
            "items_total": n_total,
            "items_done": i_done,
            "success": success,
            "skip": skip,
            "fail": fail,
            "status": status,
        }
        if extra:
            payload.update(extra)
        try:
            with open(local_heartbeat, "w") as f:
                json.dump(payload, f, indent=2)
            s3_cp_file(local_heartbeat, s3_heartbeat_uri, retries=1)
        except Exception as e:
            print(f"{tag} heartbeat push failed: {e}")

    todo = []
    finished_obj = 0
    for item in manifest:
        pending_views = progress.pending_views(item["id"], all_view_indices)
        if len(pending_views) == 0:
            finished_obj += 1
        else:
            todo.append((item, pending_views))
    print(f"[batch] finished obj: {finished_obj}; remaining obj: {len(todo)}; total obj: {len(manifest)}")

    push_heartbeat(finished_obj, len(manifest), status="starting")

    success_obj = 0
    skip_obj = finished_obj
    fail_obj = 0

    for i, (item, pending_views) in enumerate(todo):
        obj_id = item["id"]
        zip_path = item["zip_path"]
        file_path_in_zip = item["file_path_in_zip"]
        extension = item.get("extension", "")

        print(f"\n{tag} [{i+1}/{len(todo)}] id={obj_id} ext={extension} pending_views={pending_views}")

        tmp_extract_dir = os.path.join(args.tmp_dir, obj_id)
        local_obj_dir = os.path.join(args.local_output_root, obj_id)
        s3_obj_root = f"{args.s3_output_root.rstrip('/')}/{obj_id}"

        try:
            if not os.path.exists(zip_path):
                msg = f"zip not found: {zip_path}"
                print(f"  [SKIP] {msg}")
                for v in pending_views:
                    progress.update_view(obj_id, v, _entry("missing_input", error=msg), _line(obj_id, v, "missing_input", error=msg))
                fail_obj += 1
                continue

            os.makedirs(tmp_extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                if file_path_in_zip not in zf.namelist():
                    msg = f"file not in zip: {file_path_in_zip}"
                    print(f"  [SKIP] {msg}")
                    for v in pending_views:
                        progress.update_view(obj_id, v, _entry("missing_input", error=msg), _line(obj_id, v, "missing_input", error=msg))
                    fail_obj += 1
                    continue
                # Extract the full archive so accompanying textures (e.g.
                # `textures/*.png` referenced by the fbx/glb) are present.
                zf.extractall(tmp_extract_dir)

            # Stage-aware: A returns the .fbx/.glb/.gltf path; B-1 unwraps an
            # inner zip; B-2 returns the .blend path; B-3 shells out to 7z/unrar.
            extracted_path, skip_reason = resolve_object_path(item, tmp_extract_dir)
            if skip_reason:
                print(f"  [SKIP/{item.get('stage','A')}] {skip_reason}")
                for v in pending_views:
                    progress.update_view(obj_id, v,
                        _entry("extract_failed", error=skip_reason),
                        _line(obj_id, v, "extract_failed", error=skip_reason))
                fail_obj += 1
                continue

            os.makedirs(local_obj_dir, exist_ok=True)

            # Closure: handle a single [VIEW_DONE] -> upload + log.
            def on_view_done(payload):
                v = int(payload["view_index"])
                status = str(payload.get("status", "error"))
                view_local_dir = os.path.join(local_obj_dir, f"view_{v:02d}")
                s3_view_uri = f"{s3_obj_root}/view_{v:02d}"
                if status == "success":
                    ok = s3_cp_dir(view_local_dir, s3_view_uri, retries=2)
                    if ok:
                        entry = _entry(
                            "success",
                            s3_prefix=s3_view_uri,
                            render_time_s=payload.get("render_time_s"),
                        )
                        line = _line(obj_id, v, "success", render_time_s=payload.get("render_time_s"))
                        shutil.rmtree(view_local_dir, ignore_errors=True)
                    else:
                        entry = _entry("upload_failed", error="s3 cp recursive failed")
                        line = _line(obj_id, v, "upload_failed", error="s3 cp recursive failed")
                else:
                    err = payload.get("error", "")
                    entry = _entry(status, error=err)
                    line = _line(obj_id, v, status, error=err)
                progress.update_view(obj_id, v, entry, line)

            cmd = build_blender_cmd(args, item, extracted_path, local_obj_dir, pending_views)
            print(f"  [blender] launching: {cmd[0]} id={obj_id} pending_views={pending_views}")
            t0 = time.time()
            rc, stdout_tail, stderr_tail = run_blender_streaming(
                cmd,
                timeout_s=args.blender_timeout_s,
                on_view_done=on_view_done,
            )
            elapsed = time.time() - t0

            still_pending = progress.pending_views(obj_id, all_view_indices)
            still_pending = [v for v in still_pending if v in pending_views]

            if rc == 0 and len(still_pending) == 0:
                success_obj += 1
                print(f"  [OK] {elapsed:.1f}s, all views uploaded")
            else:
                fail_obj += 1
                fail_tag = "timeout" if rc == -1 else "blender_failed"
                if _is_missing_resource_error(stdout_tail, stderr_tail):
                    fail_tag = "missing_resource"
                msg = f"rc={rc} tag={fail_tag}; pending_after={still_pending}"
                print(f"  [FAIL] {msg} ({elapsed:.1f}s)")
                _upload_error_log(args.state_dir, s3_obj_root, obj_id, rc, stdout_tail, stderr_tail)
                for v in still_pending:
                    if fail_tag == "missing_resource" or progress.view_status(obj_id, v) == "":
                        progress.update_view(
                            obj_id, v, _entry(fail_tag, error=msg),
                            _line(obj_id, v, fail_tag, error=msg),
                        )

        except subprocess.TimeoutExpired:
            fail_obj += 1
            print(f"  [TIMEOUT]")
        except Exception as e:
            fail_obj += 1
            print(f"  [ERROR] {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
            shutil.rmtree(local_obj_dir, ignore_errors=True)

        # Force-flush progress + status to S3 at the obj boundary so the
        # per-view throttling never hides a fully-completed obj.
        progress.flush()

        items_done = finished_obj + i + 1
        if items_done % args.heartbeat_every_objs == 0:
            push_heartbeat(items_done, len(manifest), status="running",
                           success=success_obj, skip=skip_obj, fail=fail_obj)

        if (i + 1) % 10 == 0:
            print(f"{tag} [{i+1}/{len(todo)}] success={success_obj} skip={skip_obj} fail={fail_obj}")

    progress.flush()
    push_heartbeat(len(manifest), len(manifest), status="done",
                   success=success_obj, skip=skip_obj, fail=fail_obj)
    print(f"\n{tag} DONE. success={success_obj} skip={skip_obj} fail={fail_obj} total={len(manifest)}")


def _entry(status: str, **extra) -> dict:
    out = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    for k, v in extra.items():
        if v is not None:
            out[k] = v
    return out


def _line(obj_id: str, view_idx: int, status: str, **extra) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    parts = [ts, f"{obj_id}/view_{int(view_idx):02d}", status]
    for k, v in extra.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    return " ".join(parts)


def _is_missing_resource_error(stdout_tail: str, stderr_tail: str) -> bool:
    text = f"{stdout_tail}\n{stderr_tail}"
    return "Missing resource" in text or "Couldn't read file" in text


def _upload_error_log(state_dir: str, s3_obj_root: str, obj_id: str, rc: int, stdout_tail: str, stderr_tail: str):
    try:
        local_err = os.path.join(state_dir, f"err_{obj_id}.log")
        with open(local_err, "w") as f:
            f.write(f"=== rc={rc} ===\n\n=== STDOUT TAIL ===\n{stdout_tail}\n\n=== STDERR TAIL ===\n{stderr_tail}\n")
        s3_cp_file(local_err, f"{s3_obj_root}/error.log", retries=1)
        os.remove(local_err)
    except Exception as e:
        print(f"[batch] Failed to upload error log: {e}")


if __name__ == "__main__":
    main()
