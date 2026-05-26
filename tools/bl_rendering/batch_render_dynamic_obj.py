"""
Batch render dynamic OBJ files using Blender, with view-level S3 resume.

Per-rank state lives in S3:
    {s3_output_root}/logs/progress_{rank}.json
    {s3_output_root}/logs/status_{rank}.log

For each obj, Blender is invoked with only the views that are not yet marked
success in the progress file. The batch layer parses Blender's stdout
"[VIEW_DONE] {...json}" markers, uploads the corresponding per-view directory
to S3, and updates progress/status logs.

Usage:
    python tools/bl_rendering/batch_render_dynamic_obj.py \
        --manifest s3://.../dynamic_obj_manifest.json \
        --s3_output_root s3://.../dynamic_obj_rendered \
        --local_output_root /local-ssd/dynamic_obj_rendered \
        --blender_path /tmp/blender-4.5.1-linux-x64/blender \
        --world_size 1 --rank 0
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone


VIEW_DONE_PREFIX = "[VIEW_DONE] "


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

    def __init__(self, s3_output_root: str, rank: int, local_state_dir: str):
        os.makedirs(local_state_dir, exist_ok=True)
        self.rank = int(rank)
        self.s3_logs_root = s3_output_root.rstrip("/") + "/logs"
        self.s3_progress_uri = f"{self.s3_logs_root}/progress_{rank}.json"
        self.s3_status_uri = f"{self.s3_logs_root}/status_{rank}.log"
        self.local_progress = os.path.join(local_state_dir, f"progress_{rank}.json")
        self.local_status = os.path.join(local_state_dir, f"status_{rank}.log")
        self.progress: dict = {}

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

    def pending_views(self, sha256: str, all_view_indices) -> list:
        return [v for v in all_view_indices if not self.is_view_success(sha256, v)]

    def _save_progress_local(self):
        tmp = self.local_progress + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.progress, f)
        os.replace(tmp, self.local_progress)

    def _append_status_local(self, line: str):
        with open(self.local_status, "a") as f:
            f.write(line.rstrip("\n") + "\n")

    def update_view(self, sha256: str, view_idx: int, entry: dict, status_line: str):
        key = self.view_key(sha256, view_idx)
        self.progress[key] = entry
        self._save_progress_local()
        self._append_status_local(status_line)
        # Push to S3 (best-effort; failure is logged but not fatal).
        s3_cp_file(self.local_progress, self.s3_progress_uri, retries=1)
        s3_cp_file(self.local_status, self.s3_status_uri, retries=1)


# =====================================================================================
# Manifest loading (supports S3 paths)
# =====================================================================================


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


def build_blender_cmd(args, item, extracted_path: str, local_obj_dir: str, views_to_render):
    cmd = [
        args.blender_path,
        "--background",
        "--python", args.render_script,
        "--",
        "--object_path", extracted_path,
        "--obj_root", local_obj_dir,
        "--sha256", item["sha256"],
        "--resolution", str(args.resolution),
        "--render_engine", args.render_engine,
        "--num_cameras", str(args.num_cameras),
        "--cycles_samples", str(args.cycles_samples),
        "--cycles_device", args.cycles_device,
        "--cycles_backend", args.cycles_backend,
        "--video_fps", str(args.video_fps),
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
    """Run blender, stream stdout, call on_view_done(payload) for each [VIEW_DONE] marker.
    Returns (returncode, last_stdout_tail, last_stderr_tail).
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_lines = []
    started_at = time.time()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            stdout_lines.append(line)
            if len(stdout_lines) > 4000:
                stdout_lines = stdout_lines[-3000:]
            if line.startswith(VIEW_DONE_PREFIX):
                payload_str = line[len(VIEW_DONE_PREFIX):].strip()
                try:
                    payload = json.loads(payload_str)
                    on_view_done(payload)
                except Exception as e:
                    print(f"[batch] Failed to parse VIEW_DONE: {e}; line={line!r}")
            elif line:
                print(line)
            if timeout_s and (time.time() - started_at) > timeout_s:
                proc.kill()
                stderr_tail = proc.stderr.read() if proc.stderr else ""
                return -1, "\n".join(stdout_lines[-200:]), stderr_tail[-2000:]
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    stderr_tail = proc.stderr.read() if proc.stderr else ""
    return proc.returncode, "\n".join(stdout_lines[-200:]), stderr_tail[-2000:]


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
    args = parser.parse_args()

    if not _is_s3_uri(args.s3_output_root):
        raise SystemExit(f"--s3_output_root must be an s3:// URI, got: {args.s3_output_root}")

    os.makedirs(args.local_output_root, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)

    all_view_indices = list(range(0, args.num_cameras, args.camera_stride))
    if not all_view_indices:
        raise SystemExit("No views to render (check num_cameras / camera_stride).")
    print(f"[batch] all view indices = {all_view_indices}")

    manifest = load_manifest(args.manifest, args.state_dir)
    print(f"[batch] Loaded manifest with {len(manifest)} items")

    start = len(manifest) * args.rank // args.world_size
    end = len(manifest) * (args.rank + 1) // args.world_size
    manifest = manifest[start:end]
    print(f"[batch] Rank {args.rank}/{args.world_size}: {len(manifest)} items")

    if args.max_items is not None:
        manifest = manifest[:args.max_items]
        print(f"[batch] Limited to {len(manifest)} items (--max_items)")

    progress = ProgressStore(args.s3_output_root, args.rank, args.state_dir)
    progress.load()

    success_obj = 0
    skip_obj = 0
    fail_obj = 0

    for i, item in enumerate(manifest):
        sha256 = item["sha256"]
        zip_path = item["zip_path"]
        file_path_in_zip = item["file_path_in_zip"]
        extension = item.get("extension", "")

        pending_views = progress.pending_views(sha256, all_view_indices)
        if len(pending_views) == 0:
            skip_obj += 1
            if (i + 1) % 100 == 0:
                print(f"[batch] [{i+1}/{len(manifest)}] success={success_obj} skip={skip_obj} fail={fail_obj}")
            continue

        print(f"\n[batch] [{i+1}/{len(manifest)}] sha256={sha256[:16]}... ext={extension} pending_views={pending_views}")

        tmp_extract_dir = os.path.join(args.tmp_dir, sha256)
        local_obj_dir = os.path.join(args.local_output_root, sha256)
        s3_obj_root = f"{args.s3_output_root.rstrip('/')}/{sha256}"

        try:
            if not os.path.exists(zip_path):
                msg = f"zip not found: {zip_path}"
                print(f"  [SKIP] {msg}")
                for v in pending_views:
                    progress.update_view(sha256, v, _entry("missing_input", error=msg), _line(sha256, v, "missing_input", error=msg))
                fail_obj += 1
                continue

            os.makedirs(tmp_extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                if file_path_in_zip not in zf.namelist():
                    msg = f"file not in zip: {file_path_in_zip}"
                    print(f"  [SKIP] {msg}")
                    for v in pending_views:
                        progress.update_view(sha256, v, _entry("missing_input", error=msg), _line(sha256, v, "missing_input", error=msg))
                    fail_obj += 1
                    continue
                zf.extract(file_path_in_zip, tmp_extract_dir)

            extracted_path = os.path.join(tmp_extract_dir, file_path_in_zip)
            if not os.path.exists(extracted_path):
                msg = f"extraction failed: {extracted_path}"
                print(f"  [SKIP] {msg}")
                for v in pending_views:
                    progress.update_view(sha256, v, _entry("extract_failed", error=msg), _line(sha256, v, "extract_failed", error=msg))
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
                        line = _line(sha256, v, "success", render_time_s=payload.get("render_time_s"))
                        shutil.rmtree(view_local_dir, ignore_errors=True)
                    else:
                        entry = _entry("upload_failed", error="s3 cp recursive failed")
                        line = _line(sha256, v, "upload_failed", error="s3 cp recursive failed")
                else:
                    err = payload.get("error", "")
                    entry = _entry(status, error=err)
                    line = _line(sha256, v, status, error=err)
                progress.update_view(sha256, v, entry, line)

            cmd = build_blender_cmd(args, item, extracted_path, local_obj_dir, pending_views)
            t0 = time.time()
            rc, stdout_tail, stderr_tail = run_blender_streaming(
                cmd,
                timeout_s=args.blender_timeout_s,
                on_view_done=on_view_done,
            )
            elapsed = time.time() - t0

            still_pending = progress.pending_views(sha256, all_view_indices)
            still_pending = [v for v in still_pending if v in pending_views]

            if rc == 0 and len(still_pending) == 0:
                success_obj += 1
                print(f"  [OK] {elapsed:.1f}s, all views uploaded")
            else:
                fail_obj += 1
                tag = "timeout" if rc == -1 else "blender_failed"
                msg = f"rc={rc} tag={tag}; pending_after={still_pending}"
                print(f"  [FAIL] {msg} ({elapsed:.1f}s)")
                # Upload tail logs alongside obj for diagnostics.
                _upload_error_log(args.state_dir, s3_obj_root, sha256, rc, stdout_tail, stderr_tail)
                # Mark views that did not receive a [VIEW_DONE] this run.
                for v in still_pending:
                    if progress.view_status(sha256, v) == "":
                        progress.update_view(
                            sha256, v, _entry(tag, error=msg),
                            _line(sha256, v, tag, error=msg),
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

        if (i + 1) % 10 == 0:
            print(f"[batch] [{i+1}/{len(manifest)}] success={success_obj} skip={skip_obj} fail={fail_obj}")

    print(f"\n[batch] DONE. success={success_obj} skip={skip_obj} fail={fail_obj} total={len(manifest)}")


def _entry(status: str, **extra) -> dict:
    out = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    for k, v in extra.items():
        if v is not None:
            out[k] = v
    return out


def _line(sha256: str, view_idx: int, status: str, **extra) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    parts = [ts, f"{sha256}/view_{int(view_idx):02d}", status]
    for k, v in extra.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    return " ".join(parts)


def _upload_error_log(state_dir: str, s3_obj_root: str, sha256: str, rc: int, stdout_tail: str, stderr_tail: str):
    try:
        local_err = os.path.join(state_dir, f"err_{sha256}.log")
        with open(local_err, "w") as f:
            f.write(f"=== rc={rc} ===\n\n=== STDOUT TAIL ===\n{stdout_tail}\n\n=== STDERR TAIL ===\n{stderr_tail}\n")
        s3_cp_file(local_err, f"{s3_obj_root}/error.log", retries=1)
        os.remove(local_err)
    except Exception as e:
        print(f"[batch] Failed to upload error log: {e}")


if __name__ == "__main__":
    main()
