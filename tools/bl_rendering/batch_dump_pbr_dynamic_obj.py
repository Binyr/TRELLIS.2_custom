#!/usr/bin/env python3
"""
Batch dump PBR/topology metadata for ObjaverseXL dynamic objects.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


RENDER_VIEW_CANDIDATES = [0, 2, 4, 6, 8, 10, 12, 14]
TERMINAL_SKIP_STATUSES = {"success"}


def _is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def run_aws(args, retries: int = 2, sleep_s: float = 2.0, check: bool = False) -> subprocess.CompletedProcess:
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
            f"aws {' '.join(args)} failed rc={last.returncode}\n"
            f"stdout={last.stdout[-500:]}\nstderr={last.stderr[-500:]}"
        )
    return last


def s3_cp_file(local_path: str, s3_uri: str, retries: int = 2) -> bool:
    proc = run_aws(["s3", "cp", "--only-show-errors", local_path, s3_uri], retries=retries)
    if proc.returncode != 0:
        print(f"[s3] cp FAILED {local_path} -> {s3_uri}\n{proc.stderr.strip()[:500]}")
        return False
    return True


def s3_get_file_to_local(s3_uri: str, local_path: str, retries: int = 2) -> bool:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    proc = run_aws(["s3", "cp", "--only-show-errors", s3_uri, local_path], retries=retries)
    return proc.returncode == 0


def s3_exists_file(s3_uri: str) -> bool:
    proc = run_aws(["s3", "ls", s3_uri], retries=1)
    return proc.returncode == 0 and s3_uri.rsplit("/", 1)[-1] in proc.stdout


def load_manifest(manifest_path: str, local_state_dir: str) -> list:
    if _is_s3_uri(manifest_path):
        local = os.path.join(local_state_dir, "manifest.json")
        if not s3_get_file_to_local(manifest_path, local, retries=3):
            raise RuntimeError(f"Failed to download manifest from {manifest_path}")
        manifest_path = local
    with open(manifest_path) as f:
        data = json.load(f)
    return data["manifest"]


class ProgressStore:
    def __init__(self, s3_output_root: str, rank: int, local_state_dir: str):
        os.makedirs(local_state_dir, exist_ok=True)
        self.rank = int(rank)
        self.s3_logs_root = s3_output_root.rstrip("/") + "/logs"
        self.s3_progress_uri = f"{self.s3_logs_root}/progress_{rank}.json"
        self.s3_status_uri = f"{self.s3_logs_root}/status_{rank}.log"
        self.local_progress = os.path.join(local_state_dir, f"progress_{rank}.json")
        self.local_status = os.path.join(local_state_dir, f"status_{rank}.log")
        self.progress = {}

    def load(self):
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
        if not s3_get_file_to_local(self.s3_status_uri, self.local_status, retries=1):
            open(self.local_status, "w").close()

    def is_terminal(self, sha256: str) -> bool:
        return self.progress.get(sha256, {}).get("status") in TERMINAL_SKIP_STATUSES

    def update(self, sha256: str, entry: dict, status_line: str, push: bool = True):
        self.progress[sha256] = entry
        tmp = self.local_progress + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.progress, f)
        os.replace(tmp, self.local_progress)
        with open(self.local_status, "a") as f:
            f.write(status_line.rstrip("\n") + "\n")
        if push:
            self.flush()

    def flush(self):
        s3_cp_file(self.local_progress, self.s3_progress_uri, retries=1)
        s3_cp_file(self.local_status, self.s3_status_uri, retries=1)


def _entry(status: str, **extra) -> dict:
    out = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    for key, value in extra.items():
        if value is not None:
            out[key] = value
    return out


def _line(sha256: str, status: str, **extra) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    parts = [ts, sha256, status]
    for key, value in extra.items():
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def load_objxl_ann_views(ann_file: str) -> dict:
    if not ann_file:
        return {}
    with open(ann_file) as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("train"), list) or not isinstance(data.get("test"), list):
        raise ValueError(f"ObjXL ann_file must be a dict with train/test lists: {ann_file}")
    views_by_obj = {}
    for split in ("train", "test"):
        for ann in data[split]:
            obj_id = ann["obj_id"]
            view_id = ann["view_id"]
            if not isinstance(view_id, str) or not view_id.startswith("view_"):
                raise ValueError(f"Invalid view_id in {ann_file}: {view_id!r}")
            views_by_obj.setdefault(obj_id, set()).add(int(view_id.split("_", 1)[1]))
    return views_by_obj


def download_reference_mesh(sha256: str, s3_render_root: str, local_dir: str, view_candidates=None) -> tuple:
    candidates = list(view_candidates) if view_candidates else RENDER_VIEW_CANDIDATES
    for view_idx in candidates:
        uri = f"{s3_render_root.rstrip('/')}/{sha256}/view_{view_idx:02d}/mesh.npz"
        local = os.path.join(local_dir, f"ref_view_{view_idx:02d}_mesh.npz")
        if s3_get_file_to_local(uri, local, retries=1):
            return local, view_idx
    return "", None


def build_blender_cmd(args, object_path: str, output_path: str, reference_mesh_npz: str):
    cmd = [
        args.blender_path,
        "--background",
        "--python", args.dump_script,
        "--",
        "--object_path", object_path,
        "--output_path", output_path,
        "--max_frames", str(args.max_frames),
    ]
    if reference_mesh_npz:
        cmd += ["--reference_mesh_npz", reference_mesh_npz]
        cmd += ["--vertex_set_eps", str(args.vertex_set_eps)]
        if args.strict_reference_vertices:
            cmd.append("--strict_reference_vertices")
        else:
            cmd.append("--no_strict_reference_vertices")
    return cmd


def run_blender(cmd, timeout_s: float) -> tuple:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:]
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "")
        err = (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        return -1, out[-4000:], err[-4000:]


def write_error_log(state_dir: str, sha256: str, rc: int, stdout_tail: str, stderr_tail: str) -> str:
    path = os.path.join(state_dir, f"err_{sha256}.log")
    with open(path, "w") as f:
        f.write(f"returncode={rc}\n\n--- stdout tail ---\n{stdout_tail}\n\n--- stderr tail ---\n{stderr_tail}\n")
    return path


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--s3_render_root", type=str, required=True)
    parser.add_argument("--s3_output_root", type=str, required=True)
    parser.add_argument("--ann_file", type=str, default="")
    parser.add_argument("--local_output_root", type=str, default="/local-ssd/dynamic_obj_pbr_shared")
    parser.add_argument("--tmp_dir", type=str, default="/local-ssd/tmp_dump_pbr_objxl")
    parser.add_argument("--state_dir", type=str, default="/local-ssd/dump_pbr_objxl_state")
    parser.add_argument("--blender_path", type=str, default="/tmp/blender-4.5.1-linux-x64/blender")
    parser.add_argument("--dump_script", type=str,
                        default=os.path.join(os.path.dirname(__file__), "dump_pbr_dynamic_obj.py"))
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=121)
    parser.add_argument("--blender_timeout_s", type=float, default=3600)
    parser.add_argument("--strict_reference_vertices", action="store_true", default=True)
    parser.add_argument("--no_strict_reference_vertices", dest="strict_reference_vertices", action="store_false")
    parser.add_argument("--vertex_set_eps", type=float, default=1e-4)
    parser.add_argument("--require_reference_mesh", action="store_true", default=True)
    parser.add_argument("--no_require_reference_mesh", dest="require_reference_mesh", action="store_false")
    args = parser.parse_args()

    if not _is_s3_uri(args.s3_output_root):
        raise SystemExit(f"--s3_output_root must be s3://, got {args.s3_output_root}")
    if not _is_s3_uri(args.s3_render_root):
        raise SystemExit(f"--s3_render_root must be s3://, got {args.s3_render_root}")

    args.local_output_root = os.path.join(args.local_output_root, f"rank_{args.rank}")
    args.tmp_dir = os.path.join(args.tmp_dir, f"rank_{args.rank}")
    args.state_dir = os.path.join(args.state_dir, f"rank_{args.rank}")
    os.makedirs(args.local_output_root, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)

    manifest = load_manifest(args.manifest, args.state_dir)
    print(f"[batch_pbr] Loaded manifest with {len(manifest)} items")

    ann_views = load_objxl_ann_views(args.ann_file)
    if ann_views:
        before = len(manifest)
        manifest = [item for item in manifest if item["sha256"] in ann_views]
        num_pairs = sum(len(v) for v in ann_views.values())
        sample_obj = next(iter(ann_views))
        print(
            f"[batch_pbr] ann_file={args.ann_file} ann_objects={len(ann_views)} "
            f"ann_object_view_pairs={num_pairs} manifest={before}->{len(manifest)} "
            f"sample={sample_obj}:{sorted(ann_views[sample_obj])[:8]}"
        )

    start = len(manifest) * args.rank // args.world_size
    end = len(manifest) * (args.rank + 1) // args.world_size
    manifest = manifest[start:end]
    if args.max_items is not None:
        manifest = manifest[:args.max_items]
    print(f"[batch_pbr] rank={args.rank}/{args.world_size} assigned={len(manifest)}")

    progress = ProgressStore(args.s3_output_root, args.rank, args.state_dir)
    progress.load()

    success = skip = fail = 0
    for idx, item in enumerate(manifest):
        sha256 = item["sha256"]
        zip_path = item["zip_path"]
        file_path_in_zip = item["file_path_in_zip"]
        extension = item.get("extension", "")
        s3_out = f"{args.s3_output_root.rstrip('/')}/{sha256}.pickle"

        if progress.is_terminal(sha256) or s3_exists_file(s3_out):
            skip += 1
            continue

        print(f"\n[batch_pbr] [{idx + 1}/{len(manifest)}] sha256={sha256[:16]} ext={extension}")
        tmp_extract_dir = os.path.join(args.tmp_dir, sha256)
        local_pickle = os.path.join(args.local_output_root, f"{sha256}.pickle")
        t0 = time.time()
        try:
            if not os.path.exists(zip_path):
                status = "missing_zip"
                msg = f"zip not found: {zip_path}"
                progress.update(sha256, _entry(status, error=msg), _line(sha256, status, error=msg))
                fail += 1
                continue

            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
            os.makedirs(tmp_extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                if file_path_in_zip not in zf.namelist():
                    status = "missing_member"
                    msg = f"file not in zip: {file_path_in_zip}"
                    progress.update(sha256, _entry(status, error=msg), _line(sha256, status, error=msg))
                    fail += 1
                    continue
                zf.extractall(tmp_extract_dir)

            object_path = os.path.join(tmp_extract_dir, file_path_in_zip)
            if not os.path.exists(object_path):
                status = "extract_failed"
                msg = f"extraction failed: {object_path}"
                progress.update(sha256, _entry(status, error=msg), _line(sha256, status, error=msg))
                fail += 1
                continue

            reference_mesh, reference_view = download_reference_mesh(
                sha256,
                args.s3_render_root,
                args.state_dir,
                sorted(ann_views[sha256]) if ann_views else None,
            )
            if args.require_reference_mesh and not reference_mesh:
                status = "missing_render_mesh"
                msg = "no view_XX/mesh.npz found under render root"
                progress.update(sha256, _entry(status, error=msg), _line(sha256, status, error=msg))
                fail += 1
                continue

            if os.path.exists(local_pickle):
                os.remove(local_pickle)
            cmd = build_blender_cmd(args, object_path, local_pickle, reference_mesh)
            print(f"  [blender] launching {cmd[0]} ref_view={reference_view}")
            rc, stdout_tail, stderr_tail = run_blender(cmd, args.blender_timeout_s)
            elapsed = time.time() - t0
            if rc != 0 or not os.path.exists(local_pickle):
                err_path = write_error_log(args.state_dir, sha256, rc, stdout_tail, stderr_tail)
                s3_cp_file(err_path, f"{args.s3_output_root.rstrip('/')}/logs/errors/{sha256}.log", retries=1)
                status = "timeout" if rc == -1 else "blender_failed"
                if "reference vertices mismatch" in stdout_tail or "reference vertices mismatch" in stderr_tail:
                    status = "vertex_set_mismatch"
                msg = f"rc={rc}"
                progress.update(sha256, _entry(status, error=msg, elapsed_s=round(elapsed, 2)),
                                _line(sha256, status, error=msg, elapsed_s=round(elapsed, 2)))
                fail += 1
                continue

            if not s3_cp_file(local_pickle, s3_out, retries=2):
                status = "upload_failed"
                progress.update(sha256, _entry(status, elapsed_s=round(elapsed, 2)),
                                _line(sha256, status, elapsed_s=round(elapsed, 2)))
                fail += 1
                continue

            status = "success"
            progress.update(
                sha256,
                _entry(status, s3_uri=s3_out, reference_view=reference_view, elapsed_s=round(elapsed, 2)),
                _line(sha256, status, reference_view=reference_view, elapsed_s=round(elapsed, 2)),
            )
            success += 1
        except Exception as exc:
            status = "error"
            progress.update(sha256, _entry(status, error=f"{type(exc).__name__}: {exc}"),
                            _line(sha256, status, error=f"{type(exc).__name__}: {exc}"))
            fail += 1
        finally:
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
            if os.path.exists(local_pickle):
                try:
                    os.remove(local_pickle)
                except Exception:
                    pass

        if (idx + 1) % 10 == 0:
            print(f"[batch_pbr] progress success={success} skip={skip} fail={fail}")

    progress.flush()
    print(f"\n[batch_pbr] DONE success={success} skip={skip} fail={fail} total={len(manifest)}")


if __name__ == "__main__":
    main()
