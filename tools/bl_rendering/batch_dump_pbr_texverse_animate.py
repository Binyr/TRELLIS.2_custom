#!/usr/bin/env python3
"""
Batch dump dynamic mesh + PBR metadata for TexVerse-Animation assets.

This reuses the TexVerse render batch's stage-aware extraction logic, then
calls dump_pbr_dynamic_obj.py. The Blender dump is driven by the render-stage
mesh.npz frame_indices and checks every frame's vertex set against mesh.npz.
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

sys.path.insert(0, os.path.dirname(__file__))
from batch_render_texverse_animate import resolve_object_path


DEFAULT_VIEW_CANDIDATES = [0, 4, 8, 12]
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

    def is_terminal(self, obj_id: str) -> bool:
        return self.progress.get(obj_id, {}).get("status") in TERMINAL_SKIP_STATUSES

    def update(self, obj_id: str, entry: dict, status_line: str, push: bool = True):
        self.progress[obj_id] = entry
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


def _line(obj_id: str, status: str, **extra) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    parts = [ts, obj_id, status]
    for key, value in extra.items():
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def parse_view_candidates(value: str) -> list:
    if not value:
        return list(DEFAULT_VIEW_CANDIDATES)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_texverse_ann_views(ann_file: str) -> dict:
    if not ann_file:
        return {}
    with open(ann_file) as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("train"), list) or not isinstance(data.get("test"), list):
        raise ValueError(f"TexVerse ann_file must be a dict with train/test lists: {ann_file}")
    views_by_obj = {}
    for split in ("train", "test"):
        for ann in data[split]:
            obj_id = ann["obj_id"]
            view_id = ann["view_id"]
            if not isinstance(view_id, str) or not view_id.startswith("view_"):
                raise ValueError(f"Invalid view_id in {ann_file}: {view_id!r}")
            views_by_obj.setdefault(obj_id, set()).add(int(view_id.split("_", 1)[1]))
    return views_by_obj


def download_reference_mesh(obj_id: str, s3_render_root: str, local_dir: str, view_candidates: list) -> tuple:
    for view_idx in view_candidates:
        uri = f"{s3_render_root.rstrip('/')}/{obj_id}/view_{view_idx:02d}/mesh.npz"
        local = os.path.join(local_dir, f"ref_{obj_id}_view_{view_idx:02d}_mesh.npz")
        if s3_get_file_to_local(uri, local, retries=1):
            return local, view_idx
    return "", None


def build_blender_cmd(args, object_path: str, output_path: str, reference_mesh_npz: str):
    return [
        args.blender_path,
        "--background",
        "--python", args.dump_script,
        "--",
        "--object_path", object_path,
        "--output_path", output_path,
        "--reference_mesh_npz", reference_mesh_npz,
        "--strict_reference_vertices",
        "--vertex_set_eps", str(args.vertex_set_eps),
        "--max_frames", str(args.max_frames),
    ]


def run_blender(cmd, timeout_s: float) -> tuple:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:]
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        return -1, out[-4000:], err[-4000:]


def write_error_log(state_dir: str, obj_id: str, rc: int, stdout_tail: str, stderr_tail: str) -> str:
    path = os.path.join(state_dir, f"err_{obj_id}.log")
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
    parser.add_argument("--local_output_root", type=str, default="/local-ssd/texverse_animate_pbr_shared")
    parser.add_argument("--tmp_dir", type=str, default="/local-ssd/tmp_dump_pbr_texverse")
    parser.add_argument("--state_dir", type=str, default="/local-ssd/dump_pbr_texverse_state")
    parser.add_argument("--blender_path", type=str, default="/tmp/blender-4.5.1-linux-x64/blender")
    parser.add_argument("--dump_script", type=str,
                        default=os.path.join(os.path.dirname(__file__), "dump_pbr_dynamic_obj.py"))
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--blender_timeout_s", type=float, default=3600)
    parser.add_argument("--vertex_set_eps", type=float, default=1e-4)
    parser.add_argument("--view_candidates", type=str, default="0,4,8,12")
    parser.add_argument("--stage_filter", type=str, default="")
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
    print(f"[batch_pbr_texverse] Loaded manifest with {len(manifest)} items")

    if args.stage_filter:
        wanted = {s.strip() for s in args.stage_filter.split(",") if s.strip()}
        before = len(manifest)
        manifest = [it for it in manifest if it.get("stage", "A") in wanted]
        print(f"[batch_pbr_texverse] stage_filter={sorted(wanted)} kept {len(manifest)}/{before}")

    ann_views = load_texverse_ann_views(args.ann_file)
    if ann_views:
        before = len(manifest)
        manifest = [item for item in manifest if item["id"] in ann_views]
        num_pairs = sum(len(v) for v in ann_views.values())
        sample_obj = next(iter(ann_views))
        print(
            f"[batch_pbr_texverse] ann_file={args.ann_file} ann_objects={len(ann_views)} "
            f"ann_object_view_pairs={num_pairs} manifest={before}->{len(manifest)} "
            f"sample={sample_obj}:{sorted(ann_views[sample_obj])[:8]}"
        )

    start = len(manifest) * args.rank // args.world_size
    end = len(manifest) * (args.rank + 1) // args.world_size
    manifest = manifest[start:end]
    if args.max_items is not None:
        manifest = manifest[:args.max_items]
    print(f"[batch_pbr_texverse] rank={args.rank}/{args.world_size} assigned={len(manifest)}")

    view_candidates = parse_view_candidates(args.view_candidates)
    progress = ProgressStore(args.s3_output_root, args.rank, args.state_dir)
    progress.load()

    success = skip = fail = 0
    for idx, item in enumerate(manifest):
        obj_id = item["id"]
        zip_path = item["zip_path"]
        file_path_in_zip = item["file_path_in_zip"]
        stage = item.get("stage", "A")
        extension = item.get("extension", "")
        s3_out = f"{args.s3_output_root.rstrip('/')}/{obj_id}.pickle"

        if progress.is_terminal(obj_id) or s3_exists_file(s3_out):
            skip += 1
            continue

        print(f"\n[batch_pbr_texverse] [{idx + 1}/{len(manifest)}] id={obj_id} stage={stage} ext={extension}")
        tmp_extract_dir = os.path.join(args.tmp_dir, obj_id)
        local_pickle = os.path.join(args.local_output_root, f"{obj_id}.pickle")
        t0 = time.time()

        try:
            if not os.path.exists(zip_path):
                status = "missing_zip"
                msg = f"zip not found: {zip_path}"
                progress.update(obj_id, _entry(status, error=msg), _line(obj_id, status, error=msg))
                fail += 1
                continue

            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
            os.makedirs(tmp_extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                if file_path_in_zip not in zf.namelist():
                    status = "missing_member"
                    msg = f"file not in zip: {file_path_in_zip}"
                    progress.update(obj_id, _entry(status, error=msg), _line(obj_id, status, error=msg))
                    fail += 1
                    continue
                zf.extractall(tmp_extract_dir)

            object_path, skip_reason = resolve_object_path(item, tmp_extract_dir)
            if skip_reason:
                status = "extract_failed"
                progress.update(obj_id, _entry(status, error=skip_reason), _line(obj_id, status, error=skip_reason))
                fail += 1
                continue

            reference_mesh, reference_view = download_reference_mesh(
                obj_id,
                args.s3_render_root,
                args.state_dir,
                sorted(ann_views[obj_id]) if ann_views else view_candidates,
            )
            if not reference_mesh:
                status = "missing_render_mesh"
                msg = f"no view_XX/mesh.npz found under render root; candidates={view_candidates}"
                progress.update(obj_id, _entry(status, error=msg), _line(obj_id, status, error=msg))
                fail += 1
                continue

            if os.path.exists(local_pickle):
                os.remove(local_pickle)
            cmd = build_blender_cmd(args, object_path, local_pickle, reference_mesh)
            print(f"  [blender] launching {cmd[0]} ref_view={reference_view}")
            rc, stdout_tail, stderr_tail = run_blender(cmd, args.blender_timeout_s)
            elapsed = time.time() - t0

            if rc != 0 or not os.path.exists(local_pickle):
                err_path = write_error_log(args.state_dir, obj_id, rc, stdout_tail, stderr_tail)
                s3_cp_file(err_path, f"{args.s3_output_root.rstrip('/')}/logs/errors/{obj_id}.log", retries=1)
                status = "timeout" if rc == -1 else "blender_failed"
                text = f"{stdout_tail}\n{stderr_tail}"
                if "reference vertices mismatch" in text:
                    status = "vertex_set_mismatch"
                elif "reference mesh.npz missing required frame_indices" in text:
                    status = "missing_frame_indices"
                msg = f"rc={rc}"
                progress.update(obj_id, _entry(status, error=msg, elapsed_s=round(elapsed, 2)),
                                _line(obj_id, status, error=msg, elapsed_s=round(elapsed, 2)))
                fail += 1
                continue

            if not s3_cp_file(local_pickle, s3_out, retries=2):
                status = "upload_failed"
                progress.update(obj_id, _entry(status, elapsed_s=round(elapsed, 2)),
                                _line(obj_id, status, elapsed_s=round(elapsed, 2)))
                fail += 1
                continue

            status = "success"
            progress.update(
                obj_id,
                _entry(status, s3_uri=s3_out, reference_view=reference_view, elapsed_s=round(elapsed, 2)),
                _line(obj_id, status, reference_view=reference_view, elapsed_s=round(elapsed, 2)),
            )
            success += 1
        except Exception as exc:
            status = "error"
            progress.update(obj_id, _entry(status, error=f"{type(exc).__name__}: {exc}"),
                            _line(obj_id, status, error=f"{type(exc).__name__}: {exc}"))
            fail += 1
        finally:
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
            if os.path.exists(local_pickle):
                try:
                    os.remove(local_pickle)
                except Exception:
                    pass

        if (idx + 1) % 10 == 0:
            print(f"[batch_pbr_texverse] progress success={success} skip={skip} fail={fail}")

    progress.flush()
    print(f"\n[batch_pbr_texverse] DONE success={success} skip={skip} fail={fail} total={len(manifest)}")


if __name__ == "__main__":
    main()
