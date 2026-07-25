#!/usr/bin/env python3
"""
Voxelize ActionBench 1-view dynamic meshes into per-view dual-grid `.vxz` tars.

Input layout:
  <input_root>/<uid>/mesh.npz
  <input_root>/<uid>/result.json

Output layout:
  <output_root>/<res>/<uid>/view_00.tar
  <output_root>/<res>/<uid>/view_00_meta.json
  <output_root>/logs_<res>/voxel_progress_<rank>.json
  <output_root>/logs_<res>/voxel_status_<rank>.log
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

import o_voxel


TERMINAL_SKIP_STATUSES = {
    "success",
    "skipped_too_many_faces",
    "invalid_mesh_nonfinite",
    "missing_mesh",
    "missing_camera",
    "dual_grid_error",
    "worker_error",
}


def _is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _join_uri(root: str, *parts: str) -> str:
    if _is_s3_uri(root):
        return root.rstrip("/") + "/" + "/".join(str(p).strip("/") for p in parts)
    return os.path.join(root, *map(str, parts))


def run_aws(args, retries: int = 2, sleep_s: float = 2.0) -> subprocess.CompletedProcess:
    last = None
    for attempt in range(retries + 1):
        proc = subprocess.run(["aws"] + args, capture_output=True, text=True)
        if proc.returncode == 0:
            return proc
        last = proc
        if attempt < retries:
            time.sleep(sleep_s * (attempt + 1))
    return last


def s3_get_file(s3_uri: str, local_path: str, retries: int = 2) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    proc = run_aws(["s3", "cp", "--only-show-errors", s3_uri, local_path], retries=retries)
    if proc.returncode != 0:
        print(f"[s3] cp FAILED {s3_uri} -> {local_path}: {proc.stderr.strip()[:300]}")
        return False
    return True


def s3_cp_file(local_path: str, s3_uri: str, retries: int = 2) -> bool:
    proc = run_aws(["s3", "cp", "--only-show-errors", local_path, s3_uri], retries=retries)
    if proc.returncode != 0:
        print(f"[s3] cp FAILED {local_path} -> {s3_uri}: {proc.stderr.strip()[:300]}")
        return False
    return True


def s3_exists_file(s3_uri: str) -> bool:
    proc = run_aws(["s3", "ls", s3_uri], retries=1)
    return proc.returncode == 0 and s3_uri.rsplit("/", 1)[-1] in proc.stdout


def copy_uri_to_local(src: str, dst: str) -> bool:
    if _is_s3_uri(src):
        return s3_get_file(src, dst, retries=2)
    if not os.path.isfile(src):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def copy_local_to_uri(src: str, dst: str) -> bool:
    if _is_s3_uri(dst):
        return s3_cp_file(src, dst, retries=2)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = f"{dst}.tmp.{os.getpid()}"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return True


def uri_exists(path: str) -> bool:
    if _is_s3_uri(path):
        return s3_exists_file(path)
    return os.path.isfile(path)


def log_suffix_for_resolutions(resolutions: list[int]) -> str:
    return "_" + "_".join(str(x) for x in resolutions)


def pick_frame_sel(num_frames: int, max_frames: int, mode: str) -> list[int]:
    """Return 0-based source frame indices. max_frames <= 0 means all frames."""
    if max_frames <= 0 or mode == "all" or num_frames <= max_frames:
        return list(range(num_frames))
    if mode == "uniform":
        return [int(round(x)) for x in np.linspace(0, num_frames - 1, max_frames)]
    if mode == "center":
        start = (num_frames - max_frames) // 2
        return list(range(start, start + max_frames))
    if mode == "head":
        return list(range(max_frames))
    raise ValueError(f"unknown frame_sampling mode: {mode}")


def parse_view_idx(view_id) -> int:
    if isinstance(view_id, int):
        return int(view_id)
    if isinstance(view_id, str) and view_id.startswith("view_"):
        return int(view_id[len("view_"):])
    if isinstance(view_id, str):
        return int(view_id)
    return 0


def load_tasks(ann_file: str, split: str) -> list[tuple[str, int, dict]]:
    local_ann = ann_file
    tmp_ann = None
    if _is_s3_uri(ann_file):
        fd, tmp_ann = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        if not s3_get_file(ann_file, tmp_ann, retries=3):
            raise RuntimeError(f"failed to download ann_file: {ann_file}")
        local_ann = tmp_ann
    try:
        with open(local_ann) as f:
            raw = json.load(f)
    finally:
        if tmp_ann:
            try:
                os.unlink(tmp_ann)
            except OSError:
                pass

    if isinstance(raw, dict):
        if split:
            entries = raw[split]
        elif "test" in raw:
            entries = raw["test"]
        else:
            entries = next(v for v in raw.values() if isinstance(v, list))
    elif isinstance(raw, list):
        entries = raw
    else:
        raise RuntimeError(f"unsupported ann format: {type(raw).__name__}")

    tasks = []
    bad = 0
    for entry in entries:
        if not isinstance(entry, dict):
            bad += 1
            continue
        uid = entry.get("uid") or entry.get("obj_id") or entry.get("sha256")
        if not uid:
            bad += 1
            continue
        view_idx = parse_view_idx(entry.get("view_id", "view_00"))
        tasks.append((str(uid), int(view_idx), entry))
    if bad:
        print(f"[manifest] skipped malformed entries: {bad}")
    return tasks


def _entry(status: str, **extra) -> dict:
    out = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    for key, value in extra.items():
        if value is not None:
            out[key] = value
    return out


def _line(uid: str, view_idx: int, status: str, **extra) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    parts = [ts, f"{uid}/view_{int(view_idx):02d}", status]
    for key, value in extra.items():
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


class ProgressStore:
    def __init__(self, output_root: str, rank: int, local_state_dir: str,
                 log_suffix: str, push_every_updates: int = 4,
                 push_min_interval_s: float = 30.0):
        os.makedirs(local_state_dir, exist_ok=True)
        self.rank = int(rank)
        self.logs_root = _join_uri(output_root, f"logs{log_suffix}")
        if not _is_s3_uri(self.logs_root):
            os.makedirs(self.logs_root, exist_ok=True)
        self.remote_progress = _join_uri(self.logs_root, f"voxel_progress_{rank}.json")
        self.remote_status = _join_uri(self.logs_root, f"voxel_status_{rank}.log")
        self.local_progress = os.path.join(local_state_dir, f"voxel_progress_{rank}{log_suffix}.json")
        self.local_status = os.path.join(local_state_dir, f"voxel_status_{rank}{log_suffix}.log")
        self.progress = {}
        self.push_every_updates = max(1, int(push_every_updates))
        self.push_min_interval_s = float(push_min_interval_s)
        self._updates_since_push = 0
        self._last_push_ts = 0.0

    @staticmethod
    def view_key(uid: str, view_idx: int) -> str:
        return f"{uid}/view_{int(view_idx):02d}"

    def load(self):
        if _is_s3_uri(self.remote_progress):
            ok = s3_get_file(self.remote_progress, self.local_progress, retries=1)
            if ok:
                try:
                    with open(self.local_progress) as f:
                        self.progress = json.load(f)
                    print(f"[progress] loaded {len(self.progress)} entries from {self.remote_progress}")
                except Exception as exc:
                    print(f"[progress] parse failed ({exc}), starting empty")
                    self.progress = {}
            else:
                print(f"[progress] no progress at {self.remote_progress}, starting empty")
                self.progress = {}
            if not s3_get_file(self.remote_status, self.local_status, retries=1):
                open(self.local_status, "w").close()
        elif os.path.isfile(self.remote_progress):
            try:
                shutil.copyfile(self.remote_progress, self.local_progress)
                with open(self.local_progress) as f:
                    self.progress = json.load(f)
                print(f"[progress] loaded {len(self.progress)} entries from {self.remote_progress}")
            except Exception as exc:
                print(f"[progress] parse failed ({exc}), starting empty")
                self.progress = {}
        else:
            print(f"[progress] no progress at {self.remote_progress}, starting empty")
            self.progress = {}
        if os.path.isfile(self.remote_status):
            shutil.copyfile(self.remote_status, self.local_status)
        else:
            open(self.local_status, "w").close()

    def is_terminal(self, uid: str, view_idx: int) -> bool:
        status = self.progress.get(self.view_key(uid, view_idx), {}).get("status", "")
        return status in TERMINAL_SKIP_STATUSES

    def _save_progress_local(self):
        tmp = self.local_progress + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.progress, f)
        os.replace(tmp, self.local_progress)

    def _append_status_local(self, line: str):
        with open(self.local_status, "a") as f:
            f.write(line.rstrip("\n") + "\n")

    def update(self, uid: str, view_idx: int, entry: dict, status_line: str,
               force_push: bool = False):
        self.progress[self.view_key(uid, view_idx)] = entry
        self._save_progress_local()
        self._append_status_local(status_line)
        self._updates_since_push += 1
        now = time.time()
        if (force_push or self._updates_since_push >= self.push_every_updates
                or now - self._last_push_ts >= self.push_min_interval_s):
            self.flush()

    def flush(self):
        if _is_s3_uri(self.logs_root):
            s3_cp_file(self.local_progress, self.remote_progress, retries=1)
            s3_cp_file(self.local_status, self.remote_status, retries=1)
        else:
            os.makedirs(self.logs_root, exist_ok=True)
            shutil.copyfile(self.local_progress, self.remote_progress)
            shutil.copyfile(self.local_status, self.remote_status)
        self._updates_since_push = 0
        self._last_push_ts = time.time()


def write_error_log(output_root: str, log_suffix: str, rank: int, uid: str,
                    view_idx: int, header: str, body: str):
    log_dir = _join_uri(output_root, f"_logs{log_suffix}", f"rank_{rank}")
    local_dir = tempfile.mkdtemp(prefix="actionbench_voxel_error_")
    path = os.path.join(local_dir, f"{uid}_view_{view_idx:02d}.txt")
    with open(path, "w") as f:
        f.write(f"uid: {uid}\n")
        f.write(f"view_idx: {view_idx}\n")
        f.write(f"rank: {rank}\n")
        f.write(f"timestamp: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"header: {header}\n\n")
        f.write(body)
    if _is_s3_uri(log_dir):
        s3_cp_file(path, _join_uri(log_dir, os.path.basename(path)), retries=1)
        shutil.rmtree(local_dir, ignore_errors=True)
    else:
        os.makedirs(log_dir, exist_ok=True)
        shutil.copyfile(path, os.path.join(log_dir, os.path.basename(path)))
        shutil.rmtree(local_dir, ignore_errors=True)


def output_exists(output_root: str, resolutions: list[int], uid: str, view_idx: int) -> bool:
    for res in resolutions:
        base = _join_uri(output_root, str(res), uid, f"view_{view_idx:02d}")
        if not (uri_exists(f"{base}.tar") and uri_exists(f"{base}_meta.json")):
            return False
    return True


def copy_inputs_to_tmp(input_root: str, uid: str, view_dir: str) -> tuple[str | None, str | None]:
    src_mesh = _join_uri(input_root, uid, "mesh.npz")
    src_json = _join_uri(input_root, uid, "result.json")
    local_mesh = os.path.join(view_dir, "mesh.npz")
    local_json = os.path.join(view_dir, "result.json")
    if not copy_uri_to_local(src_mesh, local_mesh):
        return None, None
    if not copy_uri_to_local(src_json, local_json):
        return src_mesh, None
    return local_mesh, local_json


def voxelize_one_view(uid: str, view_idx: int, input_root: str, output_root: str,
                      resolutions: list[int], tmp_dir: str, max_face_count: int,
                      max_frames: int, frame_sampling: str) -> dict:
    if output_exists(output_root, resolutions, uid, view_idx):
        return {"status": "success", "num_frames": 0, "skip_reason": "all_outputs_exist"}

    view_local_dir = os.path.join(tmp_dir, f"{uid}_view{view_idx:02d}")
    os.makedirs(view_local_dir, exist_ok=True)
    timings = {
        "t_get": 0.0,
        "t_load_mesh": 0.0,
        "t_compute": 0.0,
        "t_write_vxz": 0.0,
        "t_tar": 0.0,
        "t_publish": 0.0,
    }

    try:
        t0 = time.time()
        local_mesh, local_json = copy_inputs_to_tmp(input_root, uid, view_local_dir)
        timings["t_get"] = time.time() - t0
        if local_mesh is None:
            return {"status": "missing_mesh"}
        if local_json is None:
            return {"status": "missing_camera"}

        with open(local_json) as f:
            meta = json.load(f)
        cam_info = meta.get("camera_info")
        if not cam_info or "camera_c2w" not in cam_info:
            return {"status": "missing_camera"}
        c2w = np.array(cam_info["camera_c2w"], dtype=np.float32)
        w2c_rot = np.linalg.inv(c2w)[:3, :3]

        t_lm = time.time()
        with np.load(local_mesh) as data:
            vertices_seq = data["vertices"].copy()
            faces = data["faces"].copy()
            mesh_frame_indices = data["frame_indices"].copy() if "frame_indices" in data else None
        timings["t_load_mesh"] = time.time() - t_lm

        num_frames_orig = int(vertices_seq.shape[0])
        num_faces = int(faces.shape[0])
        nonfinite_vertices = int(vertices_seq.size - np.isfinite(vertices_seq).sum())
        nonfinite_faces = int(faces.size - np.isfinite(faces).sum())
        if nonfinite_vertices or nonfinite_faces:
            return {
                "status": "invalid_mesh_nonfinite",
                "num_faces": num_faces,
                "num_frames_orig": num_frames_orig,
                "vertices_shape": list(vertices_seq.shape),
                "faces_shape": list(faces.shape),
                "nonfinite_vertices": nonfinite_vertices,
                "nonfinite_faces": nonfinite_faces,
            }
        if num_faces > max_face_count:
            return {
                "status": "skipped_too_many_faces",
                "num_faces": num_faces,
                "num_frames_orig": num_frames_orig,
            }

        frame_sel = pick_frame_sel(num_frames_orig, max_frames, frame_sampling)
        vertices_seq = vertices_seq[frame_sel]
        if mesh_frame_indices is not None:
            selected_mesh_frame_indices = [int(x) for x in mesh_frame_indices[frame_sel]]
        else:
            selected_mesh_frame_indices = None
        num_frames = int(vertices_seq.shape[0])
        faces_t = torch.from_numpy(faces).long()

        for res in resolutions:
            local_view_res_dir = os.path.join(view_local_dir, f"res_{res}")
            os.makedirs(local_view_res_dir, exist_ok=True)
            frame_files = []
            for out_idx in range(num_frames):
                local_vxz = os.path.join(local_view_res_dir, f"{out_idx:06d}.vxz")
                try:
                    vertices = vertices_seq[out_idx].astype(np.float32) @ w2c_rot.T
                    vertices = np.clip(vertices, -0.5, 0.5)
                    verts_t = torch.from_numpy(vertices)

                    tc = time.time()
                    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
                        vertices=verts_t,
                        faces=faces_t,
                        grid_size=res,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        face_weight=1.0,
                        boundary_weight=0.2,
                        regularization_weight=1e-2,
                        timing=False,
                    )
                    dual_vertices = dual_vertices * res - voxel_indices
                    if not (torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1 + 1e-3)):
                        raise RuntimeError("dual_vertices out of range")
                    dual_vertices = torch.clamp(dual_vertices, 0, 1)
                    dual_vertices = (dual_vertices * 255).type(torch.uint8)
                    intersected = (
                        intersected[:, 0:1]
                        + 2 * intersected[:, 1:2]
                        + 4 * intersected[:, 2:3]
                    ).type(torch.uint8)
                    timings["t_compute"] += time.time() - tc

                    tw = time.time()
                    o_voxel.io.write_vxz(
                        local_vxz,
                        voxel_indices,
                        {"vertices": dual_vertices, "intersected": intersected},
                        compression="zstd",
                        compression_level=5,
                    )
                    timings["t_write_vxz"] += time.time() - tw
                    frame_files.append((out_idx, local_vxz))
                except Exception:
                    return {
                        "status": "dual_grid_error",
                        "num_frames": num_frames,
                        "num_frames_orig": num_frames_orig,
                        "num_faces": num_faces,
                        "res": res,
                        "out_idx": out_idx,
                        "src_frame": int(frame_sel[out_idx]),
                        "traceback": traceback.format_exc(),
                    }
                finally:
                    try:
                        del verts_t, voxel_indices, dual_vertices, intersected
                    except UnboundLocalError:
                        pass

            local_tar = os.path.join(view_local_dir, f"view_res{res}.tar")
            tt = time.time()
            with tarfile.open(local_tar, "w") as tar:
                for out_idx, fpath in sorted(frame_files):
                    tar.add(fpath, arcname=f"{out_idx:06d}.vxz")
            timings["t_tar"] += time.time() - tt

            local_meta = os.path.join(view_local_dir, f"view_res{res}_meta.json")
            with open(local_meta, "w") as f:
                json.dump({
                    "uid": uid,
                    "view_idx": int(view_idx),
                    "view_id": f"view_{view_idx:02d}",
                    "resolution": int(res),
                    "num_frames_orig": num_frames_orig,
                    "num_frames": num_frames,
                    "frame_sampling": "all" if max_frames <= 0 else frame_sampling,
                    "frame_sel": [int(x) for x in frame_sel],
                    "mesh_frame_indices": selected_mesh_frame_indices,
                }, f)

            final_base = _join_uri(output_root, str(res), uid, f"view_{view_idx:02d}")
            tp = time.time()
            if not copy_local_to_uri(local_tar, f"{final_base}.tar"):
                return {"status": "upload_failed", "num_frames": num_frames,
                        "num_frames_orig": num_frames_orig, "res": res}
            if not copy_local_to_uri(local_meta, f"{final_base}_meta.json"):
                return {"status": "upload_failed", "num_frames": num_frames,
                        "num_frames_orig": num_frames_orig, "res": res, "stage": "meta"}
            timings["t_publish"] += time.time() - tp
            tar_size_mb = os.path.getsize(local_tar) / 1024 / 1024
            print(f"[timing] {uid[:16]}/view_{view_idx:02d} res={res} "
                  f"tar_mb={tar_size_mb:.1f} t_tar={timings['t_tar']:.2f}s "
                  f"t_publish={timings['t_publish']:.2f}s")

        return {
            "status": "success",
            "num_frames": num_frames,
            "num_frames_orig": num_frames_orig,
            "num_faces": num_faces,
            "frame_sampling": "all" if max_frames <= 0 else frame_sampling,
            "resolutions": resolutions,
            "t_get": round(timings["t_get"], 2),
            "t_load_mesh": round(timings["t_load_mesh"], 2),
            "t_compute": round(timings["t_compute"], 2),
            "t_write_vxz": round(timings["t_write_vxz"], 2),
            "t_tar": round(timings["t_tar"], 2),
            "t_publish": round(timings["t_publish"], 2),
        }
    finally:
        shutil.rmtree(view_local_dir, ignore_errors=True)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", required=True)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--resolution", default="1024",
                        help="Comma-separated resolutions, e.g. 512,1024")
    parser.add_argument("--state_dir", default="/tmp/actionbench_dual_grid_state")
    parser.add_argument("--tmp_dir", default="/tmp/actionbench_tmp_dual_grid")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_face_count", type=int, default=500_000)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="0 or negative means all frames")
    parser.add_argument("--frame_sampling", default="all",
                        choices=["all", "center", "uniform", "head"])
    parser.add_argument("--max_items", type=int, default=None)
    args = parser.parse_args()

    resolutions = [int(x) for x in args.resolution.split(",")]
    log_suffix = log_suffix_for_resolutions(resolutions)
    args.state_dir = os.path.join(args.state_dir, f"rank_{args.rank}")
    args.tmp_dir = os.path.join(args.tmp_dir, f"rank_{args.rank}")
    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    all_tasks = load_tasks(args.ann_file, args.split)
    all_tasks.sort(key=lambda x: (x[0], x[1]))
    start = len(all_tasks) * args.rank // args.world_size
    end = len(all_tasks) * (args.rank + 1) // args.world_size
    my_tasks = all_tasks[start:end]

    print("================================================================")
    print(f"[main] rank={args.rank}/{args.world_size}")
    print(f"[main] ann_file={args.ann_file}")
    print(f"[main] input_root={args.input_root}")
    print(f"[main] output_root={args.output_root}")
    print(f"[main] resolutions={resolutions}")
    print(f"[main] max_frames={args.max_frames} frame_sampling={args.frame_sampling}")
    print(f"[main] global_tasks={len(all_tasks)} rank_tasks={len(my_tasks)} idx={start}:{end}")
    print(f"[main] state={args.state_dir}")
    print(f"[main] tmp={args.tmp_dir}")
    print("================================================================")

    progress = ProgressStore(args.output_root, args.rank, args.state_dir, log_suffix=log_suffix)
    progress.load()
    to_process = [(uid, view_idx) for uid, view_idx, _ in my_tasks
                  if not progress.is_terminal(uid, view_idx)]
    skipped = len(my_tasks) - len(to_process)
    if args.max_items is not None:
        to_process = to_process[:args.max_items]
        print(f"[main] limited to {len(to_process)} (--max_items)")
    print(f"[main] terminal in progress={skipped}; to_process={len(to_process)}")

    n_success = 0
    n_skip = skipped
    n_fail = 0
    t_start = time.time()
    for i, (uid, view_idx) in enumerate(to_process):
        t0 = time.time()
        try:
            result = voxelize_one_view(
                uid=uid,
                view_idx=view_idx,
                input_root=args.input_root,
                output_root=args.output_root,
                resolutions=resolutions,
                tmp_dir=args.tmp_dir,
                max_face_count=args.max_face_count,
                max_frames=args.max_frames,
                frame_sampling=args.frame_sampling,
            )
        except Exception as exc:
            result = {"status": "worker_error", "error": f"{type(exc).__name__}: {exc}"}
            write_error_log(args.output_root, log_suffix, args.rank, uid, view_idx,
                            f"worker_error: {result['error']}", traceback.format_exc())

        status = result.get("status", "unknown")
        if status == "success":
            n_success += 1
        elif status in TERMINAL_SKIP_STATUSES:
            n_skip += 1
        else:
            n_fail += 1

        tb = result.pop("traceback", None) if isinstance(result, dict) else None
        if tb:
            write_error_log(args.output_root, log_suffix, args.rank, uid, view_idx,
                            f"status={status}", tb)

        elapsed = time.time() - t0
        line = _line(uid, view_idx, status,
                     frames=result.get("num_frames"),
                     faces=result.get("num_faces"),
                     dt=round(elapsed, 1))
        progress.update(uid, view_idx,
                        _entry(status, **{k: v for k, v in result.items() if k != "status"}),
                        line)

        done = i + 1
        overall = time.time() - t_start
        rate = done / overall if overall > 0 else 0
        eta = (len(to_process) - done) / rate if rate > 0 else float("inf")
        print(f"[main] [{done}/{len(to_process)}] {uid}/view_{view_idx:02d} "
              f"{status} dt={elapsed:.1f}s | ok={n_success} skip={n_skip} fail={n_fail} "
              f"rate={rate:.2f} view/s eta={eta/60:.1f}min")

    progress.flush()
    print(f"\n[main] DONE rank={args.rank} ok={n_success} skip={n_skip} fail={n_fail} "
          f"total_done={n_success + n_skip + n_fail} elapsed={(time.time() - t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
