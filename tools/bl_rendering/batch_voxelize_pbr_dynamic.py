#!/usr/bin/env python3
"""
Voxelize dynamic PBR pickles for ObjXL / TexVerse animation render outputs.

Inputs are the self-contained PBR pickles produced by dump_pbr_dynamic_obj.py:
    {s3_pbr_root}/{obj_id}.pickle

Each task is one ann view:
    {s3_render_root}/{obj_id}/view_XX/result.json

The rendering mesh.npz is intentionally not read. Geometry comes from the PBR
pickle's objects[0].vertices_seq and objects[0].faces.
"""

import argparse
import copy
import gc
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

import o_voxel


TERMINAL_SKIP_STATUSES = {
    "success",
    "missing_pbr",
    "missing_camera",
    "invalid_pbr",
    "missing_frame_sel",
    "frame_sel_out_of_range",
    "skipped_too_many_faces",
}
GC_EVERY_VIEWS = max(1, int(os.environ.get("VOXELIZE_PBR_GC_EVERY", "10")))


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


def load_ann_tasks(ann_file: str) -> list:
    with open(ann_file) as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("train"), list) or not isinstance(data.get("test"), list):
        raise ValueError(f"ann_file must be a dict with train/test lists: {ann_file}")

    tasks = []
    seen = {}
    for split in ("train", "test"):
        for ann in data[split]:
            obj_id = ann["obj_id"]
            view_id = ann["view_id"]
            if not isinstance(view_id, str) or not view_id.startswith("view_"):
                raise ValueError(f"Invalid view_id in {ann_file}: {view_id!r}")
            view_idx = int(view_id.split("_", 1)[1])
            key = (obj_id, view_idx)
            frame_sel = ann.get("frame_sel")
            if not isinstance(frame_sel, list) or not frame_sel:
                task = {
                    "obj_id": obj_id,
                    "view_idx": view_idx,
                    "frame_sel": None,
                    "ann_num_frames": ann.get("num_frames"),
                    "ann_num_render_frames": ann.get("num_render_frames"),
                    "error": "missing_frame_sel",
                }
            else:
                task = {
                    "obj_id": obj_id,
                    "view_idx": view_idx,
                    "frame_sel": [int(x) for x in frame_sel],
                    "ann_num_frames": ann.get("num_frames"),
                    "ann_num_render_frames": ann.get("num_render_frames"),
                    "error": None,
                }
            if key in seen:
                old = seen[key]
                if old.get("frame_sel") != task.get("frame_sel"):
                    raise ValueError(f"Duplicate ann task has inconsistent frame_sel: {obj_id}/view_{view_idx:02d}")
                continue
            seen[key] = task
            tasks.append(task)
    return tasks


class ProgressStore:
    def __init__(self, s3_output_root: str, rank: int, local_state_dir: str):
        os.makedirs(local_state_dir, exist_ok=True)
        self.s3_logs_root = s3_output_root.rstrip("/") + "/logs"
        self.s3_progress_uri = f"{self.s3_logs_root}/progress_{rank}.json"
        self.s3_status_uri = f"{self.s3_logs_root}/status_{rank}.log"
        self.local_progress = os.path.join(local_state_dir, f"progress_{rank}.json")
        self.local_status = os.path.join(local_state_dir, f"status_{rank}.log")
        self.progress = {}

    @staticmethod
    def view_key(obj_id: str, view_idx: int) -> str:
        return f"{obj_id}/view_{int(view_idx):02d}"

    def load(self):
        if s3_get_file_to_local(self.s3_progress_uri, self.local_progress, retries=1):
            try:
                with open(self.local_progress) as f:
                    self.progress = json.load(f)
                print(f"[progress] Loaded {len(self.progress)} entries from {self.s3_progress_uri}")
            except Exception as exc:
                print(f"[progress] Failed to parse {self.local_progress}: {exc}. Starting empty.")
                self.progress = {}
        else:
            print(f"[progress] No remote progress at {self.s3_progress_uri}; starting empty.")
            self.progress = {}

        if not s3_get_file_to_local(self.s3_status_uri, self.local_status, retries=1):
            open(self.local_status, "w").close()
        if not os.path.exists(self.local_progress):
            with open(self.local_progress, "w") as f:
                json.dump(self.progress, f)

    def is_terminal(self, obj_id: str, view_idx: int) -> bool:
        status = self.progress.get(self.view_key(obj_id, view_idx), {}).get("status")
        return status in TERMINAL_SKIP_STATUSES

    def update(self, obj_id: str, view_idx: int, entry: dict, status_line: str, push: bool = True):
        self.progress[self.view_key(obj_id, view_idx)] = entry
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


def _line(obj_id: str, view_idx: int, status: str, **extra) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    parts = [ts, f"{obj_id}/view_{int(view_idx):02d}", status]
    for key, value in extra.items():
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def without_status(data: dict) -> dict:
    return {k: v for k, v in data.items() if k != "status"}


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
    fn = fn / norms
    return np.stack([fn, fn, fn], axis=1).astype(np.float32)


def build_pbr_dump(pbr_shared: dict, frame_verts: np.ndarray, pbr_faces: np.ndarray) -> dict:
    dump = copy.deepcopy(pbr_shared)
    for mat in dump["materials"]:
        if mat.get("alphaTexture") is not None and mat.get("alphaMode") == "OPAQUE":
            mat["alphaMode"] = "BLEND"
    dump["materials"].append({
        "baseColorFactor": [0.8, 0.8, 0.8],
        "alphaFactor": 1.0,
        "metallicFactor": 0.0,
        "roughnessFactor": 0.5,
        "alphaMode": "OPAQUE",
        "alphaCutoff": 0.5,
        "baseColorTexture": None,
        "alphaTexture": None,
        "metallicTexture": None,
        "roughnessTexture": None,
    })
    obj_data = dump["objects"][0]
    obj_data["vertices"] = frame_verts
    obj_data["normals"] = compute_face_normals(frame_verts, pbr_faces)
    obj_data["mat_ids"] = obj_data["mat_ids"].copy()
    obj_data["mat_ids"][obj_data["mat_ids"] == -1] = len(dump["materials"]) - 1
    return dump


def load_view_w2c_rotation(result_json_path: str) -> np.ndarray:
    with open(result_json_path) as f:
        result = json.load(f)
    camera_info = result.get("camera_info")
    if not isinstance(camera_info, dict) or "camera_c2w" not in camera_info:
        raise ValueError("result.json missing camera_info.camera_c2w")
    c2w = np.asarray(camera_info["camera_c2w"], dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"camera_c2w must be shape (4,4), got {c2w.shape}")
    return np.linalg.inv(c2w)[:3, :3].astype(np.float32)


def load_pbr_pickle(path: str) -> tuple:
    with open(path, "rb") as f:
        pbr_shared = pickle.load(f)
    objects = pbr_shared.get("objects") or []
    if not objects:
        raise ValueError("PBR pickle missing objects")
    obj = objects[0]
    missing = [k for k in ("vertices_seq", "faces", "uvs", "mat_ids") if obj.get(k) is None]
    if missing:
        raise ValueError(f"PBR pickle missing required object fields: {missing}")
    vertices_seq = np.asarray(obj["vertices_seq"])
    faces = np.asarray(obj["faces"])
    if vertices_seq.ndim != 3 or vertices_seq.shape[-1] != 3:
        raise ValueError(f"vertices_seq must be (T,V,3), got {vertices_seq.shape}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"faces must be (F,3), got {faces.shape}")
    return pbr_shared, vertices_seq, faces.astype(np.int32, copy=False)


def voxelize_one_view(
    obj_id: str,
    view_idx: int,
    frame_sel: list,
    source_ann_path: str,
    ann_num_frames,
    ann_num_render_frames,
    local_pbr_path: str,
    local_result_json: str,
    local_output_dir: str,
    local_tmp_dir: str,
    resolution: int,
    debug: bool,
    vxz_compression: str,
    vxz_compression_level: int,
) -> tuple:
    t_read_start = time.time()
    pbr_shared, vertices_seq, pbr_faces = load_pbr_pickle(local_pbr_path)
    w2c_rot = load_view_w2c_rotation(local_result_json)
    t_read = time.time() - t_read_start
    num_frames_pbr = int(vertices_seq.shape[0])

    if not frame_sel:
        return "", "", {
            "status": "missing_frame_sel",
            "num_frames": 0,
            "num_frames_pbr": num_frames_pbr,
        }
    if min(frame_sel) < 0 or max(frame_sel) >= num_frames_pbr:
        return "", "", {
            "status": "frame_sel_out_of_range",
            "num_frames": 0,
            "num_frames_pbr": num_frames_pbr,
            "frame_sel_min": int(min(frame_sel)),
            "frame_sel_max": int(max(frame_sel)),
        }

    num_faces = int(pbr_faces.shape[0])
    if num_faces > 500000:
        return "", "", {
            "status": "skipped_too_many_faces",
            "num_frames": 0,
            "num_frames_pbr": num_frames_pbr,
            "num_faces": num_faces,
        }

    frame_sel_to_use = list(frame_sel)
    if debug:
        frame_sel_to_use = frame_sel_to_use[:1]
    num_frames = len(frame_sel_to_use)

    os.makedirs(local_output_dir, exist_ok=True)
    local_view_dir = os.path.join(local_tmp_dir, f"{obj_id}_view_{view_idx:02d}_{resolution}")
    shutil.rmtree(local_view_dir, ignore_errors=True)
    os.makedirs(local_view_dir, exist_ok=True)
    local_tar_path = os.path.join(local_output_dir, f"view_{view_idx:02d}.tar")
    local_meta_path = os.path.join(local_output_dir, f"view_{view_idx:02d}_meta.json")

    frame_files = []
    t_compute = 0.0
    t_write = 0.0
    try:
        for out_idx, src_idx in enumerate(frame_sel_to_use):
            frame_verts = vertices_seq[src_idx].astype(np.float32)
            frame_verts = frame_verts @ w2c_rot.T
            frame_verts = np.clip(frame_verts, -0.5, 0.5)

            local_vxz_path = os.path.join(local_view_dir, f"{out_idx:06d}.vxz")
            dump = None
            coord = None
            attr = None
            try:
                t0 = time.time()
                dump = build_pbr_dump(pbr_shared, frame_verts, pbr_faces)
                coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
                    dump,
                    grid_size=resolution,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    mip_level_offset=0,
                    verbose=False,
                    timing=False,
                )
                attr.pop("normal", None)
                attr.pop("emissive", None)
                t_compute += time.time() - t0

                t0 = time.time()
                write_kwargs = {"compression": vxz_compression}
                if vxz_compression != "none":
                    write_kwargs["compression_level"] = vxz_compression_level
                o_voxel.io.write_vxz(local_vxz_path, coord, attr, **write_kwargs)
                t_write += time.time() - t0
                frame_files.append((out_idx, local_vxz_path))
            finally:
                del dump, coord, attr

        t0 = time.time()
        with tarfile.open(local_tar_path, "w") as tar:
            for frame_idx, frame_path in frame_files:
                tar.add(frame_path, arcname=f"{frame_idx:06d}.vxz")
        t_write += time.time() - t0
        with open(local_meta_path, "w") as f:
            json.dump({
                "obj_id": obj_id,
                "view_idx": int(view_idx),
                "view_id": f"view_{int(view_idx):02d}",
                "resolution": int(resolution),
                "num_frames": int(num_frames),
                "num_frames_pbr": int(num_frames_pbr),
                "ann_num_frames": ann_num_frames,
                "ann_num_render_frames": ann_num_render_frames,
                "frame_sel": [int(x) for x in frame_sel_to_use],
                "source_ann_path": source_ann_path,
                "source": "latent_anns",
            }, f)
    finally:
        shutil.rmtree(local_view_dir, ignore_errors=True)
        del pbr_shared, vertices_seq, pbr_faces

    return local_tar_path, local_meta_path, {
        "status": "success",
        "num_frames": num_frames,
        "num_frames_pbr": num_frames_pbr,
        "num_faces": num_faces,
        "frame_sel": [int(x) for x in frame_sel_to_use],
        "t_read": round(t_read, 2),
        "t_compute": round(t_compute, 2),
        "t_write": round(t_write, 2),
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--s3_render_root", type=str, required=True)
    parser.add_argument("--s3_pbr_root", type=str, required=True)
    parser.add_argument("--s3_output_root", type=str, required=True)
    parser.add_argument("--local_cache_root", type=str, default="/local-ssd/voxelize_pbr_dynamic_cache")
    parser.add_argument("--tmp_dir", type=str, default="/local-ssd/tmp_voxelize_pbr_dynamic")
    parser.add_argument("--state_dir", type=str, default="/local-ssd/voxelize_pbr_dynamic_state")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vxz_compression", type=str, default="zstd", choices=["none", "deflate", "lzma", "zstd"])
    parser.add_argument("--vxz_compression_level", type=int, default=5)
    args = parser.parse_args()

    rank_tag = f"rank_{args.rank}"
    args.local_cache_root = os.path.join(args.local_cache_root, rank_tag)
    args.tmp_dir = os.path.join(args.tmp_dir, rank_tag)
    args.state_dir = os.path.join(args.state_dir, rank_tag)
    os.makedirs(args.local_cache_root, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)

    tasks = load_ann_tasks(args.ann_file)
    print(f"[voxelize_pbr] ann tasks={len(tasks)} from {args.ann_file}")
    start = len(tasks) * args.rank // args.world_size
    end = len(tasks) * (args.rank + 1) // args.world_size
    tasks = tasks[start:end]
    if args.max_items is not None:
        tasks = tasks[:args.max_items]
    if args.debug:
        tasks = tasks[:1]
    print(f"[voxelize_pbr] rank={args.rank}/{args.world_size} assigned={len(tasks)}")
    print(f"[voxelize_pbr] resolution={args.resolution} compression={args.vxz_compression}:{args.vxz_compression_level}")

    progress = ProgressStore(args.s3_output_root, args.rank, args.state_dir)
    progress.load()

    to_process = []
    for task in tasks:
        obj_id = task["obj_id"]
        view_idx = task["view_idx"]
        if progress.is_terminal(obj_id, view_idx):
            continue
        base = f"{args.s3_output_root.rstrip('/')}/{args.resolution}/{obj_id}/view_{view_idx:02d}"
        s3_tar = f"{base}.tar"
        s3_meta = f"{base}_meta.json"
        if s3_exists_file(s3_tar) and s3_exists_file(s3_meta):
            entry = _entry("success", s3_uri=s3_tar, skipped_existing=True)
            progress.update(obj_id, view_idx, entry, _line(obj_id, view_idx, "success", skipped_existing=True), push=False)
            continue
        to_process.append(task)
    progress.flush()
    print(f"[voxelize_pbr] to_process={len(to_process)} skipped={len(tasks) - len(to_process)}")

    success = 0
    fail = 0
    start_time = time.time()
    pbr_cache = {}

    for task in tqdm(to_process, desc="Voxelize dynamic PBR"):
        obj_id = task["obj_id"]
        view_idx = task["view_idx"]
        frame_sel = task.get("frame_sel")
        view_key = ProgressStore.view_key(obj_id, view_idx)
        t0 = time.time()
        try:
            if task.get("error") == "missing_frame_sel":
                status = "missing_frame_sel"
                progress.update(obj_id, view_idx, _entry(status, error="ann missing frame_sel"), _line(obj_id, view_idx, status, error="ann missing frame_sel"))
                fail += 1
                continue

            local_obj_cache = os.path.join(args.local_cache_root, obj_id)
            os.makedirs(local_obj_cache, exist_ok=True)

            local_pbr = pbr_cache.get(obj_id)
            if not local_pbr:
                local_pbr = os.path.join(local_obj_cache, f"{obj_id}.pickle")
                s3_pbr = f"{args.s3_pbr_root.rstrip('/')}/{obj_id}.pickle"
                if not os.path.isfile(local_pbr) and not s3_get_file_to_local(s3_pbr, local_pbr, retries=2):
                    status = "missing_pbr"
                    progress.update(obj_id, view_idx, _entry(status, error=f"missing {s3_pbr}"), _line(obj_id, view_idx, status, error=s3_pbr))
                    fail += 1
                    continue
                pbr_cache[obj_id] = local_pbr

            local_result = os.path.join(local_obj_cache, f"view_{view_idx:02d}_result.json")
            s3_result = f"{args.s3_render_root.rstrip('/')}/{obj_id}/view_{view_idx:02d}/result.json"
            if not s3_get_file_to_local(s3_result, local_result, retries=2):
                status = "missing_camera"
                progress.update(obj_id, view_idx, _entry(status, error=f"missing {s3_result}"), _line(obj_id, view_idx, status, error=s3_result))
                fail += 1
                continue

            local_output_dir = os.path.join(args.local_cache_root, "outputs", str(args.resolution), obj_id)
            local_tar, local_meta, result = voxelize_one_view(
                obj_id=obj_id,
                view_idx=view_idx,
                frame_sel=frame_sel,
                source_ann_path=args.ann_file,
                ann_num_frames=task.get("ann_num_frames"),
                ann_num_render_frames=task.get("ann_num_render_frames"),
                local_pbr_path=local_pbr,
                local_result_json=local_result,
                local_output_dir=local_output_dir,
                local_tmp_dir=args.tmp_dir,
                resolution=args.resolution,
                debug=args.debug,
                vxz_compression=args.vxz_compression,
                vxz_compression_level=args.vxz_compression_level,
            )
            status = result["status"]
            if status != "success":
                clean_result = without_status(result)
                progress.update(
                    obj_id,
                    view_idx,
                    _entry(status, **clean_result),
                    _line(obj_id, view_idx, status, **clean_result),
                )
                fail += 1
                continue

            base = f"{args.s3_output_root.rstrip('/')}/{args.resolution}/{obj_id}/view_{view_idx:02d}"
            s3_tar = f"{base}.tar"
            s3_meta = f"{base}_meta.json"
            if not s3_cp_file(local_tar, s3_tar, retries=2):
                status = "upload_failed"
                progress.update(obj_id, view_idx, _entry(status, error=s3_tar), _line(obj_id, view_idx, status, error=s3_tar))
                fail += 1
                continue
            if not s3_cp_file(local_meta, s3_meta, retries=2):
                status = "upload_failed"
                progress.update(obj_id, view_idx, _entry(status, error=s3_meta, stage="meta"), _line(obj_id, view_idx, status, error=s3_meta, stage="meta"))
                fail += 1
                continue

            elapsed = time.time() - t0
            result["s3_uri"] = s3_tar
            result["s3_meta_uri"] = s3_meta
            result["elapsed_s"] = round(elapsed, 2)
            progress.update(
                obj_id,
                view_idx,
                _entry("success", **without_status(result)),
                _line(obj_id, view_idx, "success", frames=result["num_frames"], elapsed_s=round(elapsed, 2)),
            )
            success += 1
        except ValueError as exc:
            text = str(exc)
            status = "invalid_pbr" if "PBR pickle" in text or "vertices_seq" in text or "faces" in text else "error"
            progress.update(obj_id, view_idx, _entry(status, error=text), _line(obj_id, view_idx, status, error=text))
            fail += 1
        except Exception as exc:
            traceback.print_exc()
            progress.update(obj_id, view_idx, _entry("error", error=f"{type(exc).__name__}: {exc}"), _line(obj_id, view_idx, "error", error=f"{type(exc).__name__}: {exc}"))
            fail += 1

        if (success + fail) % GC_EVERY_VIEWS == 0:
            gc.collect()

    progress.flush()
    elapsed = time.time() - start_time
    print(f"[voxelize_pbr] DONE success={success} fail={fail} processed={success + fail} elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()
