#!/usr/bin/env python3
"""
Voxelize dynamic_obj renders into per-view dual-grid `.vxz` tars.

For every `<sha256>/view_XX` entry listed in `render_finished_view.json`:

  1. Download `mesh.npz` + `result.json` from S3 to `/tmp` scratch.
  2. Convert every frame to a dual-grid voxel via
     `o_voxel.convert.mesh_to_flexible_dual_grid` (mirrors `dual_grid_v2.py`).
  3. Pack all frame `.vxz` files of a single view into one `view_XX.tar`.
  4. Upload the tar to `s3://.../dynamic_obj_voxel/{res}/<sha256>/view_XX.tar`.
  5. Delete the per-view scratch dir.

Per-rank state lives in S3:
  {s3_output_root}/logs/voxel_progress_{rank}.json   (mirrored to local)
  {s3_output_root}/logs/voxel_status_{rank}.log

3-layer skip:
  L1: `--finished_views` filters the global task pool to render-successful views.
  L2: `voxel_progress_{rank}.json` ("status"=="success") survives restarts.
  L3: `s3 ls` for `{res}/<sha>/view_XX.tar` -- physical proof of completion.

Usage (single rank, on the remote machine):
  python tools/voxel/dual_grid_dynamic_obj.py \
      --finished_views s3://.../objxl/render_finished_view.json \
      --s3_input_root  s3://.../objxl/dynamic_obj_rendered \
      --s3_output_root s3://.../objxl/dynamic_obj_voxel \
      --resolution 512 \
      --rank 0 --world_size 1
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
}


def pick_frame_sel(num_frames: int, max_frames: int, mode: str) -> list:
    """Return a 0-based list of frame indices (relative to the original mp4)."""
    if num_frames <= max_frames:
        return list(range(num_frames))
    if mode == "uniform":
        return [int(round(x)) for x in np.linspace(0, num_frames - 1, max_frames)]
    if mode == "center":
        s = (num_frames - max_frames) // 2
        return list(range(s, s + max_frames))
    if mode == "head":
        return list(range(max_frames))
    raise ValueError(f"unknown frame_sampling mode: {mode}")


# =====================================================================================
# S3 helpers (small subset of batch_render_dynamic_obj.py style)
# =====================================================================================


def _is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


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
    return proc.returncode == 0


def s3_cp_file(local_path: str, s3_uri: str, retries: int = 2) -> bool:
    proc = run_aws(["s3", "cp", "--only-show-errors", local_path, s3_uri], retries=retries)
    if proc.returncode != 0:
        print(f"[s3] cp FAILED {local_path} -> {s3_uri}: {proc.stderr.strip()[:300]}")
        return False
    return True


def log_suffix_for_resolutions(resolutions: list[int]) -> str:
    if resolutions == [512]:
        return ""
    return "_" + "_".join(str(x) for x in resolutions)


def upload_error_log(s3_output_root: str, rank: int, sha256: str, view_idx: int,
                     header: str, log_suffix: str = "", exc: BaseException = None,
                     extra_tb: str = None) -> None:
    """Upload a plaintext error log to {s3_output_root}/_logs{suffix}/rank_<rank>/..."""
    try:
        body = [
            f"sha256: {sha256}",
            f"view_idx: {view_idx}",
            f"rank: {rank}",
            f"timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"header: {header}",
            "",
        ]
        if exc is not None:
            body.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        elif extra_tb:
            body.append(extra_tb)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(body))
            tmp_path = f.name
        s3_uri = (f"{s3_output_root.rstrip('/')}/_logs{log_suffix}/rank_{rank}/"
                  f"{sha256}_view_{view_idx:02d}.txt")
        s3_cp_file(tmp_path, s3_uri, retries=1)
        os.unlink(tmp_path)
    except Exception as e:
        print(f"[s3] upload_error_log FAILED for {sha256}/view_{view_idx:02d}: {e}")


def s3_exists_file(s3_uri: str) -> bool:
    proc = run_aws(["s3", "ls", s3_uri], retries=1)
    if proc.returncode != 0:
        return False
    return s3_uri.rsplit("/", 1)[-1] in proc.stdout


# =====================================================================================
# Progress / status log (per-rank, mirrored to S3 with throttling)
# =====================================================================================


class ProgressStore:
    def __init__(self, s3_output_root: str, rank: int, local_state_dir: str,
                 log_suffix: str = "", push_every_updates: int = 8,
                 push_min_interval_s: float = 30.0):
        os.makedirs(local_state_dir, exist_ok=True)
        self.rank = int(rank)
        self.s3_logs_root = s3_output_root.rstrip("/") + f"/logs{log_suffix}"
        self.s3_progress_uri = f"{self.s3_logs_root}/voxel_progress_{rank}.json"
        self.s3_status_uri = f"{self.s3_logs_root}/voxel_status_{rank}.log"
        self.local_progress = os.path.join(local_state_dir,
                                           f"voxel_progress_{rank}{log_suffix}.json")
        self.local_status = os.path.join(local_state_dir,
                                         f"voxel_status_{rank}{log_suffix}.log")
        self.progress: dict = {}
        self.push_every_updates = max(1, int(push_every_updates))
        self.push_min_interval_s = float(push_min_interval_s)
        self._updates_since_push = 0
        self._last_push_ts = 0.0

    def load(self):
        if s3_get_file(self.s3_progress_uri, self.local_progress, retries=1):
            try:
                with open(self.local_progress) as f:
                    self.progress = json.load(f)
                print(f"[progress] loaded {len(self.progress)} entries from {self.s3_progress_uri}")
            except Exception as e:
                print(f"[progress] parse failed ({e}), starting empty")
                self.progress = {}
        else:
            print(f"[progress] no remote progress at {self.s3_progress_uri}, starting empty")
            self.progress = {}
        if not s3_get_file(self.s3_status_uri, self.local_status, retries=1):
            open(self.local_status, "w").close()

    @staticmethod
    def view_key(sha256: str, view_idx: int) -> str:
        return f"{sha256}/view_{int(view_idx):02d}"

    def status(self, sha256: str, view_idx: int) -> str:
        return self.progress.get(self.view_key(sha256, view_idx), {}).get("status", "")

    def is_terminal(self, sha256: str, view_idx: int) -> bool:
        return self.status(sha256, view_idx) in TERMINAL_SKIP_STATUSES

    def _save_progress_local(self):
        tmp = self.local_progress + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.progress, f)
        os.replace(tmp, self.local_progress)

    def _append_status_local(self, line: str):
        with open(self.local_status, "a") as f:
            f.write(line.rstrip("\n") + "\n")

    def update(self, sha256: str, view_idx: int, entry: dict, status_line: str,
               force_push: bool = False):
        key = self.view_key(sha256, view_idx)
        self.progress[key] = entry
        self._save_progress_local()
        self._append_status_local(status_line)
        self._updates_since_push += 1
        now = time.time()
        if (force_push or self._updates_since_push >= self.push_every_updates
                or (now - self._last_push_ts) >= self.push_min_interval_s):
            s3_cp_file(self.local_progress, self.s3_progress_uri, retries=1)
            s3_cp_file(self.local_status, self.s3_status_uri, retries=1)
            self._updates_since_push = 0
            self._last_push_ts = now

    def flush(self):
        s3_cp_file(self.local_progress, self.s3_progress_uri, retries=1)
        s3_cp_file(self.local_status, self.s3_status_uri, retries=1)
        self._updates_since_push = 0
        self._last_push_ts = time.time()


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


# =====================================================================================
# Manifest loading
# =====================================================================================


def load_finished_views(path_or_uri: str, local_cache_path: str) -> list:
    """Return list of (sha256, view_idx) tuples parsed from the finished-views JSON."""
    if _is_s3_uri(path_or_uri):
        if not s3_get_file(path_or_uri, local_cache_path, retries=3):
            raise RuntimeError(f"failed to download {path_or_uri}")
        local_path = local_cache_path
    else:
        local_path = path_or_uri
    with open(local_path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise RuntimeError(f"expected a list in {path_or_uri}, got {type(raw).__name__}")
    out = []
    bad = 0
    for key in raw:
        try:
            sha, view = key.split("/")
            assert view.startswith("view_")
            out.append((sha, int(view[len("view_"):])))
        except Exception:
            bad += 1
    if bad:
        print(f"[manifest] {bad} malformed entries skipped (expected 'sha/view_XX')")
    return out


def summarize_finished_views_from_logs(logs_prefix: str, local_cache_dir: str,
                                       filename_prefix: str = "progress_",
                                       statuses: tuple = ("success",),
                                       missing_ok: bool = False) -> list:
    """Dynamically scan progress_*.json under logs_prefix and return the same
    list shape as load_finished_views (list of (sha, view_idx) tuples).

    statuses     -- which status values count as "finished" (default: success).
                    Pass e.g. ("success", "no_vxz", "encode_error") to include
                    terminal failures so they're skipped cross-rank too.
    missing_ok   -- if True, return [] when no matching json files exist
                    instead of raising (useful when this prefix is the encode
                    stage's own progress dir on first run).

    Mirrors `tools/voxel/build_render_finished_views.py` but inlined so the
    voxel worker can re-summarize at startup without depending on a stale
    pre-built finished_views.json sitting on S3."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not logs_prefix.endswith("/"):
        logs_prefix = logs_prefix + "/"
    os.makedirs(local_cache_dir, exist_ok=True)
    status_set = set(statuses)

    # List progress_*.json under the prefix.
    proc = run_aws(["s3", "ls", logs_prefix], retries=2)
    if proc.returncode != 0:
        if missing_ok:
            print(f"[manifest:dyn] aws s3 ls failed for {logs_prefix} (missing_ok) -> []")
            return []
        raise RuntimeError(f"aws s3 ls failed for {logs_prefix}: {proc.stderr.strip()[:300]}")
    uris = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[-1]
        if name.startswith(filename_prefix) and name.endswith(".json"):
            uris.append(logs_prefix + name)
    uris.sort()
    print(f"[manifest:dyn] found {len(uris)} {filename_prefix}*.json under {logs_prefix}")
    if not uris:
        if missing_ok:
            return []
        raise RuntimeError(f"No {filename_prefix}*.json under {logs_prefix}")

    def _dl(uri):
        local = os.path.join(local_cache_dir, uri.rsplit("/", 1)[-1])
        ok = s3_get_file(uri, local, retries=2)
        return uri, local, ok

    matched_keys = set()
    status_counter = {}
    failed_dl = 0
    failed_parse = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(_dl, u) for u in uris]):
            uri, local, ok = fut.result()
            if not ok:
                failed_dl += 1
                continue
            try:
                with open(local) as f:
                    d = json.load(f)
            except Exception:
                failed_parse += 1
                continue
            if not isinstance(d, dict):
                failed_parse += 1
                continue
            for view_key, entry in d.items():
                st = (entry or {}).get("status", "unknown") if isinstance(entry, dict) else "unknown"
                status_counter[st] = status_counter.get(st, 0) + 1
                if st in status_set:
                    matched_keys.add(view_key)

    print(f"[manifest:dyn] download ok={len(uris)-failed_dl} failed={failed_dl} "
          f"parse_failed={failed_parse}")
    print(f"[manifest:dyn] aggregate status counts (NOT deduped across ranks): {status_counter}")
    print(f"[manifest:dyn] unique matched views (statuses={sorted(status_set)}): {len(matched_keys)}")

    out, bad = [], 0
    for key in matched_keys:
        try:
            sha, view = key.split("/")
            assert view.startswith("view_")
            out.append((sha, int(view[len("view_"):])))
        except Exception:
            bad += 1
    if bad:
        print(f"[manifest:dyn] {bad} malformed keys skipped")
    return out


# =====================================================================================
# Core: voxelize one (sha256, view_idx) -> view_XX.tar per resolution
# =====================================================================================


def voxelize_one_view(sha256: str, view_idx: int, s3_input_root: str,
                       s3_output_root: str, resolutions: list, tmp_dir: str,
                       max_face_count: int, max_frames: int,
                       frame_sampling: str) -> dict:
    view_local_dir = os.path.join(tmp_dir, f"{sha256}_view{view_idx:02d}")
    s3_view_in = f"{s3_input_root.rstrip('/')}/{sha256}/view_{view_idx:02d}"

    # L3: physical skip only when BOTH tar AND meta.json exist for every res.
    pending_res = []
    for res in resolutions:
        base = f"{s3_output_root.rstrip('/')}/{res}/{sha256}/view_{view_idx:02d}"
        if not (s3_exists_file(f"{base}.tar") and s3_exists_file(f"{base}_meta.json")):
            pending_res.append(res)
    if not pending_res:
        return {"status": "success", "num_frames": 0, "skip_reason": "all_outputs_exist"}

    os.makedirs(view_local_dir, exist_ok=True)
    local_mesh = os.path.join(view_local_dir, "mesh.npz")
    local_json = os.path.join(view_local_dir, "result.json")
    timings = {
        "t_get": 0.0,
        "t_load_mesh": 0.0,
        "t_compute": 0.0,
        "t_write_vxz": 0.0,
        "t_tar": 0.0,
        "t_upload_tar": 0.0,
        "t_upload_meta": 0.0,
        "t_l3_check": 0.0,
    }
    # Approx total of just-checked L3 (we already paid this above; reflect it).
    # NOTE: deliberately not measured here to keep the change minimal.

    try:
        t0 = time.time()
        if not s3_get_file(f"{s3_view_in}/mesh.npz", local_mesh):
            return {"status": "missing_mesh"}
        if not s3_get_file(f"{s3_view_in}/result.json", local_json):
            return {"status": "missing_camera"}
        timings["t_get"] = time.time() - t0

        with open(local_json) as f:
            meta = json.load(f)
        cam_info = meta.get("camera_info")
        if not cam_info or "camera_c2w" not in cam_info:
            return {"status": "missing_camera"}
        c2w = np.array(cam_info["camera_c2w"], dtype=np.float32)
        w2c_rot = np.linalg.inv(c2w)[:3, :3]

        t_lm = time.time()
        with np.load(local_mesh) as d:
            vertices_seq = d["vertices"].copy()  # (T, N, 3) fp16
            faces = d["faces"].copy()             # (F, 3) int32
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
            return {"status": "skipped_too_many_faces",
                    "num_faces": num_faces, "num_frames_orig": num_frames_orig}

        # Subsample to <= max_frames; vxz files use contiguous 0..K-1 names
        # and frame_sel records the corresponding 0-based index into the
        # original mp4 / mesh.npz so downstream can align rgb frames itself.
        frame_sel = pick_frame_sel(num_frames_orig, max_frames, frame_sampling)
        vertices_seq = vertices_seq[frame_sel]
        num_frames = int(vertices_seq.shape[0])

        faces_t = torch.from_numpy(faces).long()
        view_status = "success"
        view_error = None

        for res in pending_res:
            local_view_res_dir = os.path.join(view_local_dir, f"res_{res}")
            os.makedirs(local_view_res_dir, exist_ok=True)
            frame_files = []
            for out_idx in range(num_frames):
                v = vertices_seq[out_idx].astype(np.float32) @ w2c_rot.T
                v = np.clip(v, -0.5, 0.5)
                verts_t = torch.from_numpy(v)
                local_vxz = os.path.join(local_view_res_dir, f"{out_idx:06d}.vxz")
                try:
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
                    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2]
                                   + 4 * intersected[:, 2:3]).type(torch.uint8)
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
                except Exception as e:
                    print(f"[ERROR] dual_grid {sha256}/view_{view_idx:02d} "
                          f"out_idx={out_idx} src_frame={frame_sel[out_idx]} res={res}: {e}")
                    view_status = "dual_grid_error"
                    view_error = (f"out_idx={out_idx} src_frame={frame_sel[out_idx]} "
                                  f"res={res}\n" + traceback.format_exc())
                finally:
                    del verts_t
                    try:
                        del voxel_indices, dual_vertices, intersected
                    except NameError:
                        pass

            if not frame_files:
                return {"status": view_status or "dual_grid_error",
                        "num_frames": num_frames, "num_frames_orig": num_frames_orig,
                        "traceback": view_error}

            # Pack tar (contiguous 0..K-1 vxz names).
            local_tar = os.path.join(view_local_dir, f"view_res{res}.tar")
            tt = time.time()
            with tarfile.open(local_tar, "w") as tar:
                for oi, fpath in sorted(frame_files):
                    tar.add(fpath, arcname=f"{oi:06d}.vxz")
            timings["t_tar"] += time.time() - tt
            tar_size_mb = os.path.getsize(local_tar) / 1024 / 1024

            # Local meta sidecar.
            local_meta = os.path.join(view_local_dir, f"view_res{res}_meta.json")
            with open(local_meta, "w") as f:
                json.dump({
                    "sha256": sha256,
                    "view_idx": int(view_idx),
                    "resolution": int(res),
                    "num_frames_orig": num_frames_orig,
                    "num_frames": num_frames,
                    "frame_sampling": frame_sampling,
                    "frame_sel": [int(x) for x in frame_sel],
                }, f)

            base = f"{s3_output_root.rstrip('/')}/{res}/{sha256}/view_{view_idx:02d}"
            # Upload tar first, meta second -> partial failure never advertises
            # a complete output (L3 checks both).
            tup = time.time()
            if not s3_cp_file(local_tar, f"{base}.tar"):
                return {"status": "upload_failed", "num_frames": num_frames,
                        "num_frames_orig": num_frames_orig, "res": res}
            timings["t_upload_tar"] += time.time() - tup
            tum = time.time()
            if not s3_cp_file(local_meta, f"{base}_meta.json"):
                return {"status": "upload_failed", "num_frames": num_frames,
                        "num_frames_orig": num_frames_orig, "res": res,
                        "stage": "meta"}
            timings["t_upload_meta"] += time.time() - tum
            print(f"[timing] {sha256[:12]}/view_{view_idx:02d} res={res} "
                  f"tar_mb={tar_size_mb:.1f} "
                  f"t_tar={timings['t_tar']:.2f}s "
                  f"t_upload_tar={timings['t_upload_tar']:.2f}s "
                  f"t_upload_meta={timings['t_upload_meta']:.2f}s")

        return {
            "status": view_status,
            "num_frames": num_frames,
            "num_frames_orig": num_frames_orig,
            "num_faces": num_faces,
            "frame_sampling": frame_sampling,
            "resolutions": pending_res,
            "t_get": round(timings["t_get"], 2),
            "t_load_mesh": round(timings["t_load_mesh"], 2),
            "t_compute": round(timings["t_compute"], 2),
            "t_write_vxz": round(timings["t_write_vxz"], 2),
            "t_tar": round(timings["t_tar"], 2),
            "t_upload_tar": round(timings["t_upload_tar"], 2),
            "t_upload_meta": round(timings["t_upload_meta"], 2),
        }
    finally:
        shutil.rmtree(view_local_dir, ignore_errors=True)


# =====================================================================================
# Main
# =====================================================================================


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--finished_views", type=str, default="",
                        help="JSON list of 'sha/view_XX' strings (S3 URI or local path). "
                             "Mutually exclusive with --render_logs_prefix.")
    parser.add_argument("--render_logs_prefix", type=str, default="",
                        help="S3 prefix containing progress_*.json from the render stage "
                             "(e.g. s3://.../rendered_v1/logs/). When set, the worker "
                             "summarizes finished views from these files at startup "
                             "instead of reading a pre-built JSON.")
    parser.add_argument("--s3_input_root", type=str, required=True,
                        help="S3 prefix containing <sha256>/view_XX/{mesh.npz,result.json}")
    parser.add_argument("--s3_output_root", type=str, required=True,
                        help="S3 prefix to upload {res}/<sha256>/view_XX.tar under")
    parser.add_argument("--resolution", type=str, default="512",
                        help="Comma-separated resolutions (e.g. 256,512)")
    parser.add_argument("--state_dir", type=str, default="/tmp/dual_grid_state")
    parser.add_argument("--tmp_dir", type=str, default="/tmp/tmp_dual_grid")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_face_count", type=int, default=500_000)
    parser.add_argument("--max_frames", type=int, default=32,
                        help="Subsample each view down to at most this many frames")
    parser.add_argument("--frame_sampling", type=str, default="center",
                        choices=["center", "uniform", "head"],
                        help="How to pick frames when num_frames_orig > max_frames")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Debug: cap number of tasks for this rank")
    args = parser.parse_args()

    if not _is_s3_uri(args.s3_input_root):
        raise SystemExit("--s3_input_root must be s3://")
    if not _is_s3_uri(args.s3_output_root):
        raise SystemExit("--s3_output_root must be s3://")

    resolutions = [int(x) for x in args.resolution.split(",")]
    log_suffix = log_suffix_for_resolutions(resolutions)
    print(f"[main] resolutions={resolutions}")
    print(f"[main] log_suffix={log_suffix or '<default>'}")

    # Per-rank scratch / state isolation.
    args.state_dir = os.path.join(args.state_dir, f"rank_{args.rank}")
    args.tmp_dir = os.path.join(args.tmp_dir, f"rank_{args.rank}")
    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    print(f"[main] rank={args.rank}/{args.world_size} state={args.state_dir} tmp={args.tmp_dir}")

    # L1: load global finished views.
    if bool(args.finished_views) == bool(args.render_logs_prefix):
        raise SystemExit("[main] must supply exactly one of --finished_views or "
                         "--render_logs_prefix")
    if args.render_logs_prefix:
        cache_dir = os.path.join(args.state_dir, "render_progress_cache")
        all_tasks = summarize_finished_views_from_logs(args.render_logs_prefix, cache_dir)
    else:
        cache_path = os.path.join(args.state_dir, "finished_views.json")
        all_tasks = load_finished_views(args.finished_views, cache_path)
    all_tasks.sort()
    print(f"[main] global finished views: {len(all_tasks)}")

    # Shard.
    start = len(all_tasks) * args.rank // args.world_size
    end = len(all_tasks) * (args.rank + 1) // args.world_size
    my_tasks = all_tasks[start:end]
    print(f"[main] rank {args.rank}: {len(my_tasks)} tasks (idx {start}:{end})")

    # L2: load per-rank progress.
    progress = ProgressStore(args.s3_output_root, args.rank, args.state_dir,
                             log_suffix=log_suffix)
    progress.load()

    to_process = [(s, v) for s, v in my_tasks if not progress.is_terminal(s, v)]
    skipped = len(my_tasks) - len(to_process)
    print(f"[main] terminal (success/skipped) in progress: {skipped}; to process: {len(to_process)}")

    if args.max_items is not None:
        to_process = to_process[: args.max_items]
        print(f"[main] limited to {len(to_process)} (--max_items)")

    if not to_process:
        print("[main] nothing to do")
        return

    n_success = 0
    n_skip = skipped
    n_fail = 0
    t_start = time.time()
    for i, (sha256, view_idx) in enumerate(to_process):
        t0 = time.time()
        try:
            result = voxelize_one_view(
                sha256=sha256,
                view_idx=view_idx,
                s3_input_root=args.s3_input_root,
                s3_output_root=args.s3_output_root,
                resolutions=resolutions,
                tmp_dir=args.tmp_dir,
                max_face_count=args.max_face_count,
                max_frames=args.max_frames,
                frame_sampling=args.frame_sampling,
            )
        except Exception as e:
            result = {"status": "worker_error", "error": f"{type(e).__name__}: {e}"}
            print(f"[main] worker error {sha256}/view_{view_idx:02d}: {result['error']}")
            upload_error_log(args.s3_output_root, args.rank, sha256, view_idx,
                             header=f"worker_error: {result['error']}",
                             log_suffix=log_suffix, exc=e)

        status = result.get("status", "unknown")
        if status == "success":
            n_success += 1
        elif status in TERMINAL_SKIP_STATUSES:
            n_skip += 1
        else:
            n_fail += 1
            # Surface any per-frame traceback captured inside voxelize_one_view.
            tb = result.pop("traceback", None) if isinstance(result, dict) else None
            if tb:
                upload_error_log(args.s3_output_root, args.rank, sha256, view_idx,
                                 header=f"status={status}", log_suffix=log_suffix,
                                 extra_tb=tb)
            elif status not in ("worker_error",):  # worker_error already uploaded above
                upload_error_log(args.s3_output_root, args.rank, sha256, view_idx,
                                 header=f"status={status} (no python traceback)",
                                 log_suffix=log_suffix,
                                 extra_tb=json.dumps(result, indent=2, default=str))

        elapsed = time.time() - t0
        line = _line(sha256, view_idx, status,
                     frames=result.get("num_frames"),
                     faces=result.get("num_faces"),
                     dt=round(elapsed, 1))
        progress.update(sha256, view_idx, _entry(status, **{k: v for k, v in result.items()
                                                             if k != "status"}), line)

        done = i + 1
        overall = time.time() - t_start
        rate = done / overall if overall > 0 else 0
        eta = (len(to_process) - done) / rate if rate > 0 else float("inf")
        print(f"[main] [{done}/{len(to_process)}] {sha256}/view_{view_idx:02d} "
              f"{status} dt={elapsed:.1f}s | ok={n_success} skip={n_skip} fail={n_fail} "
              f"rate={rate:.2f} view/s eta={eta/60:.1f}min")

    progress.flush()
    print(f"\n[main] DONE rank={args.rank} ok={n_success} skip={n_skip} fail={n_fail} "
          f"total_done={n_success + n_skip + n_fail} elapsed={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
