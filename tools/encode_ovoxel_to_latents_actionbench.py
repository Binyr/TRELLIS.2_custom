#!/usr/bin/env python3
"""
encode_ovoxel_to_latents_actionbench.py -- Batch-encode ActionBench O-Voxel
(.vxz) view tars into per-view npz files containing shape latents + SS latents.

Layout (input/output, S3 or local filesystem):
  input :  {input_root}/{res}/<uid>/view_XX.tar    (contains {frame}.vxz)
  output:  {output_root}/{res}/<uid>/view_XX.npz   (all frames in one)

Per-view npz keys:
  num_frames
  shape_feats_<frame_id>  (float32)
  shape_coords_<frame_id> (uint8)
  ss_z_<frame_id>         (float)

Differences vs. tools/encode_ovoxel_to_latents_objxl.py:
  * Defaults to OBJXL-style S3 IO via `aws s3 cp`; local paths are still supported.
  * Task list is summarized dynamically from the voxel-stage
    `voxel_progress_*.json` files (success-only), via --voxel_logs_dir.
    No static obj_list JSON and no directory-wide data scan.
  * Path has 2 levels (<uid>/view_XX), not 3 (<shard>/<obj>/view_XX).
  * Single GPU per rank (cuda:0). Multi-rank parallelism comes from launching
    multiple ranks via koala.
  * Uses progress files under {output_root}/logs_<res>/ and per-failure
    traceback files under {output_root}/_logs_<res>/rank_<rank>/.
  * stdout is tee-d + periodically copied to the mounted output by the launcher.

Usage (single rank):
  python tools/encode_ovoxel_to_latents_actionbench.py \
      --voxel_logs_dir .../dual_grid_v1_1view_120f_uid_ovoxel/logs_1024 \
      --input_root     .../dual_grid_v1_1view_120f_uid_ovoxel \
      --output_root    .../dual_grid_v1_1view_120f_uid_ovoxel_latent \
      --resolution 1024 --ss_resolution 32 \
      --rank 0 --world_size 1
"""

import argparse
import os
import sys
import json
import shutil
import subprocess
import tarfile
import tempfile
import time
import traceback
from datetime import datetime, timezone
from collections import deque
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

import o_voxel
import trellis2.models as models
import trellis2.modules.sparse as sp


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


def _parse_view_key(view_key: str):
    uid, view = view_key.split("/")
    if not view.startswith("view_"):
        raise ValueError(view_key)
    return uid, int(view[len("view_"):])


def summarize_finished_views_from_logs(logs_dir: str, filename_prefix: str,
                                       statuses: tuple = ("success",),
                                       missing_ok: bool = False) -> list[tuple[str, int]]:
    """Read progress JSON files from a local dir or S3 prefix and return view tasks."""
    tmp_dir = None
    if _is_s3_uri(logs_dir):
        prefix = logs_dir.rstrip("/") + "/"
        proc = run_aws(["s3", "ls", prefix], retries=2)
        if proc.returncode != 0:
            if missing_ok:
                print(f"[manifest] aws s3 ls failed for {prefix} -> []")
                return []
            raise RuntimeError(f"aws s3 ls failed for {prefix}: {proc.stderr.strip()[:300]}")
        names = []
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                name = parts[-1]
                if name.startswith(filename_prefix) and name.endswith(".json"):
                    names.append(name)
        names.sort()
        tmp_dir = tempfile.mkdtemp(prefix="encode_actionbench_logs_")
        paths = []
        for name in names:
            local_path = os.path.join(tmp_dir, name)
            if s3_get_file(prefix + name, local_path, retries=2):
                paths.append(local_path)
    else:
        if not os.path.isdir(logs_dir):
            if missing_ok:
                print(f"[manifest] logs dir missing: {logs_dir} -> []")
                return []
            raise RuntimeError(f"logs dir missing: {logs_dir}")
        names = sorted(
            n for n in os.listdir(logs_dir)
            if n.startswith(filename_prefix) and n.endswith(".json")
        )
        paths = [os.path.join(logs_dir, n) for n in names]

    try:
        print(f"[manifest] found {len(names)} {filename_prefix}*.json under {logs_dir}")
        if not names:
            if missing_ok:
                return []
            raise RuntimeError(f"No {filename_prefix}*.json under {logs_dir}")

        status_set = set(statuses)
        matched = set()
        status_counter = {}
        failed_parse = 0
        for path in paths:
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:
                failed_parse += 1
                continue
            if not isinstance(data, dict):
                failed_parse += 1
                continue
            for key, entry in data.items():
                st = (entry or {}).get("status", "unknown") if isinstance(entry, dict) else "unknown"
                status_counter[st] = status_counter.get(st, 0) + 1
                if st in status_set:
                    matched.add(key)
        print(f"[manifest] parse_failed={failed_parse}")
        print(f"[manifest] aggregate status counts (not deduped): {status_counter}")
        print(f"[manifest] unique matched views (statuses={sorted(status_set)}): {len(matched)}")

        out = []
        bad = 0
        for key in matched:
            try:
                out.append(_parse_view_key(key))
            except Exception:
                bad += 1
        if bad:
            print(f"[manifest] {bad} malformed keys skipped")
        return out
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class ProgressStore:
    def __init__(self, output_root: str, rank: int, local_state_dir: str,
                 log_suffix: str = "", push_every_updates: int = 8,
                 push_min_interval_s: float = 30.0):
        os.makedirs(local_state_dir, exist_ok=True)
        self.rank = int(rank)
        self.logs_root = _join_uri(output_root, f"logs{log_suffix}")
        if not _is_s3_uri(self.logs_root):
            os.makedirs(self.logs_root, exist_ok=True)
        self.remote_progress = _join_uri(self.logs_root, f"encode_progress_{rank}.json")
        self.remote_status = _join_uri(self.logs_root, f"encode_status_{rank}.log")
        self.local_progress = os.path.join(local_state_dir, f"encode_progress_{rank}{log_suffix}.json")
        self.local_status = os.path.join(local_state_dir, f"encode_status_{rank}{log_suffix}.log")
        self.progress = {}
        self.push_every_updates = max(1, int(push_every_updates))
        self.push_min_interval_s = float(push_min_interval_s)
        self._updates_since_push = 0
        self._last_push_ts = 0.0

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
        self.progress[f"{uid}/view_{int(view_idx):02d}"] = entry
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


def upload_error_log(output_root: str, rank: int, uid: str, view_idx: int,
                     header: str, log_suffix: str = "", exc: BaseException = None,
                     extra_tb: str = None):
    log_dir = _join_uri(output_root, f"_logs{log_suffix}", f"rank_{rank}")
    tmp_dir = tempfile.mkdtemp(prefix="encode_actionbench_error_")
    path = os.path.join(tmp_dir, f"{uid}_view_{view_idx:02d}.txt")
    body = [
        f"uid: {uid}",
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
    with open(path, "w") as f:
        f.write("\n".join(body))
    if _is_s3_uri(log_dir):
        s3_cp_file(path, _join_uri(log_dir, os.path.basename(path)), retries=1)
    else:
        os.makedirs(log_dir, exist_ok=True)
        shutil.copyfile(path, os.path.join(log_dir, os.path.basename(path)))
    shutil.rmtree(tmp_dir, ignore_errors=True)


# =====================================================================================
# Encode primitives
# =====================================================================================


def _read_vxz_cpu(vxz_path: str):
    """Read one .vxz file into CPU tensors (no GPU touch, no SparseTensor)."""
    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)
    return coords, attr


def _build_chunk_cpu(items, pin_memory: bool = True):
    """CPU-side cat + (optional) pin. Safe to call from any thread.

    items: list of (coords, attr_dict) tuples; list index -> batch_idx.
    Returns (coords_cpu, vfeats_cpu, ifeats_cpu, batch_size). The three
    tensors are pinned so a later `.to(device, non_blocking=True)` can use
    the copy engine and overlap with whatever's on the compute stream.
    """
    bcoords, vfeats, ifeats = [], [], []
    for b, (coords, attr) in enumerate(items):
        n = coords.shape[0]
        batch_col = torch.full((n, 1), b, dtype=coords.dtype)
        bcoords.append(torch.cat([batch_col, coords], dim=-1))
        vfeats.append((attr['vertices'] / 255.0).float())
        ic = attr['intersected']
        ifeats.append(torch.cat([ic % 2, ic // 2 % 2, ic // 4 % 2], dim=-1).bool())
    c = torch.cat(bcoords, dim=0)
    v = torch.cat(vfeats, dim=0)
    i = torch.cat(ifeats, dim=0)
    if pin_memory:
        try:
            c = c.pin_memory(); v = v.pin_memory(); i = i.pin_memory()
        except RuntimeError:
            # CUDA may complain if too much pinned -- silently fall back.
            pass
    return c, v, i, len(items)


def _chunk_cpu_to_sparse(chunk_cpu, device):
    """Move a CPU chunk to GPU and wrap in SparseTensor pair.

    Uses non_blocking so the H2D copy can overlap with any work already
    queued on the compute stream.
    """
    coords_cpu, vfeats_cpu, ifeats_cpu, _ = chunk_cpu
    cg = coords_cpu.to(device, non_blocking=True)
    vg = vfeats_cpu.to(device, non_blocking=True)
    ig = ifeats_cpu.to(device, non_blocking=True)
    vertices = sp.SparseTensor(vg, cg)
    intersected = vertices.replace(ig)
    return vertices, intersected


def _encode_shape_chunk_cpu(encoder, chunk_cpu, device):
    """Run shape encoder for one pre-built CPU chunk. Returns
    list[(feats_np, coords_np)] of length batch_size."""
    batch_size = chunk_cpu[3]
    vertices, intersected = _chunk_cpu_to_sparse(chunk_cpu, device)
    try:
        z = encoder(vertices, intersected)
        torch.cuda.synchronize()
        if not torch.isfinite(z.feats).all():
            raise ValueError("Non-finite values in shape latent")
        feats_all = z.feats.detach().cpu().numpy().astype(np.float32)
        coords_all = z.coords.detach().cpu().numpy()
    finally:
        del vertices, intersected
    out = []
    batch_col = coords_all[:, 0]
    for b in range(batch_size):
        mask = batch_col == b
        out.append((feats_all[mask], coords_all[mask, 1:].astype(np.uint8)))
    return out


# Back-compat shim: old code path (e.g. micro-benchmarks) used
# `_encode_shape_chunk(encoder, list_of_items, device)`. Build the CPU chunk
# on the spot and forward. Kept as a thin wrapper, no longer used in the
# main loop.
def _encode_shape_chunk(encoder, items, device):
    cc = _build_chunk_cpu(items, pin_memory=False)
    return _encode_shape_chunk_cpu(encoder, cc, device)


def _encode_ss(encoder, coords_enc, ss_resolution: int, device: str):
    coords_t = torch.from_numpy(coords_enc).long()
    if coords_t.numel() == 0:
        raise ValueError("Empty shape latent (no coords)")
    cmax = int(coords_t.max().item())
    if cmax >= ss_resolution:
        raise ValueError(
            f"Shape latent coords max ({cmax}) >= ss_resolution ({ss_resolution})")
    ss = torch.zeros(1, ss_resolution, ss_resolution, ss_resolution,
                     dtype=torch.float32, device=device)
    ss[0, coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]] = 1
    z = encoder(ss[None], sample_posterior=False)
    torch.cuda.synchronize()
    if not torch.isfinite(z).all():
        raise ValueError("Non-finite values in SS latent")
    return z[0].detach().cpu().numpy()


# =====================================================================================
# Per-view pipeline
# =====================================================================================


# --- Pipeline stage 1: CPU IO ---------------------------------------------------
#
# Runs in a background thread. Copies the tar from the mounted data path, extracts to scratch,
# parses every .vxz into CPU tensors. All side-effects live in `work_dir`, which
# is handed off to the GPU stage so it can clean up after itself.


def prepare_view(
    uid: str,
    view_idx: int,
    input_root: str,
    resolution: int,
    tmp_root: str,
    frame_chunk_size: int = 8,
    pin_memory: bool = True,
) -> dict:
    """Worker-thread stage 1.

    Copy tar -> extract -> read every .vxz -> pre-build the CPU-side chunked
    SparseTensor inputs (cat + pin_memory). Returns chunks ready for an
    almost-free `_chunk_cpu_to_sparse(..., device)` on the main thread.
    """
    view_key = f"{uid}/view_{view_idx:02d}"
    tar_path = _join_uri(input_root, str(resolution), uid, f"view_{view_idx:02d}.tar")

    work_dir = os.path.join(tmp_root, uid, f"view_{view_idx:02d}")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    local_tar = os.path.join(work_dir, f"view_{view_idx:02d}.tar")

    out = {
        "uid": uid,
        "view_idx": view_idx,
        "view_key": view_key,
        "work_dir": work_dir,
        "status": "prepared",
        "frame_ids": [],
        "loaded": [],          # kept around for the per-frame fallback path
        "cpu_chunks": [],      # list of (coords_cpu, vfeats_cpu, ifeats_cpu, batch_size)
        "chunk_fids": [],      # list[list[str]], aligned with cpu_chunks
        "t_get": 0.0,
        "t_extract": 0.0,
        "t_read": 0.0,
        "t_build_cpu": 0.0,
        "error": None,
    }
    try:
        t0 = time.time()
        if not copy_uri_to_local(tar_path, local_tar):
            out["status"] = "missing_tar"
            out["t_get"] = round(time.time() - t0, 2)
            return out
        out["t_get"] = round(time.time() - t0, 2)

        t0 = time.time()
        with tarfile.open(local_tar) as tf:
            tf.extractall(work_dir)
        os.remove(local_tar)
        vxz_files = sorted(f for f in os.listdir(work_dir) if f.endswith(".vxz"))
        out["t_extract"] = round(time.time() - t0, 2)
        if not vxz_files:
            out["status"] = "no_vxz"
            return out

        t0 = time.time()
        loaded = [_read_vxz_cpu(os.path.join(work_dir, n)) for n in vxz_files]
        out["t_read"] = round(time.time() - t0, 2)
        fids = [os.path.splitext(n)[0] for n in vxz_files]
        out["frame_ids"] = fids
        out["loaded"] = loaded

        # CPU-side chunking + pin_memory. This is the ~1s/view torch.cat work
        # we want to keep off the GPU main thread.
        t0 = time.time()
        cs = max(1, int(frame_chunk_size))
        chunks_cpu = []
        chunks_fids = []
        i = 0
        while i < len(loaded):
            items = loaded[i:i + cs]
            chunks_cpu.append(_build_chunk_cpu(items, pin_memory=pin_memory))
            chunks_fids.append(fids[i:i + cs])
            i += cs
        out["cpu_chunks"] = chunks_cpu
        out["chunk_fids"] = chunks_fids
        out["t_build_cpu"] = round(time.time() - t0, 2)
        return out
    except Exception as e:
        out["status"] = "prepare_error"
        out["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return out


# --- Pipeline stage 2: GPU encode (main thread) --------------------------------


def gpu_encode_view(
    prepared: dict,
    ss_resolution: int,
    shape_encoder,
    ss_encoder,
    device: str,
    do_ss: bool,
    frame_chunk_size: int,
) -> tuple:
    """Run shape (+ optional SS) encoding for one prepared view.

    Consumes the per-chunk CPU tensors prepare_view already built; main
    thread only does H2D (overlapped) + forward + D2H.
    Returns (all_data, n_frames_encoded, t_encode_s, last_failed_frame_id).
    Raises on encode failure (caller maps to encode_error)."""
    loaded = prepared["loaded"]                  # only used in per-frame fallback
    frame_ids = prepared["frame_ids"]
    cpu_chunks = prepared["cpu_chunks"]
    chunk_fids = prepared["chunk_fids"]
    all_data = {"num_frames": np.int32(len(loaded))}
    encoded = 0
    last_failed = None

    t0 = time.time()
    for chunk_cpu, fids in zip(cpu_chunks, chunk_fids):
        try:
            outs = _encode_shape_chunk_cpu(shape_encoder, chunk_cpu, device)
        except Exception as e_chunk:
            if len(fids) == 1:
                last_failed = fids[0]
                raise RuntimeError(f"frame {fids[0]}: {e_chunk}") from e_chunk
            torch.cuda.empty_cache()
            # Per-frame fallback: rebuild from `loaded` since pinned CPU chunk is
            # already gone after a failed forward.
            outs = []
            base_idx = frame_ids.index(fids[0])
            for k in range(len(fids)):
                try:
                    cc = _build_chunk_cpu([loaded[base_idx + k]], pin_memory=False)
                    outs.append(_encode_shape_chunk_cpu(shape_encoder, cc, device)[0])
                except Exception as e_single:
                    last_failed = fids[k]
                    raise RuntimeError(
                        f"frame {fids[k]} (chunk fallback): {e_single}") from e_single
        for fid, (feats, coords_enc) in zip(fids, outs):
            all_data[f"shape_feats_{fid}"] = feats
            all_data[f"shape_coords_{fid}"] = coords_enc
            if do_ss:
                try:
                    ss_z = _encode_ss(ss_encoder, coords_enc, ss_resolution, device)
                except Exception as e_ss:
                    last_failed = fid
                    raise RuntimeError(f"frame {fid} (ss): {e_ss}") from e_ss
                all_data[f"ss_z_{fid}"] = ss_z
            encoded += 1
        torch.cuda.empty_cache()
    t_encode = round(time.time() - t0, 2)
    return all_data, encoded, t_encode, last_failed


# --- Pipeline stage 3: save npz + publish (background thread) ------------------


def save_and_upload(
    work_dir: str,
    uid: str,
    view_idx: int,
    output_root: str,
    resolution: int,
    all_data: dict,
) -> dict:
    """Write npz to scratch, publish to mounted output path, clean up work_dir.

    Returns timings + a status overlay ('success' or 'upload_failed').
    """
    local_npz = os.path.join(work_dir, f"view_{view_idx:02d}.npz")
    out_path = _join_uri(output_root, str(resolution), uid, f"view_{view_idx:02d}.npz")
    try:
        t0 = time.time()
        np.savez_compressed(local_npz, **all_data)
        t_save = round(time.time() - t0, 2)
        npz_mb = round(os.path.getsize(local_npz) / (1024 * 1024), 2)

        t0 = time.time()
        if not copy_local_to_uri(local_npz, out_path):
            return {"status": "upload_failed", "t_save_npz": t_save,
                    "t_upload_npz": round(time.time() - t0, 2), "npz_mb": npz_mb}
        t_upload = round(time.time() - t0, 2)
        return {"status": "success", "t_save_npz": t_save,
                "t_upload_npz": t_upload, "npz_mb": npz_mb}
    except Exception as exc:
        return {"status": "upload_failed", "error": str(exc)[:500],
                "t_save_npz": locals().get("t_save", 0.0),
                "t_upload_npz": locals().get("t_upload", 0.0),
                "npz_mb": locals().get("npz_mb", 0.0)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# =====================================================================================
# Main
# =====================================================================================


# Status taxonomy:
#   permanent (terminal -> skipped on resume):
#     success         OK
#     no_vxz          tar opened fine but had no .vxz inside (bad data)
#     encode_error    a frame raised inside the encoder -- almost always a
#                     bad voxel / encoder math problem, not transient
#   transient (retried on resume):
#     missing_tar     input tar does not exist under the mounted input root
#     upload_failed   output npz could not be published to the mounted output root
#     prepare_error   tar extract / vxz read raised unexpectedly (almost
#                     always a corrupt tar fetched mid-write, retried later)
#
# Every entry also carries an `attempts` counter so we can spot views that
# stay transient for too long.
TERMINAL_SKIP_STATUSES = {"success", "no_vxz", "encode_error"}
TRANSIENT_STATUSES = {"missing_tar", "upload_failed", "prepare_error"}


def _task_key(task):
    sha, vid = task
    return f"{sha}/view_{int(vid):02d}"


def _parse_view_key(view_key: str):
    sha, view = view_key.split("/")
    if not view.startswith("view_"):
        raise ValueError(view_key)
    return sha, int(view[len("view_"):])


def load_progress_snapshot(path_or_uri: str, local_cache_path: str,
                           statuses=TERMINAL_SKIP_STATUSES) -> set:
    """Load a frozen global progress snapshot.

    Accepted JSON shapes:
      * ["sha/view_00", ...]
      * {"sha/view_00": {"status": "success", ...}, ...}
      * {"views": ["sha/view_00", ...]}
      * {"views": {"sha/view_00": {"status": "success", ...}, ...}}
    """
    local_path = path_or_uri
    if _is_s3_uri(path_or_uri):
        if not s3_get_file(path_or_uri, local_cache_path, retries=3):
            raise RuntimeError(f"failed to download snapshot {path_or_uri}")
        local_path = local_cache_path

    with open(local_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "views" in raw:
        raw = raw["views"]

    status_set = set(statuses)
    keys = set()
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                keys.add(item)
            elif isinstance(item, dict):
                key = item.get("view_key")
                st = item.get("status")
                if key and (st is None or st in status_set):
                    keys.add(key)
    elif isinstance(raw, dict):
        for key, entry in raw.items():
            st = (entry or {}).get("status", "unknown") if isinstance(entry, dict) else "unknown"
            if st in status_set:
                keys.add(key)
    else:
        raise RuntimeError(
            f"expected snapshot list/dict in {path_or_uri}, got {type(raw).__name__}")

    out = set()
    bad = 0
    for key in keys:
        try:
            out.add(_parse_view_key(key))
        except Exception:
            bad += 1
    if bad:
        print(f"[snapshot] {bad} malformed keys skipped")
    print(f"[snapshot] loaded {len(out)} terminal views from {path_or_uri}")
    return out


def write_progress_snapshot(encode_logs_dir: str, output_path_or_uri: str,
                            local_cache_dir: str,
                            statuses=TERMINAL_SKIP_STATUSES) -> None:
    terminal = summarize_finished_views_from_logs(
        encode_logs_dir,
        filename_prefix="encode_progress_",
        statuses=tuple(statuses),
        missing_ok=True,
    )
    keys = sorted(_task_key(t) for t in terminal)
    local_out = os.path.join(local_cache_dir, os.path.basename(output_path_or_uri))
    if not local_out.endswith(".json"):
        local_out += ".json"
    payload = {
        "source": encode_logs_dir,
        "statuses": sorted(statuses),
        "num_views": len(keys),
        "views": keys,
    }
    os.makedirs(os.path.dirname(os.path.abspath(local_out)), exist_ok=True)
    with open(local_out, "w") as f:
        json.dump(payload, f)
    out_path = output_path_or_uri
    if not out_path.endswith(".json"):
        out_path += ".json"
    if not copy_local_to_uri(local_out, out_path):
        raise RuntimeError(f"failed to write snapshot to {out_path}")
    print(f"[snapshot] wrote {len(keys)} terminal views to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-encode ActionBench O-Voxel view tars into shape+SS latents.")
    parser.add_argument("--voxel_logs_dir", type=str, required=True,
                        help="Directory containing voxel_progress_*.json from the voxel stage.")
    parser.add_argument("--input_root", type=str, required=True,
                        help="Mounted path containing {resolution}/<uid>/view_XX.tar")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Mounted path where {resolution}/<uid>/view_XX.npz will be written")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--log_suffix", type=str, default="_1024",
                        help="Suffix for encode logs/progress, e.g. _1024.")
    parser.add_argument("--ss_resolution", type=int, default=32)
    parser.add_argument("--shape_enc_pretrained", type=str,
                        default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
    parser.add_argument("--ss_enc_pretrained", type=str,
                        default="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--task_shard_mode", type=str, default="filter_then_shard",
                        choices=("filter_then_shard",
                                 "shard_then_filter",
                                 "snapshot_filter_then_shard_then_filter"),
                        help="Task ordering mode. filter_then_shard removes live "
                             "cross-rank terminal progress before sharding. "
                             "snapshot_filter_then_shard_then_filter first removes "
                             "views in --global_progress_snapshot, shards the frozen "
                             "remainder, then applies this rank's live progress.")
    parser.add_argument("--global_progress_snapshot", type=str, default=None,
                        help="Frozen local snapshot JSON used by "
                             "snapshot_filter_then_shard_then_filter.")
    parser.add_argument("--write_global_progress_snapshot", type=str, default=None,
                        help="Summarize current encode_progress_*.json into this "
                             "local snapshot path and exit.")
    parser.add_argument("--state_dir", type=str, default="/tmp/encode_actionbench_state")
    parser.add_argument("--tmp_dir", type=str, default="/tmp/encode_actionbench_tmp")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--frame_chunk_size", type=int, default=8,
                        help="Number of frames batched into one shape-encoder forward. "
                             "On chunk failure the worker falls back to per-frame for "
                             "just that chunk so a single bad frame can't poison the view.")
    parser.add_argument("--prefetch", type=int, default=2,
                        help="Number of views to keep prepared in background threads. "
                             "Higher values overlap mounted-path IO/decode/CPU+pinned-mem "
                             "with GPU but use more scratch space and pinned host memory.")
    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Disable pin_memory in the prefetch worker.")
    args = parser.parse_args()

    if args.log_suffix == "":
        args.log_suffix = "_" + str(args.resolution)

    args.state_dir = os.path.join(args.state_dir, f"rank_{args.rank}")
    args.tmp_dir = os.path.join(args.tmp_dir, f"rank_{args.rank}")
    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    print(f"[main] rank={args.rank}/{args.world_size} state={args.state_dir} "
          f"tmp={args.tmp_dir} res={args.resolution} ss_res={args.ss_resolution}")
    print(f"[main] input_root={args.input_root}")
    print(f"[main] output_root={args.output_root}")
    print(f"[main] voxel_logs_dir={args.voxel_logs_dir}")
    print(f"[main] task_shard_mode={args.task_shard_mode} "
          f"global_progress_snapshot={args.global_progress_snapshot or '<unset>'}")

    encode_logs_dir = os.path.join(args.output_root, f"logs{args.log_suffix}")

    if args.write_global_progress_snapshot:
        snapshot_cache = os.path.join(args.state_dir, "write_progress_snapshot_cache")
        write_progress_snapshot(
            encode_logs_dir,
            args.write_global_progress_snapshot,
            snapshot_cache,
            statuses=TERMINAL_SKIP_STATUSES,
        )
        return

    # L1: gather success views from voxel-stage progress files.
    all_tasks = summarize_finished_views_from_logs(
        args.voxel_logs_dir, filename_prefix="voxel_progress_")
    all_tasks.sort()
    print(f"[main] voxel-success views: {len(all_tasks)}")

    if args.task_shard_mode == "filter_then_shard":
        # Old behavior: pull every encode_progress_*.json that any rank has
        # written so far and subtract terminal-status views from the global task
        # pool BEFORE sharding.
        cross_done = summarize_finished_views_from_logs(
            encode_logs_dir,
            filename_prefix="encode_progress_",
            statuses=tuple(TERMINAL_SKIP_STATUSES),
            missing_ok=True,
        )
        cross_done_set = set(cross_done)
        n_global = len(all_tasks)
        all_tasks = [t for t in all_tasks if t not in cross_done_set]
        print(f"[main] cross-rank terminal views to skip globally: {len(cross_done_set)} "
              f"(of {n_global} -> {len(all_tasks)} remaining)")
    elif args.task_shard_mode == "snapshot_filter_then_shard_then_filter":
        if not args.global_progress_snapshot:
            raise SystemExit(
                "--global_progress_snapshot is required for "
                "snapshot_filter_then_shard_then_filter")
        snapshot_cache = os.path.join(args.state_dir, "global_progress_snapshot.json")
        snapshot_done_set = load_progress_snapshot(
            args.global_progress_snapshot,
            snapshot_cache,
            statuses=TERMINAL_SKIP_STATUSES,
        )
        n_global = len(all_tasks)
        all_tasks = [t for t in all_tasks if t not in snapshot_done_set]
        print(f"[main] snapshot terminal views to skip globally: {len(snapshot_done_set)} "
              f"(of {n_global} -> {len(all_tasks)} remaining)")
    else:
        print("[main] cross-rank pre-shard skip disabled; sharding voxel-success pool first")

    # Shard the current task pool. In snapshot mode, the task pool is frozen by
    # --global_progress_snapshot before slicing, so rank assignment is stable
    # across all ranks in this launch.
    start = len(all_tasks) * args.rank // args.world_size
    end = len(all_tasks) * (args.rank + 1) // args.world_size
    my_tasks = all_tasks[start:end]
    print(f"[main] rank {args.rank}: {len(my_tasks)} raw shard tasks (idx {start}:{end})")

    # L2: per-rank progress (for retried-transient handling and per-rank
    # bookkeeping; cross-rank skip already excluded global terminals).
    progress = ProgressStore(args.output_root, args.rank, args.state_dir,
                             log_suffix=args.log_suffix)
    progress.load()

    def _done(sha, vid):
        return progress.progress.get(f"{sha}/view_{int(vid):02d}", {}).get("status") in TERMINAL_SKIP_STATUSES

    # Summarize the existing progress.json for visibility.
    from collections import Counter as _Counter
    prior_status = _Counter(
        (entry or {}).get("status", "?") for entry in progress.progress.values()
        if isinstance(entry, dict)
    )
    print(f"[main] prior progress entries by status: {dict(prior_status)}")

    to_process = [(s, v) for s, v in my_tasks if not _done(s, v)]
    skipped = len(my_tasks) - len(to_process)
    print(f"[main] this-rank terminal in progress (additional skip): {skipped}; "
          f"to process (incl. retried transient): {len(to_process)}")

    if args.max_items is not None:
        to_process = to_process[: args.max_items]
        print(f"[main] limited to {len(to_process)} (--max_items)")

    if not to_process:
        print("[main] nothing to do")
        return

    # Models.
    device = "cuda:0"
    torch.cuda.set_device(0)
    do_ss = args.resolution > 512
    print(f"[main] loading shape encoder: {args.shape_enc_pretrained}")
    shape_encoder = models.from_pretrained(args.shape_enc_pretrained).eval().to(device)
    if do_ss:
        print(f"[main] loading ss    encoder: {args.ss_enc_pretrained}")
        ss_encoder = models.from_pretrained(args.ss_enc_pretrained).eval().to(device)
    else:
        ss_encoder = None
        print(f"[main] resolution={args.resolution} <= 512 -> skipping SS encode")

    # ---- pipelined prefetch loop ----
    n_ok = 0
    n_fail = 0
    t_start = time.time()
    total = len(to_process)

    # Two pools: prepare (CPU-heavy, mounted-path copy + tar extract + vxz parse)
    # and save (CPU + mounted-path publish). These stages release the GIL inside
    # IO/tarfile/o_voxel, so threads are fine.
    prep_pool = ThreadPoolExecutor(max_workers=max(1, args.prefetch),
                                   thread_name_prefix="prep")
    save_pool = ThreadPoolExecutor(max_workers=max(1, args.prefetch),
                                   thread_name_prefix="save")

    prep_queue = deque()  # of (idx, sha, vid, future)
    save_inflight = []    # of (idx, sha, vid, future, t_view_start)
    cursor = 0

    def schedule_next():
        nonlocal cursor
        while len(prep_queue) < max(1, args.prefetch) and cursor < total:
            sha, vid = to_process[cursor]
            fut = prep_pool.submit(
                prepare_view, sha, vid,
                args.input_root, args.resolution, args.tmp_dir,
                args.frame_chunk_size, not args.no_pin_memory,
            )
            prep_queue.append((cursor, sha, vid, fut, time.time()))
            cursor += 1

    def finalize_result(sha, vid, result, t_view_start):
        nonlocal n_ok, n_fail
        view_key = f"{sha}/view_{int(vid):02d}"
        prev_attempts = (progress.progress.get(view_key, {}) or {}).get("attempts", 0)
        result["attempts"] = int(prev_attempts) + 1

        dt = time.time() - t_view_start
        st = result.get("status", "?")
        if st == "success":
            n_ok += 1
        else:
            n_fail += 1
            if st != "encode_error":
                upload_error_log(args.output_root, args.rank, sha, vid,
                                 header=f"soft_failure:{st}",
                                 log_suffix=args.log_suffix,
                                 extra_tb=json.dumps(result, default=str, indent=2))

        rate = (n_ok + n_fail) / max(1e-6, time.time() - t_start)
        eta_min = (total - (n_ok + n_fail)) / max(1e-6, rate) / 60.0
        progress.update(
            sha, vid, result,
            _line(sha, vid, st, dt=f"{dt:.1f}s",
                  ok=n_ok, fail=n_fail, rate=f"{rate:.2f}/s",
                  eta_min=f"{eta_min:.1f}"),
        )
        print(f"[main] {sha}/view_{vid:02d} {st} dt={dt:.1f}s "
              f"| ok={n_ok} fail={n_fail} of {total} rate={rate:.2f} view/s "
              f"eta={eta_min:.1f}min  | prep_q={len(prep_queue)} save_q={len(save_inflight)}")

    try:
        schedule_next()
        processed = 0
        while processed < total:
            # Drain any finished save futures (non-blocking-ish: check tip first).
            new_save = []
            for idx, sha, vid, fut, t_view_start, entry in save_inflight:
                if fut.done():
                    so = fut.result()
                    entry["status"] = so["status"]
                    for k in ("t_save_npz", "t_upload_npz", "npz_mb", "error"):
                        if k in so:
                            entry[k] = so[k]
                    print(f"[timing] {sha[:12]}/view_{vid:02d} frames={entry.get('num_frames','?')} "
                          f"npz_mb={entry.get('npz_mb','?')} "
                          f"t_get={entry.get('t_get','?')}s t_extract={entry.get('t_extract','?')}s "
                          f"t_read={entry.get('t_read','?')}s t_build={entry.get('t_build_cpu','?')}s "
                          f"t_encode={entry.get('t_encode','?')}s "
                          f"t_save={entry.get('t_save_npz','?')}s t_upload={entry.get('t_upload_npz','?')}s")
                    finalize_result(sha, vid, entry, t_view_start)
                    processed += 1
                else:
                    new_save.append((idx, sha, vid, fut, t_view_start, entry))
            save_inflight[:] = new_save

            if not prep_queue:
                # GPU starved -- everything drained, wait for any save to finish.
                if save_inflight:
                    time.sleep(0.05)
                    continue
                break

            idx, sha, vid, fut, t_view_start = prep_queue.popleft()
            t_wait0 = time.time()
            prepared = fut.result()  # blocks if prep hasn't finished
            t_wait = round(time.time() - t_wait0, 2)
            schedule_next()  # keep the prep pipeline full

            # Failed/skipped views: don't touch GPU, write entry directly.
            if prepared["status"] != "prepared":
                st = prepared["status"]
                if st == "prepare_error" and prepared.get("error"):
                    upload_error_log(args.output_root, args.rank, sha, vid,
                                     header="prepare_error",
                                     log_suffix=args.log_suffix,
                                     extra_tb=prepared["error"])
                entry = _entry(st, view_key=prepared["view_key"], num_frames=0,
                               t_get=prepared["t_get"], t_extract=prepared["t_extract"],
                               t_read=prepared.get("t_read", 0.0),
                               t_build_cpu=prepared.get("t_build_cpu", 0.0),
                               t_wait_prep=t_wait)
                shutil.rmtree(prepared["work_dir"], ignore_errors=True)
                finalize_result(sha, vid, entry, t_view_start)
                processed += 1
                continue

            # GPU encode (in main thread).
            try:
                with torch.no_grad():
                    all_data, encoded, t_encode, last_failed = gpu_encode_view(
                        prepared, args.ss_resolution, shape_encoder, ss_encoder,
                        device, do_ss, args.frame_chunk_size,
                    )
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[error] {sha}/view_{vid:02d}: {e}\n{tb}")
                upload_error_log(args.output_root, args.rank, sha, vid,
                                 header=str(e), log_suffix=args.log_suffix, exc=e)
                shutil.rmtree(prepared["work_dir"], ignore_errors=True)
                entry = _entry("encode_error", view_key=prepared["view_key"],
                               error=str(e)[:500],
                               t_get=prepared["t_get"], t_extract=prepared["t_extract"],
                               t_read=prepared["t_read"],
                               t_build_cpu=prepared.get("t_build_cpu", 0.0),
                               t_wait_prep=t_wait)
                finalize_result(sha, vid, entry, t_view_start)
                processed += 1
                continue

            # Hand off save+upload to background.
            entry = _entry("uploading", view_key=prepared["view_key"], num_frames=encoded,
                           t_get=prepared["t_get"], t_extract=prepared["t_extract"],
                           t_read=prepared["t_read"],
                           t_build_cpu=prepared.get("t_build_cpu", 0.0),
                           t_encode=t_encode, t_wait_prep=t_wait,
                           failed_frame=last_failed)
            save_fut = save_pool.submit(
                save_and_upload,
                prepared["work_dir"], sha, vid,
                args.output_root, args.resolution, all_data,
            )
            save_inflight.append((idx, sha, vid, save_fut, t_view_start, entry))
    finally:
        prep_pool.shutdown(wait=True)
        save_pool.shutdown(wait=True)

    # Drain any final saves that completed during shutdown.
    for idx, sha, vid, fut, t_view_start, entry in save_inflight:
        try:
            so = fut.result()
            entry["status"] = so["status"]
            for k in ("t_save_npz", "t_upload_npz", "npz_mb", "error"):
                if k in so:
                    entry[k] = so[k]
            print(f"[timing] {sha[:12]}/view_{vid:02d} frames={entry.get('num_frames','?')} "
                  f"npz_mb={entry.get('npz_mb','?')} (post-shutdown)")
        except Exception as e:
            entry["status"] = "upload_failed"
            entry["error"] = str(e)[:500]
        finalize_result(sha, vid, entry, t_view_start)

    progress.flush()
    print(f"\n[main] DONE rank={args.rank} ok={n_ok} fail={n_fail} "
          f"elapsed={(time.time() - t_start)/60.0:.1f}min")


if __name__ == "__main__":
    main()
