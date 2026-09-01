#!/usr/bin/env python3
"""Resumable material-VAE encoder for PBR O-Voxel view tars on S3.

Each output NPZ contains deterministic posterior statistics and the raw
material-encoder coordinates. The frozen task manifest guarantees that a
paired shape latent exists; an optional low-rate audit can still verify exact
coordinate equality without putting shape downloads on the hot path. Outputs
also preserve the mapping from encoded frame IDs back to source animation
frames:
  pbr_mean_<frame>   float32 [N, 32]
  pbr_logvar_<frame> float32 [N, 32]
  pbr_coords_<frame> uint8   [N, 3]
  frame_ids           int32   [T]
  source_frame_indices int32  [T]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import o_voxel
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import trellis2.models as models
import trellis2.modules.sparse as sp


ATTRS = ("base_color", "metallic", "roughness", "alpha")
TERMINAL = {
    "success",
    "no_vxz",
    "frame_set_mismatch",
    "frame_metadata_mismatch",
    "coord_mismatch",
    "encode_error",
}


class FrameSetMismatch(ValueError):
    pass


class FrameMetadataMismatch(ValueError):
    pass


class CoordinateMismatch(ValueError):
    pass


def run_aws(args: list[str], retries: int = 2) -> subprocess.CompletedProcess:
    last = None
    for attempt in range(retries + 1):
        proc = subprocess.run(["aws", *args], capture_output=True, text=True)
        if proc.returncode == 0:
            return proc
        last = proc
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    assert last is not None
    return last


def s3_get(uri: str, local: Path, retries: int = 2) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    return run_aws(["s3", "cp", "--only-show-errors", uri, str(local)], retries).returncode == 0


def s3_put(local: Path, uri: str, retries: int = 2) -> bool:
    proc = run_aws(["s3", "cp", "--only-show-errors", str(local), uri], retries)
    if proc.returncode != 0:
        print(f"[s3] upload failed: {uri}: {proc.stderr.strip()[:300]}", flush=True)
        return False
    return True


def list_object_tasks(
    s3_root: str, resolution: int, suffix: str, *, missing_ok: bool = False
) -> set[str]:
    """Return view task IDs for matching objects below an S3 resolution root."""
    prefix = f"{s3_root.rstrip('/')}/{resolution}/"
    proc = run_aws(["s3", "ls", prefix, "--recursive"], retries=3)
    if proc.returncode != 0:
        if missing_ok:
            print(f"[objects] no existing objects under {prefix}; treating as empty", flush=True)
            return set()
        raise RuntimeError(f"cannot list {prefix}: {proc.stderr.strip()[:500]}")
    root_key = s3_root.removeprefix("s3://").split("/", 1)[1].strip("/")
    key_prefix = f"{root_key}/{resolution}/"
    tasks = set()
    for line in proc.stdout.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) != 4:
            continue
        key = fields[3]
        if not key.startswith(key_prefix) or not key.endswith(suffix) or "/view_" not in key:
            continue
        rel = key[len(key_prefix) :]
        tasks.add(rel[: -len(suffix)])
    return tasks


def load_task_manifest(uri: str, local_path: Path, resolution: int) -> tuple[list[str], dict]:
    if not s3_get(uri, local_path, retries=3):
        raise RuntimeError(f"cannot download task manifest: {uri}")
    try:
        payload = json.loads(local_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid task manifest {uri}: {exc}") from exc
    tasks = payload.get("tasks")
    if payload.get("schema_version") != 1 or not isinstance(tasks, list):
        raise RuntimeError(f"unsupported task manifest schema: {uri}")
    if int(payload.get("resolution", -1)) != resolution:
        raise RuntimeError(
            f"manifest resolution={payload.get('resolution')} does not match requested {resolution}"
        )
    if len(tasks) != len(set(tasks)) or any(not isinstance(task, str) for task in tasks):
        raise RuntimeError("manifest tasks must be unique strings")
    tasks = sorted(tasks)
    checksum = hashlib.sha256(("\n".join(tasks) + "\n").encode()).hexdigest()
    if checksum != payload.get("task_checksum_sha256"):
        raise RuntimeError(
            f"manifest checksum mismatch: expected={payload.get('task_checksum_sha256')} got={checksum}"
        )
    return tasks, payload


def load_progress(progress_uri: str, local_path: Path) -> dict:
    if s3_get(progress_uri, local_path, retries=1):
        try:
            return json.loads(local_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[progress] invalid existing state ({exc}); starting empty", flush=True)
    return {}


def save_progress(progress: dict, progress_uri: str, local_path: Path) -> None:
    tmp = local_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, sort_keys=True), encoding="utf-8")
    os.replace(tmp, local_path)
    if not s3_put(local_path, progress_uri, retries=2):
        print("[progress] remote mirror failed; local state remains available", flush=True)


def sparse_input(coords: torch.Tensor, attrs: dict) -> sp.SparseTensor:
    missing = [name for name in ATTRS if name not in attrs]
    if missing:
        raise KeyError(f"missing PBR attributes: {missing}")
    feats = torch.cat([attrs[name] for name in ATTRS], dim=-1).float() / 255.0
    batch = torch.zeros_like(coords[:, :1])
    return sp.SparseTensor(feats * 2.0 - 1.0, torch.cat([batch, coords.int()], dim=1))


def should_audit_shape(task_id: str, every: int) -> bool:
    """Select a stable approximately 1/every subset, independent of sharding."""
    if every <= 0:
        return False
    code = int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:8], "big")
    return code % every == 0


def center_frames(frames: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    start = (len(frames) - max_frames) // 2
    return frames[start : start + max_frames]


def _failure(task_id: str, status: str, started: float, task_dir: Path, **extra) -> dict:
    shutil.rmtree(task_dir, ignore_errors=True)
    return {
        "status": status,
        "task": task_id,
        "seconds": round(time.perf_counter() - started, 2),
        **extra,
    }


def prepare_task(task_id: str, tar_uri: str, args) -> dict:
    """Stage 1: download, extract, validate metadata, and read VXZ on CPU."""
    task_dir = Path(args.tmp_dir) / task_id
    shutil.rmtree(task_dir, ignore_errors=True)
    task_dir.mkdir(parents=True, exist_ok=True)
    tar_path = task_dir / "input.tar"
    shape_path = task_dir / "shape.npz"
    meta_path = task_dir / "pbr_meta.json"
    audit_shape = should_audit_shape(task_id, args.shape_audit_every)
    shape_uri = (
        f"{args.s3_shape_root.rstrip('/')}/{args.resolution}/{task_id}.npz"
        if args.s3_shape_root
        else None
    )
    started = time.perf_counter()
    timings = {}
    try:
        t0 = time.perf_counter()
        if not s3_get(tar_uri, tar_path):
            return _failure(task_id, "missing_pbr_tar", started, task_dir)
        timings["t_download_pbr"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        with tarfile.open(tar_path) as tf:
            tf.extractall(task_dir)
        tar_path.unlink(missing_ok=True)
        timings["t_extract"] = round(time.perf_counter() - t0, 3)
        all_frames = list(task_dir.glob("*.vxz"))
        if not all_frames:
            return _failure(task_id, "no_vxz", started, task_dir, **timings)
        try:
            all_frames.sort(key=lambda path: int(path.stem))
        except ValueError as exc:
            raise FrameMetadataMismatch(
                f"non-numeric PBR frame IDs: {[path.stem for path in all_frames[:10]]}"
            ) from exc
        frames = center_frames(all_frames, args.max_frames)
        frame_ids = [frame_path.stem for frame_path in frames]
        try:
            frame_numbers = np.asarray([int(frame_id) for frame_id in frame_ids], dtype=np.int32)
        except ValueError as exc:
            raise FrameMetadataMismatch(f"non-numeric PBR frame IDs: {frame_ids[:10]}") from exc

        if args.frame_mapping == "meta":
            meta_uri = tar_uri[:-4] + "_meta.json"
            t0 = time.perf_counter()
            if not s3_get(meta_uri, meta_path):
                return _failure(
                    task_id, "missing_frame_meta", started, task_dir, meta_uri=meta_uri, **timings
                )
            timings["t_download_meta"] = round(time.perf_counter() - t0, 3)
            try:
                meta = json.loads(meta_path.read_text())
                source_frame_indices = np.asarray(meta["frame_sel"], dtype=np.int32)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise FrameMetadataMismatch(f"invalid frame_sel in {meta_uri}: {exc}") from exc
            all_frame_numbers = np.asarray([int(path.stem) for path in all_frames], dtype=np.int32)
            expected_ids = np.arange(len(source_frame_indices), dtype=np.int32)
            if not np.array_equal(all_frame_numbers, expected_ids):
                raise FrameMetadataMismatch(
                    f"PBR frame IDs are not contiguous 0..K-1: ids={all_frame_numbers.tolist()[:20]} "
                    f"frame_sel_len={len(source_frame_indices)}"
                )
            source_frame_indices = source_frame_indices[frame_numbers]
        else:
            source_frame_indices = frame_numbers.copy()

        if audit_shape:
            if shape_uri is None:
                raise ValueError("--s3_shape_root is required when --shape_audit_every is enabled")
            t0 = time.perf_counter()
            if not s3_get(shape_uri, shape_path):
                return _failure(task_id, "missing_shape_latent", started, task_dir, **timings)
            timings["t_download_shape"] = round(time.perf_counter() - t0, 3)

        shape_coords = None
        if audit_shape:
            shape_coords = {}
            with np.load(shape_path, allow_pickle=False) as shape:
                for frame_id in frame_ids:
                    coord_key = f"shape_coords_{frame_id}"
                    if coord_key not in shape.files:
                        raise FrameSetMismatch(f"audited shape latent is missing {coord_key}")
                    shape_coords[frame_id] = shape[coord_key].astype(np.int32, copy=True)

        t0 = time.perf_counter()
        loaded = []
        for frame_path in frames:
            coords, attrs = o_voxel.io.read_vxz(str(frame_path), num_threads=4)
            loaded.append((frame_path.stem, coords, attrs))
        timings["t_read_prepare"] = round(time.perf_counter() - t0, 3)

        return {
            "status": "prepared",
            "task": task_id,
            "task_dir": task_dir,
            "out_uri": f"{args.s3_output_root.rstrip('/')}/{args.resolution}/{task_id}.npz",
            "started": started,
            "loaded": loaded,
            "shape_coords": shape_coords,
            "frame_numbers": frame_numbers,
            "source_frame_indices": source_frame_indices,
            "num_input_frames": len(all_frames),
            "shape_audited": audit_shape,
            **timings,
        }
    except FrameSetMismatch as exc:
        return _failure(task_id, "frame_set_mismatch", started, task_dir, error=str(exc)[:500], **timings)
    except FrameMetadataMismatch as exc:
        return _failure(task_id, "frame_metadata_mismatch", started, task_dir, error=str(exc)[:500], **timings)
    except Exception as exc:
        return _failure(task_id, "prepare_error", started, task_dir, error=repr(exc)[:500], **timings)


def encode_prepared(prepared: dict, encoder) -> dict:
    """Stage 2: run the official material encoder on the main GPU thread."""
    task_id = prepared["task"]
    task_dir = prepared["task_dir"]
    t0 = time.perf_counter()
    try:
        output: dict[str, np.ndarray] = {
            "num_frames": np.int32(len(prepared["loaded"])),
            "num_input_frames": np.int32(prepared["num_input_frames"]),
            "frame_ids": prepared["frame_numbers"],
            "source_frame_indices": prepared["source_frame_indices"],
        }
        for frame_id, coords, attrs in prepared["loaded"]:
            x = sparse_input(coords, attrs).cuda()
            with torch.inference_mode():
                z, mean, logvar = encoder(x, sample_posterior=False, return_raw=True)
            material_coords = z.coords[:, 1:].int()
            if prepared["shape_coords"] is not None:
                shape_coords = torch.from_numpy(prepared["shape_coords"][frame_id]).to(
                    material_coords.device
                )
                if not torch.equal(shape_coords, material_coords):
                    raise CoordinateMismatch(
                        f"raw coordinate order differs for frame {frame_id}: "
                        f"shape={tuple(shape_coords.shape)} material={tuple(material_coords.shape)}"
                    )
            output[f"pbr_mean_{frame_id}"] = mean.detach().cpu().numpy().astype(np.float32)
            output[f"pbr_logvar_{frame_id}"] = logvar.detach().cpu().numpy().astype(np.float32)
            output[f"pbr_coords_{frame_id}"] = material_coords.detach().cpu().numpy().astype(np.uint8)
        torch.cuda.synchronize()
        prepared["loaded"] = None
        prepared["shape_coords"] = None
        prepared["output"] = output
        prepared["status"] = "encoded"
        prepared["t_encode"] = round(time.perf_counter() - t0, 3)
        return prepared
    except CoordinateMismatch as exc:
        return _failure(
            task_id, "coord_mismatch", prepared["started"], task_dir,
            error=str(exc)[:500], t_encode=round(time.perf_counter() - t0, 3)
        )
    except Exception as exc:
        return _failure(
            task_id, "encode_error", prepared["started"], task_dir,
            error=repr(exc)[:500], t_encode=round(time.perf_counter() - t0, 3)
        )


def save_and_upload(encoded: dict) -> dict:
    """Stage 3: compress and upload in a background thread."""
    task_dir = encoded["task_dir"]
    out_path = task_dir / "material.npz"
    try:
        t0 = time.perf_counter()
        np.savez_compressed(out_path, **encoded.pop("output"))
        t_save = round(time.perf_counter() - t0, 3)
        output_mb = round(out_path.stat().st_size / 1024 / 1024, 2)
        t0 = time.perf_counter()
        ok = s3_put(out_path, encoded["out_uri"])
        t_upload = round(time.perf_counter() - t0, 3)
        return {
            "status": "success" if ok else "upload_failed",
            "task": encoded["task"],
            "num_frames": len(encoded["frame_numbers"]),
            "num_input_frames": encoded["num_input_frames"],
            "shape_audited": encoded["shape_audited"],
            "output_mb": output_mb,
            "t_download_pbr": encoded.get("t_download_pbr", 0.0),
            "t_extract": encoded.get("t_extract", 0.0),
            "t_download_meta": encoded.get("t_download_meta", 0.0),
            "t_download_shape": encoded.get("t_download_shape", 0.0),
            "t_read_prepare": encoded.get("t_read_prepare", 0.0),
            "t_encode": encoded.get("t_encode", 0.0),
            "t_save": t_save,
            "t_upload": t_upload,
            "seconds": round(time.perf_counter() - encoded["started"], 2),
        }
    except Exception as exc:
        return {
            "status": "save_error",
            "task": encoded["task"],
            "error": repr(exc)[:500],
            "seconds": round(time.perf_counter() - encoded["started"], 2),
        }
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


def encode_task(task_id: str, tar_uri: str, args, encoder) -> dict:
    """Serial compatibility path used for baselines and isolated smoke tests."""
    prepared = prepare_task(task_id, tar_uri, args)
    if prepared["status"] != "prepared":
        return prepared
    encoded = encode_prepared(prepared, encoder)
    if encoded["status"] != "encoded":
        return encoded
    return save_and_upload(encoded)


def run_three_stage(pending: list[tuple[str, str]], args, encoder, on_result) -> None:
    """Bounded prepare -> GPU -> save/upload pipeline."""
    prep_depth = max(1, args.prefetch_tasks)
    save_depth = max(1, args.save_queue_depth)
    prep_queue: deque[tuple[str, str, Future]] = deque()
    save_queue: deque[Future] = deque()
    cursor = 0

    def schedule_prepares(pool: ThreadPoolExecutor) -> None:
        nonlocal cursor
        while cursor < len(pending) and len(prep_queue) < prep_depth:
            task_id, tar_uri = pending[cursor]
            prep_queue.append((task_id, tar_uri, pool.submit(prepare_task, task_id, tar_uri, args)))
            cursor += 1

    def drain_save(block: bool) -> bool:
        if not save_queue or (not block and not save_queue[0].done()):
            return False
        on_result(save_queue.popleft().result())
        return True

    with ThreadPoolExecutor(max_workers=prep_depth, thread_name_prefix="pbr-prep") as prep_pool, \
         ThreadPoolExecutor(max_workers=args.upload_workers, thread_name_prefix="pbr-save") as save_pool:
        schedule_prepares(prep_pool)
        while prep_queue:
            while drain_save(block=False):
                pass
            task_id, tar_uri, future = prep_queue.popleft()
            prepared = future.result()
            schedule_prepares(prep_pool)
            if prepared["status"] != "prepared":
                on_result(prepared)
                continue
            while len(save_queue) >= save_depth:
                drain_save(block=True)
            encoded = encode_prepared(prepared, encoder)
            if encoded["status"] != "encoded":
                on_result(encoded)
                continue
            save_queue.append(save_pool.submit(save_and_upload, encoded))
        while save_queue:
            drain_save(block=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_input_root", required=True)
    parser.add_argument("--s3_shape_root")
    parser.add_argument("--s3_output_root", required=True)
    parser.add_argument("--task_manifest", required=True)
    parser.add_argument("--frame_mapping", choices=["identity", "meta"], required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--state_dir", required=True)
    parser.add_argument("--tmp_dir", required=True)
    parser.add_argument("--max_items", type=int)
    parser.add_argument(
        "--max_frames", type=int, default=0,
        help="If positive, encode only the center max_frames frames of each PBR tar.",
    )
    parser.add_argument(
        "--shape_audit_every", type=int, default=0,
        help="Deterministically download/compare shape coordinates for about one task in N; 0 disables.",
    )
    parser.add_argument("--pipeline_mode", choices=["serial", "three_stage"], default="three_stage")
    parser.add_argument(
        "--prefetch_tasks", type=int, default=2,
        help="Number of task preparations allowed in flight in three-stage mode.",
    )
    parser.add_argument(
        "--save_queue_depth", type=int, default=2,
        help="Maximum encoded task outputs waiting for background save/upload.",
    )
    parser.add_argument("--upload_workers", type=int, default=1)
    parser.add_argument(
        "--progress_sync_every", type=int, default=1,
        help="Mirror progress to S3 every N completed tasks; outputs are always uploaded immediately.",
    )
    parser.add_argument("--encoder", default="microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("rank must satisfy 0 <= rank < world_size")
    if args.max_frames < 0 or args.shape_audit_every < 0:
        raise SystemExit("--max_frames and --shape_audit_every must be non-negative")
    if min(args.prefetch_tasks, args.save_queue_depth, args.upload_workers, args.progress_sync_every) < 1:
        raise SystemExit("pipeline depths, worker counts, and progress sync interval must be positive")
    if args.shape_audit_every and not args.s3_shape_root:
        raise SystemExit("--s3_shape_root is required when --shape_audit_every is enabled")
    state_dir = Path(args.state_dir) / f"rank_{args.rank}"
    args.tmp_dir = str(Path(args.tmp_dir) / f"rank_{args.rank}")
    state_dir.mkdir(parents=True, exist_ok=True)
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    progress_uri = f"{args.s3_output_root.rstrip('/')}/logs/pbr_encode_progress_{args.rank}.json"
    progress_path = state_dir / "progress.json"
    progress = load_progress(progress_uri, progress_path)

    all_tasks, manifest = load_task_manifest(
        args.task_manifest, state_dir / "task_manifest.json", args.resolution
    )
    # Freeze assignment before consulting mutable completion state. Round-robin
    # sharding spreads lexicographically clustered objects across ranks.
    shard_ids = all_tasks[args.rank :: args.world_size]
    completed_outputs = list_object_tasks(
        args.s3_output_root, args.resolution, ".npz", missing_ok=True
    )
    shard_ids = [task_id for task_id in shard_ids if task_id not in completed_outputs]
    pending = [
        (
            task_id,
            f"{args.s3_input_root.rstrip('/')}/{args.resolution}/{task_id}.tar",
        )
        for task_id in shard_ids
        if progress.get(task_id, {}).get("status") not in TERMINAL
    ]
    if args.max_items is not None:
        pending = pending[:args.max_items]
    print(
        f"[main] manifest={args.task_manifest} source={manifest.get('source')} "
        f"checksum={manifest.get('task_checksum_sha256')}",
        flush=True,
    )
    print(
        f"[main] eligible={len(all_tasks)} material_outputs={len(completed_outputs)} "
        f"rank_shard={len(all_tasks[args.rank::args.world_size])} "
        f"after_output_filter={len(shard_ids)} pending={len(pending)} "
        f"rank={args.rank}/{args.world_size}",
        flush=True,
    )
    if not pending:
        return

    torch.cuda.set_device(0)
    encoder = models.from_pretrained(args.encoder).eval().cuda()
    completed = 0

    def record_result(result: dict) -> None:
        nonlocal completed
        task_id = result["task"]
        previous = progress.get(task_id, {})
        result["attempts"] = int(previous.get("attempts", 0)) + 1
        progress[task_id] = result
        completed += 1
        if completed % args.progress_sync_every == 0:
            save_progress(progress, progress_uri, progress_path)
        timing = (
            f"prep={result.get('t_download_pbr', 0) + result.get('t_extract', 0) + result.get('t_read_prepare', 0):.2f}s "
            f"gpu={result.get('t_encode', 0):.2f}s save={result.get('t_save', 0):.2f}s "
            f"upload={result.get('t_upload', 0):.2f}s"
        )
        print(
            f"[{completed}/{len(pending)}] {task_id} {result['status']} "
            f"elapsed={result.get('seconds', '')}s {timing}",
            flush=True,
        )

    started = time.perf_counter()
    try:
        if args.pipeline_mode == "serial":
            for task_id, tar_uri in pending:
                record_result(encode_task(task_id, tar_uri, args, encoder))
        else:
            run_three_stage(pending, args, encoder, record_result)
    finally:
        if completed:
            save_progress(progress, progress_uri, progress_path)
    print(
        f"[main] done mode={args.pipeline_mode} completed={completed} "
        f"wall_seconds={time.perf_counter() - started:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
