#!/usr/bin/env python3
"""Resumable material-VAE encoder for PBR O-Voxel view tars on S3.

Each output NPZ contains deterministic posterior statistics, aligned to the
paired shape-latent coordinates. New outputs also preserve the mapping from
the contiguous encoded frame IDs back to source animation frames:
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


def coord_codes(coords: torch.Tensor, latent_resolution: int) -> torch.Tensor:
    xyz = coords[:, -3:].long()
    return (xyz[:, 0] * latent_resolution + xyz[:, 1]) * latent_resolution + xyz[:, 2]


def align_to_shape(z, mean: torch.Tensor, logvar: torch.Tensor, shape_coords: np.ndarray, latent_resolution: int):
    ref = torch.from_numpy(shape_coords).to(z.coords.device).int()
    got = z.coords[:, 1:].int()
    ref_codes = coord_codes(ref, latent_resolution)
    got_codes = coord_codes(got, latent_resolution)
    ref_order = torch.argsort(ref_codes)
    got_order = torch.argsort(got_codes)
    if len(ref_codes) != len(got_codes) or not torch.equal(ref_codes[ref_order], got_codes[got_order]):
        raise CoordinateMismatch(
            f"coordinate sets differ: shape={len(ref_codes)} material={len(got_codes)}"
        )
    mean_out = torch.empty_like(mean)
    logvar_out = torch.empty_like(logvar)
    mean_out[ref_order] = mean[got_order]
    logvar_out[ref_order] = logvar[got_order]
    return mean_out, logvar_out


def encode_task(task_id: str, tar_uri: str, args, encoder) -> dict:
    task_dir = Path(args.tmp_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    tar_path = task_dir / "input.tar"
    shape_path = task_dir / "shape.npz"
    meta_path = task_dir / "pbr_meta.json"
    out_path = task_dir / "material.npz"
    shape_uri = f"{args.s3_shape_root.rstrip('/')}/{args.resolution}/{task_id}.npz"
    out_uri = f"{args.s3_output_root.rstrip('/')}/{args.resolution}/{task_id}.npz"
    started = time.perf_counter()
    try:
        # The manifest is shape-conditioned, so check/download the smaller
        # shape latent before pulling the much larger PBR tar.
        if not s3_get(shape_uri, shape_path):
            return {"status": "missing_shape_latent", "task": task_id}
        if not s3_get(tar_uri, tar_path):
            return {"status": "missing_pbr_tar", "task": task_id}
        with tarfile.open(tar_path) as tf:
            tf.extractall(task_dir)
        frames = sorted(task_dir.glob("*.vxz"))
        if not frames:
            return {"status": "no_vxz", "task": task_id}
        frame_ids = [frame_path.stem for frame_path in frames]
        try:
            frame_numbers = np.asarray([int(frame_id) for frame_id in frame_ids], dtype=np.int32)
        except ValueError as exc:
            raise FrameMetadataMismatch(f"non-numeric PBR frame IDs: {frame_ids[:10]}") from exc

        if args.frame_mapping == "meta":
            meta_uri = tar_uri[:-4] + "_meta.json"
            if not s3_get(meta_uri, meta_path):
                return {"status": "missing_frame_meta", "task": task_id, "meta_uri": meta_uri}
            try:
                meta = json.loads(meta_path.read_text())
                source_frame_indices = np.asarray(meta["frame_sel"], dtype=np.int32)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise FrameMetadataMismatch(f"invalid frame_sel in {meta_uri}: {exc}") from exc
            expected_ids = np.arange(len(source_frame_indices), dtype=np.int32)
            if not np.array_equal(frame_numbers, expected_ids):
                raise FrameMetadataMismatch(
                    f"PBR frame IDs are not contiguous 0..K-1: ids={frame_numbers.tolist()[:20]} "
                    f"frame_sel_len={len(source_frame_indices)}"
                )
        else:
            source_frame_indices = frame_numbers.copy()

        with np.load(shape_path) as shape:
            shape_frame_ids = sorted(
                key[len("shape_coords_") :]
                for key in shape.files
                if key.startswith("shape_coords_")
            )
            if frame_ids != shape_frame_ids:
                only_pbr = sorted(set(frame_ids) - set(shape_frame_ids))
                only_shape = sorted(set(shape_frame_ids) - set(frame_ids))
                raise FrameSetMismatch(
                    f"frame sets differ: pbr={len(frame_ids)} shape={len(shape_frame_ids)} "
                    f"pbr_only={only_pbr[:20]} shape_only={only_shape[:20]}"
                )

            output: dict[str, np.ndarray] = {
                "num_frames": np.int32(len(frames)),
                "frame_ids": frame_numbers,
                "source_frame_indices": source_frame_indices,
            }
            latent_resolution = args.resolution // 16
            for frame_path in frames:
                frame_id = frame_path.stem
                coord_key = f"shape_coords_{frame_id}"
                coords, attrs = o_voxel.io.read_vxz(str(frame_path), num_threads=4)
                x = sparse_input(coords, attrs).cuda()
                with torch.inference_mode():
                    z, mean, logvar = encoder(x, sample_posterior=False, return_raw=True)
                mean, logvar = align_to_shape(z, mean, logvar, shape[coord_key], latent_resolution)
                output[f"pbr_mean_{frame_id}"] = mean.detach().cpu().numpy().astype(np.float32)
                output[f"pbr_logvar_{frame_id}"] = logvar.detach().cpu().numpy().astype(np.float32)
                output[f"pbr_coords_{frame_id}"] = shape[coord_key].astype(np.uint8, copy=False)
        torch.cuda.synchronize()
        np.savez_compressed(out_path, **output)
        if not s3_put(out_path, out_uri):
            return {"status": "upload_failed", "task": task_id}
        return {"status": "success", "task": task_id, "num_frames": len(frames),
                "output_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
                "seconds": round(time.perf_counter() - started, 2)}
    except FrameSetMismatch as exc:
        return {"status": "frame_set_mismatch", "task": task_id, "error": str(exc)[:500]}
    except FrameMetadataMismatch as exc:
        return {"status": "frame_metadata_mismatch", "task": task_id, "error": str(exc)[:500]}
    except CoordinateMismatch as exc:
        return {"status": "coord_mismatch", "task": task_id, "error": str(exc)[:500]}
    except Exception as exc:
        return {"status": "encode_error", "task": task_id, "error": repr(exc)[:500]}
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_input_root", required=True)
    parser.add_argument("--s3_shape_root", required=True)
    parser.add_argument("--s3_output_root", required=True)
    parser.add_argument("--task_manifest", required=True)
    parser.add_argument("--frame_mapping", choices=["identity", "meta"], required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--state_dir", required=True)
    parser.add_argument("--tmp_dir", required=True)
    parser.add_argument("--max_items", type=int)
    parser.add_argument("--encoder", default="microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("rank must satisfy 0 <= rank < world_size")
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
    for index, (task_id, tar_uri) in enumerate(pending, 1):
        result = encode_task(task_id, tar_uri, args, encoder)
        previous = progress.get(task_id, {})
        result["attempts"] = int(previous.get("attempts", 0)) + 1
        progress[task_id] = result
        save_progress(progress, progress_uri, progress_path)
        print(f"[{index}/{len(pending)}] {task_id} {result['status']} {result.get('seconds', '')}s", flush=True)


if __name__ == "__main__":
    main()
