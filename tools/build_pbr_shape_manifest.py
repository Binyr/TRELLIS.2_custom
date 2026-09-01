#!/usr/bin/env python3
"""Build a frozen material-encode manifest from completed S3 pipeline stages.

Eligible views are the intersection of:
  * PBR voxelization progress entries whose status is ``success`` and whose
    output tar still exists; and
  * shape-encode progress entries whose status is ``success`` and whose NPZ
    still exists.

The legacy 4D shape encoder has no encode-progress directory, so
``--shape_success_mode objects`` treats the actual shape NPZ inventory as the
authoritative completed set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone


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


def split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    bucket_and_key = uri[5:].split("/", 1)
    bucket = bucket_and_key[0]
    key = bucket_and_key[1].strip("/") if len(bucket_and_key) == 2 else ""
    return bucket, key


def list_view_objects(root: str, resolution: int, suffix: str) -> set[str]:
    """List task IDs below ``root/resolution`` for objects ending in suffix."""
    prefix_uri = f"{root.rstrip('/')}/{resolution}/"
    bucket, root_key = split_s3_uri(root)
    key_prefix = f"{root_key}/{resolution}/" if root_key else f"{resolution}/"
    proc = run_aws(["s3", "ls", prefix_uri, "--recursive"], retries=3)
    if proc.returncode != 0:
        raise RuntimeError(f"cannot list {prefix_uri}: {proc.stderr.strip()[:500]}")

    tasks: set[str] = set()
    for line in proc.stdout.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) != 4:
            continue
        key = fields[3]
        if not key.startswith(key_prefix) or not key.endswith(suffix):
            continue
        rel = key[len(key_prefix) :]
        if "/view_" not in rel:
            continue
        task_id = rel[: -len(suffix)]
        if task_id:
            tasks.add(task_id)
    print(f"[objects] s3://{bucket}/{key_prefix} *{suffix}: {len(tasks)}", flush=True)
    return tasks


def list_progress_uris(prefix: str, filename_prefix: str) -> list[str]:
    prefix = prefix.rstrip("/") + "/"
    proc = run_aws(["s3", "ls", prefix], retries=3)
    if proc.returncode != 0:
        raise RuntimeError(f"cannot list progress prefix {prefix}: {proc.stderr.strip()[:500]}")
    uris = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        name = fields[-1]
        if name.startswith(filename_prefix) and name.endswith(".json"):
            uris.append(prefix + name)
    uris.sort()
    if not uris:
        raise RuntimeError(f"no {filename_prefix}*.json found under {prefix}")
    return uris


def load_success_views(prefix: str, filename_prefix: str, workers: int) -> tuple[set[str], dict]:
    uris = list_progress_uris(prefix, filename_prefix)

    def load_one(uri: str):
        proc = run_aws(["s3", "cp", "--only-show-errors", uri, "-"], retries=2)
        if proc.returncode != 0:
            return uri, None, proc.stderr.strip()[:300]
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            return uri, None, f"invalid JSON: {exc}"
        if not isinstance(payload, dict):
            return uri, None, f"expected dict, got {type(payload).__name__}"
        return uri, payload, ""

    success: set[str] = set()
    status_counts: dict[str, int] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(load_one, uri) for uri in uris]
        for future in as_completed(futures):
            uri, payload, error = future.result()
            if error:
                failures.append({"uri": uri, "error": error})
                continue
            for task_id, entry in payload.items():
                status = entry.get("status", "unknown") if isinstance(entry, dict) else "invalid"
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "success":
                    success.add(task_id)

    if failures:
        sample = failures[:3]
        raise RuntimeError(f"failed to load {len(failures)}/{len(uris)} progress files: {sample}")
    print(
        f"[progress] {prefix} files={len(uris)} unique_success={len(success)} "
        f"raw_status_counts={status_counts}",
        flush=True,
    )
    return success, {"files": len(uris), "raw_status_counts": status_counts}


def upload_json(payload: dict, uri: str) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    proc = subprocess.run(
        ["aws", "s3", "cp", "--only-show-errors", "-", uri],
        input=text,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cannot upload manifest to {uri}: {proc.stderr.strip()[:500]}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["4d", "objxl", "texa", "texb"])
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--pbr_root", required=True)
    parser.add_argument("--pbr_progress_prefix", required=True)
    parser.add_argument("--shape_root", required=True)
    parser.add_argument("--shape_success_mode", choices=["progress", "objects"], required=True)
    parser.add_argument("--shape_progress_prefix")
    parser.add_argument("--output_uri", required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.shape_success_mode == "progress" and not args.shape_progress_prefix:
        raise SystemExit("--shape_progress_prefix is required with --shape_success_mode progress")

    pbr_progress_success, pbr_progress_meta = load_success_views(
        args.pbr_progress_prefix, "progress_", args.workers
    )
    pbr_objects = list_view_objects(args.pbr_root, args.resolution, ".tar")
    pbr_ready = pbr_progress_success & pbr_objects

    shape_objects = list_view_objects(args.shape_root, args.resolution, ".npz")
    shape_progress_meta = None
    if args.shape_success_mode == "progress":
        shape_progress_success, shape_progress_meta = load_success_views(
            args.shape_progress_prefix, "encode_progress_", args.workers
        )
        shape_ready = shape_progress_success & shape_objects
    else:
        shape_progress_success = None
        shape_ready = shape_objects

    eligible = sorted(pbr_ready & shape_ready)
    checksum = hashlib.sha256(("\n".join(eligible) + "\n").encode()).hexdigest()
    pbr_only = sorted(pbr_ready - shape_ready)
    shape_only = sorted(shape_ready - pbr_ready)
    counts = {
        "pbr_progress_success": len(pbr_progress_success),
        "pbr_objects": len(pbr_objects),
        "pbr_ready": len(pbr_ready),
        "pbr_success_missing_object": len(pbr_progress_success - pbr_objects),
        "shape_progress_success": None if shape_progress_success is None else len(shape_progress_success),
        "shape_objects": len(shape_objects),
        "shape_ready": len(shape_ready),
        "shape_success_missing_object": (
            None if shape_progress_success is None else len(shape_progress_success - shape_objects)
        ),
        "eligible": len(eligible),
        "pbr_only": len(pbr_only),
        "shape_only": len(shape_only),
    }
    payload = {
        "schema_version": 1,
        "source": args.source,
        "resolution": args.resolution,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_checksum_sha256": checksum,
        "inputs": {
            "pbr_root": args.pbr_root,
            "pbr_progress_prefix": args.pbr_progress_prefix,
            "shape_root": args.shape_root,
            "shape_success_mode": args.shape_success_mode,
            "shape_progress_prefix": args.shape_progress_prefix,
        },
        "progress_metadata": {
            "pbr": pbr_progress_meta,
            "shape": shape_progress_meta,
        },
        "counts": counts,
        "difference_samples": {
            "pbr_only": pbr_only[:50],
            "shape_only": shape_only[:50],
            "pbr_success_missing_object": sorted(pbr_progress_success - pbr_objects)[:50],
            "shape_success_missing_object": (
                [] if shape_progress_success is None else sorted(shape_progress_success - shape_objects)[:50]
            ),
        },
        "tasks": eligible,
    }
    print(f"[manifest] source={args.source} counts={counts}", flush=True)
    print(f"[manifest] checksum={checksum}", flush=True)
    upload_json(payload, args.output_uri)
    print(f"[manifest] uploaded: {args.output_uri}", flush=True)


if __name__ == "__main__":
    main()
