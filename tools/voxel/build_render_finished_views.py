"""
Scan dynamic_obj_rendered progress logs on S3 and emit a flat list of
`"<sha256>/view_XX"` strings for every view whose render status is "success".

Output is a JSON file shaped exactly like:

    ["sha256_a/view_00", "sha256_a/view_02", "sha256_b/view_00", ...]

so downstream voxelization can read it directly. Entries are deduplicated and
sorted for reproducibility.

Usage (on the remote machine):

    python tools/voxel/build_render_finished_views.py \
        --logs_prefix s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/objxl/dynamic_obj_rendered/logs/ \
        --output_s3   s3://arcwm-code-us-west-2/yanruibin/efs/4D_video_data_process/data/objxl/render_finished_view.json \
        --local_cache /local-ssd/dynamic_obj_progress_scan \
        --local_out   /local-ssd/render_finished_view.json \
        --num_workers 16
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_aws(args, check: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(["aws"] + args, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"aws {' '.join(args)} failed rc={proc.returncode}\n"
            f"stdout={proc.stdout[-500:]}\nstderr={proc.stderr[-500:]}"
        )
    return proc


def list_progress_files(logs_prefix: str) -> list:
    """Return list of s3://... uris pointing at progress_*.json under logs_prefix."""
    if not logs_prefix.endswith("/"):
        logs_prefix = logs_prefix + "/"
    proc = run_aws(["s3", "ls", logs_prefix], check=True)
    # Each line: "2026-06-01 12:34:56     12345 progress_0.json"
    bucket_prefix = logs_prefix  # already ends with /
    uris = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[-1]
        if name.startswith("progress_") and name.endswith(".json"):
            uris.append(bucket_prefix + name)
    return sorted(uris)


def download_one(s3_uri: str, local_dir: str) -> tuple:
    fname = s3_uri.rsplit("/", 1)[-1]
    local_path = os.path.join(local_dir, fname)
    proc = run_aws(["s3", "cp", "--only-show-errors", s3_uri, local_path])
    if proc.returncode != 0:
        return s3_uri, local_path, False, proc.stderr.strip()[:300]
    return s3_uri, local_path, True, ""


def parse_progress_file(local_path: str) -> tuple:
    """Return (success_keys: set[str], status_counts: dict[str, int])."""
    success = set()
    status_counts = {}
    try:
        with open(local_path) as f:
            data = json.load(f)
    except Exception as e:
        return success, status_counts, f"parse_failed: {e}"
    if not isinstance(data, dict):
        return success, status_counts, "not_a_dict"
    for view_key, entry in data.items():
        status = (entry or {}).get("status", "unknown") if isinstance(entry, dict) else "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "success":
            success.add(view_key)
    return success, status_counts, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_prefix", type=str, required=True,
                        help="S3 prefix containing progress_*.json")
    parser.add_argument("--output_s3", type=str, required=True,
                        help="S3 URI to upload the resulting JSON list to")
    parser.add_argument("--local_cache", type=str, default="/local-ssd/dynamic_obj_progress_scan",
                        help="Local scratch dir for downloaded progress files")
    parser.add_argument("--local_out", type=str, default="/local-ssd/render_finished_view.json",
                        help="Local path the JSON is written to before s3 upload")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Parallelism for aws s3 cp")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip uploading the result to S3")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    if not args.logs_prefix.startswith("s3://"):
        raise SystemExit("--logs_prefix must be an s3:// URI")
    if not args.output_s3.startswith("s3://"):
        raise SystemExit("--output_s3 must be an s3:// URI")

    os.makedirs(args.local_cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.local_out)), exist_ok=True)

    t0 = time.time()
    print(f"[scan] listing {args.logs_prefix}")
    progress_uris = list_progress_files(args.logs_prefix)
    print(f"[scan] found {len(progress_uris)} progress_*.json files")
    if not progress_uris:
        raise SystemExit("No progress_*.json files found under --logs_prefix")

    # Parallel download.
    print(f"[scan] downloading with {args.num_workers} workers -> {args.local_cache}")
    local_files = []
    download_failures = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(download_one, uri, args.local_cache): uri for uri in progress_uris}
        for fut in as_completed(futs):
            uri, local_path, ok, err = fut.result()
            if ok:
                local_files.append((uri, local_path))
            else:
                download_failures.append((uri, err))
                print(f"[scan] DOWNLOAD FAILED {uri}: {err}")
    print(f"[scan] downloaded {len(local_files)}/{len(progress_uris)} files in {time.time() - t0:.1f}s")
    if download_failures:
        print(f"[scan] {len(download_failures)} downloads failed; "
              f"resulting list will be missing those ranks")

    # Parse + merge.
    all_success = set()
    global_status_counts = {}
    per_file_summary = []
    parse_failures = []
    for uri, local_path in sorted(local_files):
        success, status_counts, err = parse_progress_file(local_path)
        if err:
            parse_failures.append((uri, err))
            print(f"[scan] PARSE FAILED {uri}: {err}")
            continue
        before = len(all_success)
        all_success.update(success)
        added = len(all_success) - before
        for k, v in status_counts.items():
            global_status_counts[k] = global_status_counts.get(k, 0) + v
        per_file_summary.append((uri.rsplit("/", 1)[-1], sum(status_counts.values()),
                                 status_counts.get("success", 0), added))

    # Print summary.
    print("\n[scan] per-rank summary (filename, total_entries, success, newly_added_to_union):")
    for name, total, succ, added in per_file_summary:
        print(f"  {name:30s} total={total:>7d} success={succ:>7d} added={added:>7d}")
    print(f"\n[scan] global status counts (sum across rank files, NOT deduped):")
    for k, v in sorted(global_status_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:25s} {v}")

    finished_list = sorted(all_success)
    n_unique_objs = len({k.split("/")[0] for k in finished_list})
    print(f"\n[scan] unique success views    : {len(finished_list)}")
    print(f"[scan] unique objs with success: {n_unique_objs}")

    # Write local.
    with open(args.local_out, "w") as f:
        json.dump(finished_list, f)
    size_mb = os.path.getsize(args.local_out) / 1024 / 1024
    print(f"\n[scan] wrote {args.local_out} ({size_mb:.2f} MB)")

    # Upload.
    if args.dry_run:
        print("[scan] --dry_run set, skipping s3 upload")
        return
    print(f"[scan] uploading -> {args.output_s3}")
    proc = run_aws(["s3", "cp", "--only-show-errors", args.local_out, args.output_s3])
    if proc.returncode != 0:
        raise SystemExit(f"upload failed: {proc.stderr[:500]}")
    print(f"[scan] DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
