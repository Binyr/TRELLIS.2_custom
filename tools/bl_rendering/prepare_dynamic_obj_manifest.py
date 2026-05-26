"""
Prepare a manifest of dynamic OBJ files for rendering.

Scans all zip files downloaded by objaverse.xl, reads .objaverse-file-hashes.json
from each zip, and builds a mapping: sha256 -> (zip_path, file_path_in_zip).
Intersects with the target sha256 list and outputs a manifest JSON.

Usage:
    python tools/bl_rendering/prepare_dynamic_obj_manifest.py \
        --sha256_list claude_tmp/uuid_github_intersection_sha256.txt \
        --data_root /threed-code/yanruibin/trellis.2_data/dynamic_obj/raw/github/repos \
        --output /threed-code/yanruibin/efs/4D_video_data_process/data/objxl/dynamic_obj_manifest.json
"""

import argparse
import glob
import json
import os
import sys
import time
import zipfile
from pathlib import Path


SUPPORTED_EXTENSIONS = {".fbx", ".glb", ".gltf", ".dae", ".obj"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha256_list", type=str, required=True,
                        help="Path to sha256 list file (one per line)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing repos/<user>/<repo>.zip")
    parser.add_argument("--output", type=str, required=True,
                        help="Output manifest JSON path")
    args = parser.parse_args()

    # Load target sha256 set
    with open(args.sha256_list) as f:
        target_sha256 = set(line.strip() for line in f if line.strip())
    print(f"[manifest] Target sha256 count: {len(target_sha256)}")

    # Find all zip files
    zip_pattern = os.path.join(args.data_root, "*", "*.zip")
    zip_files = sorted(glob.glob(zip_pattern))
    print(f"[manifest] Found {len(zip_files)} zip files")

    # Scan zips and build manifest
    manifest = {}  # sha256 -> {zip_path, file_path_in_zip, file_identifier, extension}
    errors = []
    t0 = time.time()

    for i, zip_path in enumerate(zip_files):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"[manifest] Scanned {i+1}/{len(zip_files)} zips ({elapsed:.1f}s)")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Find .objaverse-file-hashes.json
                hash_files = [n for n in zf.namelist() if n.endswith(".objaverse-file-hashes.json")]
                if not hash_files:
                    continue

                with zf.open(hash_files[0]) as hf:
                    hashes = json.load(hf)

                namelist_set = set(zf.namelist())

                for item in hashes:
                    sha = item.get("sha256", "")
                    if sha not in target_sha256:
                        continue

                    file_id = item.get("fileIdentifier", "")
                    # Extract relative path from GitHub URL
                    # Format: https://github.com/<user>/<repo>/blob/<commit>/<path>
                    if "/blob/" in file_id:
                        relative_path = file_id.split("/blob/")[1].split("/", 1)[-1]
                    else:
                        continue

                    ext = os.path.splitext(relative_path)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        continue

                    # Check if file exists in zip
                    if relative_path not in namelist_set:
                        continue

                    manifest[sha] = {
                        "sha256": sha,
                        "zip_path": zip_path,
                        "file_path_in_zip": relative_path,
                        "file_identifier": file_id,
                        "extension": ext,
                    }

        except (zipfile.BadZipFile, Exception) as e:
            errors.append({"zip_path": zip_path, "error": str(e)})

    elapsed = time.time() - t0
    print(f"[manifest] Scan complete in {elapsed:.1f}s")
    print(f"[manifest] Matched: {len(manifest)} / {len(target_sha256)} target sha256")
    print(f"[manifest] Errors: {len(errors)}")

    # Extension distribution
    ext_counts = {}
    for item in manifest.values():
        ext = item["extension"]
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print(f"[manifest] Extension distribution: {ext_counts}")

    # Missing sha256
    found_sha256 = set(manifest.keys())
    missing_sha256 = target_sha256 - found_sha256
    print(f"[manifest] Missing (not found in any zip): {len(missing_sha256)}")

    # Save manifest
    output_data = {
        "metadata": {
            "target_sha256_count": len(target_sha256),
            "matched_count": len(manifest),
            "missing_count": len(missing_sha256),
            "extension_distribution": ext_counts,
            "data_root": args.data_root,
            "sha256_list": args.sha256_list,
            "scan_errors": len(errors),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "manifest": list(manifest.values()),
        "errors": errors[:100],  # Keep first 100 errors for debugging
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[manifest] Saved to: {args.output}")


if __name__ == "__main__":
    main()
