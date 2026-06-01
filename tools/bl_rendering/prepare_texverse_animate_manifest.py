"""
Prepare a manifest of TexVerse-Animation zips for rendering.

Given:
  - id_list_txt:    list of target ids (one per line, no extension)
  - model_paths_txt: lines like `models/<shard>/<id>.zip` (relative to data_root)
  - data_root:      e.g. /local-ssd/data/texverse_1k_animate

For each target id present in model_paths, open its zip and pick the main
model file inside (preferring formats Blender can ingest directly). Filter
out zips whose only model is .blend / nested .zip / .dae / .obj since the
current `dynamic_obj_rendering.py` does not support those (handled in a
follow-up stage).

Output JSON schema (manifest entries are compatible with
`batch_render_texverse_animate.py`):

  {
    "metadata": {...stats...},
    "manifest": [
      {
        "id": "<id>",
        "zip_path": "/abs/path/to/<id>.zip",
        "file_path_in_zip": "source/foo.fbx",
        "extension": ".fbx",
        "source": "texverse_animate"
      },
      ...
    ],
    "skipped": [
      {"id": "...", "reason": "not_in_model_paths" | "unsupported_ext" |
                                "no_model_in_zip" | "zip_open_failed",
       "detail": "...optional..."},
      ...
    ]
  }

Usage (on the remote machine):
    python tools/bl_rendering/prepare_texverse_animate_manifest.py \
        --data_root /local-ssd/data/texverse_1k_animate \
        --id_list   /local-ssd/data/texverse_1k_animate/TexVerse-Animation_id_list.txt \
        --model_paths /local-ssd/data/texverse_1k_animate/model_paths.txt \
        --output    /local-ssd/data/texverse_1k_animate/texverse_animate_manifest.json \
        --num_workers 16
"""

import argparse
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed


SUPPORTED_EXTENSIONS_PRIORITY = [".glb", ".gltf", ".fbx"]
DEFERRED_EXTENSIONS = {".blend", ".zip", ".dae", ".obj"}
ALL_KNOWN_EXTENSIONS = set(SUPPORTED_EXTENSIONS_PRIORITY) | DEFERRED_EXTENSIONS


def pick_main_file(namelist):
    """Pick best model file from a zip namelist. Returns (file_path, ext) or (None, None)."""
    by_ext = {}
    for name in namelist:
        if name.endswith("/"):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in ALL_KNOWN_EXTENSIONS:
            by_ext.setdefault(ext, []).append(name)

    for ext in SUPPORTED_EXTENSIONS_PRIORITY:
        if ext in by_ext:
            files = sorted(by_ext[ext], key=lambda p: (p.count("/"), len(p), p))
            return files[0], ext

    for ext in DEFERRED_EXTENSIONS:
        if ext in by_ext:
            return None, ext

    return None, None


def inspect_zip(arg):
    """Worker: open zip, pick main file. Returns (id, zip_path, status, file_path, ext, detail)."""
    obj_id, zip_path = arg
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
    except Exception as e:
        return obj_id, zip_path, "zip_open_failed", None, None, f"{type(e).__name__}: {e}"

    file_path, ext = pick_main_file(names)
    if file_path is None:
        if ext is None:
            return obj_id, zip_path, "no_model_in_zip", None, None, ""
        return obj_id, zip_path, "unsupported_ext", None, ext, f"only_found={ext}"
    return obj_id, zip_path, "ok", file_path, ext, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--id_list", type=str, required=True)
    parser.add_argument("--model_paths", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0,
                        help="Debug: only process first N candidate zips.")
    args = parser.parse_args()

    with open(args.id_list) as f:
        target_ids = set(line.strip() for line in f if line.strip())
    print(f"[manifest] target ids: {len(target_ids)}")

    id_to_zip = {}
    with open(args.model_paths) as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            obj_id = os.path.splitext(os.path.basename(rel))[0]
            if obj_id in target_ids:
                id_to_zip[obj_id] = os.path.join(args.data_root, rel)
    print(f"[manifest] target ∩ model_paths: {len(id_to_zip)}")

    missing_in_model_paths = sorted(target_ids - id_to_zip.keys())
    print(f"[manifest] missing in model_paths: {len(missing_in_model_paths)}")

    candidates = sorted(id_to_zip.items())
    if args.limit > 0:
        candidates = candidates[: args.limit]
        print(f"[manifest] limit={args.limit}: processing {len(candidates)} zips")

    manifest = []
    skipped = [{"id": oid, "reason": "not_in_model_paths"} for oid in missing_in_model_paths]
    ext_counts = {}
    reason_counts = {"not_in_model_paths": len(missing_in_model_paths)}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(inspect_zip, c): c[0] for c in candidates}
        done = 0
        for fut in as_completed(futures):
            obj_id, zip_path, status, file_path, ext, detail = fut.result()
            done += 1
            if done % 2000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"[manifest] inspected {done}/{len(candidates)} "
                      f"({rate:.0f} zip/s, ok={len(manifest)})")
            if status == "ok":
                manifest.append({
                    "id": obj_id,
                    "zip_path": zip_path,
                    "file_path_in_zip": file_path,
                    "extension": ext,
                    "source": "texverse_animate",
                })
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
            else:
                skipped.append({"id": obj_id, "reason": status,
                                "detail": detail, "zip_path": zip_path})
                reason_counts[status] = reason_counts.get(status, 0) + 1

    elapsed = time.time() - t0
    print(f"[manifest] inspection done in {elapsed:.1f}s")
    print(f"[manifest] manifest entries: {len(manifest)}")
    print(f"[manifest] ext distribution : {ext_counts}")
    print(f"[manifest] skip reasons     : {reason_counts}")

    manifest.sort(key=lambda x: x["id"])

    output_data = {
        "metadata": {
            "data_root": args.data_root,
            "id_list": args.id_list,
            "model_paths": args.model_paths,
            "target_id_count": len(target_ids),
            "candidate_count": len(candidates),
            "manifest_count": len(manifest),
            "skipped_count": len(skipped),
            "extension_distribution": ext_counts,
            "skip_reasons": reason_counts,
            "supported_extensions": SUPPORTED_EXTENSIONS_PRIORITY,
            "deferred_extensions": sorted(DEFERRED_EXTENSIONS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "manifest": manifest,
        "skipped": skipped,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[manifest] saved to {args.output}")


if __name__ == "__main__":
    main()
