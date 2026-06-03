"""
Prepare a manifest of TexVerse-Animation zips for rendering.

Given:
  - id_list_txt:    list of target ids (one per line, no extension)
  - model_paths_txt: lines like `models/<shard>/<id>.zip` (relative to data_root)
  - data_root:      e.g. /local-ssd/data/texverse_1k_animate

For each target id present in model_paths, open its zip and pick the main
model file inside. Entries are tagged with a `stage` field so downstream
launchers can run them in waves with different renderer support:

  - stage="A"             : top-level .fbx / .glb / .gltf (current renderer)
  - stage="B-1-nested"    : top-level only nested .zip, but inner zip
                            contains .fbx / .glb / .gltf
  - stage="B-2-blend"     : top-level .blend (needs blend importer)
  - stage="B-3-archive"   : .7z / .rar inside (needs 7z/unrar)

Anything else is recorded in `skipped` (no model / unsupported / corrupted /
nested zip where inner is also .blend or another archive, etc.).

Output JSON schema (manifest entries are compatible with
`batch_render_texverse_animate.py`):

  {
    "metadata": {...stats...},
    "manifest": [
      {
        "id": "<id>",
        "stage": "A" | "B-1-nested" | "B-2-blend" | "B-3-archive",
        "zip_path": "/abs/path/to/<id>.zip",
        "file_path_in_zip": "source/foo.fbx" | "source/foo.zip" | "source/foo.blend" | "source/foo.7z",
        "extension": ".fbx" | ".glb" | ".gltf" | ".blend",
        "nested_zip_path": "EgyptianStyleTreasureChest.fbx",  # only for B-1
        "archive_kind": "7z" | "rar",                          # only for B-3
        "source": "texverse_animate"
      },
      ...
    ],
    "skipped": [
      {"id": "...", "reason": "...", "detail": "...optional..."},
      ...
    ]
  }
"""

import argparse
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed


DIRECT_EXTENSIONS_PRIORITY = [".glb", ".gltf", ".fbx"]  # stage A
BLEND_EXT = ".blend"                                     # stage B-2
NESTED_ZIP_EXT = ".zip"                                  # stage B-1 trigger
ARCHIVE_EXTS = (".7z", ".rar")                           # stage B-3
DAE_OBJ = {".dae", ".obj"}                               # always skipped
ALL_KNOWN_EXTENSIONS = (set(DIRECT_EXTENSIONS_PRIORITY)
                        | {BLEND_EXT, NESTED_ZIP_EXT}
                        | set(ARCHIVE_EXTS)
                        | DAE_OBJ)


def _index_namelist(namelist):
    """Group entries in a zip namelist by lowercase extension."""
    by_ext = {}
    for name in namelist:
        if name.endswith("/"):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in ALL_KNOWN_EXTENSIONS:
            by_ext.setdefault(ext, []).append(name)
    return by_ext


def _pick_first(files):
    return sorted(files, key=lambda p: (p.count("/"), len(p), p))[0]


def _peek_nested_zip(outer_zf, nested_zip_member):
    """Open one level of nested zip; return its sorted namelist, or None on failure."""
    try:
        with outer_zf.open(nested_zip_member) as fp:
            data = fp.read()
        import io
        with zipfile.ZipFile(io.BytesIO(data)) as inner:
            return inner.namelist()
    except Exception:
        return None


def classify_zip(arg):
    """Worker: open one outer zip and decide its stage.

    Returns one of:
        ("ok",         id, zip_path, entry_dict)                 # any stage A/B-*
        ("skip",       id, zip_path, reason, detail)
    """
    obj_id, zip_path = arg
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            by_ext = _index_namelist(names)

            # ---- stage A: directly-renderable model at top level ----
            for ext in DIRECT_EXTENSIONS_PRIORITY:
                if ext in by_ext:
                    return ("ok", obj_id, zip_path, {
                        "id": obj_id,
                        "stage": "A",
                        "zip_path": zip_path,
                        "file_path_in_zip": _pick_first(by_ext[ext]),
                        "extension": ext,
                        "source": "texverse_animate",
                    })

            # ---- stage B-2: top-level .blend ----
            if BLEND_EXT in by_ext:
                return ("ok", obj_id, zip_path, {
                    "id": obj_id,
                    "stage": "B-2-blend",
                    "zip_path": zip_path,
                    "file_path_in_zip": _pick_first(by_ext[BLEND_EXT]),
                    "extension": BLEND_EXT,
                    "source": "texverse_animate",
                })

            # ---- stage B-1: nested .zip whose inner has a direct-renderable ----
            if NESTED_ZIP_EXT in by_ext:
                # Try the first inner zip (sorted by shallowness/length) first.
                inner_zip_name = _pick_first(by_ext[NESTED_ZIP_EXT])
                inner_names = _peek_nested_zip(zf, inner_zip_name)
                if inner_names is None:
                    return ("skip", obj_id, zip_path,
                            "nested_zip_open_failed", f"inner={inner_zip_name}")
                inner_by_ext = _index_namelist(inner_names)
                # User decision: B-1 ONLY accepts fbx/glb/gltf (not .blend) for now.
                for ext in DIRECT_EXTENSIONS_PRIORITY:
                    if ext in inner_by_ext:
                        return ("ok", obj_id, zip_path, {
                            "id": obj_id,
                            "stage": "B-1-nested",
                            "zip_path": zip_path,
                            "file_path_in_zip": inner_zip_name,  # the inner .zip member
                            "extension": ext,
                            "nested_inner_path": _pick_first(inner_by_ext[ext]),
                            "source": "texverse_animate",
                        })
                # Inner zip exists but has no renderable. Skip.
                inner_kinds = {k: len(v) for k, v in inner_by_ext.items()}
                return ("skip", obj_id, zip_path,
                        "nested_no_renderable", f"inner_ext={inner_kinds}")

            # ---- stage B-3: top-level .7z or .rar ----
            for kind_ext in ARCHIVE_EXTS:
                if kind_ext in by_ext:
                    return ("ok", obj_id, zip_path, {
                        "id": obj_id,
                        "stage": "B-3-archive",
                        "zip_path": zip_path,
                        "file_path_in_zip": _pick_first(by_ext[kind_ext]),
                        "extension": kind_ext,
                        "archive_kind": kind_ext.lstrip("."),
                        "source": "texverse_animate",
                    })

            # ---- nothing usable ----
            if DAE_OBJ & set(by_ext.keys()):
                only = sorted(DAE_OBJ & set(by_ext.keys()))
                return ("skip", obj_id, zip_path,
                        "unsupported_ext", f"only_found={','.join(only)}")
            return ("skip", obj_id, zip_path, "no_model_in_zip", "")

    except Exception as e:
        return ("skip", obj_id, zip_path,
                "zip_open_failed", f"{type(e).__name__}: {e}")


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
    stage_counts = {}
    reason_counts = {"not_in_model_paths": len(missing_in_model_paths)}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(classify_zip, c): c[0] for c in candidates}
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if done % 2000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"[manifest] inspected {done}/{len(candidates)} "
                      f"({rate:.0f} zip/s, ok={len(manifest)})")
            if res[0] == "ok":
                _, obj_id, zip_path, entry = res
                manifest.append(entry)
                stage_counts[entry["stage"]] = stage_counts.get(entry["stage"], 0) + 1
            else:
                _, obj_id, zip_path, reason, detail = res
                skipped.append({"id": obj_id, "reason": reason,
                                "detail": detail, "zip_path": zip_path})
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    elapsed = time.time() - t0
    print(f"[manifest] inspection done in {elapsed:.1f}s")
    print(f"[manifest] manifest entries: {len(manifest)}")
    print(f"[manifest] per-stage counts : {stage_counts}")
    print(f"[manifest] skip reasons     : {reason_counts}")

    manifest.sort(key=lambda x: (x["stage"], x["id"]))

    output_data = {
        "metadata": {
            "data_root": args.data_root,
            "id_list": args.id_list,
            "model_paths": args.model_paths,
            "target_id_count": len(target_ids),
            "candidate_count": len(candidates),
            "manifest_count": len(manifest),
            "skipped_count": len(skipped),
            "stage_counts": stage_counts,
            "skip_reasons": reason_counts,
            "stages": {
                "A":            "top-level .fbx/.glb/.gltf, renderable today",
                "B-1-nested":   "outer .zip wraps a nested .zip with .fbx/.glb/.gltf inside",
                "B-2-blend":    "top-level .blend (needs blend importer)",
                "B-3-archive":  "top-level .7z or .rar (needs 7z/unrar)",
            },
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
