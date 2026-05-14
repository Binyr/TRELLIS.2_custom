#!/usr/bin/env python3
"""
voxelize_pbr_v2.py - Voxelize 4D animated objects PBR frame-by-frame.

Combines pbr_shared pickle (materials, faces, UVs, mat_ids) with
result_mesh.npz (vertices_seq per frame) to produce per-frame .vxz files
via o_voxel.convert.blender_dump_to_volumetric_attr().

Features:
- Multi-process via Pool(maxtasksperchild=1)
- Checkpoint-based resume via progress json
- Status log with timing/ETA
- Filter-before-shard for balanced workload
- Priority list support

Usage:
    python data_toolkit/voxelize_pbr_v2.py \
        --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
        --pbr_shared_root data/trellis.2/pbr_shared \
        --rendered_root data/objverse_minghao_4d_mine_40075/rendering_v5 \
        --output_root data/trellis.2/pbr_voxels_4d \
        --log_root data/trellis.2/logs/voxelize_pbr_4d \
        --resolution 512 \
        --max_workers 8 \
        --rank 0 --world_size 1
"""

import argparse
import copy
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
import o_voxel


def parse_entry(entry: str):
    parts = Path(entry).parts
    obj_id = parts[-1]
    shard_with_suffix = parts[-2]
    shard_id = shard_with_suffix.split('_static_camera_distance_v3')[0]
    return shard_id, obj_id


def load_progress(progress_path: str) -> dict:
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress_path: str, progress: dict):
    with open(progress_path, 'w') as f:
        json.dump(progress, f)


def append_status_log(status_log_path: str, line: str):
    """Append a line to status log. Uses read+write instead of 'a' mode for S3 compatibility."""
    existing = ''
    if os.path.exists(status_log_path):
        try:
            with open(status_log_path, 'r') as f:
                existing = f.read()
        except Exception:
            pass
    with open(status_log_path, 'w') as f:
        f.write(existing + line + '\n')


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-face normals, expanded to (F, 3, 3) for o_voxel compatibility."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
    fn = fn / norms
    return np.stack([fn, fn, fn], axis=1).astype(np.float32)


def voxelize_pbr_one_object(
    args_tuple,
    pbr_shared_root,
    rendered_root,
    output_root,
    resolutions,
):
    """Worker function: voxelize all frames of one object."""
    shard_id, obj_id = args_tuple

    # Load pbr shared data
    pbr_path = os.path.join(pbr_shared_root, shard_id, f'{obj_id}.pickle')
    if not os.path.exists(pbr_path):
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'missing_pbr', 'num_frames': 0}

    with open(pbr_path, 'rb') as f:
        pbr_shared = pickle.load(f)

    # Load mesh sequence
    rendered_dir = os.path.join(rendered_root, f'{shard_id}_static_camera_distance_v3', obj_id)
    mesh_npz_path = os.path.join(rendered_dir, 'result_mesh.npz')
    if not os.path.exists(mesh_npz_path):
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'missing_mesh', 'num_frames': 0}

    with np.load(mesh_npz_path) as mesh_data:
        vertices_seq = mesh_data['vertices'].copy()
        mesh_faces = mesh_data['faces'].copy()

    # Check face count
    num_faces = mesh_faces.shape[0]
    if num_faces > 500000:
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'skipped_too_many_faces', 'num_frames': 0, 'num_faces': num_faces}

    # Verify face consistency
    pbr_faces = pbr_shared['objects'][0]['faces']
    if mesh_faces.shape != pbr_faces.shape:
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'face_mismatch', 'num_frames': 0}

    num_frames = vertices_seq.shape[0]

    for res in resolutions:
        output_dir = os.path.join(output_root, str(res), shard_id, obj_id)
        os.makedirs(output_dir, exist_ok=True)

        for frame_idx in range(num_frames):
            output_path = os.path.join(output_dir, f'{frame_idx:06d}.vxz')
            if os.path.exists(output_path):
                continue

            frame_verts = np.clip(vertices_seq[frame_idx].astype(np.float32), -0.5, 0.5)
            normals = compute_face_normals(frame_verts, pbr_faces)

            # Build dump
            dump = copy.deepcopy(pbr_shared)
            for mat in dump['materials']:
                if mat.get('alphaTexture') is not None and mat['alphaMode'] == 'OPAQUE':
                    mat['alphaMode'] = 'BLEND'
            dump['materials'].append({
                'baseColorFactor': [0.8, 0.8, 0.8], 'alphaFactor': 1.0,
                'metallicFactor': 0.0, 'roughnessFactor': 0.5,
                'alphaMode': 'OPAQUE', 'alphaCutoff': 0.5,
                'baseColorTexture': None, 'alphaTexture': None,
                'metallicTexture': None, 'roughnessTexture': None,
            })
            obj_data = dump['objects'][0]
            obj_data['vertices'] = frame_verts
            obj_data['normals'] = normals
            obj_data['mat_ids'] = obj_data['mat_ids'].copy()
            obj_data['mat_ids'][obj_data['mat_ids'] == -1] = len(dump['materials']) - 1

            try:
                coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
                    dump, grid_size=res,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    mip_level_offset=0, verbose=False, timing=False,
                )
                del attr['normal']
                del attr['emissive']
                o_voxel.io.write_vxz(output_path, coord, attr)
            except Exception as e:
                print(f"[ERROR] voxelize_pbr failed: {shard_id}/{obj_id} frame={frame_idx} res={res}: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue
            finally:
                try:
                    del coord, attr, dump, normals
                except NameError:
                    pass

    del vertices_seq, mesh_faces, pbr_shared
    return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'success', 'num_frames': num_frames}


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--pbr_shared_root', type=str, default='data/trellis.2/pbr_shared')
    parser.add_argument('--rendered_root', type=str, default='data/objverse_minghao_4d_mine_40075/rendering_v5')
    parser.add_argument('--output_root', type=str, default='data/trellis.2/pbr_voxels_4d')
    parser.add_argument('--log_root', type=str, default='data/trellis.2/logs/voxelize_pbr_4d')
    parser.add_argument('--resolution', type=str, default='512')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'test', 'all'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=1)
    parser.add_argument('--priority_list', type=str, default=None)
    args = parser.parse_args()

    resolutions = [int(x) for x in args.resolution.split(',')]
    print(f"Resolutions: {resolutions}")

    with open(args.ann_file, 'r') as f:
        ann_data = json.load(f)

    entries = []
    if args.split in ('train', 'all'):
        entries.extend(ann_data.get('train', []))
    if args.split in ('test', 'all'):
        entries.extend(ann_data.get('test', []))

    print(f"Total entries: {len(entries)}")

    # Load progress from ALL ranks
    os.makedirs(args.log_root, exist_ok=True)
    res_tag = args.resolution.replace(',', '_')
    all_progress = {}
    if res_tag == '512':
        progress_files = [f for f in glob.glob(os.path.join(args.log_root, 'progress_*.json'))
                          if os.path.basename(f).replace('progress_', '').replace('.json', '').isdigit()]
    else:
        progress_files = glob.glob(os.path.join(args.log_root, f'progress_{res_tag}_*.json'))
    for p_path in progress_files:
        try:
            with open(p_path, 'r') as f:
                all_progress.update(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to read {p_path}: {e}")
    print(f"Loaded global progress: {len(all_progress)} completed objects from {len(progress_files)} files")

    # Filter out completed FIRST
    to_process = []
    for entry in entries:
        shard_id, obj_id = parse_entry(entry)
        obj_key = f"{shard_id}/{obj_id}"
        if obj_key in all_progress and all_progress[obj_key].get('status') == 'success':
            continue
        to_process.append((shard_id, obj_id))

    print(f"To process (after filtering): {len(to_process)}")

    # Sort by priority
    if args.priority_list and os.path.exists(args.priority_list):
        with open(args.priority_list, 'r') as f:
            priority_ids = set(line.strip() for line in f if line.strip())
        priority_objs = [(s, o) for s, o in to_process if o in priority_ids]
        non_priority_objs = [(s, o) for s, o in to_process if o not in priority_ids]
        to_process = priority_objs + non_priority_objs
        print(f"Priority list: {len(priority_ids)} ids, {len(priority_objs)} matched in to_process")

    # THEN shard
    start = len(to_process) * args.rank // args.world_size
    end = len(to_process) * (args.rank + 1) // args.world_size
    to_process = to_process[start:end]
    print(f"Rank {args.rank}/{args.world_size}: assigned {len(to_process)} entries")

    # Per-rank progress
    progress_path = os.path.join(args.log_root, f'progress_{args.rank}.json') if res_tag == '512' else os.path.join(args.log_root, f'progress_{res_tag}_{args.rank}.json')
    progress = load_progress(progress_path)

    if len(to_process) == 0:
        print("Nothing to do.")
        return

    status_log_path = os.path.join(args.log_root, f'status_{args.rank}.log') if res_tag == '512' else os.path.join(args.log_root, f'status_{res_tag}_{args.rank}.log')
    total_to_process = len(to_process)
    completed_count = 0
    start_time = time.time()

    worker_fn = partial(
        voxelize_pbr_one_object,
        pbr_shared_root=args.pbr_shared_root,
        rendered_root=args.rendered_root,
        output_root=args.output_root,
        resolutions=resolutions,
    )

    if args.max_workers <= 1:
        for item in to_process:
            result = worker_fn(item)
            obj_key = f"{result['shard_id']}/{result['obj_id']}"
            progress[obj_key] = result
            save_progress(progress_path, progress)
            completed_count += 1
            elapsed = time.time() - start_time
            avg = elapsed / completed_count
            eta = avg * (total_to_process - completed_count)
            append_status_log(status_log_path, f"{obj_key} {result['status']} frames={result.get('num_frames', 0)} done={completed_count}/{total_to_process} avg={avg:.1f}s/obj eta={eta:.0f}s")
            print(f"[{completed_count}/{total_to_process}] {obj_key} {result['status']} avg={avg:.1f}s eta={eta:.0f}s")
    else:
        with Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
            results_iter = pool.imap_unordered(worker_fn, to_process)
            for result in results_iter:
                obj_key = f"{result['shard_id']}/{result['obj_id']}"
                progress[obj_key] = result
                save_progress(progress_path, progress)
                completed_count += 1
                elapsed = time.time() - start_time
                avg = elapsed / completed_count
                eta = avg * (total_to_process - completed_count)
                append_status_log(status_log_path, f"{obj_key} {result['status']} frames={result.get('num_frames', 0)} done={completed_count}/{total_to_process} avg={avg:.1f}s/obj eta={eta:.0f}s")
                print(f"[{completed_count}/{total_to_process}] {obj_key} {result['status']} avg={avg:.1f}s eta={eta:.0f}s")

    statuses = {}
    for v in progress.values():
        s = v.get('status', 'unknown')
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nFinal summary: {statuses}")


if __name__ == '__main__':
    main()
