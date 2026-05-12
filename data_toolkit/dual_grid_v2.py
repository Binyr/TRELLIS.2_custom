#!/usr/bin/env python3
"""
dual_grid_v2.py - Convert 4D animated mesh sequences to geometry O-Voxels frame-by-frame.

Reads result_mesh.npz (vertices_seq + faces) and produces per-frame .vxz files
via o_voxel.convert.mesh_to_flexible_dual_grid().

Features:
- Multi-process parallel processing (--max_workers)
- Checkpoint-based resume via progress_{rank}.json (avoids mass network disk access)
- Distributed sharding (--rank / --world_size)

Usage:
    python data_toolkit/dual_grid_v2.py \
        --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
        --rendered_root data/objverse_minghao_4d_mine_40075/rendering_v5 \
        --output_root data/trellis.2/dual_grid_4d \
        --resolution 1024 \
        --max_workers 8 \
        --rank 0 --world_size 1
"""

import argparse
import json
import os
import time
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

import o_voxel


def parse_entry(entry: str):
    """
    Parse a json entry path into shard_id and obj_id.
    Entry: /efs/.../000-000_static_camera_distance_v3/00a1d892548542c7ab83565070737d6b
    """
    parts = Path(entry).parts
    obj_id = parts[-1]
    shard_with_suffix = parts[-2]
    shard_id = shard_with_suffix.split('_static_camera_distance_v3')[0]
    return shard_id, obj_id


def dual_grid_one_object(
    shard_id: str,
    obj_id: str,
    rendered_root: str,
    output_root: str,
    resolutions: list,
):
    """Convert all frames of one object to geometry O-Voxels at given resolutions."""

    # Load mesh sequence
    rendered_dir = os.path.join(rendered_root, f'{shard_id}_static_camera_distance_v3', obj_id)
    mesh_npz_path = os.path.join(rendered_dir, 'result_mesh.npz')
    if not os.path.exists(mesh_npz_path):
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'missing_mesh', 'num_frames': 0}

    with np.load(mesh_npz_path) as mesh_data:
        vertices_seq = mesh_data['vertices'].copy()  # (T, N, 3) float16
        faces = mesh_data['faces'].copy()             # (F, 3) int32

    num_frames = vertices_seq.shape[0]
    faces_t = torch.from_numpy(faces).long()

    for res in resolutions:
        output_dir = os.path.join(output_root, str(res), shard_id, obj_id)
        os.makedirs(output_dir, exist_ok=True)

        for frame_idx in range(num_frames):
            output_path = os.path.join(output_dir, f'{frame_idx:06d}.vxz')
            # Skip already processed frames (for partially completed objs)
            if os.path.exists(output_path):
                continue

            # Get frame vertices, clamp to [-0.5, 0.5]
            frame_verts = np.clip(vertices_seq[frame_idx].astype(np.float32), -0.5, 0.5)
            verts_t = torch.from_numpy(frame_verts)

            try:
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

                # Encode dual vertices and intersected (same as data_toolkit/dual_grid.py)
                dual_vertices = dual_vertices * res - voxel_indices
                assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1 + 1e-3), \
                    'dual_vertices out of range'
                dual_vertices = torch.clamp(dual_vertices, 0, 1)
                dual_vertices = (dual_vertices * 255).type(torch.uint8)
                intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

                o_voxel.io.write_vxz(
                    output_path,
                    voxel_indices,
                    {'vertices': dual_vertices, 'intersected': intersected},
                )
            except Exception as e:
                print(f"[ERROR] dual_grid failed: {shard_id}/{obj_id} frame={frame_idx} res={res}: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue
            finally:
                # Explicitly free large tensors to prevent memory leak
                del verts_t
                try:
                    del voxel_indices, dual_vertices, intersected
                except NameError:
                    pass

    del vertices_seq, faces, faces_t
    return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'success', 'num_frames': num_frames}


def load_progress(progress_path: str) -> dict:
    """Load progress file. Returns dict of {obj_key: info}."""
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress_path: str, progress: dict):
    """Atomically save progress file."""
    tmp_path = progress_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(progress, f)
    os.replace(tmp_path, progress_path)


def _worker_wrapper(args_tuple, rendered_root, output_root, resolutions):
    """Wrapper for Pool.imap_unordered: unpacks tuple and calls dual_grid_one_object."""
    shard_id, obj_id = args_tuple
    try:
        return dual_grid_one_object(
            shard_id=shard_id,
            obj_id=obj_id,
            rendered_root=rendered_root,
            output_root=output_root,
            resolutions=resolutions,
        )
    except Exception as e:
        print(f"[ERROR] {shard_id}/{obj_id}: {e}")
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'error', 'error': str(e)}


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, required=True,
                        help='Path to rendering_v5_anns_8cam.json')
    parser.add_argument('--rendered_root', type=str,
                        default='data/objverse_minghao_4d_mine_40075/rendering_v5',
                        help='Root directory of rendered data (result_mesh.npz)')
    parser.add_argument('--output_root', type=str, default='data/trellis.2/dual_grid_4d',
                        help='Output root for .vxz files')
    parser.add_argument('--resolution', type=str, default='1024',
                        help='Comma-separated resolutions (e.g. 256,512,1024)')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'test', 'all'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Number of parallel processes')
    args = parser.parse_args()

    resolutions = [int(x) for x in args.resolution.split(',')]
    print(f"Resolutions: {resolutions}")

    # Load annotations
    with open(args.ann_file, 'r') as f:
        ann_data = json.load(f)

    entries = []
    if args.split in ('train', 'all'):
        entries.extend(ann_data.get('train', []))
    if args.split in ('test', 'all'):
        entries.extend(ann_data.get('test', []))

    print(f"Total entries: {len(entries)}")

    # Shard
    start = len(entries) * args.rank // args.world_size
    end = len(entries) * (args.rank + 1) // args.world_size
    entries = entries[start:end]
    print(f"Rank {args.rank}/{args.world_size}: processing {len(entries)} entries")

    # Load progress (keyed by resolution to allow separate runs)
    os.makedirs(args.output_root, exist_ok=True)
    res_tag = args.resolution.replace(',', '_')
    progress_path = os.path.join(args.output_root, f'progress_{args.rank}.json') if res_tag == '512' else os.path.join(args.output_root, f'progress_{res_tag}_{args.rank}.json')
    progress = load_progress(progress_path)
    print(f"Loaded progress ({progress_path}): {len(progress)} completed objects")

    # Filter out completed objects
    to_process = []
    for entry in entries:
        shard_id, obj_id = parse_entry(entry)
        obj_key = f"{shard_id}/{obj_id}"
        if obj_key in progress and progress[obj_key].get('status') == 'success':
            continue
        to_process.append((shard_id, obj_id))

    print(f"To process (after filtering): {len(to_process)}")

    if len(to_process) == 0:
        print("Nothing to do.")
        return

    status_log_path = os.path.join(args.output_root, f'status_{args.rank}.log') if res_tag == '512' else os.path.join(args.output_root, f'status_{res_tag}_{args.rank}.log')
    total_to_process = len(to_process)
    completed_count = 0
    start_time = time.time()

    # Process
    if args.max_workers <= 1:
        # Single process
        for shard_id, obj_id in tqdm(to_process, desc="Dual grid 4D"):
            result = dual_grid_one_object(
                shard_id=shard_id,
                obj_id=obj_id,
                rendered_root=args.rendered_root,
                output_root=args.output_root,
                resolutions=resolutions,
            )
            obj_key = f"{shard_id}/{obj_id}"
            progress[obj_key] = result
            save_progress(progress_path, progress)
            completed_count += 1
            elapsed = time.time() - start_time
            avg_per_obj = elapsed / completed_count
            eta = avg_per_obj * (total_to_process - completed_count)
            with open(status_log_path, 'a') as f:
                f.write(f"{obj_key} {result['status']} frames={result.get('num_frames', 0)} done={completed_count}/{total_to_process} avg={avg_per_obj:.1f}s/obj eta={eta:.0f}s\n")
    else:
        # Multi-process with worker recycling to prevent memory leaks
        worker_fn = partial(
            _worker_wrapper,
            rendered_root=args.rendered_root,
            output_root=args.output_root,
            resolutions=resolutions,
        )
        with Pool(processes=args.max_workers, maxtasksperchild=4) as pool:
            results_iter = pool.imap_unordered(worker_fn, to_process)
            with tqdm(total=len(to_process), desc="Dual grid 4D") as pbar:
                for result in results_iter:
                    obj_key = f"{result['shard_id']}/{result['obj_id']}"
                    progress[obj_key] = result
                    save_progress(progress_path, progress)
                    completed_count += 1
                    elapsed = time.time() - start_time
                    avg_per_obj = elapsed / completed_count
                    eta = avg_per_obj * (total_to_process - completed_count)
                    with open(status_log_path, 'a') as f:
                        f.write(f"{obj_key} {result['status']} frames={result.get('num_frames', 0)} done={completed_count}/{total_to_process} avg={avg_per_obj:.1f}s/obj eta={eta:.0f}s\n")
                    pbar.update(1)

    # Summary
    statuses = {}
    for v in progress.values():
        s = v.get('status', 'unknown')
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nFinal summary: {statuses}")


if __name__ == '__main__':
    main()
