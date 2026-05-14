#!/usr/bin/env python3
"""
dual_grid_v2.py - Convert 4D animated mesh sequences to geometry O-Voxels frame-by-frame.

Reads result_mesh.npz (vertices_seq + faces) and produces per-frame .vxz files
via o_voxel.convert.mesh_to_flexible_dual_grid().

Features:
- Per-view scheduling granularity: each task = one (object, view) pair
- View-level checkpoint-based resume via progress_{rank}.json
- Multi-process parallel processing (--max_workers)
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
import io
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

import o_voxel

# Expected views: stride=2, start=0, 16 cameras -> views 0,2,4,6,8,10,12,14
EXPECTED_VIEWS = [0, 2, 4, 6, 8, 10, 12, 14]


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


def load_camera_w2c_rotations(rendered_dir: str, view_start=0, view_stride=2):
    """
    Load camera w2c rotation matrices from result.json.
    Returns dict of {view_index: w2c_rot_np} for selected views.
    """
    result_json_path = os.path.join(rendered_dir, 'result.json')
    if not os.path.exists(result_json_path):
        return None
    with open(result_json_path, 'r') as f:
        data = json.load(f)
    cameras = data['_global']['static_cameras']
    view_dict = {}
    for cam in cameras:
        view_idx = cam['view_index']
        if view_idx % view_stride == view_start:
            c2w = np.array(cam['camera_c2w'], dtype=np.float32)
            w2c = np.linalg.inv(c2w)
            w2c_rot = w2c[:3, :3]
            view_dict[view_idx] = w2c_rot
    return view_dict


def dual_grid_one_view(
    shard_id: str,
    obj_id: str,
    view_idx: int,
    rendered_root: str,
    output_root: str,
    resolutions: list,
    debug: bool = False,
):
    """
    Convert all frames of one object for ONE camera view to geometry O-Voxels.
    Returns a single result dict.
    """

    # Load mesh sequence
    rendered_dir = os.path.join(rendered_root, f'{shard_id}_static_camera_distance_v3', obj_id)
    mesh_npz_path = os.path.join(rendered_dir, 'result_mesh.npz')
    if not os.path.exists(mesh_npz_path):
        return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': 'missing_mesh', 'num_frames': 0}

    # Load camera rotation for this view
    t_read_start = time.time()
    camera_views = load_camera_w2c_rotations(rendered_dir)
    if camera_views is None or view_idx not in camera_views:
        return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': 'missing_camera', 'num_frames': 0}

    w2c_rot = camera_views[view_idx]

    with np.load(mesh_npz_path) as mesh_data:
        vertices_seq = mesh_data['vertices'].copy()  # (T, N, 3) float16
        faces = mesh_data['faces'].copy()             # (F, 3) int32
    t_read = time.time() - t_read_start

    num_faces = faces.shape[0]
    if num_faces > 500000:
        return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': 'skipped_too_many_faces', 'num_frames': 0, 'num_faces': num_faces}

    num_frames = vertices_seq.shape[0]
    if debug:
        num_frames = min(num_frames, 1)
    faces_t = torch.from_numpy(faces).long()

    view_status = 'success'
    t_compute = 0.0
    t_write = 0.0
    for res in resolutions:
        output_dir = os.path.join(output_root, str(res), shard_id, obj_id)
        os.makedirs(output_dir, exist_ok=True)
        tar_path = os.path.join(output_dir, f'view_{view_idx:02d}.tar')

        # Skip if tar already exists
        if os.path.exists(tar_path):
            continue

        # Compute all frames, collect vxz bytes in memory
        frame_buffers = {}  # frame_idx -> bytes
        for frame_idx in range(num_frames):
            # Get frame vertices, rotate to camera space, then clamp
            frame_verts = vertices_seq[frame_idx].astype(np.float32)
            frame_verts = frame_verts @ w2c_rot.T  # world -> camera (rotation only)
            frame_verts = np.clip(frame_verts, -0.5, 0.5)
            verts_t = torch.from_numpy(frame_verts)

            try:
                t0 = time.time()
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

                dual_vertices = dual_vertices * res - voxel_indices
                assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1 + 1e-3), \
                    'dual_vertices out of range'
                dual_vertices = torch.clamp(dual_vertices, 0, 1)
                dual_vertices = (dual_vertices * 255).type(torch.uint8)
                intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
                t_compute += time.time() - t0

                # Write vxz to a temp file, then read bytes
                t0 = time.time()
                tmp_fd, tmp_path = tempfile.mkstemp(suffix='.vxz')
                os.close(tmp_fd)
                o_voxel.io.write_vxz(
                    tmp_path,
                    voxel_indices,
                    {'vertices': dual_vertices, 'intersected': intersected},
                )
                with open(tmp_path, 'rb') as f:
                    frame_buffers[frame_idx] = f.read()
                os.remove(tmp_path)
                t_write += time.time() - t0
            except Exception as e:
                print(f"[ERROR] dual_grid failed: {shard_id}/{obj_id} view={view_idx} frame={frame_idx} res={res}: {e}")
                view_status = 'error'
                continue
            finally:
                del verts_t
                try:
                    del voxel_indices, dual_vertices, intersected
                except NameError:
                    pass

        # Write all frames as a single tar to S3 (one write op)
        if frame_buffers:
            t0 = time.time()
            with open(tar_path, 'wb') as fout:
                with tarfile.open(fileobj=fout, mode='w') as tar:
                    for fi in sorted(frame_buffers.keys()):
                        data = frame_buffers[fi]
                        info = tarfile.TarInfo(name=f'{fi:06d}.vxz')
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
            t_write += time.time() - t0
            del frame_buffers

    del vertices_seq, faces, faces_t
    print(f"[TIMING] {shard_id}/{obj_id}/view_{view_idx:02d} read={t_read:.1f}s compute={t_compute:.1f}s write={t_write:.1f}s total={t_read+t_compute+t_write:.1f}s frames={num_frames}")
    return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': view_status, 'num_frames': num_frames,
            't_read': round(t_read, 2), 't_compute': round(t_compute, 2), 't_write': round(t_write, 2)}


def load_progress(progress_path: str) -> dict:
    """Load progress file. Returns dict of {view_key: info}."""
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


def _worker_wrapper(args_tuple, rendered_root, output_root, resolutions, debug=False):
    """Wrapper for Pool.imap_unordered: processes one (shard_id, obj_id, view_idx) task."""
    shard_id, obj_id, view_idx = args_tuple
    try:
        return dual_grid_one_view(
            shard_id=shard_id,
            obj_id=obj_id,
            view_idx=view_idx,
            rendered_root=rendered_root,
            output_root=output_root,
            resolutions=resolutions,
            debug=debug,
        )
    except Exception as e:
        print(f"[ERROR] {shard_id}/{obj_id}/view_{view_idx:02d}: {e}")
        return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': 'error', 'error': str(e)}


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
    parser.add_argument('--priority_list', type=str, default=None,
                        help='Path to file with priority obj_ids (one per line), these will be processed first')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: only process 1 view and 1 frame per object')
    args = parser.parse_args()

    resolutions = [int(x) for x in args.resolution.split(',')]
    print(f"Resolutions: {resolutions}")
    if args.debug:
        print("[DEBUG MODE] Only 1 view and 1 frame per object")

    # Load annotations
    with open(args.ann_file, 'r') as f:
        ann_data = json.load(f)

    entries = []
    if args.split in ('train', 'all'):
        entries.extend(ann_data.get('train', []))
    if args.split in ('test', 'all'):
        entries.extend(ann_data.get('test', []))

    print(f"Total entries (objects): {len(entries)}")

    # Log directory: output_root/log_{resolution}/
    res_tag = args.resolution.replace(',', '_')
    log_dir = os.path.join(args.output_root, f'log_{res_tag}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)

    # Load progress from ALL ranks to get global completion status (view-level)
    all_progress = {}
    import glob
    progress_files = glob.glob(os.path.join(log_dir, 'progress_*.json'))
    for p_path in progress_files:
        try:
            with open(p_path, 'r') as f:
                all_progress.update(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to read {p_path}: {e}")
    print(f"Loaded global progress: {len(all_progress)} completed views from {len(progress_files)} files")

    # Build per-view task list, filtering out completed views
    views_to_use = EXPECTED_VIEWS if not args.debug else EXPECTED_VIEWS[:1]
    to_process = []  # list of (shard_id, obj_id, view_idx)
    skipped_views = 0
    for entry in entries:
        shard_id, obj_id = parse_entry(entry)
        for v in views_to_use:
            view_key = f"{shard_id}/{obj_id}/view_{v:02d}"
            if view_key in all_progress and all_progress[view_key].get('status') == 'success':
                skipped_views += 1
                continue
            to_process.append((shard_id, obj_id, v))

    print(f"To process: {len(to_process)} views ({skipped_views} already completed)")

    # Sort by priority: priority_list obj_ids first
    if args.priority_list and os.path.exists(args.priority_list):
        with open(args.priority_list, 'r') as f:
            priority_ids = set(line.strip() for line in f if line.strip())
        priority_views = [(s, o, v) for s, o, v in to_process if o in priority_ids]
        non_priority_views = [(s, o, v) for s, o, v in to_process if o not in priority_ids]
        to_process = priority_views + non_priority_views
        print(f"Priority list: {len(priority_ids)} ids, {len(priority_views)} views matched")

    # THEN shard
    start = len(to_process) * args.rank // args.world_size
    end = len(to_process) * (args.rank + 1) // args.world_size
    to_process = to_process[start:end]
    print(f"Rank {args.rank}/{args.world_size}: assigned {len(to_process)} views")

    # Per-rank progress file
    progress_path = os.path.join(log_dir, f'progress_{args.rank}.json')
    progress = load_progress(progress_path)

    if len(to_process) == 0:
        print("Nothing to do.")
        return

    status_log_path = os.path.join(log_dir, f'status_{args.rank}.log')
    total_to_process = len(to_process)
    completed_count = 0
    start_time = time.time()

    # Process
    if args.max_workers <= 1:
        # Single process
        for shard_id, obj_id, view_idx in tqdm(to_process, desc="Dual grid 4D"):
            result = dual_grid_one_view(
                shard_id=shard_id,
                obj_id=obj_id,
                view_idx=view_idx,
                rendered_root=args.rendered_root,
                output_root=args.output_root,
                resolutions=resolutions,
                debug=args.debug,
            )
            view_key = f"{shard_id}/{obj_id}/view_{view_idx:02d}"
            progress[view_key] = result
            save_progress(progress_path, progress)
            completed_count += 1
            elapsed = time.time() - start_time
            avg_per_view = elapsed / completed_count
            eta = avg_per_view * (total_to_process - completed_count)
            append_status_log(status_log_path, f"{view_key} {result['status']} frames={result.get('num_frames', 0)} done={completed_count}/{total_to_process} avg={avg_per_view:.1f}s/view eta={eta:.0f}s")
    else:
        # Multi-process with worker recycling to prevent memory leaks
        worker_fn = partial(
            _worker_wrapper,
            rendered_root=args.rendered_root,
            output_root=args.output_root,
            resolutions=resolutions,
            debug=args.debug,
        )
        with Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
            results_iter = pool.imap_unordered(worker_fn, to_process)
            with tqdm(total=total_to_process, desc="Dual grid 4D") as pbar:
                for result in results_iter:
                    view_key = f"{result['shard_id']}/{result['obj_id']}/view_{result['view_idx']:02d}"
                    progress[view_key] = result
                    save_progress(progress_path, progress)
                    completed_count += 1
                    elapsed = time.time() - start_time
                    avg_per_view = elapsed / completed_count
                    eta = avg_per_view * (total_to_process - completed_count)
                    append_status_log(status_log_path, f"{view_key} {result['status']} frames={result.get('num_frames', 0)} done={completed_count}/{total_to_process} avg={avg_per_view:.1f}s/view eta={eta:.0f}s")
                    pbar.set_postfix_str(f"avg={avg_per_view:.1f}s/view eta={eta:.0f}s")
                    pbar.update(1)

    # Summary
    statuses = {}
    for v in progress.values():
        s = v.get('status', 'unknown')
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nFinal summary (view-level): {statuses}")
    print(f"Total views tracked: {len(progress)}")


if __name__ == '__main__':
    main()
