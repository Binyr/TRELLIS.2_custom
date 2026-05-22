#!/usr/bin/env python3
"""
voxelize_pbr_v2.py - Voxelize 4D animated objects PBR frame-by-frame.

Combines pbr_shared pickle (materials, faces, UVs, mat_ids) with
result_mesh.npz (vertices_seq per frame) to produce per-view .tar files
containing per-frame .vxz via o_voxel.convert.blender_dump_to_volumetric_attr().

Features:
- Per-view scheduling granularity: each task = one (object, view) pair
- View-level checkpoint-based resume via progress_{rank}.json
- Multi-process parallel processing (--max_workers)
- Distributed sharding (--rank / --world_size)

Usage:
    python data_toolkit/voxelize_pbr_v2.py \
        --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
        --pbr_shared_root data/trellis.2/pbr_shared \
        --rendered_root data/objverse_minghao_4d_mine_40075/rendering_v5 \
        --output_root data/trellis.2/pbr_voxels_4d \
        --resolution 512 \
        --max_workers 8 \
        --rank 0 --world_size 1
"""

import argparse
import copy
import json
import os
import pickle
import shutil
import sys
import tarfile
import time
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
from tqdm import tqdm

import o_voxel

# Expected views: stride=2, start=0, 16 cameras -> views 0,2,4,6,8,10,12,14
EXPECTED_VIEWS = [0, 2, 4, 6, 8, 10, 12, 14]

# View statuses that should not be retried on resume.
SKIP_STATUSES = {'success', 'missing_pbr', 'missing_mesh', 'face_mismatch'}


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


def _build_pbr_dump(pbr_shared: dict, frame_verts: np.ndarray, pbr_faces: np.ndarray) -> dict:
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
    obj_data['normals'] = compute_face_normals(frame_verts, pbr_faces)
    obj_data['mat_ids'] = obj_data['mat_ids'].copy()
    obj_data['mat_ids'][obj_data['mat_ids'] == -1] = len(dump['materials']) - 1
    return dump


def voxelize_pbr_one_view(
    shard_id: str,
    obj_id: str,
    view_idx: int,
    pbr_shared_root: str,
    rendered_root: str,
    output_root: str,
    resolutions: list,
    tmp_dir: str = '/tmp',
    debug: bool = False,
):
    """
    Voxelize all frames of one object for ONE camera view to PBR O-Voxels.
    Returns a single result dict.
    """
    rendered_dir = os.path.join(rendered_root, f'{shard_id}_static_camera_distance_v3', obj_id)

    t_read_start = time.time()
    pbr_path = os.path.join(pbr_shared_root, shard_id, f'{obj_id}.pickle')
    if not os.path.exists(pbr_path):
        return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': 'missing_pbr', 'num_frames': 0}

    with open(pbr_path, 'rb') as f:
        pbr_shared = pickle.load(f)

    mesh_npz_path = os.path.join(rendered_dir, 'result_mesh.npz')
    if not os.path.exists(mesh_npz_path):
        return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': 'missing_mesh', 'num_frames': 0}

    camera_views = load_camera_w2c_rotations(rendered_dir)
    if camera_views is None or view_idx not in camera_views:
        return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': 'missing_camera', 'num_frames': 0}

    w2c_rot = camera_views[view_idx]

    with np.load(mesh_npz_path) as mesh_data:
        vertices_seq = mesh_data['vertices'].copy()
        mesh_faces = mesh_data['faces'].copy()
    t_read = time.time() - t_read_start

    num_faces = mesh_faces.shape[0]
    if num_faces > 500000:
        return {
            'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx,
            'status': 'skipped_too_many_faces', 'num_frames': 0, 'num_faces': num_faces,
        }

    pbr_faces = pbr_shared['objects'][0]['faces']
    if mesh_faces.shape != pbr_faces.shape:
        return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': 'face_mismatch', 'num_frames': 0}

    num_frames = vertices_seq.shape[0]
    if debug:
        num_frames = min(num_frames, 1)

    view_status = 'success'
    failed_frame = None
    error_msg = None
    t_compute = 0.0
    t_write = 0.0

    for res in resolutions:
        if view_status != 'success':
            break

        output_dir = os.path.join(output_root, str(res), shard_id, obj_id)
        os.makedirs(output_dir, exist_ok=True)
        tar_path = os.path.join(output_dir, f'view_{view_idx:02d}.tar')

        if os.path.exists(tar_path):
            continue

        local_view_dir = os.path.join(tmp_dir, f'{shard_id}_{obj_id}_view{view_idx:02d}_res{res}')
        os.makedirs(local_view_dir, exist_ok=True)

        frame_files = []
        for frame_idx in range(num_frames):
            frame_verts = vertices_seq[frame_idx].astype(np.float32)
            frame_verts = frame_verts @ w2c_rot.T
            frame_verts = np.clip(frame_verts, -0.5, 0.5)

            local_vxz_path = os.path.join(local_view_dir, f'{frame_idx:06d}.vxz')
            dump = None
            coord = None
            attr = None
            try:
                t0 = time.time()
                dump = _build_pbr_dump(pbr_shared, frame_verts, pbr_faces)
                coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
                    dump, grid_size=res,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    mip_level_offset=0, verbose=False, timing=False,
                )
                del attr['normal']
                del attr['emissive']
                t_compute += time.time() - t0
                t_compute_cur = time.time() - t0

                t0 = time.time()
                o_voxel.io.write_vxz(local_vxz_path, coord, attr)
                t_write_cur = time.time() - t0
                t_write += t_write_cur
                frame_files.append((frame_idx, local_vxz_path))
                print(f"frame_idx: {frame_idx}, t_write: {t_write_cur:.3f}s, t_compute_cur: {t_compute_cur:.3f}s")
            except Exception as e:
                view_status = 'error'
                failed_frame = frame_idx
                error_msg = f"frame={frame_idx} res={res}: {e}"
                print(f"[ERROR] voxelize_pbr failed: {shard_id}/{obj_id} view={view_idx} {error_msg}")
                break
            finally:
                try:
                    del dump, coord, attr
                except NameError:
                    pass

        if view_status != 'success':
            shutil.rmtree(local_view_dir, ignore_errors=True)
            break

        if len(frame_files) == num_frames:
            t0 = time.time()
            local_tar_path = os.path.join(local_view_dir, 'view.tar')
            with tarfile.open(local_tar_path, 'w') as tar:
                for fi, fpath in sorted(frame_files):
                    tar.add(fpath, arcname=f'{fi:06d}.vxz')
            t_tar = time.time() - t0
            t0 = time.time()
            os.system(f'cp "{local_tar_path}" "{tar_path}"')
            t_cp = time.time() - t0
            t_write += t_tar + t_cp
            print(f"[TIMING] tar={t_tar:.3f}s cp={t_cp:.3f}s")

        shutil.rmtree(local_view_dir, ignore_errors=True)

    del vertices_seq, mesh_faces, pbr_shared
    print(
        f"[TIMING] {shard_id}/{obj_id}/view_{view_idx:02d} read={t_read:.1f}s "
        f"compute={t_compute:.1f}s write={t_write:.1f}s total={t_read + t_compute + t_write:.1f}s "
        f"frames={num_frames}"
    )

    result = {
        'shard_id': shard_id,
        'obj_id': obj_id,
        'view_idx': view_idx,
        'status': view_status,
        'num_frames': num_frames,
        't_read': round(t_read, 2),
        't_compute': round(t_compute, 2),
        't_write': round(t_write, 2),
    }
    if failed_frame is not None:
        result['failed_frame'] = failed_frame
    if error_msg is not None:
        result['error'] = error_msg
    return result


def _worker_wrapper(
    args_tuple,
    pbr_shared_root,
    rendered_root,
    output_root,
    resolutions,
    tmp_dir='/tmp',
    debug=False,
):
    """Wrapper for Pool.imap_unordered: processes one (shard_id, obj_id, view_idx) task."""
    shard_id, obj_id, view_idx = args_tuple
    try:
        return voxelize_pbr_one_view(
            shard_id=shard_id,
            obj_id=obj_id,
            view_idx=view_idx,
            pbr_shared_root=pbr_shared_root,
            rendered_root=rendered_root,
            output_root=output_root,
            resolutions=resolutions,
            tmp_dir=tmp_dir,
            debug=debug,
        )
    except Exception as e:
        print(f"[ERROR] {shard_id}/{obj_id}/view_{view_idx:02d}: {e}")
        return {
            'shard_id': shard_id,
            'obj_id': obj_id,
            'view_idx': view_idx,
            'status': 'error',
            'error': str(e),
        }


def _format_status_log_line(view_key: str, result: dict, completed_count: int, total_to_process: int, avg_per_view: float, eta: float) -> str:
    line = (
        f"{view_key} {result['status']} frames={result.get('num_frames', 0)} "
        f"done={completed_count}/{total_to_process} avg={avg_per_view:.1f}s/view eta={eta:.0f}s"
    )
    if result.get('failed_frame') is not None:
        line += f" failed_frame={result['failed_frame']}"
    if result.get('error'):
        line += f" error={result['error']}"
    return line


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, required=True,
                        help='Path to rendering_v5_anns_8cam.json')
    parser.add_argument('--pbr_shared_root', type=str, default='data/trellis.2/pbr_shared',
                        help='Root directory of pbr_shared pickle files')
    parser.add_argument('--rendered_root', type=str,
                        default='data/objverse_minghao_4d_mine_40075/rendering_v5',
                        help='Root directory of rendered data (result_mesh.npz)')
    parser.add_argument('--output_root', type=str, default='data/trellis.2/pbr_voxels_4d',
                        help='Output root for view tar files')
    parser.add_argument('--resolution', type=str, default='512',
                        help='Comma-separated resolutions (e.g. 256,512,1024)')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'test', 'all'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Number of parallel processes')
    parser.add_argument('--priority_list', type=str, default=None,
                        help='Path to file with priority obj_ids (one per line), these will be processed first')
    parser.add_argument('--finished_views', type=str, default=None,
                        help='Path to JSON file with list of already finished view keys to skip before sharding')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: only process 1 view and 1 frame per object')
    parser.add_argument('--tmp_dir', type=str, default='/local-ssd/tmp_voxelize_pbr',
                        help='Local SSD path for temp files (default: /local-ssd/tmp_voxelize_pbr)')
    args = parser.parse_args()

    resolutions = [int(x) for x in args.resolution.split(',')]
    print(f"Resolutions: {resolutions}")
    if args.debug:
        print("[DEBUG MODE] Only 1 view and 1 frame per object")
    os.makedirs(args.tmp_dir, exist_ok=True)
    print(f"Temp dir: {args.tmp_dir}")

    with open(args.ann_file, 'r') as f:
        ann_data = json.load(f)

    entries = []
    if args.split in ('train', 'all'):
        entries.extend(ann_data.get('train', []))
    if args.split in ('test', 'all'):
        entries.extend(ann_data.get('test', []))

    print(f"Total entries (objects): {len(entries)}")

    res_tag = args.resolution.replace(',', '_')
    log_dir = os.path.join(args.output_root, f'log_{res_tag}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)

    views_to_use = EXPECTED_VIEWS if not args.debug else EXPECTED_VIEWS[:1]
    all_views = []
    for entry in entries:
        shard_id, obj_id = parse_entry(entry)
        for v in views_to_use:
            all_views.append((shard_id, obj_id, v))

    print(f"Total views: {len(all_views)}")

    if args.priority_list and os.path.exists(args.priority_list):
        with open(args.priority_list, 'r') as f:
            priority_ids = set(line.strip() for line in f if line.strip())
        priority_views = [(s, o, v) for s, o, v in all_views if o in priority_ids]
        non_priority_views = [(s, o, v) for s, o, v in all_views if o not in priority_ids]
        all_views = priority_views + non_priority_views
        print(f"Priority list: {len(priority_ids)} ids, {len(priority_views)} views matched")

    if args.finished_views and os.path.exists(args.finished_views):
        with open(args.finished_views, 'r') as f:
            finished_set = set(json.load(f))
        before_count = len(all_views)
        all_views = [(s, o, v) for s, o, v in all_views if f"{s}/{o}/view_{v:02d}" not in finished_set]
        print(f"Finished views filter: {before_count} -> {len(all_views)} ({before_count - len(all_views)} removed)")

    start = len(all_views) * args.rank // args.world_size
    end = len(all_views) * (args.rank + 1) // args.world_size
    my_views = all_views[start:end]
    print(f"Rank {args.rank}/{args.world_size}: assigned {len(my_views)} views")

    progress_path = os.path.join(log_dir, f'progress_{args.rank}.json')
    progress = load_progress(progress_path)

    to_process = []
    skipped_views = 0
    for s, o, v in my_views:
        view_key = f"{s}/{o}/view_{v:02d}"
        if view_key in progress and progress[view_key].get('status') in SKIP_STATUSES:
            skipped_views += 1
            continue
        to_process.append((s, o, v))

    print(f"To process: {len(to_process)} views ({skipped_views} skipped by progress)")

    if len(to_process) == 0:
        print("Nothing to do.")
        return

    status_log_path = os.path.join(log_dir, f'status_{args.rank}.log')
    total_to_process = len(to_process)
    completed_count = 0
    start_time = time.time()

    if args.max_workers <= 1:
        for shard_id, obj_id, view_idx in tqdm(to_process, desc="Voxelize PBR 4D"):
            result = voxelize_pbr_one_view(
                shard_id=shard_id,
                obj_id=obj_id,
                view_idx=view_idx,
                pbr_shared_root=args.pbr_shared_root,
                rendered_root=args.rendered_root,
                output_root=args.output_root,
                resolutions=resolutions,
                tmp_dir=args.tmp_dir,
                debug=args.debug,
            )
            view_key = f"{shard_id}/{obj_id}/view_{view_idx:02d}"
            progress[view_key] = result
            save_progress(progress_path, progress)
            completed_count += 1
            elapsed = time.time() - start_time
            avg_per_view = elapsed / completed_count
            eta = avg_per_view * (total_to_process - completed_count)
            append_status_log(
                status_log_path,
                _format_status_log_line(view_key, result, completed_count, total_to_process, avg_per_view, eta),
            )
    else:
        worker_fn = partial(
            _worker_wrapper,
            pbr_shared_root=args.pbr_shared_root,
            rendered_root=args.rendered_root,
            output_root=args.output_root,
            resolutions=resolutions,
            tmp_dir=args.tmp_dir,
            debug=args.debug,
        )
        with Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
            results_iter = pool.imap_unordered(worker_fn, to_process)
            with tqdm(total=total_to_process, desc="Voxelize PBR 4D") as pbar:
                for result in results_iter:
                    view_key = f"{result['shard_id']}/{result['obj_id']}/view_{result['view_idx']:02d}"
                    progress[view_key] = result
                    save_progress(progress_path, progress)
                    completed_count += 1
                    elapsed = time.time() - start_time
                    avg_per_view = elapsed / completed_count
                    eta = avg_per_view * (total_to_process - completed_count)
                    append_status_log(
                        status_log_path,
                        _format_status_log_line(view_key, result, completed_count, total_to_process, avg_per_view, eta),
                    )
                    pbar.set_postfix_str(f"avg={avg_per_view:.1f}s/view eta={eta:.0f}s")
                    pbar.update(1)

    statuses = {}
    for v in progress.values():
        s = v.get('status', 'unknown')
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nFinal summary (view-level): {statuses}")
    print(f"Total views tracked: {len(progress)}")


if __name__ == '__main__':
    main()
