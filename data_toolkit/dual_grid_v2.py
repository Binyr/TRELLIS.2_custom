#!/usr/bin/env python3
"""
dual_grid_v2.py - Convert 4D animated mesh sequences to geometry O-Voxels frame-by-frame.

Reads result_mesh.npz (vertices_seq + faces) and produces per-frame .vxz files
via o_voxel.convert.mesh_to_flexible_dual_grid().

Features:
- Per-view scheduling granularity: each task = one (object, view) pair
- View-level checkpoint-based resume via progress_{rank}.json
- Single-process per rank (matching dual_grid_dynamic_obj.py)
- Distributed sharding (--rank / --world_size)

Usage:
    python data_toolkit/dual_grid_v2.py \
        --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
        --rendered_root data/objverse_minghao_4d_mine_40075/rendering_v5 \
        --output_root data/trellis.2/dual_grid_4d \
        --resolution 1024 \
        --rank 0 --world_size 1
"""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import o_voxel

# Expected views: stride=2, start=0, 16 cameras -> views 0,2,4,6,8,10,12,14
EXPECTED_VIEWS = [0, 2, 4, 6, 8, 10, 12, 14]

TERMINAL_SKIP_STATUSES = {
    'success',
    'skipped_too_many_faces',
    'invalid_mesh_nonfinite',
    'missing_mesh',
    'missing_camera',
    'frame_metadata_mismatch',
    'dual_grid_error',
    'worker_error',
}

ERROR_STATUSES = {
    'invalid_mesh_nonfinite',
    'frame_metadata_mismatch',
    'dual_grid_error',
    'worker_error',
    'upload_failed',
}


def s3_uri_for_path(path: str) -> str | None:
    """Map the /threed-code mount to its backing S3 URI."""
    if path.startswith('s3://'):
        return path
    mount_prefix = '/threed-code/'
    if path.startswith(mount_prefix):
        return 's3://arcwm-code-us-west-2/' + path[len(mount_prefix):]
    return None


def aws_s3_cp(src: str, dst: str, retries: int = 2) -> None:
    """Copy with the native AWS CLI when either endpoint is S3-backed."""
    src_arg = s3_uri_for_path(src) or src
    dst_arg = s3_uri_for_path(dst) or dst
    last_error = ''
    for attempt in range(retries + 1):
        proc = subprocess.run(
            ['aws', 's3', 'cp', '--only-show-errors', src_arg, dst_arg],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return
        last_error = proc.stderr.strip()
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(
        f'aws s3 cp failed after {retries + 1} attempts: '
        f'{src_arg} -> {dst_arg}: {last_error[:500]}'
    )


def publish_file(local_path: str, output_path: str) -> None:
    """Publish output without writing through an S3 FUSE mount."""
    if s3_uri_for_path(output_path) is not None:
        aws_s3_cp(local_path, output_path)
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    shutil.copyfile(local_path, output_path)


def fetch_file(remote_path: str, local_path: str) -> bool:
    """Fetch prior state; a missing remote file is a normal first-run case."""
    if s3_uri_for_path(remote_path) is None:
        if not os.path.exists(remote_path):
            return False
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        shutil.copyfile(remote_path, local_path)
        return True
    try:
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        aws_s3_cp(remote_path, local_path, retries=0)
        return True
    except RuntimeError:
        if os.path.exists(local_path):
            os.remove(local_path)
        return False


def output_file_exists(path: str) -> bool:
    """Check S3-backed outputs through the native AWS CLI."""
    s3_uri = s3_uri_for_path(path)
    if s3_uri is None:
        return os.path.exists(path)
    filename = s3_uri.rsplit('/', 1)[-1]
    last_error = ''
    for attempt in range(2):
        proc = subprocess.run(
            ['aws', 's3', 'ls', s3_uri],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return filename in proc.stdout
        last_error = proc.stderr.strip()
        if attempt == 0:
            time.sleep(2)
    if last_error:
        print(f'[WARN] aws s3 ls failed for {s3_uri}: {last_error[:300]}')
    return False


def pick_frame_sel(num_frames: int, max_frames: int, mode: str) -> list[int]:
    """Select mesh/RGB positions while preserving their original 0-based IDs."""
    if max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))
    if mode == 'center':
        start = (num_frames - max_frames) // 2
        return list(range(start, start + max_frames))
    if mode == 'uniform':
        return [int(round(x)) for x in np.linspace(0, num_frames - 1, max_frames)]
    if mode == 'head':
        return list(range(max_frames))
    raise ValueError(f'unknown frame_sampling mode: {mode}')


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
    tmp_dir: str = '/tmp',
    debug: bool = False,
    max_face_count: int = 500_000,
    max_frames: int = 0,
    frame_sampling: str = 'center',
    write_frame_meta: bool = False,
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

    # This source is already mounted, so t_get is zero; camera/NPZ reads are
    # accounted as t_load_mesh to match dual_grid_dynamic_obj timing fields.
    timings = {
        't_get': 0.0,
        't_load_mesh': 0.0,
        't_compute': 0.0,
        't_write_vxz': 0.0,
        't_tar': 0.0,
        't_upload_tar': 0.0,
        't_upload_meta': 0.0,
    }
    t_load_start = time.time()
    camera_views = load_camera_w2c_rotations(rendered_dir)
    if camera_views is None or view_idx not in camera_views:
        return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': 'missing_camera', 'num_frames': 0}

    w2c_rot = camera_views[view_idx]

    with np.load(mesh_npz_path) as mesh_data:
        vertices_seq = mesh_data['vertices'].copy()  # (T, N, 3) float16
        faces = mesh_data['faces'].copy()             # (F, 3) int32
        animation_frame_indices = (
            mesh_data['frame_indices'].copy()
            if 'frame_indices' in mesh_data.files
            else None
        )
    timings['t_load_mesh'] = time.time() - t_load_start

    num_faces = int(faces.shape[0])
    nonfinite_vertices = int(vertices_seq.size - np.isfinite(vertices_seq).sum())
    nonfinite_faces = int(faces.size - np.isfinite(faces).sum())
    if nonfinite_vertices or nonfinite_faces:
        return {
            'shard_id': shard_id,
            'obj_id': obj_id,
            'view_idx': view_idx,
            'status': 'invalid_mesh_nonfinite',
            'num_frames': 0,
            'num_faces': num_faces,
            'vertices_shape': list(vertices_seq.shape),
            'faces_shape': list(faces.shape),
            'nonfinite_vertices': nonfinite_vertices,
            'nonfinite_faces': nonfinite_faces,
        }
    if num_faces > max_face_count:
        return {'shard_id': shard_id, 'obj_id': obj_id, 'view_idx': view_idx, 'status': 'skipped_too_many_faces', 'num_frames': 0, 'num_faces': num_faces}

    num_frames_orig = int(vertices_seq.shape[0])
    if animation_frame_indices is not None and len(animation_frame_indices) != num_frames_orig:
        return {
            'shard_id': shard_id,
            'obj_id': obj_id,
            'view_idx': view_idx,
            'status': 'frame_metadata_mismatch',
            'num_frames': 0,
            'num_frames_orig': num_frames_orig,
            'error': (
                f'frame_indices has {len(animation_frame_indices)} entries but '
                f'vertices has {num_frames_orig} frames'
            ),
        }
    frame_sel = pick_frame_sel(num_frames_orig, max_frames, frame_sampling)
    if debug:
        frame_sel = frame_sel[:1]
    num_frames = len(frame_sel)
    faces_t = torch.from_numpy(faces).long()

    view_status = 'success'
    view_errors = []
    for res in resolutions:
        output_dir = os.path.join(output_root, str(res), shard_id, obj_id)
        if s3_uri_for_path(output_dir) is None:
            os.makedirs(output_dir, exist_ok=True)
        tar_path = os.path.join(output_dir, f'view_{view_idx:02d}.tar')
        meta_path = os.path.join(output_dir, f'view_{view_idx:02d}_meta.json')

        # Frame-aware outputs use the sidecar as the completion marker. Existing
        # all-frame pipelines retain their historical tar-only resume behavior.
        output_complete = output_file_exists(tar_path) and (
            not write_frame_meta or output_file_exists(meta_path)
        )
        if output_complete:
            continue

        # Local SSD temp directory for this view's vxz files
        local_view_dir = os.path.join(tmp_dir, f'{shard_id}_{obj_id}_view{view_idx:02d}_res{res}')
        os.makedirs(local_view_dir, exist_ok=True)

        # Compute all frames, write vxz to local SSD
        frame_files = []  # list of (frame_idx, local_path)
        for frame_idx in frame_sel:
            frame_verts = vertices_seq[frame_idx].astype(np.float32)
            frame_verts = frame_verts @ w2c_rot.T  # world -> camera (rotation only)
            frame_verts = np.clip(frame_verts, -0.5, 0.5)
            verts_t = torch.from_numpy(frame_verts)

            local_vxz_path = os.path.join(local_view_dir, f'{frame_idx:06d}.vxz')
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
                timings['t_compute'] += time.time() - t0
                t_compute_cur = time.time() - t0
                t0 = time.time()
                o_voxel.io.write_vxz(
                    local_vxz_path,
                    voxel_indices,
                    {'vertices': dual_vertices, 'intersected': intersected},
                    compression='zstd',
                    compression_level=5,
                )
                t_write_cur = time.time() - t0
                timings['t_write_vxz'] += t_write_cur
                frame_files.append((frame_idx, local_vxz_path))
                print(f"t_write: {t_write_cur:.3f}s, t_compute_cur: {t_compute_cur:.3f}s")
            except Exception as e:
                print(f"[ERROR] dual_grid failed: {shard_id}/{obj_id} view={view_idx} frame={frame_idx} res={res}: {e}")
                view_status = 'dual_grid_error'
                view_errors.append({
                    'frame_idx': int(frame_idx),
                    'resolution': int(res),
                    'error': f'{type(e).__name__}: {e}',
                    'traceback': traceback.format_exc(),
                })
                continue
            finally:
                del verts_t
                try:
                    del voxel_indices, dual_vertices, intersected
                except NameError:
                    pass

        # With frame metadata enabled, never publish a partial-frame archive.
        have_all_selected_frames = len(frame_files) == num_frames
        if frame_files and (have_all_selected_frames or not write_frame_meta):
            t0 = time.time()
            local_tar_path = os.path.join(local_view_dir, 'view.tar')
            with tarfile.open(local_tar_path, 'w') as tar:
                for fi, fpath in sorted(frame_files):
                    tar.add(fpath, arcname=f'{fi:06d}.vxz')
            t_tar = time.time() - t0
            timings['t_tar'] += t_tar
            if write_frame_meta:
                local_meta_path = os.path.join(local_view_dir, 'view_meta.json')
                source_animation_frames = (
                    None
                    if animation_frame_indices is None
                    else [int(animation_frame_indices[i]) for i in frame_sel]
                )
                with open(local_meta_path, 'w') as f:
                    json.dump({
                        'schema_version': 1,
                        'shard_id': shard_id,
                        'obj_id': obj_id,
                        'view_idx': int(view_idx),
                        'resolution': int(res),
                        'num_frames_orig': num_frames_orig,
                        'num_frames': num_frames,
                        'max_frames': int(max_frames),
                        'frame_sampling': frame_sampling,
                        'frame_sel': [int(i) for i in frame_sel],
                        'rgb_frame_indices_0based': [int(i) for i in frame_sel],
                        'rgb_frame_numbers_1based': [int(i) + 1 for i in frame_sel],
                        'source_animation_frame_indices': source_animation_frames,
                        'vxz_frame_ids': [int(i) for i in frame_sel],
                        'rgb_mp4': f'result_rgb_mp4/view_{view_idx:02d}.mp4',
                    }, f)
                # Publish tar first and metadata last; both are required for
                # frame-aware resume, so a partial upload is never complete.
                t0 = time.time()
                try:
                    publish_file(local_tar_path, tar_path)
                except Exception as exc:
                    timings['t_upload_tar'] += time.time() - t0
                    shutil.rmtree(local_view_dir, ignore_errors=True)
                    return {
                        'shard_id': shard_id, 'obj_id': obj_id,
                        'view_idx': view_idx, 'status': 'upload_failed',
                        'stage': 'tar', 'resolution': int(res),
                        'error': str(exc), 'num_frames': num_frames,
                        'num_frames_orig': num_frames_orig,
                        'frame_sampling': frame_sampling,
                        'frame_sel': [int(i) for i in frame_sel],
                    }
                timings['t_upload_tar'] += time.time() - t0
                t0 = time.time()
                try:
                    publish_file(local_meta_path, meta_path)
                except Exception as exc:
                    timings['t_upload_meta'] += time.time() - t0
                    shutil.rmtree(local_view_dir, ignore_errors=True)
                    return {
                        'shard_id': shard_id, 'obj_id': obj_id,
                        'view_idx': view_idx, 'status': 'upload_failed',
                        'stage': 'meta', 'resolution': int(res),
                        'error': str(exc), 'num_frames': num_frames,
                        'num_frames_orig': num_frames_orig,
                        'frame_sampling': frame_sampling,
                        'frame_sel': [int(i) for i in frame_sel],
                    }
                timings['t_upload_meta'] += time.time() - t0
            else:
                t0 = time.time()
                try:
                    publish_file(local_tar_path, tar_path)
                except Exception as exc:
                    timings['t_upload_tar'] += time.time() - t0
                    shutil.rmtree(local_view_dir, ignore_errors=True)
                    return {
                        'shard_id': shard_id, 'obj_id': obj_id,
                        'view_idx': view_idx, 'status': 'upload_failed',
                        'stage': 'tar', 'resolution': int(res),
                        'error': str(exc), 'num_frames': num_frames,
                        'num_frames_orig': num_frames_orig,
                        'frame_sampling': frame_sampling,
                        'frame_sel': [int(i) for i in frame_sel],
                    }
                timings['t_upload_tar'] += time.time() - t0
            print(
                f"[TIMING] res={res} tar={t_tar:.3f}s "
                f"upload_tar={timings['t_upload_tar']:.3f}s "
                f"upload_meta={timings['t_upload_meta']:.3f}s"
            )
        elif write_frame_meta:
            print(
                f'[ERROR] refusing partial output: {shard_id}/{obj_id} '
                f'view={view_idx} res={res} frames={len(frame_files)}/{num_frames}'
            )
            view_status = 'dual_grid_error'

        # Clean up local temp dir
        shutil.rmtree(local_view_dir, ignore_errors=True)

    del vertices_seq, faces, faces_t
    total_time = sum(timings.values())
    print(
        f"[TIMING] {shard_id}/{obj_id}/view_{view_idx:02d} "
        f"get={timings['t_get']:.1f}s load={timings['t_load_mesh']:.1f}s "
        f"compute={timings['t_compute']:.1f}s "
        f"write_vxz={timings['t_write_vxz']:.1f}s "
        f"tar={timings['t_tar']:.1f}s "
        f"upload_tar={timings['t_upload_tar']:.1f}s "
        f"upload_meta={timings['t_upload_meta']:.1f}s "
        f"total={total_time:.1f}s frames={num_frames}"
    )
    return {
        'shard_id': shard_id,
        'obj_id': obj_id,
        'view_idx': view_idx,
        'status': view_status,
        'num_frames': num_frames,
        'num_frames_orig': num_frames_orig,
        'frame_sampling': frame_sampling,
        'frame_sel': [int(i) for i in frame_sel],
        'errors': view_errors or None,
        **{key: round(value, 2) for key, value in timings.items()},
    }


def load_progress(progress_path: str) -> dict:
    """Load progress file. Returns dict of {view_key: info}."""
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress_path: str, progress: dict):
    tmp_path = progress_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(progress, f)
    os.replace(tmp_path, progress_path)


def append_status_log(status_log_path: str, line: str):
    """Append a line to the local status log."""
    with open(status_log_path, 'a') as f:
        f.write(line.rstrip('\n') + '\n')


def publish_error_log(
    local_log_dir: str,
    remote_log_dir: str,
    rank: int,
    view_key: str,
    result: dict,
) -> None:
    """Publish a structured per-view diagnostic without failing the worker."""
    safe_key = view_key.replace('/', '__')
    local_dir = os.path.join(local_log_dir, 'errors', f'rank_{rank}')
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, f'{safe_key}.json')
    remote_path = os.path.join(
        remote_log_dir, 'errors', f'rank_{rank}', f'{safe_key}.json'
    )
    payload = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'rank': int(rank),
        'view_key': view_key,
        'result': result,
    }
    with open(local_path, 'w') as f:
        json.dump(payload, f, indent=2)
    try:
        publish_file(local_path, remote_path)
    except Exception as exc:
        print(f'[WARN] error-log upload failed for {view_key}: {exc}')


def _worker_wrapper(args_tuple, rendered_root, output_root, resolutions, tmp_dir='/tmp', debug=False,
                    max_face_count=500_000, max_frames=0, frame_sampling='center',
                    write_frame_meta=False):
    """Run one task with worker-level error capture and scratch cleanup."""
    shard_id, obj_id, view_idx = args_tuple
    try:
        return dual_grid_one_view(
            shard_id=shard_id,
            obj_id=obj_id,
            view_idx=view_idx,
            rendered_root=rendered_root,
            output_root=output_root,
            resolutions=resolutions,
            tmp_dir=tmp_dir,
            debug=debug,
            max_face_count=max_face_count,
            max_frames=max_frames,
            frame_sampling=frame_sampling,
            write_frame_meta=write_frame_meta,
        )
    except Exception as e:
        print(f"[ERROR] {shard_id}/{obj_id}/view_{view_idx:02d}: {e}")
        return {
            'shard_id': shard_id,
            'obj_id': obj_id,
            'view_idx': view_idx,
            'status': 'worker_error',
            'error': f'{type(e).__name__}: {e}',
            'traceback': traceback.format_exc(),
        }
    finally:
        prefix = f'{shard_id}_{obj_id}_view{view_idx:02d}_res'
        for path in Path(tmp_dir).glob(prefix + '*'):
            shutil.rmtree(path, ignore_errors=True)


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
                        help='Deprecated compatibility option; processing is single-process')
    parser.add_argument('--max_face_count', type=int, default=500_000,
                        help='Skip meshes with more than this many faces')
    parser.add_argument('--priority_list', type=str, default=None,
                        help='Path to file with priority obj_ids (one per line), these will be processed first')
    parser.add_argument('--finished_views', type=str, default=None,
                        help='Path to JSON file with list of already finished view keys to skip before sharding')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: only process 1 view and 1 frame per object')
    parser.add_argument('--tmp_dir', type=str, default='/local-ssd/tmp_dual_grid',
                        help='Local SSD path for temp files (default: /local-ssd/tmp_dual_grid)')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Maximum selected frames per view; 0 preserves all frames')
    parser.add_argument('--frame_sampling', type=str, default='center',
                        choices=['center', 'uniform', 'head'],
                        help='Frame selection used when num_frames exceeds --max_frames')
    parser.add_argument('--write_frame_meta', action='store_true',
                        help='Write view_XX_meta.json and require tar+meta for completion')
    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error('--max_frames must be >= 0')
    if args.max_workers != 1:
        print(
            f'[WARN] --max_workers={args.max_workers} is deprecated and ignored; '
            'dual-grid processing is single-process'
        )

    resolutions = [int(x) for x in args.resolution.split(',')]
    print(f"Resolutions: {resolutions}")
    print(
        f"Frame selection: mode={args.frame_sampling} max_frames={args.max_frames} "
        f"write_meta={args.write_frame_meta}"
    )
    if args.debug:
        print("[DEBUG MODE] Only 1 view and 1 frame per object")
    os.makedirs(args.tmp_dir, exist_ok=True)
    print(f"Temp dir: {args.tmp_dir}")

    # Load annotations
    with open(args.ann_file, 'r') as f:
        ann_data = json.load(f)

    entries = []
    if args.split in ('train', 'all'):
        entries.extend(ann_data.get('train', []))
    if args.split in ('test', 'all'):
        entries.extend(ann_data.get('test', []))

    print(f"Total entries (objects): {len(entries)}")

    # Keep mutable rank state local and publish snapshots through native S3.
    res_tag = args.resolution.replace(',', '_')
    remote_log_dir = os.path.join(args.output_root, f'log_{res_tag}')
    log_dir = os.path.join(args.tmp_dir, '_state', f'log_{res_tag}')
    os.makedirs(log_dir, exist_ok=True)
    if s3_uri_for_path(args.output_root) is None:
        os.makedirs(args.output_root, exist_ok=True)

    # Build full per-view task list (deterministic order)
    views_to_use = EXPECTED_VIEWS if not args.debug else EXPECTED_VIEWS[:1]
    all_views = []  # list of (shard_id, obj_id, view_idx)
    for entry in entries:
        shard_id, obj_id = parse_entry(entry)
        for v in views_to_use:
            all_views.append((shard_id, obj_id, v))

    print(f"Total views: {len(all_views)}")

    # Sort by priority BEFORE sharding (so priority objects go first in all ranks)
    if args.priority_list and os.path.exists(args.priority_list):
        with open(args.priority_list, 'r') as f:
            priority_ids = set(line.strip() for line in f if line.strip())
        priority_views = [(s, o, v) for s, o, v in all_views if o in priority_ids]
        non_priority_views = [(s, o, v) for s, o, v in all_views if o not in priority_ids]
        all_views = priority_views + non_priority_views
        print(f"Priority list: {len(priority_ids)} ids, {len(priority_views)} views matched")

    # Filter out globally finished views BEFORE sharding
    if args.finished_views and os.path.exists(args.finished_views):
        with open(args.finished_views, 'r') as f:
            finished_set = set(json.load(f))
        before_count = len(all_views)
        all_views = [(s, o, v) for s, o, v in all_views if f"{s}/{o}/view_{v:02d}" not in finished_set]
        print(f"Finished views filter: {before_count} -> {len(all_views)} ({before_count - len(all_views)} removed)")

    # Shard FIRST (deterministic, rank always gets the same chunk)
    start = len(all_views) * args.rank // args.world_size
    end = len(all_views) * (args.rank + 1) // args.world_size
    my_views = all_views[start:end]
    print(f"Rank {args.rank}/{args.world_size}: assigned {len(my_views)} views")

    # Per-rank progress file
    progress_path = os.path.join(log_dir, f'progress_{args.rank}.json')
    remote_progress_path = os.path.join(remote_log_dir, f'progress_{args.rank}.json')
    fetch_file(remote_progress_path, progress_path)
    progress = load_progress(progress_path)

    # THEN filter out completed views (only check this rank's progress)
    to_process = []
    skipped_views = 0
    for s, o, v in my_views:
        view_key = f"{s}/{o}/view_{v:02d}"
        progress_status = progress.get(view_key, {}).get('status', '')
        if progress_status in TERMINAL_SKIP_STATUSES:
            skipped_views += 1
            continue
        to_process.append((s, o, v))

    print(f"To process: {len(to_process)} views ({skipped_views} already completed)")

    if len(to_process) == 0:
        print("Nothing to do.")
        return

    status_log_path = os.path.join(log_dir, f'status_{args.rank}.log')
    remote_status_log_path = os.path.join(remote_log_dir, f'status_{args.rank}.log')
    fetch_file(remote_status_log_path, status_log_path)
    total_to_process = len(to_process)
    completed_count = 0
    start_time = time.time()
    state_updates_since_push = 0
    last_state_push_time = 0.0

    def publish_state(force: bool = False):
        nonlocal state_updates_since_push, last_state_push_time
        now = time.time()
        if not force and state_updates_since_push < 8 and now - last_state_push_time < 30:
            return
        try:
            publish_file(progress_path, remote_progress_path)
            publish_file(status_log_path, remote_status_log_path)
        except Exception as exc:
            print(f'[WARN] progress/status upload failed: {exc}')
        state_updates_since_push = 0
        last_state_push_time = now

    def record_result(view_key: str, result: dict, status_line: str):
        nonlocal state_updates_since_push
        result.setdefault('updated_at', datetime.now(timezone.utc).isoformat())
        progress[view_key] = result
        save_progress(progress_path, progress)
        append_status_log(status_log_path, status_line)
        if result.get('status') in ERROR_STATUSES:
            publish_error_log(log_dir, remote_log_dir, args.rank, view_key, result)
        state_updates_since_push += 1
        publish_state()

    # Process one view at a time, matching dual_grid_dynamic_obj.py.
    for shard_id, obj_id, view_idx in tqdm(to_process, desc="Dual grid 4D"):
        result = _worker_wrapper(
            (shard_id, obj_id, view_idx),
            rendered_root=args.rendered_root,
            output_root=args.output_root,
            resolutions=resolutions,
            tmp_dir=args.tmp_dir,
            debug=args.debug,
            max_face_count=args.max_face_count,
            max_frames=args.max_frames,
            frame_sampling=args.frame_sampling,
            write_frame_meta=args.write_frame_meta,
        )
        view_key = f"{shard_id}/{obj_id}/view_{view_idx:02d}"
        completed_count += 1
        elapsed = time.time() - start_time
        avg_per_view = elapsed / completed_count
        eta = avg_per_view * (total_to_process - completed_count)
        record_result(
            view_key,
            result,
            f"{datetime.now(timezone.utc).isoformat()} {view_key} {result['status']} "
            f"frames={result.get('num_frames', 0)} "
            f"done={completed_count}/{total_to_process} avg={avg_per_view:.1f}s/view eta={eta:.0f}s",
        )

    publish_state(force=True)

    # Summary
    statuses = {}
    for v in progress.values():
        s = v.get('status', 'unknown')
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nFinal summary (view-level): {statuses}")
    print(f"Total views tracked: {len(progress)}")


if __name__ == '__main__':
    main()
