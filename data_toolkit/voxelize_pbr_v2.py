#!/usr/bin/env python3
"""
voxelize_pbr_v2.py - Voxelize 4D animated objects frame-by-frame.

Combines:
  - pbr_shared pickle (materials, faces, UVs, mat_ids)
  - result_mesh.npz (vertices_seq per frame)
to produce per-frame .vxz files via o_voxel.convert.blender_dump_to_volumetric_attr().

Usage:
    python data_toolkit/voxelize_pbr_v2.py \
        --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
        --pbr_shared_root data/trellis.2/pbr_shared \
        --rendered_root data/objverse_minghao_4d_mine_40075/rendering_v5 \
        --output_root data/trellis.2/pbr_voxels_4d \
        --resolution 256 \
        --rank 0 --world_size 1
"""

import argparse
import json
import os
import pickle
import copy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import o_voxel


def parse_entry(entry: str):
    """
    Parse a json entry path into shard_id and obj_id.
    Entry: /efs/.../000-000_static_camera_distance_v3/00a1d892548542c7ab83565070737d6b
    Returns: shard_id='000-000', obj_id='00a1d892548542c7ab83565070737d6b'
    """
    parts = Path(entry).parts
    obj_id = parts[-1]
    shard_with_suffix = parts[-2]
    shard_id = shard_with_suffix.split('_static_camera_distance_v3')[0]
    return shard_id, obj_id


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-face normals and expand to (F, 3, 3) format expected by o_voxel.
    Each face's 3 vertices get the same face normal.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    face_normals = face_normals / norms
    # Expand to (F, 3, 3): each vertex of the face gets the face normal
    return np.stack([face_normals, face_normals, face_normals], axis=1).astype(np.float32)


def prepare_dump_for_frame(pbr_shared: dict, frame_vertices: np.ndarray) -> dict:
    """
    Construct a full 'dump' dict compatible with o_voxel.convert.blender_dump_to_volumetric_attr().

    pbr_shared: {'materials': [...], 'objects': [{'faces', 'uvs', 'mat_ids'}]}
    frame_vertices: (N, 3) float32, already normalized to ~[-0.5, 0.5]
    """
    dump = copy.deepcopy(pbr_shared)

    # Fix alpha mode (same as voxelize_pbr.py)
    for mat in dump['materials']:
        if mat['alphaTexture'] is not None and mat['alphaMode'] == 'OPAQUE':
            mat['alphaMode'] = 'BLEND'

    # Append default material (same as voxelize_pbr.py)
    dump['materials'].append({
        "baseColorFactor": [0.8, 0.8, 0.8],
        "alphaFactor": 1.0,
        "metallicFactor": 0.0,
        "roughnessFactor": 0.5,
        "alphaMode": "OPAQUE",
        "alphaCutoff": 0.5,
        "baseColorTexture": None,
        "alphaTexture": None,
        "metallicTexture": None,
        "roughnessTexture": None,
    })

    # Build the single merged object with vertices
    obj_data = dump['objects'][0]
    obj_data['vertices'] = frame_vertices

    # Compute normals (required by blender_dump_to_volumetric_attr, but deleted after)
    obj_data['normals'] = compute_face_normals(frame_vertices, obj_data['faces'])

    # Fix mat_ids: -1 -> default material
    obj_data['mat_ids'] = obj_data['mat_ids'].copy()
    obj_data['mat_ids'][obj_data['mat_ids'] == -1] = len(dump['materials']) - 1
    assert np.all(obj_data['mat_ids'] >= 0), 'invalid mat_ids'

    # Verify vertices range
    assert np.all(frame_vertices >= -0.501) and np.all(frame_vertices <= 0.501), \
        f'vertices out of range: min={frame_vertices.min()}, max={frame_vertices.max()}'

    # Clamp to exact [-0.5, 0.5] for safety
    obj_data['vertices'] = np.clip(frame_vertices, -0.5, 0.5)

    return dump


def voxelize_one_object(
    shard_id: str,
    obj_id: str,
    pbr_shared_root: str,
    rendered_root: str,
    output_root: str,
    resolutions: list,
):
    """Voxelize all frames of one object at given resolutions."""

    # Load pbr shared data
    pbr_path = os.path.join(pbr_shared_root, shard_id, f'{obj_id}.pickle')
    if not os.path.exists(pbr_path):
        print(f"[SKIP] PBR shared not found: {pbr_path}")
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'missing_pbr'}

    with open(pbr_path, 'rb') as f:
        pbr_shared = pickle.load(f)

    # Load mesh sequence
    rendered_dir = os.path.join(rendered_root, f'{shard_id}_static_camera_distance_v3', obj_id)
    mesh_npz_path = os.path.join(rendered_dir, 'result_mesh.npz')
    if not os.path.exists(mesh_npz_path):
        print(f"[SKIP] mesh.npz not found: {mesh_npz_path}")
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'missing_mesh'}

    mesh_data = np.load(mesh_npz_path)
    vertices_seq = mesh_data['vertices']  # (T, N, 3) float16
    frame_indices = mesh_data['frame_indices']  # (T,) int32

    # Verify face count consistency
    shared_faces_from_npz = mesh_data['faces']  # (F, 3) from 4D_video_data.py
    pbr_faces = pbr_shared['objects'][0]['faces']
    if shared_faces_from_npz.shape != pbr_faces.shape:
        print(f"[ERROR] Face count mismatch for {shard_id}/{obj_id}: "
              f"mesh.npz={shared_faces_from_npz.shape}, pbr={pbr_faces.shape}")
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'face_mismatch'}

    num_frames = vertices_seq.shape[0]

    for res in resolutions:
        output_dir = os.path.join(output_root, str(res), shard_id, obj_id)
        os.makedirs(output_dir, exist_ok=True)

        for frame_idx in range(num_frames):
            output_path = os.path.join(output_dir, f'{frame_idx:06d}.vxz')
            if os.path.exists(output_path):
                continue

            # Get frame vertices
            frame_verts = vertices_seq[frame_idx].astype(np.float32)

            # Build dump
            dump = prepare_dump_for_frame(pbr_shared, frame_verts)

            # Voxelize
            try:
                coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
                    dump,
                    grid_size=res,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    mip_level_offset=0,
                    verbose=False,
                    timing=False,
                )
                del attr['normal']
                del attr['emissive']
                o_voxel.io.write_vxz(output_path, coord, attr)
            except Exception as e:
                print(f"[ERROR] Voxelize failed: {shard_id}/{obj_id} frame={frame_idx} res={res}: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue

    return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'success', 'num_frames': num_frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, required=True,
                        help='Path to rendering_v5_anns_8cam.json')
    parser.add_argument('--pbr_shared_root', type=str, default='data/trellis.2/pbr_shared',
                        help='Root directory of pbr_shared pickle files')
    parser.add_argument('--rendered_root', type=str,
                        default='data/objverse_minghao_4d_mine_40075/rendering_v5',
                        help='Root directory of rendered data (result_mesh.npz)')
    parser.add_argument('--output_root', type=str, default='data/trellis.2/pbr_voxels_4d',
                        help='Output root for .vxz files')
    parser.add_argument('--resolution', type=str, default='256',
                        help='Comma-separated resolutions (e.g. 256,512)')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'test', 'all'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
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

    # Process
    results = []
    for entry in tqdm(entries, desc="Voxelizing PBR"):
        shard_id, obj_id = parse_entry(entry)
        result = voxelize_one_object(
            shard_id=shard_id,
            obj_id=obj_id,
            pbr_shared_root=args.pbr_shared_root,
            rendered_root=args.rendered_root,
            output_root=args.output_root,
            resolutions=resolutions,
        )
        results.append(result)

    # Summary
    statuses = {}
    for r in results:
        s = r['status']
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nSummary: {statuses}")


if __name__ == '__main__':
    main()
