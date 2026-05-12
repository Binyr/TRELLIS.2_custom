#!/usr/bin/env python3
"""
run_dump_pbr_4d.py - Orchestrator to run dump_pbr_4d.py on all objects listed in
rendering_v5_anns_8cam.json.

Usage:
    python tools/run_dump_pbr_4d.py \
        --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
        --glb_root /threed-code/yanruibin/yanruibin/glbs \
        --output_root data/trellis.2/pbr_shared \
        --split all \
        --rank 0 --world_size 1
"""

import argparse
import json
import os
from pathlib import Path
from subprocess import call, DEVNULL
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


BLENDER_LINK = 'https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64/blender'


def install_blender():
    if not os.path.exists(BLENDER_PATH):
        print("[blender] Installing Blender 4.5.1...")
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
    else:
        print(f"[blender] Found at: {BLENDER_PATH}")


def parse_entry(entry: str):
    """
    Parse a json entry path into shard_id and obj_id.
    Entry format: /efs/yanruibin/data/40075/objverse_minghao_4d_mine/000-000_static_camera_distance_v3/00a1d892548542c7ab83565070737d6b
    Returns: shard_id='000-000', obj_id='00a1d892548542c7ab83565070737d6b'
    """
    parts = Path(entry).parts
    obj_id = parts[-1]
    shard_with_suffix = parts[-2]  # e.g. '000-000_static_camera_distance_v3'
    # Remove '_static_camera_distance_v3' suffix
    shard_id = shard_with_suffix.split('_static_camera_distance_v3')[0]
    return shard_id, obj_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, required=True,
                        help='Path to rendering_v5_anns_8cam.json')
    parser.add_argument('--glb_root', type=str, default='/threed-code/yanruibin/yanruibin/glbs',
                        help='Root directory containing GLB files')
    parser.add_argument('--output_root', type=str, default='data/trellis.2/pbr_shared',
                        help='Output root for pickle files')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'test', 'all'],
                        help='Which split to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Number of parallel Blender processes')
    parser.add_argument('--blender_path', type=str, default=None,
                        help='Custom blender path (default: auto-install)')
    args = parser.parse_args()

    # Install blender
    blender_path = args.blender_path or BLENDER_PATH
    if args.blender_path is None:
        install_blender()

    # Load annotations
    with open(args.ann_file, 'r') as f:
        ann_data = json.load(f)

    # Collect entries
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

    # Filter already processed
    to_process = []
    for entry in entries:
        shard_id, obj_id = parse_entry(entry)
        output_path = os.path.join(args.output_root, shard_id, f'{obj_id}.pickle')
        if os.path.exists(output_path):
            continue
        glb_path = os.path.join(args.glb_root, shard_id, f'{obj_id}.glb')
        if not os.path.exists(glb_path):
            print(f"[WARN] GLB not found: {glb_path}")
            continue
        to_process.append((shard_id, obj_id, glb_path, output_path))

    print(f"To process (after filtering): {len(to_process)}")

    script_path = os.path.join(os.path.dirname(__file__), 'dump_pbr_4d.py')

    def process_one(item):
        shard_id, obj_id, glb_path, output_path = item
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cmd = [
            blender_path, '-b', '-P', script_path,
            '--',
            '--object_path', glb_path,
            '--output_path', output_path,
        ]
        ret = call(cmd, stdout=DEVNULL, stderr=DEVNULL)
        if ret != 0:
            print(f"[ERROR] Failed: {shard_id}/{obj_id} (exit code {ret})")
            # Remove partial output
            if os.path.exists(output_path):
                os.remove(output_path)
        return ret

    # Process
    if args.max_workers <= 1:
        for item in tqdm(to_process, desc="Dumping PBR"):
            process_one(item)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            list(tqdm(executor.map(process_one, to_process), total=len(to_process), desc="Dumping PBR"))

    print("[Done]")


if __name__ == '__main__':
    main()
