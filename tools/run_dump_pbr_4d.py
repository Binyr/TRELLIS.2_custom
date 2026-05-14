#!/usr/bin/env python3
"""
run_dump_pbr_4d.py - Batch extract PBR materials/UVs/mat_ids from animated GLB files.

Features:
- Multi-process via Pool(maxtasksperchild=1)
- Checkpoint-based resume via progress json
- Status log with timing/ETA
- Filter-before-shard for balanced workload
- Priority list support

Usage:
    python tools/run_dump_pbr_4d.py \
        --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
        --glb_root /threed-code/yanruibin/yanruibin/glbs \
        --output_root data/trellis.2/pbr_shared \
        --log_root data/trellis.2/logs/dump_pbr_4d \
        --max_workers 8 \
        --rank 0 --world_size 1
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from subprocess import call, DEVNULL


BLENDER_LINK = 'https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64/blender'


def install_blender():
    if not os.path.exists(BLENDER_PATH):
        print("[blender] Installing Blender 4.5.1...")
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1')
        os.system(f'wget -q {BLENDER_LINK} -O {BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64.tar.xz')
        os.system(f'tar -xf {BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
        os.system(f'{BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64/4.5/python/bin/python3.11 -m pip install -q pillow')
    else:
        print(f"[blender] Found at: {BLENDER_PATH}")


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


def dump_pbr_one_object(args_tuple, glb_root, output_root, blender_path, script_path):
    """Worker function: run Blender to extract PBR for one object."""
    shard_id, obj_id = args_tuple
    glb_path = os.path.join(glb_root, shard_id, f'{obj_id}.glb')
    output_path = os.path.join(output_root, shard_id, f'{obj_id}.pickle')

    if not os.path.exists(glb_path):
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'missing_glb'}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        blender_path, '-b', '-P', script_path,
        '--',
        '--object_path', glb_path,
        '--output_path', output_path,
    ]
    ret = call(cmd, stdout=DEVNULL, stderr=DEVNULL)
    if ret != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'error', 'error': f'blender exit={ret}'}

    if os.path.exists(output_path):
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'success'}
    else:
        return {'shard_id': shard_id, 'obj_id': obj_id, 'status': 'error', 'error': 'output not created'}


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--glb_root', type=str, default='/threed-code/yanruibin/yanruibin/glbs')
    parser.add_argument('--output_root', type=str, default='data/trellis.2/pbr_shared')
    parser.add_argument('--log_root', type=str, default='data/trellis.2/logs/dump_pbr_4d')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'test', 'all'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=1)
    parser.add_argument('--priority_list', type=str, default=None)
    parser.add_argument('--blender_path', type=str, default=None)
    args = parser.parse_args()

    # Install blender
    blender_path = args.blender_path or BLENDER_PATH
    if args.blender_path is None:
        install_blender()

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dump_pbr_4d.py')

    # Load annotations
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
    all_progress = {}
    progress_files = [f for f in glob.glob(os.path.join(args.log_root, 'progress_*.json'))
                      if os.path.basename(f).replace('progress_', '').replace('.json', '').isdigit()]
    for p_path in progress_files:
        try:
            with open(p_path, 'r') as f:
                all_progress.update(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to read {p_path}: {e}")
    print(f"Loaded global progress: {len(all_progress)} completed objects from {len(progress_files)} files")

    # Filter out completed objects FIRST
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
    progress_path = os.path.join(args.log_root, f'progress_{args.rank}.json')
    progress = load_progress(progress_path)

    if len(to_process) == 0:
        print("Nothing to do.")
        return

    status_log_path = os.path.join(args.log_root, f'status_{args.rank}.log')
    total_to_process = len(to_process)
    completed_count = 0
    start_time = time.time()

    worker_fn = partial(
        dump_pbr_one_object,
        glb_root=args.glb_root,
        output_root=args.output_root,
        blender_path=blender_path,
        script_path=script_path,
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
            append_status_log(status_log_path, f"{obj_key} {result['status']} done={completed_count}/{total_to_process} avg={avg:.1f}s/obj eta={eta:.0f}s")
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
                append_status_log(status_log_path, f"{obj_key} {result['status']} done={completed_count}/{total_to_process} avg={avg:.1f}s/obj eta={eta:.0f}s")
                print(f"[{completed_count}/{total_to_process}] {obj_key} {result['status']} avg={avg:.1f}s eta={eta:.0f}s")

    statuses = {}
    for v in progress.values():
        s = v.get('status', 'unknown')
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nFinal summary: {statuses}")


if __name__ == '__main__':
    main()
