"""
Download ObjaverseXL GitHub objects by sha256 list.
Usage: python data_toolkit/download_by_sha256.py --sha256_list claude_tmp/uuid_github_intersection_sha256.txt --root trellis.2_data/ObjaverseXL_github
"""
import os
import sys
import time
import argparse
import pandas as pd
import objaverse.xl as oxl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha256_list', type=str, required=True, help='Path to sha256 list file')
    parser.add_argument('--root', type=str, required=True, help='Root directory with metadata.csv')
    parser.add_argument('--download_root', type=str, default=None, help='Download directory (defaults to root)')
    parser.add_argument('--processes', type=int, default=16, help='Number of download processes')
    parser.add_argument('--max_retries', type=int, default=1, help='Max retry attempts')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt.download_root = opt.download_root or opt.root

    # Load sha256 list
    with open(opt.sha256_list) as f:
        target_sha256 = set(line.strip() for line in f if line.strip())
    print(f"Target sha256 list: {len(target_sha256)} entries")

    # Load metadata and filter
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv'))
    metadata = metadata[metadata['sha256'].isin(target_sha256)]
    print(f"Matched in metadata: {len(metadata)}")

    # Shard by rank
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata.iloc[start:end]
    print(f"Rank {opt.rank}/{opt.world_size}: processing {len(metadata)} objects")

    # Get annotations and filter
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]
    print(f"Matched annotations: {len(annotations)}")

    # Download with retry
    os.makedirs(os.path.join(opt.download_root, 'raw', 'new_records'), exist_ok=True)

    file_paths = {}
    for attempt in range(opt.max_retries):
        remaining = annotations[~annotations['fileIdentifier'].isin(file_paths.keys())]
        if len(remaining) == 0:
            break
        print(f"[download] attempt {attempt+1}/{opt.max_retries}, {len(remaining)} objects remaining")
        try:
            new_paths = oxl.download_objects(
                remaining,
                download_dir=os.path.join(opt.download_root, "raw"),
                save_repo_format="zip",
                processes=opt.processes,
            )
            file_paths.update(new_paths)
        except Exception as e:
            print(f"[download] attempt {attempt+1} crashed: {e}")
            print(f"[download] {len(file_paths)} downloaded so far, sleeping 30s...")
            time.sleep(30)
            continue

    print(f"[download] finished with {len(file_paths)} objects downloaded")

    # Build records
    downloaded = {}
    meta_indexed = metadata.set_index("file_identifier")
    for k, v in file_paths.items():
        if k in meta_indexed.index:
            sha256 = meta_indexed.loc[k, "sha256"]
            downloaded[sha256] = os.path.relpath(v, opt.download_root)

    records = pd.DataFrame(list(downloaded.items()), columns=['sha256', 'local_path'])
    records.to_csv(os.path.join(opt.download_root, 'raw', 'new_records', f'part_{opt.rank}.csv'), index=False)
    print(f"Saved {len(records)} records")


if __name__ == '__main__':
    main()
