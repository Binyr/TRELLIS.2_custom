import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata
        

def download(metadata, output_dir, **kwargs):
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]

    # download with retry - objaverse has a bug where download errors
    # produce unpicklable exceptions that crash multiprocessing
    import time
    file_paths = {}
    max_retries = 1000
    for attempt in range(max_retries):
        remaining = annotations[~annotations['fileIdentifier'].isin(file_paths.keys())]
        if len(remaining) == 0:
            break
        print(f"[download] attempt {attempt+1}/{max_retries}, {len(remaining)} objects remaining")
        try:
            new_paths = oxl.download_objects(
                remaining,
                download_dir=os.path.join(output_dir, "raw"),
                save_repo_format="zip",
                processes=16,
            )
            file_paths.update(new_paths)
        except Exception as e:
            print(f"[download] attempt {attempt+1} crashed: {e}")
            print(f"[download] {len(file_paths)} objects downloaded so far, sleeping 120s before retry...")
            time.sleep(120)
            continue

    print(f"[download] finished with {len(file_paths)} objects downloaded")

    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    for k, v in file_paths.items():
        if k in metadata.index:
            sha256 = metadata.loc[k, "sha256"]
            downloaded[sha256] = os.path.relpath(v, output_dir)

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    import tempfile
    import zipfile
    
    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    if local_path.startswith('raw/github/repos/'):
                        path_parts = local_path.split('/')
                        file_name = os.path.join(*path_parts[5:])
                        zip_file = os.path.join(output_dir, *path_parts[:5])
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                zip_ref.extractall(tmp_dir)
                            file = os.path.join(tmp_dir, file_name)
                            record = func(file, sha256)
                    else:
                        file = os.path.join(output_dir, local_path)
                        record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)
