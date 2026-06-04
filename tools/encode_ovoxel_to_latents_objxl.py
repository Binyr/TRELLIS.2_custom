#!/usr/bin/env python3
"""
encode_ovoxel_to_latents_objxl.py -- Batch-encode O-Voxel (.vxz) view tars
produced by `tools/voxel/dual_grid_dynamic_obj.py` into per-view npz files
containing shape latents + SS latents.

Layout (input/output, S3):
  input :  {s3_input_root}/{res}/<sha256>/view_XX.tar    (contains {frame}.vxz)
  output:  {s3_output_root}/{res}/<sha256>/view_XX.npz   (all frames in one)

Per-view npz keys:
  num_frames
  shape_feats_<frame_id>  (float32)
  shape_coords_<frame_id> (uint8)
  ss_z_<frame_id>         (float)

Differences vs. trellis.2_private/tools/encode_ovoxel_to_latents.py:
  * Reads/writes S3, not network disk. All blob IO uses `aws s3 cp`; tar is
    extracted under /local-ssd, npz is written under /local-ssd then uploaded.
  * Task list is summarized DYNAMICALLY from the voxel-stage
    `voxel_progress_*.json` files (success-only), via --voxel_logs_prefix.
    No static obj_list JSON.
  * Path has 2 levels (<sha>/view_XX), not 3 (<shard>/<obj>/view_XX).
  * Single GPU per rank (cuda:0). Multi-rank parallelism comes from launching
    multiple ranks via koala.
  * Uses the same ProgressStore (throttled-mirror to S3) + per-failure
    traceback upload to {s3_output_root}/_logs/rank_<rank>/<sha>_view_XX.txt
    as `tools/voxel/dual_grid_dynamic_obj.py`.
  * stdout is tee-d + periodically synced to S3 by the launcher shell, not by
    this script.

Usage (single rank):
  python tools/encode_ovoxel_to_latents_objxl.py \
      --voxel_logs_prefix s3://.../dynamic_obj_voxel_32f/logs/ \
      --s3_input_root     s3://.../dynamic_obj_voxel_32f \
      --s3_output_root    s3://.../dynamic_obj_voxel_32f_latent \
      --resolution 512 --ss_resolution 32 \
      --rank 0 --world_size 1
"""

import argparse
import os
import sys
import json
import shutil
import tarfile
import tempfile
import time
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

import o_voxel
import trellis2.models as models
import trellis2.modules.sparse as sp

# Reuse helpers from the voxel worker (S3 cp, ProgressStore, error log upload,
# manifest summarization).
from tools.voxel.dual_grid_dynamic_obj import (  # noqa: E402
    _is_s3_uri,
    s3_get_file,
    s3_cp_file,
    upload_error_log,
    ProgressStore,
    summarize_finished_views_from_logs,
    _entry,
    _line,
)


# =====================================================================================
# Encode primitives
# =====================================================================================


def _read_vxz_cpu(vxz_path: str):
    """Read one .vxz file into CPU tensors (no GPU touch, no SparseTensor)."""
    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)
    return coords, attr


def _build_sparse_chunk(items, device):
    """Stack a list of per-frame (coords, attr) into a single batched SparseTensor pair.

    items: list of (coords_cpu, attr_cpu) tuples; index in list -> batch_idx.
    Returns (vertices, intersected) SparseTensors on `device`.
    """
    vert_feats = []
    inter_feats = []
    bcoords_list = []
    for b, (coords, attr) in enumerate(items):
        n = coords.shape[0]
        batch_col = torch.full((n, 1), b, dtype=coords.dtype)
        bcoords_list.append(torch.cat([batch_col, coords], dim=-1))
        vert_feats.append((attr['vertices'] / 255.0).float())
        ic = attr['intersected']
        inter_feats.append(torch.cat([ic % 2, ic // 2 % 2, ic // 4 % 2], dim=-1).bool())

    coords_cat = torch.cat(bcoords_list, dim=0)
    vertices = sp.SparseTensor(
        torch.cat(vert_feats, dim=0),
        coords_cat,
    )
    intersected = vertices.replace(torch.cat(inter_feats, dim=0))
    return vertices.to(device), intersected.to(device)


def _encode_shape_chunk(encoder, items, device):
    """Encode a chunk of frames in one forward. Returns list of (feats_np, coords_np)
    aligned with `items`."""
    if len(items) == 1:
        # path-of-least-resistance for fallback
        vertices, intersected = _build_sparse_chunk(items, device)
    else:
        vertices, intersected = _build_sparse_chunk(items, device)
    try:
        z = encoder(vertices, intersected)
        torch.cuda.synchronize()
        if not torch.isfinite(z.feats).all():
            raise ValueError("Non-finite values in shape latent")
        feats_all = z.feats.detach().cpu().numpy().astype(np.float32)
        coords_all = z.coords.detach().cpu().numpy()  # uint? long; col0 = batch
    finally:
        del vertices, intersected
    out = []
    batch_col = coords_all[:, 0]
    for b in range(len(items)):
        mask = batch_col == b
        feats = feats_all[mask]
        coords = coords_all[mask, 1:].astype(np.uint8)
        out.append((feats, coords))
    return out


def _encode_ss(encoder, coords_enc, ss_resolution: int, device: str):
    coords_t = torch.from_numpy(coords_enc).long()
    if coords_t.numel() == 0:
        raise ValueError("Empty shape latent (no coords)")
    cmax = int(coords_t.max().item())
    if cmax >= ss_resolution:
        raise ValueError(
            f"Shape latent coords max ({cmax}) >= ss_resolution ({ss_resolution})")
    ss = torch.zeros(1, ss_resolution, ss_resolution, ss_resolution,
                     dtype=torch.float32, device=device)
    ss[0, coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]] = 1
    z = encoder(ss[None], sample_posterior=False)
    torch.cuda.synchronize()
    if not torch.isfinite(z).all():
        raise ValueError("Non-finite values in SS latent")
    return z[0].detach().cpu().numpy()


# =====================================================================================
# Per-view pipeline
# =====================================================================================


def encode_one_view(
    sha256: str,
    view_idx: int,
    s3_input_root: str,
    s3_output_root: str,
    resolution: int,
    ss_resolution: int,
    tmp_root: str,
    shape_encoder,
    ss_encoder,
    device: str,
    do_ss: bool = True,
    frame_chunk_size: int = 8,
) -> dict:
    view_key = f"{sha256}/view_{view_idx:02d}"
    tar_uri = f"{s3_input_root.rstrip('/')}/{resolution}/{sha256}/view_{view_idx:02d}.tar"
    out_uri = f"{s3_output_root.rstrip('/')}/{resolution}/{sha256}/view_{view_idx:02d}.npz"

    work_dir = os.path.join(tmp_root, sha256, f"view_{view_idx:02d}")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    local_tar = os.path.join(work_dir, f"view_{view_idx:02d}.tar")
    local_npz = os.path.join(work_dir, f"view_{view_idx:02d}.npz")

    t = {"t_get": 0.0, "t_extract": 0.0, "t_encode": 0.0,
         "t_save_npz": 0.0, "t_upload_npz": 0.0}
    try:
        # 1) pull tar
        t0 = time.time()
        if not s3_get_file(tar_uri, local_tar, retries=2):
            return _entry("missing_tar", view_key=view_key, num_frames=0, **t)
        t["t_get"] = round(time.time() - t0, 2)

        # 2) extract
        t0 = time.time()
        with tarfile.open(local_tar) as tf:
            tf.extractall(work_dir)
        os.remove(local_tar)
        vxz_files = sorted(f for f in os.listdir(work_dir) if f.endswith(".vxz"))
        t["t_extract"] = round(time.time() - t0, 2)
        if not vxz_files:
            return _entry("no_vxz", view_key=view_key, num_frames=0, **t)

        # 3) per-chunk encode (chunked shape encode, then per-frame SS if needed)
        all_data = {"num_frames": np.int32(len(vxz_files))}
        encoded = 0
        t0 = time.time()
        last_failed = None
        frame_ids = [os.path.splitext(n)[0] for n in vxz_files]

        # Pre-load all .vxz to CPU (fast, sequential, /local-ssd).
        loaded = []
        for vxz_name in vxz_files:
            loaded.append(_read_vxz_cpu(os.path.join(work_dir, vxz_name)))

        cs = max(1, int(frame_chunk_size))
        i = 0
        while i < len(loaded):
            chunk_items = loaded[i:i + cs]
            chunk_fids = frame_ids[i:i + cs]
            try:
                outs = _encode_shape_chunk(shape_encoder, chunk_items, device)
            except Exception as e_chunk:
                # Fallback: re-do this chunk one frame at a time so a single bad
                # frame doesn't kill the whole chunk.
                if len(chunk_items) == 1:
                    last_failed = chunk_fids[0]
                    t["t_encode"] = round(time.time() - t0, 2)
                    raise RuntimeError(f"frame {chunk_fids[0]}: {e_chunk}") from e_chunk
                torch.cuda.empty_cache()
                outs = []
                for k, item in enumerate(chunk_items):
                    try:
                        outs.append(_encode_shape_chunk(shape_encoder, [item], device)[0])
                    except Exception as e_single:
                        last_failed = chunk_fids[k]
                        t["t_encode"] = round(time.time() - t0, 2)
                        raise RuntimeError(
                            f"frame {chunk_fids[k]} (chunk fallback): {e_single}") from e_single
            for fid, (feats, coords_enc) in zip(chunk_fids, outs):
                all_data[f"shape_feats_{fid}"] = feats
                all_data[f"shape_coords_{fid}"] = coords_enc
                if do_ss:
                    try:
                        ss_z = _encode_ss(ss_encoder, coords_enc, ss_resolution, device)
                    except Exception as e_ss:
                        last_failed = fid
                        t["t_encode"] = round(time.time() - t0, 2)
                        raise RuntimeError(f"frame {fid} (ss): {e_ss}") from e_ss
                    all_data[f"ss_z_{fid}"] = ss_z
                encoded += 1
            torch.cuda.empty_cache()
            i += cs
        t["t_encode"] = round(time.time() - t0, 2)

        # 4) save npz local
        t0 = time.time()
        np.savez_compressed(local_npz, **all_data)
        t["t_save_npz"] = round(time.time() - t0, 2)

        # 5) upload npz
        t0 = time.time()
        if not s3_cp_file(local_npz, out_uri, retries=2):
            return _entry("upload_failed", view_key=view_key, num_frames=encoded,
                          failed_frame=last_failed, **t)
        t["t_upload_npz"] = round(time.time() - t0, 2)

        npz_mb = round(os.path.getsize(local_npz) / (1024 * 1024), 2)
        print(f"[timing] {sha256[:12]}/view_{view_idx:02d} "
              f"frames={encoded} npz_mb={npz_mb} "
              f"t_get={t['t_get']}s t_extract={t['t_extract']}s "
              f"t_encode={t['t_encode']}s t_save={t['t_save_npz']}s "
              f"t_upload={t['t_upload_npz']}s")

        return _entry("success", view_key=view_key, num_frames=encoded,
                      npz_mb=npz_mb, **t)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# =====================================================================================
# Main
# =====================================================================================


# Status taxonomy:
#   permanent (terminal -> skipped on resume):
#     success         OK
#     no_vxz          tar opened fine but had no .vxz inside (bad data)
#     encode_error    a frame raised inside the encoder -- almost always a
#                     bad voxel / encoder math problem, not transient
#   transient (retried on resume):
#     missing_tar     `aws s3 cp` for the input tar returned non-zero
#     upload_failed   `aws s3 cp` for the output npz returned non-zero
#
# Every entry also carries an `attempts` counter so we can spot views that
# stay transient for too long.
TERMINAL_SKIP_STATUSES = {"success", "no_vxz", "encode_error"}
TRANSIENT_STATUSES = {"missing_tar", "upload_failed"}


def main():
    parser = argparse.ArgumentParser(
        description="Batch-encode O-Voxel view tars into shape+SS latents (objxl S3 layout).")
    parser.add_argument("--voxel_logs_prefix", type=str, required=True,
                        help="S3 prefix containing voxel_progress_*.json from the voxel stage "
                             "(e.g. s3://.../dynamic_obj_voxel_32f/logs/). success-only views "
                             "are used as the task pool.")
    parser.add_argument("--s3_input_root", type=str, required=True,
                        help="s3://.../dynamic_obj_voxel_32f")
    parser.add_argument("--s3_output_root", type=str, required=True,
                        help="s3://.../dynamic_obj_voxel_32f_latent")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--ss_resolution", type=int, default=32)
    parser.add_argument("--shape_enc_pretrained", type=str,
                        default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
    parser.add_argument("--ss_enc_pretrained", type=str,
                        default="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--state_dir", type=str, default="/local-ssd/encode_objxl_state")
    parser.add_argument("--tmp_dir", type=str, default="/local-ssd/encode_objxl_tmp")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--frame_chunk_size", type=int, default=8,
                        help="Number of frames batched into one shape-encoder forward. "
                             "On chunk failure the worker falls back to per-frame for "
                             "just that chunk so a single bad frame can't poison the view.")
    args = parser.parse_args()

    if not _is_s3_uri(args.s3_input_root):
        raise SystemExit("--s3_input_root must be s3://")
    if not _is_s3_uri(args.s3_output_root):
        raise SystemExit("--s3_output_root must be s3://")

    args.state_dir = os.path.join(args.state_dir, f"rank_{args.rank}")
    args.tmp_dir = os.path.join(args.tmp_dir, f"rank_{args.rank}")
    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    print(f"[main] rank={args.rank}/{args.world_size} state={args.state_dir} "
          f"tmp={args.tmp_dir} res={args.resolution} ss_res={args.ss_resolution}")

    # L1: gather success views from voxel-stage progress files.
    cache_dir = os.path.join(args.state_dir, "voxel_progress_cache")
    all_tasks = summarize_finished_views_from_logs(
        args.voxel_logs_prefix, cache_dir, filename_prefix="voxel_progress_")
    all_tasks.sort()
    print(f"[main] voxel-success views: {len(all_tasks)}")

    # shard
    start = len(all_tasks) * args.rank // args.world_size
    end = len(all_tasks) * (args.rank + 1) // args.world_size
    my_tasks = all_tasks[start:end]
    print(f"[main] rank {args.rank}: {len(my_tasks)} tasks (idx {start}:{end})")

    # L2: per-rank progress.
    progress = ProgressStore(args.s3_output_root, args.rank, args.state_dir)
    # rename the per-rank progress URIs so they don't collide with voxel-stage files.
    progress.s3_progress_uri = (
        f"{args.s3_output_root.rstrip('/')}/logs/encode_progress_{args.rank}.json")
    progress.s3_status_uri = (
        f"{args.s3_output_root.rstrip('/')}/logs/encode_status_{args.rank}.log")
    progress.local_progress = os.path.join(args.state_dir, f"encode_progress_{args.rank}.json")
    progress.local_status = os.path.join(args.state_dir, f"encode_status_{args.rank}.log")
    progress.load()

    def _done(sha, vid):
        return progress.progress.get(f"{sha}/view_{int(vid):02d}", {}).get("status") in TERMINAL_SKIP_STATUSES

    # Summarize the existing progress.json for visibility.
    from collections import Counter as _Counter
    prior_status = _Counter(
        (entry or {}).get("status", "?") for entry in progress.progress.values()
        if isinstance(entry, dict)
    )
    print(f"[main] prior progress entries by status: {dict(prior_status)}")

    to_process = [(s, v) for s, v in my_tasks if not _done(s, v)]
    skipped = len(my_tasks) - len(to_process)
    print(f"[main] terminal in progress (skip): {skipped}; "
          f"to process (incl. retried transient): {len(to_process)}")

    if args.max_items is not None:
        to_process = to_process[: args.max_items]
        print(f"[main] limited to {len(to_process)} (--max_items)")

    if not to_process:
        print("[main] nothing to do")
        return

    # Models.
    device = "cuda:0"
    torch.cuda.set_device(0)
    do_ss = args.resolution > 512
    print(f"[main] loading shape encoder: {args.shape_enc_pretrained}")
    shape_encoder = models.from_pretrained(args.shape_enc_pretrained).eval().to(device)
    if do_ss:
        print(f"[main] loading ss    encoder: {args.ss_enc_pretrained}")
        ss_encoder = models.from_pretrained(args.ss_enc_pretrained).eval().to(device)
    else:
        ss_encoder = None
        print(f"[main] resolution={args.resolution} <= 512 -> skipping SS encode")

    n_ok = 0
    n_fail = 0
    t_start = time.time()
    total = len(to_process)
    for i, (sha, vid) in enumerate(to_process):
        t0 = time.time()
        try:
            with torch.no_grad():
                result = encode_one_view(
                    sha256=sha, view_idx=vid,
                    s3_input_root=args.s3_input_root,
                    s3_output_root=args.s3_output_root,
                    resolution=args.resolution,
                    ss_resolution=args.ss_resolution,
                    tmp_root=args.tmp_dir,
                    shape_encoder=shape_encoder,
                    ss_encoder=ss_encoder,
                    device=device,
                    do_ss=do_ss,
                    frame_chunk_size=args.frame_chunk_size,
                )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[error] {sha}/view_{vid:02d}: {e}\n{tb}")
            upload_error_log(args.s3_output_root, args.rank, sha, vid,
                             header=str(e), exc=e)
            result = _entry("encode_error", view_key=f"{sha}/view_{vid:02d}",
                            error=str(e)[:500])

        # carry forward an attempts counter so re-trying transient failures is visible
        view_key = f"{sha}/view_{int(vid):02d}"
        prev_attempts = (progress.progress.get(view_key, {}) or {}).get("attempts", 0)
        result["attempts"] = int(prev_attempts) + 1

        dt = time.time() - t0
        st = result.get("status", "?")
        if st == "success":
            n_ok += 1
        else:
            n_fail += 1
            # Soft failures from encode_one_view itself (missing_tar / no_vxz /
            # upload_failed) won't have a python traceback, dump the result dict
            # instead so the S3 _logs entry is still useful.
            if st != "encode_error":
                upload_error_log(args.s3_output_root, args.rank, sha, vid,
                                 header=f"soft_failure:{st}",
                                 extra_tb=json.dumps(result, default=str, indent=2))

        rate = (n_ok + n_fail) / max(1e-6, time.time() - t_start)
        eta_min = (total - (i + 1)) / max(1e-6, rate) / 60.0
        progress.update(
            sha, vid, result,
            _line(sha, vid, st, dt=f"{dt:.1f}s",
                  ok=n_ok, fail=n_fail, rate=f"{rate:.2f}/s",
                  eta_min=f"{eta_min:.1f}"),
        )
        print(f"[main] [{i+1}/{total}] {sha}/view_{vid:02d} {st} dt={dt:.1f}s "
              f"| ok={n_ok} fail={n_fail} rate={rate:.2f} view/s eta={eta_min:.1f}min")

    progress.flush()
    print(f"\n[main] DONE rank={args.rank} ok={n_ok} fail={n_fail} "
          f"elapsed={ (time.time() - t_start)/60.0:.1f}min")


if __name__ == "__main__":
    main()
