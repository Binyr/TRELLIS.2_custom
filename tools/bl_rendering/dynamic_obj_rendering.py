#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dynamic_obj_rendering.py

Renders animated 3D files (FBX/GLB/GLTF/DAE/OBJ) using Blender.
Based on 4D_video_data.py with the following changes:
1. Added FBX/DAE import support.
2. Removed motion trimming (umeyama) - uses full animation range.
3. If >121 frames, takes the center 121 frames.
4. Simplified for batch processing.

Usage (called by Blender):
    blender --background --python dynamic_obj_rendering.py -- \
        --object_path /path/to/file.fbx \
        --output_file /path/to/output/result.json \
        --hdr_dir /path/to/hdr \
        --render_engine CYCLES \
        --transparent_bg
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

try:
    import av
except Exception:
    av = None


# =====================================================================================
# 0. UTILS / TIMING / DEBUG
# =====================================================================================


def get_cli_argv() -> List[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


class StageTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0

    def log(self, name: str):
        now = time.perf_counter()
        print(f"[timing] {name}: +{now - self.last:.3f}s | total={now - self.t0:.3f}s")
        self.last = now


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def natural_key(s: str):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


def atomic_write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


# =====================================================================================
# 0.1 VIDEO EXPORT HELPERS
# =====================================================================================


def ensure_pyav_available():
    if av is None:
        raise ImportError(
            "PyAV is not available in the current Blender Python environment. "
            "Please install `av` into Blender's Python before using PNG->MP4 export."
        )


def get_render_view_indices(num_cameras: int, camera_stride: int) -> List[int]:
    if num_cameras <= 0:
        raise ValueError("num_cameras must be > 0")
    if camera_stride <= 0:
        raise ValueError("camera_stride must be > 0")
    view_indices = list(range(0, num_cameras, camera_stride))
    if len(view_indices) == 0:
        raise ValueError(
            f"No camera will be rendered: num_cameras={num_cameras}, camera_stride={camera_stride}"
        )
    return view_indices


def all_view_mp4_exist(video_dir: str, view_indices: List[int]) -> bool:
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        return False
    for view_idx in view_indices:
        mp4_path = video_dir / f"view_{view_idx:02d}.mp4"
        if not mp4_path.is_file():
            return False
    return True


def view_dir_for(obj_root: str, view_idx: int) -> str:
    return os.path.join(obj_root, f"view_{view_idx:02d}")


def view_outputs_complete(obj_root: str, view_idx: int, need_normal: bool) -> bool:
    """Check whether a view's local outputs are all present."""
    vd = view_dir_for(obj_root, view_idx)
    if not os.path.isfile(os.path.join(vd, "result.json")):
        return False
    if not os.path.isfile(os.path.join(vd, "mesh.npz")):
        return False
    if not os.path.isfile(os.path.join(vd, "rgb.mp4")):
        return False
    if need_normal and not os.path.isfile(os.path.join(vd, "normal.mp4")):
        return False
    return True


def list_png_files_natural(view_dir: str) -> List[str]:
    view_dir = Path(view_dir)
    if not view_dir.is_dir():
        return []
    pngs = [str(p) for p in view_dir.glob("*.png") if p.is_file()]
    pngs = sorted(pngs, key=lambda x: natural_key(Path(x).name))
    return pngs


def load_png_as_float01_chw_with_blender(image_path: str) -> np.ndarray:
    """Load PNG (8-bit or 16-bit) as float01 CHW with PIL.
    Uses PIL (not bpy.data.images) because bpy is not thread-safe and the
    encode pool runs in background threads concurrently with Blender's main
    render loop.
    """
    from PIL import Image  # local import keeps top-level cheap
    with Image.open(image_path) as img:
        mode = img.mode
        arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] >= 3:
        rgb = arr[:, :, :3]
    else:
        rgb = np.repeat(arr[:, :, :1], 3, axis=2)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    elif rgb.dtype == np.uint16:
        rgb = rgb.astype(np.float32) / 65535.0
    else:
        rgb = rgb.astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32, copy=False)


def write_16bit_depth_video_streaming_from_pngs(image_paths: List[str], save_path: str, fps=24, modal="rgb"):
    ensure_pyav_available()
    if len(image_paths) == 0:
        raise RuntimeError(f"No PNG frames found for video export: {save_path}")

    first = load_png_as_float01_chw_with_blender(image_paths[0])
    if first.shape[0] != 3:
        raise RuntimeError(f"Expected 3 channels, got shape={first.shape} for {image_paths[0]}")

    _, height, width = first.shape
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    output = av.open(save_path, mode="w")
    stream = output.add_stream("hevc", rate=fps)
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = "yuv420p10le"
    # Respect cgroup CPU quota: x265 otherwise spawns threads based on host nproc.
    # BLENDER_NUM_THREADS is exported by shs/render_dynamic_obj_mp.sh as
    # (cgroup_cpu_quota / workers_per_pod). Fall back to a small constant in
    # standalone runs.
    _enc_threads = int(os.environ.get("BLENDER_NUM_THREADS", "0")) or 4
    stream.options = {"crf": "10"}
    # libx265 caps frame-level threads at X265_MAX_FRAME_THREADS (=16). Going
    # over that triggers `avcodec_open2("libx265", {...})` InvalidData. Clamp
    # the wrapper-level thread_count accordingly; x265's own internal pool
    # (sized from host nproc, but cgroup-throttled) still handles the heavy
    # parallelism.
    _enc_threads = max(1, min(int(_enc_threads), 16))
    try:
        stream.thread_count = _enc_threads
        stream.thread_type = "FRAME"
    except Exception:
        pass

    def encode_one_frame(chw: np.ndarray):
        hwc = np.transpose(chw, (1, 2, 0))
        if modal == "rgb":
            frame = np.clip(hwc * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            yuv_frame = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv420p10le")
        elif modal == "normal":
            frame = np.clip(hwc * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)
            yuv_frame = av.VideoFrame.from_ndarray(frame, format="rgb48le").reformat(format="yuv420p10le")
        else:
            raise ValueError(f"Unsupported modal for PNG sequence export: {modal}")
        for packet in stream.encode(yuv_frame):
            output.mux(packet)

    try:
        encode_one_frame(first)
        for image_path in image_paths[1:]:
            chw = load_png_as_float01_chw_with_blender(image_path)
            if chw.shape != first.shape:
                raise RuntimeError(
                    f"Frame shape mismatch: expected {first.shape}, got {chw.shape} from {image_path}"
                )
            encode_one_frame(chw)
        for packet in stream.encode():
            output.mux(packet)
    except Exception:
        output.close()
        try:
            if os.path.isfile(save_path):
                os.remove(save_path)
        except Exception:
            pass
        raise
    else:
        output.close()


def encode_view_pngs_to_mp4(png_dir: str, mp4_path: str, fps: int, modal: str):
    """Encode all PNGs in `png_dir` (natural-sorted) into a single mp4."""
    pngs = list_png_files_natural(png_dir)
    if len(pngs) == 0:
        raise RuntimeError(f"No PNG frames to encode in {png_dir}")
    print(f"[video] Encoding {modal} mp4: {len(pngs)} frames -> {mp4_path}")
    write_16bit_depth_video_streaming_from_pngs(pngs, mp4_path, fps=fps, modal=modal)


# =====================================================================================
# 1. SCENE / CAMERA HELPERS
# =====================================================================================

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "glb": lambda filepath: bpy.ops.import_scene.gltf(filepath=filepath),
    "gltf": lambda filepath: bpy.ops.import_scene.gltf(filepath=filepath),
    "fbx": lambda filepath: bpy.ops.import_scene.fbx(filepath=filepath),
    "dae": lambda filepath: bpy.ops.wm.collada_import(filepath=filepath),
    "obj": lambda filepath: bpy.ops.wm.obj_import(filepath=filepath),
}


def init_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material, do_unlink=True)
    for texture in list(bpy.data.textures):
        if texture.users == 0:
            bpy.data.textures.remove(texture, do_unlink=True)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh, do_unlink=True)
    for world in list(bpy.data.worlds):
        if world.users == 0:
            bpy.data.worlds.remove(world, do_unlink=True)
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass


def load_object(object_path: str) -> None:
    file_extension = object_path.rsplit(".", 1)[-1].lower()
    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path} (ext={file_extension})")
    IMPORT_FUNCTIONS[file_extension](filepath=object_path)


def look_at(
    cam_pos: np.ndarray,
    target: np.ndarray,
    up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
):
    forward = normalize(target - cam_pos)
    right = normalize(np.cross(forward, up))
    if np.linalg.norm(right) < 1e-6:
        up_alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = normalize(np.cross(forward, up_alt))
    true_up = normalize(np.cross(right, forward))
    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, 0] = right
    cam2world[:3, 1] = true_up
    cam2world[:3, 2] = -forward
    cam2world[:3, 3] = cam_pos
    return cam2world


def orbit_offset(radius: float, azim: float, elev: float):
    x = radius * math.cos(elev) * math.cos(azim)
    y = radius * math.cos(elev) * math.sin(azim)
    z = radius * math.sin(elev)
    return np.array([x, y, z], dtype=np.float32)


def create_camera(name="TrackingCamera", lens=50.0, sensor_width=36.0, sensor_height=36.0):
    cam_data = bpy.data.cameras.new(name)
    cam_data.type = "PERSP"
    cam_data.lens = lens
    cam_data.sensor_width = sensor_width
    cam_data.sensor_height = sensor_height
    cam_data.sensor_fit = "HORIZONTAL"
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def set_camera_from_cam2world(cam_obj, cam2world: np.ndarray):
    cam_obj.matrix_world = Matrix(cam2world.tolist())


def set_camera_intrinsics_from_fov(cam_obj, fov_deg: float, sensor_size: float = 36.0):
    fov_deg = float(np.clip(fov_deg, 1.0, 179.0))
    fov_rad = math.radians(fov_deg)
    cam_data = cam_obj.data
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = float(sensor_size)
    cam_data.sensor_height = float(sensor_size)
    lens_mm = 0.5 * float(sensor_size) / max(math.tan(0.5 * fov_rad), 1e-8)
    cam_data.lens = float(lens_mm)


def get_camera_intrinsics_dict(cam_obj, resolution: int):
    cam_data = cam_obj.data
    angle_x = float(cam_data.angle_x)
    angle_y = float(cam_data.angle_y)
    fx = 0.5 * float(resolution) / max(math.tan(0.5 * angle_x), 1e-8)
    fy = 0.5 * float(resolution) / max(math.tan(0.5 * angle_y), 1e-8)
    cx = 0.5 * float(resolution)
    cy = 0.5 * float(resolution)
    return {
        "lens_mm": float(cam_data.lens),
        "sensor_width_mm": float(cam_data.sensor_width),
        "sensor_height_mm": float(cam_data.sensor_height),
        "fov_x_deg": float(math.degrees(angle_x)),
        "fov_y_deg": float(math.degrees(angle_y)),
        "fx_px": float(fx),
        "fy_px": float(fy),
        "cx_px": float(cx),
        "cy_px": float(cy),
    }


# =====================================================================================
# 2. FAST GEOMETRY EXTRACTION
# =====================================================================================


def extract_merged_mesh_world_fast(mesh_objs, depsgraph=None):
    """Extract (vertices, faces) for the current frame. Triangulation is
    re-derived from loop_triangles; only safe to call on the reference frame
    because Blender's evaluated-mesh triangulation is not guaranteed to be
    deterministic across frames (e.g. armature deform can flip the quad
    diagonal of an isolated triangle).
    """
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    all_vertices = []
    all_faces = []
    vert_offset = 0

    for obj in mesh_objs:
        if obj.type != "MESH":
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        temp_mesh = obj_eval.to_mesh()
        if temp_mesh is None:
            continue
        try:
            temp_mesh.calc_loop_triangles()
            num_verts = len(temp_mesh.vertices)
            num_tris = len(temp_mesh.loop_triangles)
            if num_verts == 0 or num_tris == 0:
                continue
            co = np.empty(num_verts * 3, dtype=np.float32)
            temp_mesh.vertices.foreach_get("co", co)
            co = co.reshape(num_verts, 3)
            tri = np.empty(num_tris * 3, dtype=np.int32)
            temp_mesh.loop_triangles.foreach_get("vertices", tri)
            tri = tri.reshape(num_tris, 3)
            world_mat = obj_eval.matrix_world.copy()
            R = np.array(world_mat.to_3x3(), dtype=np.float32)
            t = np.array(world_mat.translation[:], dtype=np.float32)
            verts_world = co @ R.T + t[None, :]
            all_vertices.append(verts_world)
            all_faces.append(tri + vert_offset)
            vert_offset += num_verts
        finally:
            obj_eval.to_mesh_clear()

    if len(all_vertices) == 0:
        raise RuntimeError("No valid mesh found in current frame.")

    merged_vertices = np.concatenate(all_vertices, axis=0)
    merged_faces = np.concatenate(all_faces, axis=0)
    return merged_vertices.astype(np.float32, copy=False), merged_faces.astype(np.int32, copy=False)


def extract_merged_vertices_world_fast(mesh_objs, depsgraph=None):
    """Extract only world-space vertices for the current frame, in the SAME
    object/vertex order as the reference call to extract_merged_mesh_world_fast.
    This skips loop_triangle recomputation, which is both faster and avoids
    the per-frame triangulation drift Blender exhibits on armature-driven
    quads (see Topology-changed bug in fd4d).
    """
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    all_vertices = []
    for obj in mesh_objs:
        if obj.type != "MESH":
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        temp_mesh = obj_eval.to_mesh()
        if temp_mesh is None:
            continue
        try:
            num_verts = len(temp_mesh.vertices)
            if num_verts == 0:
                continue
            co = np.empty(num_verts * 3, dtype=np.float32)
            temp_mesh.vertices.foreach_get("co", co)
            co = co.reshape(num_verts, 3)
            world_mat = obj_eval.matrix_world.copy()
            R = np.array(world_mat.to_3x3(), dtype=np.float32)
            t = np.array(world_mat.translation[:], dtype=np.float32)
            verts_world = co @ R.T + t[None, :]
            all_vertices.append(verts_world)
        finally:
            obj_eval.to_mesh_clear()

    if len(all_vertices) == 0:
        raise RuntimeError("No valid mesh found in current frame.")
    return np.concatenate(all_vertices, axis=0).astype(np.float32, copy=False)


def collect_keyframe_frames(frame_start: int, frame_end: int) -> List[int]:
    keyframes = set()
    for action in bpy.data.actions:
        for fcurve in action.fcurves:
            for kp in fcurve.keyframe_points:
                frame = int(round(float(kp.co.x)))
                if frame_start <= frame <= frame_end:
                    keyframes.add(frame)
    return sorted(keyframes)


def save_mesh_as_ply(vertices: np.ndarray, faces: np.ndarray, ply_path: str):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.astype("<f4", copy=False).tofile(f)
        face_dtype = np.dtype([("count", "u1"), ("idx", "<i4", (3,))])
        face_data = np.empty(len(faces), dtype=face_dtype)
        face_data["count"] = 3
        face_data["idx"] = faces.astype("<i4", copy=False)
        face_data.tofile(f)


# =====================================================================================
# 3. SEQUENCE NORMALIZATION + RAW CACHE
# =====================================================================================


def create_sequence_normalizer():
    root_objs = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(root_objs) == 0:
        raise RuntimeError("No root objects found in the scene.")
    normalizer = bpy.data.objects.new("SequenceNormalizer", None)
    bpy.context.scene.collection.objects.link(normalizer)
    normalizer.location = (0.0, 0.0, 0.0)
    normalizer.rotation_euler = (0.0, 0.0, 0.0)
    normalizer.scale = (1.0, 1.0, 1.0)
    for obj in root_objs:
        world_mat = obj.matrix_world.copy()
        obj.parent = normalizer
        obj.matrix_parent_inverse = normalizer.matrix_world.inverted()
        obj.matrix_world = world_mat
    bpy.context.view_layer.update()
    return normalizer


def compute_sequence_normalization_params_and_cache(
    mesh_objs,
    frame_indices: np.ndarray,
    cache_raw_vertices: bool = True,
    raw_cache_dtype=np.float16,
):
    scene = bpy.context.scene

    shared_faces = None
    ref_vert_count = None
    raw_vertices_cache = [] if cache_raw_vertices else None

    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    for i in frame_indices:
        scene.frame_set(int(i))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        if shared_faces is None:
            # First frame: also extract triangulation as the reference topology.
            raw_vertices, raw_faces = extract_merged_mesh_world_fast(mesh_objs, depsgraph=depsgraph)
            shared_faces = raw_faces.copy()
            ref_vert_count = int(raw_vertices.shape[0])
            print(
                f"Reference topology set from frame {int(i)}: "
                f"{shared_faces.shape[0]} faces, {ref_vert_count} vertices"
            )
        else:
            # Subsequent frames: only re-extract vertices; reuse shared_faces.
            # Blender's evaluated-mesh triangulation can drift between frames
            # (armature deform flipping quad diagonals), so we never recompute
            # it here. Only the per-object vertex count must stay constant.
            raw_vertices = extract_merged_vertices_world_fast(mesh_objs, depsgraph=depsgraph)
            if raw_vertices.shape[0] != ref_vert_count:
                raise RuntimeError(
                    f"Vertex count changed at frame {int(i)}: "
                    f"reference={ref_vert_count}, current={raw_vertices.shape[0]}."
                )

        frame_min = raw_vertices.min(axis=0)
        frame_max = raw_vertices.max(axis=0)
        global_min = np.minimum(global_min, frame_min)
        global_max = np.maximum(global_max, frame_max)

        if cache_raw_vertices:
            raw_vertices_cache.append(raw_vertices.astype(raw_cache_dtype))

    shared_faces = np.asarray(shared_faces, dtype=np.int32)
    global_center = 0.5 * (global_min + global_max)
    extent = global_max - global_min
    box_size = float(np.max(extent))
    sequence_scale = 1.0 / box_size if box_size > 1e-6 else 1.0

    canonical_bbox_min = ((global_min - global_center) * sequence_scale).astype(np.float32)
    canonical_bbox_max = ((global_max - global_center) * sequence_scale).astype(np.float32)

    return (
        shared_faces,
        global_center.astype(np.float32),
        sequence_scale,
        canonical_bbox_min,
        canonical_bbox_max,
        raw_vertices_cache,
    )


def apply_sequence_normalization(normalizer_obj, global_center: np.ndarray, global_scale: float):
    normalizer_obj.scale = (global_scale, global_scale, global_scale)
    normalizer_obj.location = tuple((-global_scale * global_center).tolist())


def precompute_normalized_mesh_sequence_from_cache(
    raw_vertices_cache: List[np.ndarray],
    global_center: np.ndarray,
    global_scale: float,
    shared_faces: np.ndarray,
    frame_indices: np.ndarray,
):
    if raw_vertices_cache is None:
        raise RuntimeError("raw_vertices_cache is None.")

    num_frames = len(raw_vertices_cache)
    num_vertices = raw_vertices_cache[0].shape[0]
    vertices_seq = np.empty((num_frames, num_vertices, 3), dtype=np.float16)

    for local_frame_idx, raw_vertices in enumerate(raw_vertices_cache):
        raw32 = np.asarray(raw_vertices, dtype=np.float32)
        norm_vertices = ((raw32 - global_center[None, :]) * global_scale).astype(np.float32)
        vertices_seq[local_frame_idx] = norm_vertices.astype(np.float16)

    return vertices_seq


# =====================================================================================
# 4. RENDERING / LIGHTING
# =====================================================================================


def enable_cycles_device(device: str = "GPU", backend: str = "CUDA"):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    device = str(device).upper()
    backend = str(backend).upper()
    prefs = bpy.context.preferences.addons["cycles"].preferences

    if device == "CPU":
        try:
            prefs.compute_device_type = "NONE"
        except Exception:
            pass
        try:
            prefs.refresh_devices()
        except AttributeError:
            try:
                prefs.get_devices()
            except Exception:
                pass
        for d in getattr(prefs, "devices", []):
            d.use = (d.type == "CPU")
        scene.cycles.device = "CPU"
        return

    prefs.compute_device_type = backend
    try:
        prefs.refresh_devices()
    except AttributeError:
        prefs.get_devices()

    found_gpu = False
    for d in prefs.devices:
        d.use = (d.type != "CPU")
        if d.type != "CPU" and d.use:
            found_gpu = True

    scene.cycles.device = "GPU"
    if not found_gpu:
        raise RuntimeError(f"No usable GPU found for Cycles backend={backend}.")


def setup_renderer(
    resolution=512,
    engine="BLENDER_EEVEE",
    transparent_bg=True,
    cycles_samples: int = 64,
    cycles_use_denoising: bool = False,
    cycles_device: str = "GPU",
):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent_bg else "RGB"
    scene.render.film_transparent = transparent_bg
    scene.render.use_file_extension = True
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.cycles.tile_size = 8192

    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    bpy.context.scene.render.filter_size = 1.5

    if engine == "CYCLES":
        scene.cycles.device = str(cycles_device).upper()
        scene.cycles.samples = int(cycles_samples)
        scene.cycles.use_denoising = bool(cycles_use_denoising)
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = 0.02


def _ensure_world_node_tree():
    """Ensure scene.world exists with nodes enabled; clear nodes and return (world, nodes, links)."""
    scene = bpy.context.scene
    if scene.world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    else:
        world = scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    return world, nodes, links


def _delete_all_lights():
    """Delete every LIGHT object currently in the scene."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()


def init_uniform_lighting():
    """Pure white environment background (used once at scene init)."""
    _delete_all_lights()
    _, nodes, links = _ensure_world_node_tree()

    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def init_random_lighting(camera_dir: np.ndarray, rng: np.random.Generator) -> dict:
    """Per-view lighting: 1-3 random POINT lights with energy balanced against camera direction,
    plus a residual white environment background. Mirrors data_toolkit/blender_script/render_cond.py.

    Returns a metadata dict describing the sampled lights for the output JSON.
    """
    _delete_all_lights()
    _, nodes, links = _ensure_world_node_tree()

    num_lights = int(rng.integers(low=1, high=4))  # 1, 2, or 3 (high is exclusive)
    total_strength = 1.5
    lights_meta = []
    cam_dir_np = np.asarray(camera_dir, dtype=np.float64)
    for i in range(num_lights):
        new_light = bpy.data.objects.new(
            f"Light_{i}", bpy.data.lights.new(f"Light_{i}", type="POINT")
        )
        bpy.context.collection.objects.link(new_light)

        new_light_distance = 1.0 / float(rng.uniform(1.0 / 100.0, 1.0 / 10.0))
        new_light_dir = rng.standard_normal(3)
        new_light_dir[2] += 0.6
        new_light_dir = new_light_dir / np.linalg.norm(new_light_dir)
        new_light_location = new_light_dir * new_light_distance
        dot = float(np.sum(cam_dir_np * new_light_dir))
        new_light_camera_strength_ratio = max(dot * 0.5 + 0.5, 0.0)
        new_light_max_energy = total_strength / (dot * 0.45 + 0.55)
        new_light_strength = float(np.sqrt(rng.uniform(0.01, 1.0))) * new_light_max_energy
        new_light_camera_strength = new_light_camera_strength_ratio * new_light_strength
        total_strength -= new_light_camera_strength

        new_light.location = (
            float(new_light_location[0]),
            float(new_light_location[1]),
            float(new_light_location[2]),
        )
        new_light.data.color = (1.0, 1.0, 1.0)
        new_light.data.energy = float(new_light_strength * new_light_distance ** 2 * 31.4)
        new_light.data.shadow_soft_size = float(rng.uniform(0.1, 0.1 * new_light_distance))

        lights_meta.append({
            "location": [float(x) for x in new_light_location],
            "distance": float(new_light_distance),
            "energy": float(new_light.data.energy),
            "shadow_soft_size": float(new_light.data.shadow_soft_size),
        })

    bg_strength = float(max(total_strength, 0.3))
    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = bg_strength
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    return {
        "lighting_type": "random_point_lights",
        "num_lights": int(num_lights),
        "bg_strength": bg_strength,
        "lights": lights_meta,
    }


def render_frame(output_path: str):
    scene = bpy.context.scene
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def setup_normal_output(normal_root_dir: str):
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    view_layer.use_pass_normal = True

    scene.use_nodes = True
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True

    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    rlayers = nodes.new(type="CompositorNodeRLayers")
    rlayers.location = (-500, 0)

    composite = nodes.new(type="CompositorNodeComposite")
    composite.location = (350, 120)
    links.new(rlayers.outputs["Image"], composite.inputs["Image"])

    normal_mul = nodes.new(type="CompositorNodeMixRGB")
    normal_mul.blend_type = "MULTIPLY"
    normal_mul.inputs[0].default_value = 1.0
    normal_mul.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    normal_mul.location = (-150, -80)
    links.new(rlayers.outputs["Normal"], normal_mul.inputs[1])

    normal_add = nodes.new(type="CompositorNodeMixRGB")
    normal_add.blend_type = "ADD"
    normal_add.inputs[0].default_value = 1.0
    normal_add.inputs[2].default_value = (0.5, 0.5, 0.5, 0.0)
    normal_add.location = (80, -80)
    links.new(normal_mul.outputs["Image"], normal_add.inputs[1])

    set_alpha = nodes.new(type="CompositorNodeSetAlpha")
    set_alpha.location = (300, -80)
    links.new(normal_add.outputs["Image"], set_alpha.inputs["Image"])
    links.new(rlayers.outputs["Alpha"], set_alpha.inputs["Alpha"])

    file_output = nodes.new(type="CompositorNodeOutputFile")
    file_output.location = (550, -80)
    file_output.base_path = normal_root_dir
    slot = file_output.file_slots[0]
    slot.path = "frame_"
    slot.use_node_format = True
    slot.save_as_render = False

    file_output.format.file_format = "PNG"
    file_output.format.color_mode = "RGBA"
    file_output.format.color_depth = "16"

    links.new(set_alpha.outputs["Image"], file_output.inputs[0])
    return file_output


def update_normal_output_path(file_output_node, base_dir: str, prefix: str):
    os.makedirs(base_dir, exist_ok=True)
    file_output_node.base_path = base_dir
    file_output_node.file_slots[0].path = prefix


# =====================================================================================
# 5. CAMERA FITTING
# =====================================================================================


def compute_camera_axes_from_angles(azim: float, elev: float):
    cam_pos_unit = orbit_offset(1.0, azim, elev)
    forward = normalize(-cam_pos_unit)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize(np.cross(forward, world_up))
    if np.linalg.norm(right) < 1e-6:
        up_alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = normalize(np.cross(forward, up_alt))
    true_up = normalize(np.cross(right, forward))
    return right, true_up, forward


def compute_world_to_camera_aligned_rotation(azim: float, elev: float):
    right, up, forward = compute_camera_axes_from_angles(azim, elev)
    R = np.stack([right, up, forward], axis=0).astype(np.float32)
    return R, right, up, forward


def get_bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray):
    x0, y0, z0 = bbox_min.tolist()
    x1, y1, z1 = bbox_max.tolist()
    return np.array([
        [x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
        [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1],
    ], dtype=np.float32)


def compute_camera_space_scene_box_from_vertices_seq(vertices_seq: np.ndarray, rot_world_to_cam_aligned: np.ndarray):
    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    rot_t = rot_world_to_cam_aligned.T.astype(np.float32)

    for f in range(vertices_seq.shape[0]):
        verts32 = np.asarray(vertices_seq[f], dtype=np.float32)
        verts_cam = verts32 @ rot_t
        global_min = np.minimum(global_min, verts_cam.min(axis=0))
        global_max = np.maximum(global_max, verts_cam.max(axis=0))

    return global_min.astype(np.float32), global_max.astype(np.float32)


def compute_bbox_unit_normalization_scale(bbox_min: np.ndarray, bbox_max: np.ndarray):
    extent = np.asarray(bbox_max, dtype=np.float32) - np.asarray(bbox_min, dtype=np.float32)
    max_extent = float(np.max(extent))
    scale = 1.0 / max_extent if max_extent > 1e-6 else 1.0
    return float(scale), extent.astype(np.float32)


def compute_tight_camera_distance_for_aligned_bbox(cam_obj, bbox_min_aligned: np.ndarray, bbox_max_aligned: np.ndarray, frame_padding: float = 0.03, fit_safety: float = 1.02):
    corners = get_bbox_corners(np.asarray(bbox_min_aligned, dtype=np.float32), np.asarray(bbox_max_aligned, dtype=np.float32))

    half_fov_x = 0.5 * float(cam_obj.data.angle_x)
    half_fov_y = 0.5 * float(cam_obj.data.angle_y)
    tan_half_fov_x = max(math.tan(half_fov_x), 1e-6)
    tan_half_fov_y = max(math.tan(half_fov_y), 1e-6)

    fill_ratio = max(1e-3, 1.0 - float(frame_padding))
    d_required = 0.0
    for p in corners:
        px = abs(float(p[0]))
        py = abs(float(p[1]))
        pz = float(p[2])
        req_x = px / (fill_ratio * tan_half_fov_x) - pz
        req_y = py / (fill_ratio * tan_half_fov_y) - pz
        d_required = max(d_required, req_x, req_y)

    min_pz = float(np.min(corners[:, 2]))
    d_required = max(d_required, -min_pz + 1e-4)
    d_required = max(d_required * float(fit_safety), 1e-4)
    return float(d_required)


def create_static_multiview_cameras(
    num_cameras: int,
    seed: int,
    normalized_vertices_seq: np.ndarray,
    resolution: int,
    elev_min_deg: float = 0.0,
    elev_max_deg: float = 80.0,
    frame_padding: float = 0.03,
    fit_safety: float = 1.02,
    distance_jitter_scale: float = 1.04,
    camera_sensor_size: float = 36.0,
):
    rng = np.random.default_rng(seed)
    distance_jitter_scale = max(1.0, float(distance_jitter_scale))

    azim0 = float(rng.uniform(0.0, 2.0 * math.pi))
    azims = azim0 + np.arange(num_cameras, dtype=np.float32) * (2.0 * math.pi / num_cameras)
    elevs_deg = rng.uniform(elev_min_deg, elev_max_deg, size=num_cameras).astype(np.float32)

    camera_objs = []
    camera_infos = []
    target = np.zeros(3, dtype=np.float32)

    for k in range(num_cameras):
        cam_obj = create_camera(
            name=f"TrackingCamera_{k:02d}",
            lens=50.0,
            sensor_width=float(camera_sensor_size),
            sensor_height=float(camera_sensor_size),
        )

        azim = float(azims[k] % (2.0 * math.pi))
        elev_deg = float(elevs_deg[k])
        elev = math.radians(elev_deg)

        rot_world_to_cam_aligned, right, up, forward = compute_world_to_camera_aligned_rotation(azim=azim, elev=elev)
        camera_space_bbox_min, camera_space_bbox_max = compute_camera_space_scene_box_from_vertices_seq(
            normalized_vertices_seq, rot_world_to_cam_aligned=rot_world_to_cam_aligned,
        )

        view_specific_scale, camera_space_extent = compute_bbox_unit_normalization_scale(camera_space_bbox_min, camera_space_bbox_max)
        normalized_camera_space_bbox_min = (camera_space_bbox_min * view_specific_scale).astype(np.float32)
        normalized_camera_space_bbox_max = (camera_space_bbox_max * view_specific_scale).astype(np.float32)

        tight_distance_in_view_normalized_space = compute_tight_camera_distance_for_aligned_bbox(
            cam_obj=cam_obj,
            bbox_min_aligned=normalized_camera_space_bbox_min,
            bbox_max_aligned=normalized_camera_space_bbox_max,
            frame_padding=frame_padding,
            fit_safety=fit_safety,
        )

        tight_distance = float(tight_distance_in_view_normalized_space / max(view_specific_scale, 1e-8))
        distance = tight_distance * float(rng.uniform(1.0, distance_jitter_scale))
        cam_pos = orbit_offset(distance, azim, elev)
        cam2world = look_at(cam_pos, target)
        set_camera_from_cam2world(cam_obj, cam2world)
        intrinsics = get_camera_intrinsics_dict(cam_obj, resolution=resolution)

        camera_infos.append({
            "camera_name": cam_obj.name,
            "view_index": int(k),
            "azimuth_deg": float(np.degrees(azim)),
            "elevation_deg": float(elev_deg),
            "distance": float(distance),
            "camera_c2w": cam2world.tolist(),
            "camera_pos": cam_pos.tolist(),
            "intrinsics": intrinsics,
        })
        camera_objs.append(cam_obj)

    return camera_objs, camera_infos


# =====================================================================================
# 6. CORE WORKER
# =====================================================================================


def emit_view_done(sha256: str, view_idx: int, status: str, **extra):
    """Emit a single-line marker to stdout for the batch layer to parse."""
    payload = {"sha256": sha256, "view_index": int(view_idx), "status": status}
    payload.update(extra)
    print("[VIEW_DONE] " + json.dumps(payload, ensure_ascii=False), flush=True)


def process_geometry(args):
    timer = StageTimer()
    print("--- Starting Dynamic OBJ Rendering ---")

    sha256 = args.sha256
    obj_root = args.obj_root  # local dir for this obj's outputs
    os.makedirs(obj_root, exist_ok=True)

    all_view_indices = get_render_view_indices(args.num_cameras, args.camera_stride)
    print(f"All view indices = {all_view_indices}")

    if args.render_view_indices is not None and len(args.render_view_indices) > 0:
        requested = [int(v) for v in args.render_view_indices]
        invalid = [v for v in requested if v not in all_view_indices]
        if invalid:
            raise ValueError(f"--render_view_indices contains invalid view ids: {invalid}")
        render_view_indices = requested
    else:
        render_view_indices = all_view_indices
    print(f"To render view indices = {render_view_indices}")

    if len(render_view_indices) == 0:
        print("Nothing to render (empty render_view_indices). Exiting.")
        return

    print(f"Loading object from: {args.object_path}")
    init_scene()
    load_object(args.object_path)
    timer.log("scene init + load object")

    sequence_normalizer = create_sequence_normalizer()
    timer.log("create sequence normalizer")

    if args.render_engine == "CYCLES":
        enable_cycles_device(device=args.cycles_device, backend=args.cycles_backend)

    setup_renderer(
        resolution=args.resolution,
        engine=args.render_engine,
        transparent_bg=args.transparent_bg,
        cycles_samples=args.cycles_samples,
        cycles_use_denoising=args.cycles_use_denoising,
        cycles_device=args.cycles_device,
    )

    # Compositor output for normals (re-routed per view in the render loop).
    normal_output_node = None
    if args.render_normal_map:
        # Use obj_root as a placeholder base; we will override per view.
        normal_output_node = setup_normal_output(obj_root)
    timer.log("renderer setup")

    mesh_objs = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.hide_render:
            continue
        if not obj.visible_get(view_layer=bpy.context.view_layer):
            continue
        mesh_objs.append(obj)
    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects found after loading.")
    print(f"Found {len(mesh_objs)} mesh objects")

    # Resolve animation frame range
    actions = bpy.data.actions
    raw_frame_start, raw_frame_end = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
    if actions:
        ranges = [action.frame_range for action in actions]
        raw_frame_start = int(min(r[0] for r in ranges))
        raw_frame_end = int(max(r[1] for r in ranges))

    frame_indices = np.arange(raw_frame_start, raw_frame_end + 1, dtype=np.int32)
    print(f"Raw animation range: [{raw_frame_start}, {raw_frame_end}] ({len(frame_indices)} frames)")

    MAX_FRAMES = int(args.max_frames)
    if len(frame_indices) > MAX_FRAMES:
        crop_start = (len(frame_indices) - MAX_FRAMES) // 2
        frame_indices = frame_indices[crop_start : crop_start + MAX_FRAMES]
        print(f"Center-cropped to {MAX_FRAMES} frames: [{int(frame_indices[0])}, {int(frame_indices[-1])}]")

    # First pass: topology check + normalization
    (
        shared_faces,
        global_center,
        sequence_scale,
        canonical_bbox_min,
        canonical_bbox_max,
        raw_vertices_cache,
    ) = compute_sequence_normalization_params_and_cache(
        mesh_objs, frame_indices, cache_raw_vertices=True, raw_cache_dtype=np.float16,
    )
    timer.log("first pass: topology + normalization + cache")
    print(f"Sequence normalization: center={global_center.tolist()}, scale={sequence_scale}")

    vertices_seq = precompute_normalized_mesh_sequence_from_cache(
        raw_vertices_cache=raw_vertices_cache,
        global_center=global_center,
        global_scale=sequence_scale,
        shared_faces=shared_faces,
        frame_indices=frame_indices,
    )
    timer.log("normalize cached mesh sequence")

    # Cameras: deterministic for the WHOLE obj (use num_cameras so all_view ids match across resumes).
    camera_seed = int(args.traj_seed + args.traj_id * 9973)
    camera_objs, camera_infos = create_static_multiview_cameras(
        num_cameras=args.num_cameras,
        seed=camera_seed,
        normalized_vertices_seq=vertices_seq,
        resolution=args.resolution,
        elev_min_deg=args.camera_elev_min_deg,
        elev_max_deg=args.camera_elev_max_deg,
        frame_padding=args.camera_frame_padding,
        fit_safety=args.camera_fit_safety,
        distance_jitter_scale=args.camera_distance_jitter_scale,
        camera_sensor_size=args.camera_sensor_size,
    )
    timer.log("create static cameras")

    # Lighting: uniform white env once; sample random POINT lights deterministically per view.
    init_uniform_lighting()
    bpy.context.scene.camera = camera_objs[render_view_indices[0]]
    timer.log("lighting setup")

    # Common object-level metadata, baked into every per-view result.json.
    common_meta = {
        "object_path": args.object_path,
        "sha256": sha256,
        "num_frames": int(len(frame_indices)),
        "frame_start": int(frame_indices[0]),
        "frame_end": int(frame_indices[-1]),
        "raw_frame_start": int(raw_frame_start),
        "raw_frame_end": int(raw_frame_end),
        "num_vertices": int(vertices_seq.shape[1]),
        "num_faces": int(shared_faces.shape[0]),
        "global_center": global_center.tolist(),
        "sequence_scale": float(sequence_scale),
        "canonical_bbox_min": canonical_bbox_min.tolist(),
        "canonical_bbox_max": canonical_bbox_max.tolist(),
        "num_cameras": int(args.num_cameras),
        "camera_stride": int(args.camera_stride),
        "all_view_indices": [int(x) for x in all_view_indices],
        "resolution": int(args.resolution),
        "render_engine": args.render_engine,
        "video_fps": int(args.video_fps),
    }

    lighting_seed_base = int(args.traj_seed + args.traj_id * 10007 + 424242)

    # ============================ Render per view ============================
    # Post-processing (mp4 encode + mesh.npz + result.json + emit_view_done) is
    # offloaded to a background thread pool so it overlaps with rendering of
    # subsequent views. emit_view_done only fires once that view's post-process
    # is fully on disk, so the batch layer's S3 upload sees a complete view dir.
    encode_workers = max(1, int(getattr(args, "encode_workers", 2)))
    encode_pool = ThreadPoolExecutor(max_workers=encode_workers, thread_name_prefix="encode")
    encode_futures: List = []
    encode_blocked_total = 0.0
    encode_task_total = 0.0
    render_stage_t0 = time.perf_counter()
    sync_total = 0.0
    render_total = 0.0

    def _post_process_view(
        view_idx: int,
        view_out_dir: str,
        view_png_rgb_dir: str,
        view_png_normal_dir: str,
        lighting_meta: dict,
        view_t0: float,
    ) -> float:
        view_name = f"view_{view_idx:02d}"
        bg_t0 = time.perf_counter()
        try:
            rgb_mp4_path = os.path.join(view_out_dir, "rgb.mp4")
            encode_view_pngs_to_mp4(view_png_rgb_dir, rgb_mp4_path, fps=args.video_fps, modal="rgb")
            shutil.rmtree(view_png_rgb_dir, ignore_errors=True)

            if args.render_normal_map:
                normal_mp4_path = os.path.join(view_out_dir, "normal.mp4")
                encode_view_pngs_to_mp4(view_png_normal_dir, normal_mp4_path, fps=args.video_fps, modal="normal")
                shutil.rmtree(view_png_normal_dir, ignore_errors=True)

            mesh_npz_path = os.path.join(view_out_dir, "mesh.npz")
            np.savez(mesh_npz_path, vertices=vertices_seq, faces=shared_faces, frame_indices=frame_indices)

            view_result = dict(common_meta)
            view_result.update({
                "view_index": int(view_idx),
                "camera_info": camera_infos[view_idx],
                "lighting": lighting_meta,
                "rgb_mp4": "rgb.mp4",
                "normal_mp4": "normal.mp4" if args.render_normal_map else None,
                "mesh_npz": "mesh.npz",
                "render_time_s": float(time.perf_counter() - view_t0),
                "status": "success",
            })
            atomic_write_json(os.path.join(view_out_dir, "result.json"), view_result)
            emit_view_done(
                sha256=sha256,
                view_idx=view_idx,
                status="success",
                render_time_s=float(time.perf_counter() - view_t0),
            )
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[ERROR] view {view_name} post-process failed: {err_msg}")
            for p in (view_png_rgb_dir, view_png_normal_dir):
                shutil.rmtree(p, ignore_errors=True)
            for p in ("rgb.mp4", "normal.mp4", "mesh.npz", "result.json"):
                fp = os.path.join(view_out_dir, p)
                if os.path.isfile(fp):
                    try:
                        os.remove(fp)
                    except Exception:
                        pass
            emit_view_done(
                sha256=sha256,
                view_idx=view_idx,
                status="error",
                error=err_msg,
            )
        return time.perf_counter() - bg_t0

    try:
        for view_idx in render_view_indices:
            view_t0 = time.perf_counter()
            view_name = f"view_{view_idx:02d}"
            view_out_dir = view_dir_for(obj_root, view_idx)
            os.makedirs(view_out_dir, exist_ok=True)

            view_lighting_rng = np.random.default_rng(lighting_seed_base + int(view_idx))
            cam_pos_np = np.asarray(camera_infos[view_idx]["camera_pos"], dtype=np.float64)
            cam_pos_norm = float(np.linalg.norm(cam_pos_np))
            cam_dir_np = cam_pos_np / cam_pos_norm if cam_pos_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
            lighting_meta = init_random_lighting(cam_dir_np, view_lighting_rng)
            lighting_meta["view_index"] = int(view_idx)

            bpy.context.scene.camera = camera_objs[view_idx]

            view_png_rgb_dir = os.path.join(view_out_dir, "_png_rgb")
            view_png_normal_dir = os.path.join(view_out_dir, "_png_normal")
            os.makedirs(view_png_rgb_dir, exist_ok=True)
            if args.render_normal_map:
                os.makedirs(view_png_normal_dir, exist_ok=True)
                update_normal_output_path(normal_output_node, base_dir=view_png_normal_dir, prefix="frame_")

            try:
                for local_frame_idx, source_frame in enumerate(frame_indices):
                    frame_int = int(source_frame)
                    sync_t0 = time.perf_counter()
                    bpy.context.scene.frame_set(frame_int)
                    apply_sequence_normalization(
                        sequence_normalizer, global_center=global_center, global_scale=sequence_scale,
                    )
                    bpy.context.view_layer.update()
                    sync_total += time.perf_counter() - sync_t0

                    rt0 = time.perf_counter()
                    rgb_path = os.path.join(view_png_rgb_dir, f"frame_{frame_int:04d}.png")
                    render_frame(rgb_path)
                    render_total += time.perf_counter() - rt0
                    if (local_frame_idx + 1) % 10 == 0 or local_frame_idx == 0:
                        print(f"  [{view_name}] frame {local_frame_idx + 1}/{len(frame_indices)} (#{frame_int})")

                # Throttle so disk/mem stays bounded if encode falls behind.
                while sum(1 for f in encode_futures if not f.done()) >= encode_workers:
                    wt0 = time.perf_counter()
                    for f in encode_futures:
                        if not f.done():
                            try:
                                f.result()
                            except Exception as e:
                                print(f"[ERROR] encode task raised: {e}")
                            break
                    encode_blocked_total += time.perf_counter() - wt0

                fut = encode_pool.submit(
                    _post_process_view,
                    view_idx,
                    view_out_dir,
                    view_png_rgb_dir,
                    view_png_normal_dir,
                    lighting_meta,
                    view_t0,
                )
                encode_futures.append(fut)
                print(f"  [{view_name}] render done in {time.perf_counter() - view_t0:.2f}s, encode queued")
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                print(f"[ERROR] view {view_name} render failed: {err_msg}")
                for p in (view_png_rgb_dir, view_png_normal_dir):
                    shutil.rmtree(p, ignore_errors=True)
                for p in ("rgb.mp4", "normal.mp4", "mesh.npz", "result.json"):
                    fp = os.path.join(view_out_dir, p)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
                emit_view_done(
                    sha256=sha256,
                    view_idx=view_idx,
                    status="error",
                    error=err_msg,
                )

        render_only_t = time.perf_counter() - render_stage_t0
        print(
            f"[timing] view-major render-only: {render_only_t:.3f}s "
            f"(sync_total={sync_total:.2f}s, render_total={render_total:.2f}s, "
            f"encode_blocked={encode_blocked_total:.2f}s, "
            f"frames={len(frame_indices)}, views={len(render_view_indices)})"
        )
        drain_t0 = time.perf_counter()
        for f in encode_futures:
            try:
                encode_task_total += f.result()
            except Exception as e:
                print(f"[ERROR] encode task raised: {e}")
        drain_t = time.perf_counter() - drain_t0
        loop_t = time.perf_counter() - render_stage_t0
        print(
            f"[timing] view-major loop: {loop_t:.3f}s "
            f"(encode_drain_wait={drain_t:.2f}s, encode_task_total={encode_task_total:.2f}s, "
            f"encode_workers={encode_workers})"
        )
    finally:
        encode_pool.shutdown(wait=True)
    timer.log("render all views")
    print("--- Processing Complete ---")


# =====================================================================================
# 7. MAIN
# =====================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render animated 3D files with Blender.")
    parser.add_argument("--object_path", type=str, required=True)
    parser.add_argument("--obj_root", type=str, required=True,
                        help="Local output directory for this obj (contains per-view subdirs).")
    parser.add_argument("--sha256", type=str, required=True,
                        help="Object identifier; emitted in [VIEW_DONE] markers.")
    parser.add_argument("--render_view_indices", type=int, nargs="*", default=None,
                        help="Subset of view indices to render (default: all views).")

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--render_engine", type=str, default="CYCLES", choices=["BLENDER_EEVEE", "CYCLES"])
    parser.add_argument("--transparent_bg", action="store_true")

    parser.add_argument("--traj_id", type=int, default=0)
    parser.add_argument("--traj_seed", type=int, default=0)

    parser.add_argument("--num_cameras", type=int, default=16)
    parser.add_argument("--camera_stride", type=int, default=1)
    parser.add_argument("--camera_elev_min_deg", type=float, default=0.0)
    parser.add_argument("--camera_elev_max_deg", type=float, default=80.0)
    parser.add_argument("--camera_frame_padding", type=float, default=0.03)
    parser.add_argument("--camera_fit_safety", type=float, default=1.02)
    parser.add_argument("--camera_distance_jitter_scale", type=float, default=1.04)
    parser.add_argument("--camera_sensor_size", type=float, default=36.0)

    parser.add_argument("--render_normal_map", dest="render_normal_map", action="store_true")
    parser.add_argument("--no_render_normal_map", dest="render_normal_map", action="store_false")
    parser.set_defaults(render_normal_map=True)

    parser.add_argument("--cycles_backend", type=str, default="OPTIX", choices=["CUDA", "OPTIX"])
    parser.add_argument("--cycles_samples", type=int, default=256)
    parser.add_argument("--cycles_use_denoising", action="store_true")
    parser.add_argument("--cycles_device", type=str, default="GPU", choices=["GPU", "CPU"])

    parser.add_argument("--video_fps", type=int, default=24)
    parser.add_argument("--max_frames", type=int, default=121,
                        help="Cap on per-obj animation length (center-cropped). Default 121.")
    parser.add_argument("--encode_workers", type=int, default=2,
                        help="Background threads for mp4 encode (overlaps with rendering).")

    args = parser.parse_args(get_cli_argv())
    process_geometry(args)
