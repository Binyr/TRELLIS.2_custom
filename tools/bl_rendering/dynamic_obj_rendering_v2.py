#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dynamic_obj_rendering_v2.py

Frame-major (inverted) variant of dynamic_obj_rendering.py.

Key differences vs v1:
- Outer loop iterates frames, inner loop iterates views, so each frame's mesh
  is uploaded / BVH-built only once per obj instead of once per view.
- All views in an obj are rendered as a single unit (obj-level resume only).
- Output layout:
    <obj_root>/
        mesh.npz                   # single shared geometry for the whole obj
        result.json                # single obj-level metadata, "views" sub-dict
        result_rgb_mp4/
            view_00.mp4
            view_02.mp4
            ...
- Emits a single [OBJ_DONE] marker on stdout when finished (success or error).
- Normal-map rendering is intentionally not supported in v2.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import bpy
import numpy as np
from mathutils import Matrix

try:
    import av
except Exception:
    av = None


# =====================================================================================
# 0. UTILS / TIMING
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


def list_png_files_natural(view_dir: str) -> List[str]:
    view_dir = Path(view_dir)
    if not view_dir.is_dir():
        return []
    pngs = [str(p) for p in view_dir.glob("*.png") if p.is_file()]
    pngs = sorted(pngs, key=lambda x: natural_key(Path(x).name))
    return pngs


def load_png_as_float01_chw_with_blender(image_path: str) -> np.ndarray:
    img = bpy.data.images.load(image_path, check_existing=False)
    try:
        width = int(img.size[0])
        height = int(img.size[1])
        channels = int(img.channels)
        pixels = np.array(img.pixels[:], dtype=np.float32)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid image size for {image_path}: {(width, height)}")
        pixels = pixels.reshape(height, width, channels)
        pixels = pixels[::-1, :, :]
        if channels >= 3:
            rgb = pixels[:, :, :3]
        else:
            rgb = np.repeat(pixels[:, :, :1], 3, axis=2)
        rgb = np.clip(rgb, 0.0, 1.0)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32, copy=False)
        return chw
    finally:
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass


def write_rgb_video_streaming_from_pngs(image_paths: List[str], save_path: str, fps: int = 24):
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
    stream.options = {"crf": "10"}

    def encode_one_frame(chw: np.ndarray):
        hwc = np.transpose(chw, (1, 2, 0))
        frame = np.clip(hwc * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        yuv_frame = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv420p10le")
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


def encode_view_pngs_to_mp4(png_dir: str, mp4_path: str, fps: int):
    pngs = list_png_files_natural(png_dir)
    if len(pngs) == 0:
        raise RuntimeError(f"No PNG frames to encode in {png_dir}")
    print(f"[video] Encoding rgb mp4: {len(pngs)} frames -> {mp4_path}")
    write_rgb_video_streaming_from_pngs(pngs, mp4_path, fps=fps)


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
    depsgraph = bpy.context.evaluated_depsgraph_get()

    shared_faces = None
    raw_vertices_cache = [] if cache_raw_vertices else None

    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    for i in frame_indices:
        scene.frame_set(int(i))
        bpy.context.view_layer.update()
        raw_vertices, raw_faces = extract_merged_mesh_world_fast(mesh_objs, depsgraph=depsgraph)

        if shared_faces is None:
            shared_faces = raw_faces.copy()
            print(
                f"Reference topology set from frame {int(i)}: "
                f"{shared_faces.shape[0]} faces, {raw_vertices.shape[0]} vertices"
            )
        else:
            if raw_faces.shape != shared_faces.shape or not np.array_equal(raw_faces, shared_faces):
                raise RuntimeError(
                    f"Topology changed at frame {int(i)}. "
                    f"Reference faces shape={shared_faces.shape}, current faces shape={raw_faces.shape}."
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
    engine="CYCLES",
    transparent_bg=True,
    cycles_samples: int = 256,
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
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()


def init_uniform_lighting():
    _delete_all_lights()
    _, nodes, links = _ensure_world_node_tree()
    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def sample_view_lighting_config(camera_dir: np.ndarray, rng: np.random.Generator) -> dict:
    """Pre-compute (without touching the scene) the lighting config for one view,
    matching the random-point-lights distribution used in v1.
    Returns a dict with keys: lights (list of {location, energy, shadow_soft_size}),
    bg_strength, num_lights, lighting_type.
    """
    num_lights = int(rng.integers(low=1, high=4))
    total_strength = 1.5
    lights = []
    cam_dir_np = np.asarray(camera_dir, dtype=np.float64)
    for _ in range(num_lights):
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

        energy = float(new_light_strength * new_light_distance ** 2 * 31.4)
        shadow_soft_size = float(rng.uniform(0.1, 0.1 * new_light_distance))

        lights.append({
            "location": [float(x) for x in new_light_location],
            "distance": float(new_light_distance),
            "energy": energy,
            "shadow_soft_size": shadow_soft_size,
        })

    bg_strength = float(max(total_strength, 0.0))
    return {
        "lighting_type": "random_point_lights",
        "num_lights": int(num_lights),
        "bg_strength": bg_strength,
        "lights": lights,
    }


def apply_view_lighting(lighting_cfg: dict):
    """Tear down current lights and instantiate the lighting described by `lighting_cfg`."""
    _delete_all_lights()
    _, nodes, links = _ensure_world_node_tree()

    for i, light in enumerate(lighting_cfg["lights"]):
        new_light = bpy.data.objects.new(
            f"Light_{i}", bpy.data.lights.new(f"Light_{i}", type="POINT")
        )
        bpy.context.collection.objects.link(new_light)
        loc = light["location"]
        new_light.location = (float(loc[0]), float(loc[1]), float(loc[2]))
        new_light.data.color = (1.0, 1.0, 1.0)
        new_light.data.energy = float(light["energy"])
        new_light.data.shadow_soft_size = float(light["shadow_soft_size"])

    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = float(lighting_cfg["bg_strength"])
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def render_frame(output_path: str):
    scene = bpy.context.scene
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


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

        view_specific_scale, _ = compute_bbox_unit_normalization_scale(camera_space_bbox_min, camera_space_bbox_max)
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
# 6. CORE WORKER (frame-major)
# =====================================================================================


def emit_obj_done(sha256: str, status: str, **extra):
    """Emit a single-line marker to stdout for the batch layer to parse."""
    payload = {"sha256": sha256, "status": status}
    payload.update(extra)
    print("[OBJ_DONE] " + json.dumps(payload, ensure_ascii=False), flush=True)


def process_geometry(args):
    timer = StageTimer()
    print("--- Starting Dynamic OBJ Rendering (v2, frame-major) ---")

    sha256 = args.sha256
    obj_root = args.obj_root
    os.makedirs(obj_root, exist_ok=True)
    rgb_mp4_dir = os.path.join(obj_root, "result_rgb_mp4")
    os.makedirs(rgb_mp4_dir, exist_ok=True)

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

    actions = bpy.data.actions
    raw_frame_start, raw_frame_end = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
    if actions:
        ranges = [action.frame_range for action in actions]
        raw_frame_start = int(min(r[0] for r in ranges))
        raw_frame_end = int(max(r[1] for r in ranges))

    frame_indices = np.arange(raw_frame_start, raw_frame_end + 1, dtype=np.int32)
    print(f"Raw animation range: [{raw_frame_start}, {raw_frame_end}] ({len(frame_indices)} frames)")

    MAX_FRAMES = 121
    if len(frame_indices) > MAX_FRAMES:
        crop_start = (len(frame_indices) - MAX_FRAMES) // 2
        frame_indices = frame_indices[crop_start : crop_start + MAX_FRAMES]
        print(f"Center-cropped to {MAX_FRAMES} frames: [{int(frame_indices[0])}, {int(frame_indices[-1])}]")

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
    )
    timer.log("normalize cached mesh sequence")

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

    # Pre-sample lighting per view (deterministic). The scene's actual lights
    # are swapped in the inner view loop via apply_view_lighting().
    lighting_seed_base = int(args.traj_seed + args.traj_id * 10007 + 424242)
    init_uniform_lighting()  # bootstrap, replaced before the first frame anyway
    view_lighting_cfgs: Dict[int, dict] = {}
    for view_idx in render_view_indices:
        view_lighting_rng = np.random.default_rng(lighting_seed_base + int(view_idx))
        cam_pos_np = np.asarray(camera_infos[view_idx]["camera_pos"], dtype=np.float64)
        cam_pos_norm = float(np.linalg.norm(cam_pos_np))
        cam_dir_np = cam_pos_np / cam_pos_norm if cam_pos_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
        cfg = sample_view_lighting_config(cam_dir_np, view_lighting_rng)
        cfg["view_index"] = int(view_idx)
        view_lighting_cfgs[int(view_idx)] = cfg
    bpy.context.scene.camera = camera_objs[render_view_indices[0]]
    timer.log("lighting precompute")

    # Per-view scratch dirs for PNGs.
    view_png_dirs: Dict[int, str] = {}
    for view_idx in render_view_indices:
        d = os.path.join(obj_root, f"_png_view_{view_idx:02d}")
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        view_png_dirs[int(view_idx)] = d

    # ============================ Frame-major render loop ============================
    render_stage_t0 = time.perf_counter()
    sync_total = 0.0
    render_total = 0.0
    num_frames = len(frame_indices)
    num_views = len(render_view_indices)

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

            for view_idx in render_view_indices:
                vlt0 = time.perf_counter()
                apply_view_lighting(view_lighting_cfgs[view_idx])
                bpy.context.scene.camera = camera_objs[view_idx]
                # Light/camera swap counts as scene sync, but it should be tiny.
                sync_total += time.perf_counter() - vlt0

                rt0 = time.perf_counter()
                rgb_path = os.path.join(view_png_dirs[view_idx], f"frame_{frame_int:04d}.png")
                render_frame(rgb_path)
                render_total += time.perf_counter() - rt0

            if (local_frame_idx + 1) % 10 == 0 or local_frame_idx == 0:
                print(
                    f"  frame {local_frame_idx + 1}/{num_frames} (#{frame_int}) "
                    f"sync={sync_total:.1f}s render={render_total:.1f}s"
                )
    except Exception:
        # Clean up scratch PNG dirs on render failure so resume is fresh.
        for d in view_png_dirs.values():
            shutil.rmtree(d, ignore_errors=True)
        raise

    render_loop_t = time.perf_counter() - render_stage_t0
    print(
        f"[timing] frame-major loop: {render_loop_t:.3f}s "
        f"(sync_total={sync_total:.2f}s, render_total={render_total:.2f}s, "
        f"frames={num_frames}, views={num_views})"
    )
    timer.log("render loop")

    # Encode mp4 per view + cleanup.
    encode_t0 = time.perf_counter()
    views_meta: Dict[str, dict] = {}
    for view_idx in render_view_indices:
        png_dir = view_png_dirs[view_idx]
        mp4_path = os.path.join(rgb_mp4_dir, f"view_{view_idx:02d}.mp4")
        encode_view_pngs_to_mp4(png_dir, mp4_path, fps=args.video_fps)
        shutil.rmtree(png_dir, ignore_errors=True)
        views_meta[str(int(view_idx))] = {
            "view_index": int(view_idx),
            "camera_info": camera_infos[view_idx],
            "lighting": view_lighting_cfgs[view_idx],
            "rgb_mp4": f"result_rgb_mp4/view_{view_idx:02d}.mp4",
        }
    encode_t = time.perf_counter() - encode_t0
    print(f"[timing] mp4 encode: {encode_t:.3f}s")
    timer.log("mp4 encode")

    # Write shared mesh.npz + obj-level result.json.
    mesh_npz_path = os.path.join(obj_root, "mesh.npz")
    np.savez(mesh_npz_path, vertices=vertices_seq, faces=shared_faces, frame_indices=frame_indices)

    obj_result = {
        "sha256": sha256,
        "object_path": args.object_path,
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
        "rendered_view_indices": [int(x) for x in render_view_indices],
        "resolution": int(args.resolution),
        "render_engine": args.render_engine,
        "video_fps": int(args.video_fps),
        "render_time_s": float(time.perf_counter() - render_stage_t0),
        "sync_time_s": float(sync_total),
        "render_pt_time_s": float(render_total),
        "encode_time_s": float(encode_t),
        "mesh_npz": "mesh.npz",
        "views": views_meta,
        "status": "success",
    }
    atomic_write_json(os.path.join(obj_root, "result.json"), obj_result)
    timer.log("write mesh + result")

    emit_obj_done(
        sha256=sha256,
        status="success",
        num_frames=int(len(frame_indices)),
        num_views=int(num_views),
        render_time_s=float(time.perf_counter() - render_stage_t0),
    )
    print("--- Processing Complete ---")


# =====================================================================================
# 7. MAIN
# =====================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render animated 3D files with Blender (frame-major).")
    parser.add_argument("--object_path", type=str, required=True)
    parser.add_argument("--obj_root", type=str, required=True,
                        help="Local output directory for this obj.")
    parser.add_argument("--sha256", type=str, required=True,
                        help="Object identifier; emitted in [OBJ_DONE] marker.")
    parser.add_argument("--render_view_indices", type=int, nargs="*", default=None,
                        help="Subset of view indices to render (default: all views).")

    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--render_engine", type=str, default="CYCLES", choices=["BLENDER_EEVEE", "CYCLES"])
    parser.add_argument("--transparent_bg", action="store_true")

    parser.add_argument("--traj_id", type=int, default=0)
    parser.add_argument("--traj_seed", type=int, default=0)

    parser.add_argument("--num_cameras", type=int, default=16)
    parser.add_argument("--camera_stride", type=int, default=2)
    parser.add_argument("--camera_elev_min_deg", type=float, default=0.0)
    parser.add_argument("--camera_elev_max_deg", type=float, default=80.0)
    parser.add_argument("--camera_frame_padding", type=float, default=0.03)
    parser.add_argument("--camera_fit_safety", type=float, default=1.02)
    parser.add_argument("--camera_distance_jitter_scale", type=float, default=1.04)
    parser.add_argument("--camera_sensor_size", type=float, default=36.0)

    parser.add_argument("--cycles_backend", type=str, default="OPTIX", choices=["CUDA", "OPTIX"])
    parser.add_argument("--cycles_samples", type=int, default=256)
    parser.add_argument("--cycles_use_denoising", action="store_true")
    parser.add_argument("--cycles_device", type=str, default="GPU", choices=["GPU", "CPU"])

    parser.add_argument("--video_fps", type=int, default=24)

    args = parser.parse_args(get_cli_argv())

    try:
        process_geometry(args)
    except Exception as e:
        import traceback as _tb
        err_msg = f"{type(e).__name__}: {e}"
        print(f"[ERROR] obj {args.sha256} failed: {err_msg}")
        _tb.print_exc()
        emit_obj_done(sha256=args.sha256, status="error", error=err_msg)
        sys.exit(0)
