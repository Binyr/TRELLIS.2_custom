#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump dynamic mesh + PBR metadata for one dynamic ObjaverseXL asset.

The output is self-contained for material voxelization: animated vertices,
faces, UVs, material ids, and materials are all produced from the same Blender
load/evaluation path. A render-stage mesh.npz may still be supplied as a
sanity check that each frame's vertex set matches the rendered geometry.
"""

import argparse
import io
import json
import os
import pickle
import sys
from typing import Callable, Dict, List, Optional, Tuple

import bpy
import numpy as np
from PIL import Image

try:
    import dynamic_obj_rendering as render_mod
except Exception:
    render_mod = None


IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "glb": lambda filepath: bpy.ops.import_scene.gltf(filepath=filepath),
    "gltf": lambda filepath: bpy.ops.import_scene.gltf(filepath=filepath),
    "fbx": lambda filepath: bpy.ops.import_scene.fbx(filepath=filepath),
    "dae": lambda filepath: bpy.ops.wm.collada_import(filepath=filepath),
    "obj": lambda filepath: bpy.ops.wm.obj_import(filepath=filepath),
}


def init_scene() -> None:
    if render_mod is not None:
        render_mod.init_scene()
        return
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in list(bpy.data.textures):
        try:
            bpy.data.textures.remove(texture, do_unlink=True)
        except TypeError:
            bpy.data.textures.remove(texture)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image, do_unlink=True)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def _load_blend(filepath: str) -> None:
    bpy.ops.wm.open_mainfile(filepath=filepath)
    for obj in list(bpy.data.objects):
        if obj.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(obj, do_unlink=True)
    if bpy.context.scene.world is not None:
        bpy.context.scene.world = None


IMPORT_FUNCTIONS["blend"] = _load_blend


def load_object(object_path: str) -> None:
    if render_mod is not None:
        render_mod.load_object(object_path)
        return
    file_extension = object_path.rsplit(".", 1)[-1].lower()
    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path} (ext={file_extension})")
    IMPORT_FUNCTIONS[file_extension](filepath=object_path)


def resolve_frame_indices(max_frames: int) -> np.ndarray:
    scene = bpy.context.scene
    raw_start, raw_end = int(scene.frame_start), int(scene.frame_end)
    if bpy.data.actions:
        ranges = [action.frame_range for action in bpy.data.actions]
        raw_start = int(min(r[0] for r in ranges))
        raw_end = int(max(r[1] for r in ranges))
    frame_indices = np.arange(raw_start, raw_end + 1, dtype=np.int32)
    if len(frame_indices) > int(max_frames):
        crop_start = (len(frame_indices) - int(max_frames)) // 2
        frame_indices = frame_indices[crop_start: crop_start + int(max_frames)]
    if len(frame_indices) == 0:
        frame_indices = np.array([int(scene.frame_start)], dtype=np.int32)
    return frame_indices


def extract_image(tex_node, channels):
    image = tex_node.image
    pixels = np.array(image.pixels[:])
    data = pixels.reshape(image.size[1], image.size[0], -1)
    data = data[..., channels]

    if data.dtype != np.uint8:
        data = np.clip(data, 0.0, 1.0)
        data = (data * 255).astype(np.uint8)

    if len(data.shape) == 2:
        pil_image = Image.fromarray(data, mode="L")
    elif data.shape[2] == 3:
        pil_image = Image.fromarray(data, mode="RGB")
    elif data.shape[2] == 4:
        pil_image = Image.fromarray(data, mode="RGBA")
    else:
        raise ValueError("Unsupported channel shape for image")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return {
        "image": buffer.getvalue(),
        "interpolation": tex_node.interpolation,
        "extension": tex_node.extension,
    }


def try_extract_image(link, expected_channel="RGB"):
    assert expected_channel in ["RGB", "R", "G", "B", "A"], "Unsupported channel"

    if expected_channel == "RGB":
        assert link.from_node.type == "TEX_IMAGE", "Material is not supported"
        assert link.from_socket.name == "Color", "Material is not supported"
        return extract_image(link.from_node, [0, 1, 2])

    if expected_channel in ["R", "G", "B"]:
        socket_name = {"R": "Red", "G": "Green", "B": "Blue"}[expected_channel]
        assert link.from_node.type == "SEPARATE_COLOR" and link.from_node.mode == "RGB", \
            f"Material is not supported, {link.from_node.type}, {link.from_node.mode}"
        assert link.from_socket.name == socket_name, "Material is not supported"
        sep_node = link.from_node
        assert sep_node.inputs[0].is_linked and sep_node.inputs[0].links[0].from_node.type == "TEX_IMAGE", \
            "Material is not supported"
        assert sep_node.inputs[0].links[0].from_socket.name == "Color", "Material is not supported"
        channel_index = {"R": 0, "G": 1, "B": 2}[expected_channel]
        return extract_image(sep_node.inputs[0].links[0].from_node, channel_index)

    assert link.from_node.type == "TEX_IMAGE", "Material is not supported"
    assert link.from_socket.name == "Alpha", "Material is not supported"
    return extract_image(link.from_node, 3)


def try_extract_factor(link, mode="color"):
    assert mode in ["color", "scalar"], "Unsupported mode"

    if mode == "color":
        if link.from_node.type == "MIX":
            mix_node = link.from_node
            assert mix_node.data_type == "RGBA" and mix_node.blend_type == "MULTIPLY", \
                f"Material is not supported, {mix_node.data_type}, {mix_node.blend_type}"
            assert not mix_node.inputs["Factor"].is_linked and mix_node.inputs["Factor"].default_value == 1.0, \
                "Material is not supported"
            if mix_node.inputs["A"].is_linked:
                assert not mix_node.inputs["B"].is_linked, "Material is not supported"
                return list(mix_node.inputs["B"].default_value)[:3], mix_node.inputs["A"].links[0]
            assert not mix_node.inputs["A"].is_linked, "Material is not supported"
            assert mix_node.inputs["B"].is_linked, "Material is not supported"
            return list(mix_node.inputs["A"].default_value)[:3], mix_node.inputs["B"].links[0]
        return [1.0, 1.0, 1.0], link

    if link.from_node.type == "MATH":
        math_node = link.from_node
        assert math_node.operation == "MULTIPLY", "Material is not supported"
        assert math_node.inputs[0].is_linked, "Material is not supported"
        assert not math_node.inputs[1].is_linked, "Material is not supported"
        return math_node.inputs[1].default_value, math_node.inputs[0].links[0]
    return 1.0, link


def try_extract_image_with_factor(link, expected_channel="RGB"):
    factor, link = try_extract_factor(link, "color" if expected_channel == "RGB" else "scalar")
    image = try_extract_image(link, expected_channel)
    return factor, image


def extract_materials():
    materials = []
    for mat in bpy.data.materials:
        pack = {
            "baseColorFactor": [1.0, 1.0, 1.0],
            "alphaFactor": 1.0,
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0,
            "alphaMode": "OPAQUE",
            "alphaCutoff": 0.5,
            "baseColorTexture": None,
            "alphaTexture": None,
            "metallicTexture": None,
            "roughnessTexture": None,
        }
        if not mat.use_nodes:
            pack.update({
                "baseColorFactor": [0.8, 0.8, 0.8],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.5,
            })
            materials.append(pack)
            continue

        try:
            principled_node = mat.node_tree.nodes.get("Principled BSDF")
            if principled_node is None:
                materials.append(pack)
                continue

            if not principled_node.inputs["Base Color"].is_linked:
                pack["baseColorFactor"] = list(principled_node.inputs["Base Color"].default_value)[:3]
            else:
                link = principled_node.inputs["Base Color"].links[0]
                if link.from_node.type == "RGB":
                    pack["baseColorFactor"] = list(link.from_node.outputs[0].default_value)[:3]
                else:
                    factor, image = try_extract_image_with_factor(link, "RGB")
                    pack["baseColorFactor"] = factor
                    pack["baseColorTexture"] = image

            if not principled_node.inputs["Alpha"].is_linked:
                pack["alphaFactor"] = principled_node.inputs["Alpha"].default_value
                if pack["alphaFactor"] < 1.0:
                    pack["alphaMode"] = "BLEND"
            else:
                link = principled_node.inputs["Alpha"].links[0]
                node = link.from_node
                if node.type == "VALUE":
                    pack["alphaFactor"] = node.outputs[0].default_value
                    if pack["alphaFactor"] < 1.0:
                        pack["alphaMode"] = "BLEND"
                else:
                    pack["alphaMode"] = "BLEND"
                    if node.type == "MATH":
                        if node.operation == "ROUND":
                            assert node.inputs[0].is_linked, "Material is not supported"
                            pack["alphaMode"] = "MASK"
                            link = node.inputs[0].links[0]
                        elif node.operation == "SUBTRACT":
                            assert node.inputs[0].default_value == 1.0 and \
                                node.inputs[1].is_linked and \
                                node.inputs[1].links[0].from_node.type == "MATH" and \
                                node.inputs[1].links[0].from_node.operation == "LESS_THAN", \
                                "Material is not supported"
                            pack["alphaMode"] = "MASK"
                            pack["alphaCutoff"] = node.inputs[1].links[0].from_node.inputs[1].default_value
                            link = node.inputs[1].links[0].from_node.inputs[0].links[0]
                    factor, image = try_extract_image_with_factor(link, "A")
                    pack["alphaFactor"] = factor
                    pack["alphaTexture"] = image

            if not principled_node.inputs["Metallic"].is_linked:
                pack["metallicFactor"] = principled_node.inputs["Metallic"].default_value
            else:
                link = principled_node.inputs["Metallic"].links[0]
                node = link.from_node
                if node.type == "VALUE":
                    pack["metallicFactor"] = node.outputs[0].default_value
                else:
                    factor, image = try_extract_image_with_factor(link, "B")
                    pack["metallicFactor"] = factor
                    pack["metallicTexture"] = image

            if not principled_node.inputs["Roughness"].is_linked:
                pack["roughnessFactor"] = principled_node.inputs["Roughness"].default_value
            else:
                link = principled_node.inputs["Roughness"].links[0]
                node = link.from_node
                if node.type == "VALUE":
                    pack["roughnessFactor"] = node.outputs[0].default_value
                else:
                    factor, image = try_extract_image_with_factor(link, "G")
                    pack["roughnessFactor"] = factor
                    pack["roughnessTexture"] = image
        except Exception as exc:
            print(f"[WARN] Failed to parse material '{mat.name}': {exc}")
        materials.append(pack)
    return materials


def get_mesh_objects():
    mesh_objs = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.hide_render:
            continue
        if not obj.visible_get(view_layer=bpy.context.view_layer):
            continue
        mesh_objs.append(obj)
    return mesh_objs


def create_sequence_normalizer():
    if render_mod is not None:
        return render_mod.create_sequence_normalizer()
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


def extract_uv_mat_for_faces(mesh_objs):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    all_uvs = []
    all_mat_ids = []

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

            mat_indices = np.empty(num_tris, dtype=np.int32)
            temp_mesh.loop_triangles.foreach_get("material_index", mat_indices)
            global_mat_ids = np.full(num_tris, -1, dtype=np.int32)
            for i, local_mat_idx in enumerate(mat_indices):
                if 0 <= local_mat_idx < len(obj.material_slots) and obj.material_slots[local_mat_idx].material is not None:
                    global_mat_ids[i] = bpy.data.materials.find(obj.material_slots[local_mat_idx].name)
            all_mat_ids.append(global_mat_ids)

            uv_layer = temp_mesh.uv_layers.active
            if uv_layer is not None:
                loop_indices = np.empty(num_tris * 3, dtype=np.int32)
                temp_mesh.loop_triangles.foreach_get("loops", loop_indices)
                uv_data = np.empty(len(temp_mesh.loops) * 2, dtype=np.float32)
                uv_layer.data.foreach_get("uv", uv_data)
                uv_data = uv_data.reshape(-1, 2)
                all_uvs.append(uv_data[loop_indices].reshape(num_tris, 3, 2))
            else:
                all_uvs.append(np.zeros((num_tris, 3, 2), dtype=np.float32))
        finally:
            obj_eval.to_mesh_clear()

    if len(all_uvs) == 0:
        raise RuntimeError("No valid mesh found.")
    return (
        np.concatenate(all_uvs, axis=0).astype(np.float32),
        np.concatenate(all_mat_ids, axis=0).astype(np.int32),
    )


def extract_reference_mesh(mesh_objs) -> Tuple[np.ndarray, np.ndarray]:
    if render_mod is not None:
        vertices, faces = render_mod.extract_merged_mesh_world_fast(
            mesh_objs, depsgraph=bpy.context.evaluated_depsgraph_get()
        )
        return vertices.astype(np.float32, copy=False), faces.astype(np.int32, copy=False)

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
            world_mat = obj_eval.matrix_world.copy()
            R = np.array(world_mat.to_3x3(), dtype=np.float32)
            t = np.array(world_mat.translation[:], dtype=np.float32)
            all_vertices.append(co @ R.T + t[None, :])
            all_faces.append(tri.reshape(num_tris, 3) + vert_offset)
            vert_offset += num_verts
        finally:
            obj_eval.to_mesh_clear()
    if not all_faces:
        raise RuntimeError("No valid mesh found.")
    return (
        np.concatenate(all_vertices, axis=0).astype(np.float32),
        np.concatenate(all_faces, axis=0).astype(np.int32),
    )


def extract_vertices_only(mesh_objs) -> np.ndarray:
    if render_mod is not None:
        return render_mod.extract_merged_vertices_world_fast(
            mesh_objs, depsgraph=bpy.context.evaluated_depsgraph_get()
        ).astype(np.float32, copy=False)

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
            all_vertices.append(co @ R.T + t[None, :])
        finally:
            obj_eval.to_mesh_clear()
    if not all_vertices:
        raise RuntimeError("No valid mesh found.")
    return np.concatenate(all_vertices, axis=0).astype(np.float32)


def compute_normalized_mesh_sequence(mesh_objs, frame_indices: np.ndarray):
    if render_mod is not None:
        (
            faces,
            global_center,
            sequence_scale,
            canonical_bbox_min,
            canonical_bbox_max,
            raw_vertices_cache,
        ) = render_mod.compute_sequence_normalization_params_and_cache(
            mesh_objs, frame_indices, cache_raw_vertices=True, raw_cache_dtype=np.float16,
        )
        vertices_seq = render_mod.precompute_normalized_mesh_sequence_from_cache(
            raw_vertices_cache=raw_vertices_cache,
            global_center=global_center,
            global_scale=sequence_scale,
            shared_faces=faces,
            frame_indices=frame_indices,
        )
        return (
            vertices_seq.astype(np.float16, copy=False),
            faces.astype(np.int32, copy=False),
            global_center.astype(np.float32, copy=False),
            float(sequence_scale),
            canonical_bbox_min.astype(np.float32, copy=False),
            canonical_bbox_max.astype(np.float32, copy=False),
        )

    raw_vertices_cache = []
    faces = None
    ref_vert_count = None
    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for frame in frame_indices:
        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        if faces is None:
            raw_vertices, faces = extract_reference_mesh(mesh_objs)
            ref_vert_count = int(raw_vertices.shape[0])
        else:
            raw_vertices = extract_vertices_only(mesh_objs)
            if raw_vertices.shape[0] != ref_vert_count:
                raise RuntimeError(
                    f"Vertex count changed at frame {int(frame)}: "
                    f"reference={ref_vert_count}, current={raw_vertices.shape[0]}."
                )
        global_min = np.minimum(global_min, raw_vertices.min(axis=0))
        global_max = np.maximum(global_max, raw_vertices.max(axis=0))
        raw_vertices_cache.append(raw_vertices.astype(np.float16))

    global_center = (0.5 * (global_min + global_max)).astype(np.float32)
    extent = global_max - global_min
    sequence_scale = 1.0 / float(np.max(extent)) if float(np.max(extent)) > 1e-6 else 1.0
    canonical_bbox_min = ((global_min - global_center) * sequence_scale).astype(np.float32)
    canonical_bbox_max = ((global_max - global_center) * sequence_scale).astype(np.float32)
    vertices_seq = np.empty((len(raw_vertices_cache), raw_vertices_cache[0].shape[0], 3), dtype=np.float16)
    for i, raw_vertices in enumerate(raw_vertices_cache):
        raw32 = np.asarray(raw_vertices, dtype=np.float32)
        vertices_seq[i] = ((raw32 - global_center[None, :]) * sequence_scale).astype(np.float16)
    return vertices_seq, faces.astype(np.int32), global_center, float(sequence_scale), canonical_bbox_min, canonical_bbox_max


def load_reference_mesh_npz(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not path:
        return None, None
    with np.load(path) as data:
        vertices = data["vertices"].astype(np.float32, copy=False)
        frame_indices = data["frame_indices"].astype(np.int32, copy=False) if "frame_indices" in data else None
        return vertices, frame_indices


def _sorted_quantized_rows(vertices: np.ndarray, eps: float) -> np.ndarray:
    quantized = np.rint(np.asarray(vertices, dtype=np.float32) / float(eps)).astype(np.int64)
    order = np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))
    return quantized[order]


def reference_vertices_mismatch_message(
    vertices_seq: np.ndarray,
    ref_vertices_seq: np.ndarray,
    eps: float,
) -> Optional[str]:
    if vertices_seq.shape != ref_vertices_seq.shape:
        return f"reference vertices mismatch: dumped={vertices_seq.shape}, reference={ref_vertices_seq.shape}"

    for frame_idx in range(vertices_seq.shape[0]):
        dumped = _sorted_quantized_rows(vertices_seq[frame_idx], eps)
        reference = _sorted_quantized_rows(ref_vertices_seq[frame_idx], eps)
        if np.array_equal(dumped, reference):
            continue

        diff = np.nonzero(np.any(dumped != reference, axis=1))[0]
        first_diff = int(diff[0]) if diff.size else -1
        dump_bbox_min = np.asarray(vertices_seq[frame_idx], dtype=np.float32).min(axis=0).tolist()
        dump_bbox_max = np.asarray(vertices_seq[frame_idx], dtype=np.float32).max(axis=0).tolist()
        ref_bbox_min = np.asarray(ref_vertices_seq[frame_idx], dtype=np.float32).min(axis=0).tolist()
        ref_bbox_max = np.asarray(ref_vertices_seq[frame_idx], dtype=np.float32).max(axis=0).tolist()
        return (
            "reference vertices mismatch: "
            f"frame={frame_idx}, shape={vertices_seq.shape}, eps={eps}, first_sorted_diff={first_diff}, "
            f"dumped_key={dumped[first_diff].tolist() if first_diff >= 0 else None}, "
            f"reference_key={reference[first_diff].tolist() if first_diff >= 0 else None}, "
            f"dumped_bbox=({dump_bbox_min}, {dump_bbox_max}), "
            f"reference_bbox=({ref_bbox_min}, {ref_bbox_max})"
        )
    return None


def main(args):
    print(f"[dump_pbr_dynamic_obj] object_path={args.object_path}")
    init_scene()
    load_object(args.object_path)
    create_sequence_normalizer()

    ref_vertices, ref_frame_indices = load_reference_mesh_npz(args.reference_mesh_npz)
    if ref_vertices is None:
        raise RuntimeError("--reference_mesh_npz is required for dynamic ObjXL PBR dump")
    if ref_frame_indices is None:
        raise RuntimeError("reference mesh.npz missing required frame_indices")
    if int(ref_vertices.shape[0]) != int(ref_frame_indices.shape[0]):
        raise RuntimeError(
            "reference frame count mismatch: "
            f"vertices_frames={int(ref_vertices.shape[0])}, frame_indices={int(ref_frame_indices.shape[0])}"
        )
    frame_indices = ref_frame_indices
    frame_source = "reference_mesh_npz"

    ref_frame = int(frame_indices[0])
    bpy.context.scene.frame_set(ref_frame)
    bpy.context.view_layer.update()
    print(
        f"[dump_pbr_dynamic_obj] reference_frame={ref_frame} "
        f"num_frames_after_crop={len(frame_indices)} frame_source={frame_source}"
    )

    materials = extract_materials()
    mesh_objs = get_mesh_objects()
    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects found after loading.")

    (
        vertices_seq,
        faces,
        global_center,
        sequence_scale,
        canonical_bbox_min,
        canonical_bbox_max,
    ) = compute_normalized_mesh_sequence(mesh_objs, frame_indices)

    bpy.context.scene.frame_set(ref_frame)
    bpy.context.view_layer.update()
    uvs, mat_ids = extract_uv_mat_for_faces(mesh_objs)
    if faces.shape[0] != uvs.shape[0] or faces.shape[0] != mat_ids.shape[0]:
        raise RuntimeError(
            f"topology attribute count mismatch: faces={faces.shape}, uvs={uvs.shape}, mat_ids={mat_ids.shape}"
        )
    if faces.size and int(faces.max()) >= int(vertices_seq.shape[1]):
        raise RuntimeError(f"faces reference invalid vertex index: max_face_index={int(faces.max())}, V={vertices_seq.shape[1]}")
    if mat_ids.size and (int(mat_ids.max()) >= len(materials) or int(mat_ids.min()) < -1):
        raise RuntimeError(
            f"material index out of range: min={int(mat_ids.min())}, max={int(mat_ids.max())}, num_materials={len(materials)}"
        )

    vertex_set_match = None
    if ref_vertices is not None:
        mismatch = reference_vertices_mismatch_message(vertices_seq, ref_vertices, args.vertex_set_eps)
        vertex_set_match = mismatch is None
        if mismatch is not None and args.strict_reference_vertices:
            raise RuntimeError(mismatch)

    output = {
        "materials": materials,
        "objects": [{
            "vertices_seq": vertices_seq,
            "faces": faces,
            "uvs": uvs,
            "mat_ids": mat_ids,
        }],
        "meta": {
            "object_path": args.object_path,
            "reference_frame": ref_frame,
            "frame_indices": frame_indices.tolist(),
            "frame_source": frame_source,
            "num_materials": len(materials),
            "num_frames": int(vertices_seq.shape[0]),
            "num_vertices": int(vertices_seq.shape[1]),
            "num_faces": int(faces.shape[0]),
            "vertex_set_match_reference": vertex_set_match,
            "vertex_set_eps": float(args.vertex_set_eps),
            "sequence_center": global_center.tolist(),
            "sequence_scale": float(sequence_scale),
            "canonical_bbox_min": canonical_bbox_min.tolist(),
            "canonical_bbox_max": canonical_bbox_max.tolist(),
        },
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(output, f)
    print(f"[dump_pbr_dynamic_obj] saved={args.output_path}")
    print(json.dumps(output["meta"], sort_keys=True))


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--reference_mesh_npz", type=str, default="")
    parser.add_argument("--strict_reference_vertices", action="store_true")
    parser.add_argument("--no_strict_reference_vertices", dest="strict_reference_vertices", action="store_false")
    parser.set_defaults(strict_reference_vertices=True)
    parser.add_argument("--vertex_set_eps", type=float, default=1e-4)
    parser.add_argument("--max_frames", type=int, default=121)
    main(parser.parse_args(argv))
