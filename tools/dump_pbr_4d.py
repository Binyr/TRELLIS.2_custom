#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_pbr_4d.py - Blender script to extract PBR materials + UVs + mat_ids + faces
from an animated GLB file. Extracts topology-related data only (shared across frames).
Vertices are NOT extracted since they come from mesh.npz.

Usage:
    blender -b -P tools/dump_pbr_4d.py -- --object_path xxx.glb --output_path xxx.pickle

Output pickle format:
{
    'materials': [...],   # same as blender_script/dump_pbr.py
    'objects': [
        {
            'faces': (F, 3) int32,
            'uvs': (F, 3, 2) float32,
            'mat_ids': (F,) int32,
        }
    ]
}

IMPORTANT: Object iteration order and triangulation method (calc_loop_triangles)
must match 4D_video_data.py's extract_merged_mesh_world_fast() to ensure face
indices align with vertices_seq from mesh.npz.
"""

import argparse
import io
import os
import pickle
import sys
from typing import Dict, Callable, List, Tuple

import bpy
import numpy as np
from PIL import Image


# =====================================================================================
# IMPORT
# =====================================================================================

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
}


def init_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in list(bpy.data.textures):
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    file_extension = object_path.split(".")[-1].lower()
    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path}")
    IMPORT_FUNCTIONS[file_extension](filepath=object_path)


# =====================================================================================
# MATERIAL EXTRACTION (reused from blender_script/dump_pbr.py)
# =====================================================================================

def extract_image(tex_node, channels):
    image = tex_node.image
    pixels = np.array(image.pixels[:])
    data = pixels.reshape(image.size[1], image.size[0], -1)
    data = data[..., channels]

    if data.dtype != np.uint8:
        data = np.clip(data, 0.0, 1.0)
        data = (data * 255).astype(np.uint8)

    if len(data.shape) == 2:
        pil_image = Image.fromarray(data, mode='L')
    elif data.shape[2] == 3:
        pil_image = Image.fromarray(data, mode='RGB')
    elif data.shape[2] == 4:
        pil_image = Image.fromarray(data, mode='RGBA')
    else:
        raise ValueError("Unsupported channel shape for image")

    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()

    return {
        'image': png_bytes,
        'interpolation': tex_node.interpolation,
        'extension': tex_node.extension,
    }


def try_extract_image(link, expected_channel='RGB'):
    assert expected_channel in ['RGB', 'R', 'G', 'B', 'A'], "Unsupported channel"

    if expected_channel == 'RGB':
        assert link.from_node.type == 'TEX_IMAGE', "Material is not supported"
        assert link.from_socket.name == 'Color', "Material is not supported"
        tex_node = link.from_node
        return extract_image(tex_node, [0, 1, 2])

    if expected_channel in ['R', 'G', 'B']:
        socket_name = {'R': 'Red', 'G': 'Green', 'B': 'Blue'}[expected_channel]
        assert link.from_node.type == 'SEPARATE_COLOR' and link.from_node.mode == 'RGB', \
            f"Material is not supported, {link.from_node.type}, {link.from_node.mode}"
        assert link.from_socket.name == socket_name, "Material is not supported"
        sep_node = link.from_node
        assert sep_node.inputs[0].is_linked and sep_node.inputs[0].links[0].from_node.type == 'TEX_IMAGE', \
            "Material is not supported"
        assert sep_node.inputs[0].links[0].from_socket.name == 'Color', "Material is not supported"
        tex_node = sep_node.inputs[0].links[0].from_node
        channel_index = {'R': 0, 'G': 1, 'B': 2}[expected_channel]
        return extract_image(tex_node, channel_index)

    if expected_channel == 'A':
        assert link.from_node.type == 'TEX_IMAGE', "Material is not supported"
        assert link.from_socket.name == 'Alpha', "Material is not supported"
        tex_node = link.from_node
        return extract_image(tex_node, 3)


def try_extract_factor(link, mode='color'):
    assert mode in ['color', 'scalar'], "Unsupported mode"

    if mode == 'color':
        if link.from_node.type == 'MIX':
            mix_node = link.from_node
            assert mix_node.data_type == 'RGBA' and mix_node.blend_type == 'MULTIPLY', \
                f"Material is not supported, {mix_node.data_type}, {mix_node.blend_type}"
            assert not mix_node.inputs['Factor'].is_linked and mix_node.inputs['Factor'].default_value == 1.0, \
                "Material is not supported"
            if mix_node.inputs['A'].is_linked:
                assert not mix_node.inputs['B'].is_linked, "Material is not supported"
                return (list(mix_node.inputs['B'].default_value)[:3], mix_node.inputs['A'].links[0])
            else:
                assert not mix_node.inputs['A'].is_linked, "Material is not supported"
                assert mix_node.inputs['B'].is_linked, "Material is not supported"
                return (list(mix_node.inputs['A'].default_value)[:3], mix_node.inputs['B'].links[0])
        return ([1.0, 1.0, 1.0], link)

    if mode == 'scalar':
        if link.from_node.type == 'MATH':
            math_node = link.from_node
            assert math_node.operation == 'MULTIPLY', "Material is not supported"
            assert math_node.inputs[0].is_linked, "Material is not supported"
            assert not math_node.inputs[1].is_linked, "Material is not supported"
            return (math_node.inputs[1].default_value, math_node.inputs[0].links[0])
        return (1.0, link)


def try_extract_image_with_factor(link, expected_channel='RGB'):
    factor, link = try_extract_factor(link, 'color' if expected_channel in ['RGB'] else 'scalar')
    image = try_extract_image(link, expected_channel)
    return (factor, image)


def extract_materials():
    """Extract all materials from the scene. Returns list of material dicts."""
    materials = []
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            # Fallback for non-node materials
            materials.append({
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
            continue

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

        try:
            principled_node = mat.node_tree.nodes.get('Principled BSDF')
            if principled_node is None:
                materials.append(pack)
                continue

            # Base Color
            if not principled_node.inputs['Base Color'].is_linked:
                pack["baseColorFactor"] = list(principled_node.inputs['Base Color'].default_value)[:3]
            else:
                link = principled_node.inputs['Base Color'].links[0]
                if link.from_node.type == 'RGB':
                    pack["baseColorFactor"] = list(link.from_node.outputs[0].default_value)[:3]
                else:
                    factor, image = try_extract_image_with_factor(link, 'RGB')
                    pack["baseColorFactor"] = factor
                    pack["baseColorTexture"] = image

            # Alpha
            if not principled_node.inputs['Alpha'].is_linked:
                pack["alphaFactor"] = principled_node.inputs['Alpha'].default_value
                if pack["alphaFactor"] < 1.0:
                    pack["alphaMode"] = "BLEND"
            else:
                link = principled_node.inputs['Alpha'].links[0]
                node = link.from_node
                if node.type == 'VALUE':
                    pack["alphaFactor"] = node.outputs[0].default_value
                    if pack["alphaFactor"] < 1.0:
                        pack["alphaMode"] = "BLEND"
                else:
                    pack["alphaMode"] = "BLEND"
                    if node.type == 'MATH':
                        if node.operation == 'ROUND':
                            assert node.inputs[0].is_linked, "Material is not supported"
                            pack["alphaMode"] = "MASK"
                            link = node.inputs[0].links[0]
                        elif node.operation == 'SUBTRACT':
                            assert node.inputs[0].default_value == 1.0 and \
                                node.inputs[1].is_linked and \
                                node.inputs[1].links[0].from_node.type == 'MATH' and \
                                node.inputs[1].links[0].from_node.operation == 'LESS_THAN', \
                                "Material is not supported"
                            assert node.inputs[1].links[0].from_node.inputs[0].is_linked, "Material is not supported"
                            pack["alphaMode"] = "MASK"
                            pack["alphaCutoff"] = node.inputs[1].links[0].from_node.inputs[1].default_value
                            link = node.inputs[1].links[0].from_node.inputs[0].links[0]
                    factor, image = try_extract_image_with_factor(link, 'A')
                    pack["alphaFactor"] = factor
                    pack["alphaTexture"] = image

            # Metallic
            if not principled_node.inputs['Metallic'].is_linked:
                pack["metallicFactor"] = principled_node.inputs['Metallic'].default_value
            else:
                link = principled_node.inputs['Metallic'].links[0]
                node = link.from_node
                if node.type == 'VALUE':
                    pack["metallicFactor"] = node.outputs[0].default_value
                else:
                    factor, image = try_extract_image_with_factor(link, 'B')
                    pack["metallicFactor"] = factor
                    pack["metallicTexture"] = image

            # Roughness
            if not principled_node.inputs['Roughness'].is_linked:
                pack["roughnessFactor"] = principled_node.inputs['Roughness'].default_value
            else:
                link = principled_node.inputs['Roughness'].links[0]
                node = link.from_node
                if node.type == 'VALUE':
                    pack["roughnessFactor"] = node.outputs[0].default_value
                else:
                    factor, image = try_extract_image_with_factor(link, 'G')
                    pack["roughnessFactor"] = factor
                    pack["roughnessTexture"] = image

            materials.append(pack)
        except Exception as e:
            print(f"[WARN] Failed to parse material '{mat.name}': {e}")
            materials.append(pack)

    return materials


# =====================================================================================
# MESH EXTRACTION (using calc_loop_triangles to match 4D_video_data.py)
# =====================================================================================

def get_mesh_objects():
    """Get mesh objects in the same order as 4D_video_data.py"""
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


def extract_merged_topology(mesh_objs):
    """
    Extract merged faces, UVs, and mat_ids using calc_loop_triangles.
    This matches the triangulation in 4D_video_data.py's extract_merged_mesh_world_fast().
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    all_faces = []
    all_uvs = []
    all_mat_ids = []
    vert_offset = 0

    for obj in mesh_objs:
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

            # Faces (same as 4D_video_data.py)
            tri = np.empty(num_tris * 3, dtype=np.int32)
            temp_mesh.loop_triangles.foreach_get("vertices", tri)
            tri = tri.reshape(num_tris, 3)
            all_faces.append(tri + vert_offset)

            # Material indices per triangle
            mat_indices = np.empty(num_tris, dtype=np.int32)
            temp_mesh.loop_triangles.foreach_get("material_index", mat_indices)

            # Map local material_index to global material index
            global_mat_ids = np.full(num_tris, -1, dtype=np.int32)
            for i, lt in enumerate(temp_mesh.loop_triangles):
                local_mat_idx = mat_indices[i]
                if len(obj.material_slots) > 0 and obj.material_slots[local_mat_idx].material is not None:
                    global_mat_ids[i] = bpy.data.materials.find(
                        obj.material_slots[local_mat_idx].name
                    )
                else:
                    global_mat_ids[i] = -1
            all_mat_ids.append(global_mat_ids)

            # UVs per triangle vertex
            uv_layer = temp_mesh.uv_layers.active
            if uv_layer is not None:
                # Extract UVs via loop indices from loop_triangles
                loop_indices = np.empty(num_tris * 3, dtype=np.int32)
                temp_mesh.loop_triangles.foreach_get("loops", loop_indices)

                uv_data = np.empty(len(temp_mesh.loops) * 2, dtype=np.float32)
                uv_layer.data.foreach_get("uv", uv_data)
                uv_data = uv_data.reshape(-1, 2)

                tri_uvs = uv_data[loop_indices].reshape(num_tris, 3, 2)
                all_uvs.append(tri_uvs)
            else:
                # No UV, fill with zeros
                all_uvs.append(np.zeros((num_tris, 3, 2), dtype=np.float32))

            vert_offset += num_verts
        finally:
            obj_eval.to_mesh_clear()

    if len(all_faces) == 0:
        raise RuntimeError("No valid mesh found.")

    merged_faces = np.concatenate(all_faces, axis=0).astype(np.int32)
    merged_uvs = np.concatenate(all_uvs, axis=0).astype(np.float32)
    merged_mat_ids = np.concatenate(all_mat_ids, axis=0).astype(np.int32)

    return merged_faces, merged_uvs, merged_mat_ids


# =====================================================================================
# MAIN
# =====================================================================================

def main(args):
    print(f"[dump_pbr_4d] Loading: {args.object_path}")
    init_scene()
    load_object(args.object_path)

    # Set to first frame
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()

    # Extract materials
    print("[dump_pbr_4d] Extracting materials...")
    materials = extract_materials()
    print(f"[dump_pbr_4d] Found {len(materials)} materials")

    # Extract mesh topology
    print("[dump_pbr_4d] Extracting mesh topology (faces, UVs, mat_ids)...")
    mesh_objs = get_mesh_objects()
    print(f"[dump_pbr_4d] Found {len(mesh_objs)} mesh objects")
    faces, uvs, mat_ids = extract_merged_topology(mesh_objs)
    print(f"[dump_pbr_4d] Merged: {faces.shape[0]} faces, UVs shape={uvs.shape}, mat_ids shape={mat_ids.shape}")

    # Build output
    output = {
        'materials': materials,
        'objects': [
            {
                'faces': faces,
                'uvs': uvs,
                'mat_ids': mat_ids,
            }
        ]
    }

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"[dump_pbr_4d] Saved to: {args.output_path}")
    print(f"[dump_pbr_4d] Done.")


if __name__ == '__main__':
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args(argv)

    main(args)
