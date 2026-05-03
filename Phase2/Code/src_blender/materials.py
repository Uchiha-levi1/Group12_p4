"""Blender material and texture helpers (Phase2/src_blender).

Intended to be imported by blender scripts run inside Blender (requires bpy).
"""

import os
from typing import Tuple

import bpy

CropUV = Tuple[float, float, float, float]
Vec3 = Tuple[float, float, float]


def load_image(image_path: str):
    """Load an image into Blender and return the bpy.data.images reference.

    In Blender: img = bpy.data.images.load(image_path)
    """
    # Implement inside Blender.
    path = os.path.abspath(os.path.expanduser(image_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Texture image not found: {path}")
    # check_existing=True avoids duplicate datablocks for the same path
    return bpy.data.images.load(path, check_existing=True)


def create_material_with_image(
    name: str,
    image_path: str,
    *,
    crop_uv: CropUV = (0.0, 0.0, 1.0, 1.0),
    uv_scale: Vec3 = (1.0, 1.0, 1.0),
    uv_rotation_z: float = 0.0,
    uv_offset: Vec3 = (0.0, 0.0, 0.0),
) -> bpy.types.Material:
    """UV crop (u_min, v_min, u_max, v_max) then tile / rotate / offset on the plane.
    Graph: TexCoord[UV] -> Mapping[crop] -> Mapping[tile] -> ImageTexture -> PrincipledBSDF -> Output
    """
    u0, v0, u1, v1 = crop_uv
    if not (0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0):
        raise ValueError(
            f"crop_uv must satisfy 0 <= u_min < u_max <= 1 (same for v); got {crop_uv!r}"
        )
    img = load_image(image_path)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    x, y = 0, 300
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (x + 900, y)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (x + 600, y)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.location = (x + 300, y)
    if hasattr(tex, "colorspace_settings"):
        tex.colorspace_settings.name = "sRGB"
    map_tile = nodes.new("ShaderNodeMapping")
    map_tile.location = (x + 150, y)
    map_tile.inputs["Location"].default_value = (*uv_offset[:2], uv_offset[2])
    map_tile.inputs["Rotation"].default_value = (0.0, 0.0, uv_rotation_z)
    map_tile.inputs["Scale"].default_value = (*uv_scale[:2], uv_scale[2])
    du, dv = u1 - u0, v1 - v0
    map_crop = nodes.new("ShaderNodeMapping")
    map_crop.location = (x, y)
    map_crop.inputs["Location"].default_value = (u0, v0, 0.0)
    map_crop.inputs["Rotation"].default_value = (0.0, 0.0, 0.0)
    map_crop.inputs["Scale"].default_value = (du, dv, 1.0)
    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (x - 200, y)
    links.new(texcoord.outputs["UV"], map_crop.inputs["Vector"])
    links.new(map_crop.outputs["Vector"], map_tile.inputs["Vector"])
    links.new(map_tile.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> bool:
    if getattr(obj, "data", None) is None or not hasattr(obj.data, "materials"):
        return False
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return True
