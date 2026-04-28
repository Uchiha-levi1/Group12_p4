"""Blender material and texture helpers (Phase2/src_blender).

Intended to be imported by blender scripts run inside Blender (requires bpy).
"""

# NOTE: actual implementations below should run inside Blender where `bpy` is available.
# These are starter stubs with usage notes.

from typing import Optional


def load_image(image_path: str):
    """Load an image into Blender and return the bpy.data.images reference.

    In Blender: img = bpy.data.images.load(image_path)
    """
    # Implement inside Blender.
    return image_path


def create_material_with_image(name: str, image_path: str):
    """Create a new material that maps image_path as a texture (use nodes).

    Blender notes:
    - mat = bpy.data.materials.new(name)
    - mat.use_nodes = True
    - tex_node = mat.node_tree.nodes.new('ShaderNodeTexImage'); tex_node.image = img
    - link nodes to Principled BSDF
    """
    return name


def assign_material(obj, material):
    """Assign material to object: obj.data.materials.append(material) in Blender.
    """
    return True
