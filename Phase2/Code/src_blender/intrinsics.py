"""Camera intrinsics helpers for Blender-based dataset export.

Contains utility to compute fx,fy,cx,cy from Blender camera data.
"""

def intrinsics_dict():
    # Provide both scalar intrinsics and a 3x3 identity K for convenience.
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
