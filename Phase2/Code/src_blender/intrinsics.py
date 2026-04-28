"""Camera intrinsics helpers for Blender-based dataset export.

Contains utility to compute fx,fy,cx,cy from Blender camera data.
"""

from typing import Tuple


def fx_fy_cx_cy_from_blender(camera, resolution_x: int, resolution_y: int) -> Tuple[float,float,float,float]:
    """Compute intrinsics from a bpy.types.Camera instance.

    In Blender run-time:
      lens = camera.lens
      sensor_width = camera.sensor_width
      shift_x = camera.shift_x
      shift_y = camera.shift_y
      fx = lens * (resolution_x / sensor_width)
      fy = fx  # assume square pixels; refine with sensor_height if needed
      cx = resolution_x * 0.5 - shift_x * resolution_x
      cy = resolution_y * 0.5 - shift_y * resolution_y
    """
    # Placeholder return; actual function should run under Blender and use camera fields.
    return 0.0, 0.0, 0.0, 0.0


def intrinsics_dict_from_camera(camera, res_x: int, res_y: int):
    fx, fy, cx, cy = fx_fy_cx_cy_from_blender(camera, res_x, res_y)
    return {'fx':fx, 'fy':fy, 'cx':cx, 'cy':cy}
