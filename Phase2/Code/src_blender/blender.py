import bpy
import os
import sys
import random
import argparse
from mathutils import Vector, Quaternion

# Starter blender runner: setup scene, sample trajectories, render frames, export poses.
# This file augments the minimal blender.py that creates a plane. Implement details using bpy.

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

PHASE2_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
DEFAULT_FLOOR_TEXTURE = os.path.join(
    PHASE2_ROOT,
    'static',
    'austin-scherbarth-qSrFTyh-IB0-unsplash.jpg'
)

try:
    from . import materials, intrinsics, trajectories
except ImportError:
    # Allow running as a standalone Blender script via --python path/to/blender.py
    import materials  # type: ignore
    import intrinsics  # type: ignore
    import trajectories  # type: ignore


def parse_args(argv=None):
    """Parse CLI args passed after '--' in Blender command."""
    if argv is None:
        argv = sys.argv

    user_argv = []
    if '--' in argv:
        user_argv = argv[argv.index('--') + 1:]

    parser = argparse.ArgumentParser(description="Phase 2 Blender data generator")
    parser.add_argument('--output_dir', default=os.path.join(PHASE2_ROOT, 'static', './tmp/'))
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--imu_rate', type=float, default=200.0)
    parser.add_argument('--cam_rate', type=float, default=20.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--engine', default='BLENDER_EEVEE')
    parser.add_argument('--fps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke_test', action='store_true')

    parser.add_argument(
        '--floor_texture',
        default=DEFAULT_FLOOR_TEXTURE,
        help='Image path for floor diffuse texture',
    )
    parser.add_argument(
        '--crop_uv',
        nargs=4,
        type=float,
        metavar=('UMIN', 'VMIN', 'UMAX', 'VMAX'),
        default=None,
        help='Normalized UV crop u_min v_min u_max v_max (default: full image)',
    )
    parser.add_argument('--floor_size', type=float, default=100.0)
    parser.add_argument('--uv_scale', nargs=2, type=float, default=[1.0, 1.0],
                        metavar=('SX', 'SY'))
    parser.add_argument('--uv_rotate_z', type=float, default=0.0)
    parser.add_argument(
          '--motion_mode',
          choices=['trajectory', 'preview_diagonal'],
          default='trajectory',
          help='Use the existing trajectory path or a simple diagonal preview motion',
      )

    return parser.parse_args(user_argv)


def reset_scene():
    """Remove existing objects/materials/images for deterministic generation."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for block in bpy.data.materials:
        bpy.data.materials.remove(block, do_unlink=True)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def setup_scene(render_resolution=(640, 480), engine='BLENDER_EEVEE', fps=100):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = render_resolution[0]
    scene.render.resolution_y = render_resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    return scene


def ensure_camera(scene, location=(0.0, 0.0, 3.0), euler_xyz=(0.0, 0.0, 0.0)):
    """Create camera if needed and make it active.

    Convention: default Euler (0, 0, 0) points the camera downward (−Z) at the XY floor plane.
    """
    cam = next((obj for obj in scene.objects if obj.type == 'CAMERA'), None)
    if cam is None:
        bpy.ops.object.camera_add(location=location, rotation=euler_xyz)
        cam = bpy.context.active_object
    scene.camera = cam
    return cam


def ensure_sun(scene, energy=3.0):
    """Add a sun lamp so the textured floor is visible in EEVEE/Cycles smoke renders."""
    existing = next((obj for obj in scene.objects if obj.type == 'LIGHT'), None)
    if existing is not None:
        return existing
    bpy.ops.object.light_add(type='SUN', location=(10.0, -10.0, 20.0))
    sun = bpy.context.active_object
    sun.data.energy = energy
    return sun


def create_floor(name='Floor', size=100.0, location=(0.0, 0.0, 0.0)):
    """Large ground plane in XY (matches downward-looking camera). Edge length ``size`` (meters)."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    floor_obj = bpy.context.active_object
    floor_obj.name = name
    return floor_obj


def set_camera_pose(camera, position, quaternion_xyzw):
    """Set camera world pose. Input quaternion expected in xyzw order."""
    camera.location = Vector(position)
    camera.rotation_mode = 'QUATERNION'
    qx, qy, qz, qw = quaternion_xyzw
    camera.rotation_quaternion = Quaternion((qw, qx, qy, qz))

def preview_diagonal_pose(
      t,
      duration,
      floor_size,
      z=3.0,
      margin_ratio=0.15,
  ):
      """Simple placeholder VO motion: fixed camera translating diagonally across most of the floor."""
      if duration <= 0.0:
          s = 0.0
      else:
          s = max(0.0, min(1.0, t / duration))

      half_extent = floor_size * 0.5
      margin = floor_size * margin_ratio
      usable = max(0.0, half_extent - margin)

      start_x, start_y = -usable, -usable
      end_x, end_y = usable, usable

      x = start_x + s * (end_x - start_x)
      y = start_y + s * (end_y - start_y)

      position = (x, y, z)
      quaternion_xyzw = (0.0, 0.0, 0.0, 1.0)
      return position, quaternion_xyzw


def smoke_test_render(scene, camera, output_dir):
    """Render a single frame to validate pipeline wiring."""
    os.makedirs(output_dir, exist_ok=True)
    smoke_path = os.path.join(output_dir, "smoke_test.png")
    # Quaternion xyzw identity => Blender camera default forward axis looks along −Z.
    set_camera_pose(camera, (0.0, 0.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    scene.render.filepath = smoke_path
    bpy.ops.render.render(write_still=True)
    print(f"Smoke test render written to {smoke_path}")


def export_pose_file(path, timestamp, position, quaternion):
    # Append a single line: timestamp tx ty tz qx qy qz qw
    with open(path, 'a') as f:
        f.write(f"{timestamp:.9f} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} {quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n")


def run_demo(
    output_dir: str = '/tmp/phase2_out',
    duration: float = 5.0,
    imu_rate: float = 200.0,
    cam_rate: float = 20.0,
    render_resolution=(640, 480),
    engine='BLENDER_EEVEE',
    fps=100,
    seed=42,
    smoke_test=False,
    floor_texture=DEFAULT_FLOOR_TEXTURE,
    crop_uv=None,
    floor_size: float = 100.0,
    uv_scale=(1.0, 1.0),
    uv_rotate_z: float = 0.0,
    traj_cfg=None,
    motion_mode='trajectory'
):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    cam_dir = os.path.join(output_dir, 'cam0', 'data')
    os.makedirs(cam_dir, exist_ok=True)
    pose_csv = os.path.join(output_dir, 'groundtruth.csv')

    reset_scene()
    scene = setup_scene(render_resolution=render_resolution, engine=engine, fps=fps)
    camera = ensure_camera(scene)
    crop = tuple(crop_uv) if crop_uv is not None else (0.0, 0.0, 1.0, 1.0)
    ensure_sun(scene)
    sx, sy = uv_scale
    floor_mat = materials.create_material_with_image(
        'FloorMaterial',
        floor_texture,
        crop_uv=crop,
        uv_scale=(sx, sy, 1.0),
        uv_rotation_z=uv_rotate_z,
    )
    materials.assign_material(create_floor(size=floor_size), floor_mat)
    traj = traj_cfg if traj_cfg is not None else trajectories.TrajectoryConfig()

    if smoke_test:
        smoke_test_render(scene, camera, output_dir)
        return

    t = 0.0
    dt_imu = 1.0 / imu_rate
    dt_cam = 1.0 / cam_rate
    next_cam = 0.0
    frame_idx = 0

    # Clear pose CSV and write header optional
    with open(pose_csv, 'w') as f:
        f.write('# timestamp tx ty tz qx qy qz qw\n')

    if motion_mode == 'preview_diagonal':
        while t < duration:
            ts = t
            pos, quat = preview_diagonal_pose(ts, duration, floor_size)
            filename = os.path.join(cam_dir, f"{int(ts*1e9):016d}.png")
            set_camera_pose(camera, pos, quat)
            scene.render.filepath = filename
            bpy.ops.render.render(write_still=True)
            export_pose_file(pose_csv, ts, pos, quat)
            frame_idx += 1
            t += dt_cam

        print(f"Wrote {frame_idx} frames to {cam_dir}")
        return

    while t < duration:
        pos, quat, _vel, _acc = trajectories.figure8(t, traj)
        if t >= next_cam - 1e-9:
            ts = t
            filename = os.path.join(cam_dir, f"{int(ts*1e9):016d}.png")
            set_camera_pose(camera, pos, quat)
            scene.render.filepath = filename
            bpy.ops.render.render(write_still=True)
            export_pose_file(pose_csv, ts, pos, quat)
            next_cam += dt_cam
            frame_idx += 1
        t += dt_imu

    print(f"Wrote {frame_idx} frames to {cam_dir}")


if __name__ == '__main__':
    args = parse_args()
    run_demo(
        output_dir=args.output_dir,
        duration=args.duration,
        imu_rate=args.imu_rate,
        cam_rate=args.cam_rate,
        render_resolution=(args.width, args.height),
        engine=args.engine,
        fps=args.fps,
        seed=args.seed,
        smoke_test=args.smoke_test,
        floor_texture=args.floor_texture,
        crop_uv=args.crop_uv,
        floor_size=args.floor_size,
        uv_scale=args.uv_scale,
        uv_rotate_z=args.uv_rotate_z,
        motion_mode=args.motion_mode
    )

