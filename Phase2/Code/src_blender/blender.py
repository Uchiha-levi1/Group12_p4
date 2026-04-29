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
    parser.add_argument('--output_dir', default='/tmp/phase2_out')
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--imu_rate', type=float, default=200.0)
    parser.add_argument('--cam_rate', type=float, default=20.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--engine', default='BLENDER_EEVEE')
    parser.add_argument('--fps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke_test', action='store_true')
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


def ensure_camera(scene, location=(0.0, 0.0, 3.0), euler_xyz=(3.14159, 0.0, 0.0)):
    """Create camera if needed and make it active."""
    cam = next((obj for obj in scene.objects if obj.type == 'CAMERA'), None)
    if cam is None:
        bpy.ops.object.camera_add(location=location, rotation=euler_xyz)
        cam = bpy.context.active_object
    scene.camera = cam
    return cam


def set_camera_pose(camera, position, quaternion_xyzw):
    """Set camera world pose. Input quaternion expected in xyzw order."""
    camera.location = Vector(position)
    camera.rotation_mode = 'QUATERNION'
    qx, qy, qz, qw = quaternion_xyzw
    camera.rotation_quaternion = Quaternion((qw, qx, qy, qz))


def smoke_test_render(scene, camera, output_dir):
    """Render a single frame to validate pipeline wiring."""
    os.makedirs(output_dir, exist_ok=True)
    smoke_path = os.path.join(output_dir, "smoke_test.png")
    set_camera_pose(camera, (0.0, 0.0, 3.0), (1.0, 0.0, 0.0, 0.0))
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
    smoke_test=False
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

    while t < duration:
        pos, quat = trajectories.figure8(t)
        if t >= next_cam - 1e-9:
            ts = t
            filename = os.path.join(cam_dir, f"{int(ts*1e9):016d}.png")
            # In Blender: set camera transform and render to filename
            # Example (to implement): bpy.context.scene.camera.matrix_world = matrix_from_pos_quat(pos, quat)
            # bpy.context.scene.render.filepath = filename
            # bpy.ops.render.render(write_still=True)
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
        smoke_test=args.smoke_test
    )

