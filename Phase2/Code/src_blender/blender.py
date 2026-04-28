import bpy
import os
import sys
import math

# Starter blender runner: setup scene, sample trajectories, render frames, export poses.
# This file augments the minimal blender.py that creates a plane. Implement details using bpy.

from . import materials, intrinsics, trajectories


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv
    if '--' in argv:
        idx = argv.index('--')
        return argv[idx+1:]
    return []


def setup_scene(render_resolution=(640,480), engine='BLENDER_EEVEE'):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = render_resolution[0]
    scene.render.resolution_y = render_resolution[1]
    scene.render.fps = 100
    return scene


def export_pose_file(path, timestamp, position, quaternion):
    # Append a single line: timestamp tx ty tz qx qy qz qw
    with open(path, 'a') as f:
        f.write(f"{timestamp:.9f} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} {quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n")


def run_demo(output_dir: str = '/tmp/phase2_out', duration: float = 5.0, imu_rate: float = 200.0, cam_rate: float = 20.0):
    os.makedirs(output_dir, exist_ok=True)
    cam_dir = os.path.join(output_dir, 'cam0', 'data')
    os.makedirs(cam_dir, exist_ok=True)
    pose_csv = os.path.join(output_dir, 'groundtruth.csv')

    scene = setup_scene()

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
    run_demo()

