import bpy
import csv
import os
import sys
import random
import argparse
from mathutils import Vector, Quaternion

# Blender runner: textured floor + poses from traj_gen_v4 trajectory.csv, 10 Hz renders.

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

PHASE2_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
# Dataset lives under static/tmp after moving phase2_data; default renders one train split.
DEFAULT_OUTPUT_DIR = os.path.join(
    PHASE2_ROOT, 'static', 'tmp', 'phase2_data', 'train', 'traj_000'
)
DEFAULT_FLOOR_TEXTURE = os.path.join(
    PHASE2_ROOT,
    'static',
    'Wheres-Waldo-Space-Station-Super-High-Resolution-scaled.jpg',
)

TRAJECTORY_CSV_NAME = 'trajectory.csv'
# traj_gen_v4 exports ~100 Hz; fixed 10 Hz camera => every 10th sample
CAMERA_SUBSAMPLE_STRIDE = 10

try:
    from . import materials
except ImportError:
    import materials  # type: ignore


def resolve_trajectory_csv(output_dir):
    """Find trajectory.csv: output_dir first, then cwd. Returns (path, tried_paths)."""
    tried = []
    out_abs = os.path.abspath(os.path.expanduser(output_dir))
    candidates = [
        os.path.join(out_abs, TRAJECTORY_CSV_NAME),
        os.path.join(os.getcwd(), TRAJECTORY_CSV_NAME),
    ]
    for p in candidates:
        tried.append(os.path.abspath(p))
        if os.path.isfile(p):
            return os.path.abspath(p), tried
    return None, tried


def load_trajectory_v4_csv(path):
    """Parse traj_gen_v4 trajectory.csv into list of (t, (x,y,z), (qx,qy,qz,qw))."""
    required = {'t', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'}
    rows_out = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {path}")
        headers = {h.strip() for h in reader.fieldnames}
        missing = required - headers
        if missing:
            raise ValueError(
                f"{path} missing columns {sorted(missing)}; have {sorted(headers)}"
            )
        for row in reader:
            t = float(row['t'])
            pos = (float(row['x']), float(row['y']), float(row['z']))
            qw = float(row['qw'])
            qx = float(row['qx'])
            qy = float(row['qy'])
            qz = float(row['qz'])
            quat_xyzw = (qx, qy, qz, qw)
            rows_out.append((t, pos, quat_xyzw))
    if not rows_out:
        raise ValueError(f"No data rows in {path}")
    return rows_out


def subsample_camera_rows(rows, stride=CAMERA_SUBSAMPLE_STRIDE):
    """Deterministic 10 Hz from uniform high-rate rows: indices 0, stride, 2*stride, ..."""
    return [rows[i] for i in range(0, len(rows), stride)]


def parse_args(argv=None):
    """Parse CLI args passed after '--' in Blender command."""
    if argv is None:
        argv = sys.argv

    user_argv = []
    if '--' in argv:
        user_argv = argv[argv.index('--') + 1:]

    parser = argparse.ArgumentParser(description="Phase 2 Blender data generator")
    parser.add_argument(
        '--output_dir',
        default=DEFAULT_OUTPUT_DIR,
        help='Renders and groundtruth go here; must contain trajectory.csv (default: train/traj_000)',
    )
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
    parser.add_argument('--floor_size', type=float, default=150.0)
    parser.add_argument('--uv_scale', nargs=2, type=float, default=[1.0, 1.0],
                        metavar=('SX', 'SY'))
    parser.add_argument('--uv_rotate_z', type=float, default=0.0)

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


def smoke_test_render(scene, camera, output_dir):
    """Render a single frame to validate pipeline wiring."""
    os.makedirs(output_dir, exist_ok=True)
    smoke_path = os.path.join(output_dir, "smoke_test.png")
    set_camera_pose(camera, (0.0, 0.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    scene.render.filepath = smoke_path
    bpy.ops.render.render(write_still=True)
    print(f"Smoke test render written to {smoke_path}")


def export_pose_file(path, timestamp, position, quaternion_xyzw):
    # Append a single line: timestamp tx ty tz qx qy qz qw
    qx, qy, qz, qw = quaternion_xyzw
    with open(path, 'a') as f:
        f.write(
            f"{timestamp:.9f} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
            f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
        )


def run_demo(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    render_resolution=(640, 480),
    engine='BLENDER_EEVEE',
    fps=100,
    seed=42,
    smoke_test=False,
    floor_texture=DEFAULT_FLOOR_TEXTURE,
    crop_uv=None,
    floor_size: float = 150.0,
    uv_scale=(1.0, 1.0),
    uv_rotate_z: float = 0.0,
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
    camera.data.lens_unit = 'FOV'
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.angle = 1.57
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

    if smoke_test:
        smoke_test_render(scene, camera, output_dir)
        return

    traj_path, tried = resolve_trajectory_csv(output_dir)
    if traj_path is None:
        raise FileNotFoundError(
            "trajectory.csv not found. Tried:\n  " + "\n  ".join(tried)
        )
    print(f"Using trajectory: {traj_path}")

    all_rows = load_trajectory_v4_csv(traj_path)
    cam_rows = subsample_camera_rows(all_rows)
    if not cam_rows:
        raise ValueError("Subsampled camera trajectory is empty")

    with open(pose_csv, 'w') as f:
        f.write('# timestamp tx ty tz qx qy qz qw\n')

    frame_idx = 0
    for ts, pos, quat_xyzw in cam_rows:
        filename = os.path.join(cam_dir, f"{int(ts*1e9):016d}.png")
        set_camera_pose(camera, pos, quat_xyzw)
        scene.render.filepath = filename
        bpy.ops.render.render(write_still=True)
        export_pose_file(pose_csv, ts, pos, quat_xyzw)
        frame_idx += 1

    print(f"Wrote {frame_idx} frames to {cam_dir} (10 Hz subsample, stride={CAMERA_SUBSAMPLE_STRIDE})")


if __name__ == '__main__':
    args = parse_args()
    run_demo(
        output_dir=args.output_dir,
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
    )
