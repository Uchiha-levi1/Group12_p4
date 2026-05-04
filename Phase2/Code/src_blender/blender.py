import bpy
import csv
import os
import re
import sys
import random
import argparse
from mathutils import Vector, Quaternion

# Blender runner: textured floor + poses from traj_gen_v4 trajectory.csv, 10 Hz renders.

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

PHASE2_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
DEFAULT_DATASET_ROOT = os.path.join(PHASE2_ROOT, 'static', 'phase2_data')
DEFAULT_FLOOR_TEXTURE = os.path.join(
    PHASE2_ROOT,
    'static',
    'Wheres-Waldo-Space-Station-Super-High-Resolution-scaled.jpg',
)

TRAJECTORY_CSV_NAME = 'trajectory.csv'
TRAJ_DIR_RE = re.compile(r'^traj_(\d+)$')
# Batch order: repeat rounds of 80/10/10 — 8*y train, y val, y test (fixed y here).
RENDER_CHUNK_VAL = 1
# traj_gen_v4 exports ~100 Hz; fixed 10 Hz camera => every 10th sample
CAMERA_SUBSAMPLE_STRIDE = 10

try:
    from . import materials
except ImportError:
    import materials  # type: ignore


def _collect_split_trajs(root, split):
    """Sorted absolute paths: traj_* under root/<split>/ that contain trajectory.csv."""
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        return []
    rows = []
    for name in os.listdir(split_dir):
        m = TRAJ_DIR_RE.match(name)
        if not m:
            continue
        path = os.path.join(split_dir, name)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, TRAJECTORY_CSV_NAME)):
            rows.append((int(m.group(1)), os.path.abspath(path)))
    rows.sort(key=lambda t: t[0])
    return [p for _, p in rows]


def discover_traj_dirs(dataset_root):
    """List traj_* dirs with trajectory.csv.

    Order: rounds of 80/10/10 — 8*y train, y val, y test, repeat until all splits
    are exhausted. y = RENDER_CHUNK_VAL (1 => 8 train, 1 val, 1 test per round);
    within each split, trajectories are ordered by traj index.
    """
    root = os.path.abspath(os.path.expanduser(dataset_root))
    y = max(1, RENDER_CHUNK_VAL)
    x = 8 * y

    train_paths = _collect_split_trajs(root, 'train')
    val_paths = _collect_split_trajs(root, 'val')
    test_paths = _collect_split_trajs(root, 'test')

    it_t = it_v = it_te = 0
    out = []
    while it_t < len(train_paths) or it_v < len(val_paths) or it_te < len(test_paths):
        for _ in range(x):
            if it_t < len(train_paths):
                out.append(train_paths[it_t])
                it_t += 1
        for _ in range(y):
            if it_v < len(val_paths):
                out.append(val_paths[it_v])
                it_v += 1
        for _ in range(y):
            if it_te < len(test_paths):
                out.append(test_paths[it_te])
                it_te += 1
    return out


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


def _count_png_files(directory):
    if not os.path.isdir(directory):
        return 0
    return sum(1 for name in os.listdir(directory) if name.lower().endswith('.png'))


def _count_groundtruth_pose_lines(path):
    """Number of non-empty, non-comment data lines in groundtruth.csv."""
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                n += 1
    return n


def render_output_complete(output_dir, expected_frames):
    """True if images/ has expected PNG count and groundtruth.csv has same pose rows."""
    if expected_frames <= 0:
        return False
    images_dir = os.path.join(output_dir, 'images')
    pose_csv = os.path.join(output_dir, 'groundtruth.csv')
    if _count_png_files(images_dir) != expected_frames:
        return False
    if _count_groundtruth_pose_lines(pose_csv) != expected_frames:
        return False
    return True


def parse_args(argv=None):
    """Parse CLI args passed after '--' in Blender command."""
    if argv is None:
        argv = sys.argv

    user_argv = []
    if '--' in argv:
        user_argv = argv[argv.index('--') + 1:]

    parser = argparse.ArgumentParser(description="Phase 2 Blender data generator")
    parser.add_argument(
        '--dataset_root',
        default=DEFAULT_DATASET_ROOT,
        help='Root with train/val/test/traj_*; used when --output_dir is omitted (batch all)',
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Render a single trajectory folder only (must contain trajectory.csv). '
        'If omitted, renders every traj_* under --dataset_root.',
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
    output_dir: str,
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
    cam_dir = os.path.join(output_dir, 'images')
    pose_csv = os.path.join(output_dir, 'groundtruth.csv')

    if smoke_test:
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

    n_frames = len(cam_rows)
    if render_output_complete(output_dir, n_frames):
        print(
            f"Skip (already complete): {n_frames} PNGs and {n_frames} GT poses in "
            f"{cam_dir}"
        )
        return

    os.makedirs(cam_dir, exist_ok=True)

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


def _smoke_output_dir():
    return os.path.join(DEFAULT_DATASET_ROOT, '__smoke__')


if __name__ == '__main__':
    args = parse_args()
    common = dict(
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

    if args.smoke_test:
        run_demo(output_dir=_smoke_output_dir(), **common)
    elif args.output_dir is not None:
        run_demo(output_dir=os.path.abspath(os.path.expanduser(args.output_dir)), **common)
    else:
        traj_dirs = discover_traj_dirs(args.dataset_root)
        if not traj_dirs:
            raise FileNotFoundError(
                f"No traj_* folders with {TRAJECTORY_CSV_NAME} under {args.dataset_root}"
            )
        print(f"Batch render: {len(traj_dirs)} trajectories under {args.dataset_root}")
        for i, traj_dir in enumerate(traj_dirs):
            print(f"\n[{i + 1}/{len(traj_dirs)}] {traj_dir}")
            run_demo(output_dir=traj_dir, **common)
