"""
Trajectory generator v4 for Deep VIO project.

Adds vs v3:
  - Per-segment duration proportional to step length:
        t_seg = step_length * time_factor
    where time_factor varies in [0.5, 1.0] s/m per trajectory
    (slow → aggressive). Reference: 0.667 s/m → 3m takes 2s.
  - Per-trajectory parameter variation:
        N_CPS in U(15, 25)
        heading cone angle in U(70°, 120°)
        yaw step min in [0°, 30°]
        time_factor in U(0.5, 1.0)
  - Dataset generator: 50 train + 10 val + 10 test = 70 trajectories.
    Each trajectory in its own folder with:
        trajectory.csv / trajectory.npz   (GT physics)
        imu.csv / imu.npz                 (true + measured IMU)
        meta.json                         (config + diagnostics)
        plots/traj.png, plots/imu.png
    Top-level dataset_meta.json with global info.

IMU profile fixed: vibration-affected MPU-6050 (vibration_mpu6050).
Bounds fixed:    |v|≤3, |a|≤4, |ω|≤1.
Workspace fixed: xy ±55 m, z [5,10] m, plane 150×150.
"""

import os
import json
import csv
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import minsnap_trajectories as ms
from scipy.spatial.transform import Rotation as R

# GNSS-INS-Sim noise model (applies bias + Gauss-Markov drift + white noise)
from gnss_ins_sim.pathgen.pathgen import acc_gen, gyro_gen
from gnss_ins_sim.sim.imu_model import IMU as GnssIMU

_PHASE2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_TRAJ_DATA_ROOT = os.path.join(_PHASE2_ROOT, 'static', 'phase2_data')

# ============================================================
# FIXED CONFIG (same for all trajectories)
# ============================================================

# Workspace (plane 150x150, centered at origin; margin 20m -> ±55)
XY_BOUND = 55.0
Z_MIN, Z_MAX = 5.0, 10.0
Z_CLAMP_MARGIN = 1.0
PLANE_SIZE = 150.0

# Step
STEP_MIN, STEP_MAX = 2.0, 6.0

# Bounds (moderate; lets aggressive time_factors actually use them)
V_MAX = 5.0
A_MAX = 6.0
OMEGA_MAX = 2.0
RP_MAX_DEG = 40.0
RP_MAX_RAD = np.deg2rad(RP_MAX_DEG)

# Sampling rates
DENSE_HZ = 1000
OUT_HZ = 100

# Backtrack
N_RETRY_PER_CP = 100
N_BACKTRACK_BUDGET = 50

# Gravity
G = np.array([0.0, 0.0, -9.81])

# IMU profile: vibration-affected MPU-6050
IMU_PROFILE_NAME = 'vibration_mpu6050'
IMU_CUSTOM_ERR = {
    'accel': {
        'b':       np.array([0.0, 0.0, 0.0]),
        'b_drift': np.array([9.81e-4, 9.81e-4, 9.81e-4]),     # 0.1 mg
        'b_corr':  np.array([100.0, 100.0, 100.0]),
        'vrw':     np.array([0.015, 0.015, 0.015]),           # σ_per_sample=0.15 @ 100Hz
    },
    'gyro': {
        'b':       np.array([0.0, 0.0, 0.0]),
        'b_drift': np.array([9.696e-5, 9.696e-5, 9.696e-5]),  # 20°/hr
        'b_corr':  np.array([100.0, 100.0, 100.0]),
        'arw':     np.array([0.0008, 0.0008, 0.0008]),        # σ_per_sample=0.008 @ 100Hz
    },
}

# ============================================================
# PER-TRAJECTORY CONFIG
# ============================================================

@dataclass
class TrajConfig:
    """Configuration that varies per trajectory."""
    seed: int
    n_cps: int = 20
    cone_angle_deg: float = 90.0     # heading cone half-angle
    yaw_step_min_deg: float = 5.0    # min yaw step magnitude
    yaw_step_max_deg: float = 30.0   # max yaw step magnitude
    yaw_same_sign_prob: float = 0.7
    time_factor: float = 0.667       # s/m, t_seg = step_length * time_factor

    @property
    def heading_cos_threshold(self) -> float:
        return float(np.cos(np.deg2rad(self.cone_angle_deg)))


def sample_traj_config(seed: int, rng: np.random.Generator) -> TrajConfig:
    """Sample a per-trajectory config from the variation ranges."""
    return TrajConfig(
        seed=seed,
        n_cps=int(rng.integers(15, 26)),                       # U[15, 25]
        cone_angle_deg=float(rng.uniform(70.0, 120.0)),        # U(70, 120)
        yaw_step_min_deg=0.0,                                  # fixed at 0
        yaw_step_max_deg=30.0,                                 # fixed at 30
        yaw_same_sign_prob=0.7,
        time_factor=float(rng.uniform(0.5, 1.0)),              # U(0.5, 1.0)
    )

# -------------------------- CP SAMPLING --------------------------

def sample_unit_vector_3d():
    """Uniform on 3D sphere."""
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)


def in_workspace(p):
    return (-XY_BOUND <= p[0] <= XY_BOUND
            and -XY_BOUND <= p[1] <= XY_BOUND
            and Z_MIN + Z_CLAMP_MARGIN <= p[2] <= Z_MAX - Z_CLAMP_MARGIN)


def sample_initial_cps():
    """Sample CP[0] in inner box, CP[1] = CP[0] + step*d.
    Returns p0, p1, d_init, step01."""
    while True:
        p0 = np.array([
            np.random.uniform(-0.6 * XY_BOUND, 0.6 * XY_BOUND),
            np.random.uniform(-0.6 * XY_BOUND, 0.6 * XY_BOUND),
            np.random.uniform(Z_MIN + Z_CLAMP_MARGIN, Z_MAX - Z_CLAMP_MARGIN),
        ])
        for _ in range(50):
            d = sample_unit_vector_3d()
            step = np.random.uniform(STEP_MIN, STEP_MAX)
            p1 = p0 + step * d
            if in_workspace(p1):
                return p0, p1, d, step
        # Couldn't extend from this p0, resample p0


def propose_next_cp(p_prev, d_prev, cos_threshold):
    """Try one proposal of next CP via heading-cone sampling.
    Returns (p_new, d_new, step) or (None, None, None)."""
    d = sample_unit_vector_3d()
    if np.dot(d, d_prev) < cos_threshold:
        return None, None, None
    step = np.random.uniform(STEP_MIN, STEP_MAX)
    p_new = p_prev + step * d
    if not in_workspace(p_new):
        return None, None, None
    return p_new, d, step


def generate_position_cps(cfg: TrajConfig):
    """Generate cfg.n_cps position control points + the steps between them."""
    cos_thresh = cfg.heading_cos_threshold
    while True:
        backtracks_left = N_BACKTRACK_BUDGET
        p0, p1, d_init, step01 = sample_initial_cps()
        cps = [p0, p1]
        dirs = [d_init]                # len = len(cps) - 1
        steps = [step01]                # len = len(cps) - 1
        i = 2

        while i < cfg.n_cps:
            success = False
            for _ in range(N_RETRY_PER_CP):
                p_new, d_new, step_new = propose_next_cp(
                    cps[-1], dirs[-1], cos_thresh)
                if p_new is not None:
                    cps.append(p_new)
                    dirs.append(d_new)
                    steps.append(step_new)
                    i += 1
                    success = True
                    break

            if not success:
                if backtracks_left <= 0 or len(cps) <= 2:
                    break
                cps.pop()
                dirs.pop()
                steps.pop()
                i -= 1
                backtracks_left -= 1

        if len(cps) == cfg.n_cps:
            return np.array(cps), np.array(steps)


def generate_yaw_cps(n, cfg: TrajConfig, seg_durations=None):
    """Generate n yaw waypoints. If seg_durations given, cap each yaw step
    so yaw rate stays under OMEGA_MAX with margin (e.g., 0.7 * OMEGA_MAX)."""
    yaws = [np.random.uniform(-np.pi, np.pi)]
    last_sign = np.random.choice([-1, 1])
    for i in range(n - 1):
        mag = np.deg2rad(np.random.uniform(
            cfg.yaw_step_min_deg, cfg.yaw_step_max_deg))
        if seg_durations is not None:
            # Cap step so peak yaw rate ≈ 1.875 * step / t_seg < 0.7 * OMEGA_MAX
            # (1.875 is min-snap rest-to-rest peak factor; conservative cap)
            cap = 0.7 * OMEGA_MAX * seg_durations[i] / 1.875
            mag = min(mag, cap)
        if np.random.random() < cfg.yaw_same_sign_prob:
            sign = last_sign
        else:
            sign = -last_sign
        yaws.append(yaws[-1] + sign * mag)
        last_sign = sign
    return np.array(yaws)


# -------------------------- MIN-SNAP FITTING --------------------------

def fit_minsnap_position(cps, times):
    """Fit one global piecewise min-snap polynomial through CPs.
    Endpoints rest. Interior derivatives free."""
    n = len(cps)
    refs = []
    for i, (t, p) in enumerate(zip(times, cps)):
        if i == 0 or i == n - 1:
            refs.append(ms.Waypoint(
                time=t, position=p,
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
            ))
        else:
            # Position only — library QP fills in vel/accel
            refs.append(ms.Waypoint(time=t, position=p))
    polys = ms.generate_trajectory(
        refs, degree=7, idx_minimized_orders=4, num_continuous_orders=3
    )
    return polys


def fit_minsnap_yaw(yaws, times):
    """Fit 1D min-acceleration polynomial through yaw waypoints. Endpoints rest."""
    n = len(yaws)
    refs = []
    for i, (t, y) in enumerate(zip(times, yaws)):
        if i == 0 or i == n - 1:
            refs.append(ms.Waypoint(
                time=t, position=np.array([y]),
                velocity=np.array([0.0]),
                acceleration=np.array([0.0]),
            ))
        else:
            refs.append(ms.Waypoint(time=t, position=np.array([y])))
    polys = ms.generate_trajectory(
        refs, degree=5, idx_minimized_orders=2, num_continuous_orders=3
    )
    return polys


def eval_polys(polys, t_samples, order):
    """Returns derivs of shape (order+1, n_samples, dim)."""
    return ms.compute_trajectory_derivatives(polys, t_samples, order)


# -------------------------- ORIENTATION --------------------------

def differential_flatness_rpy(accel_world, yaw):
    """Quadrotor flat-output -> roll, pitch."""
    thrust = accel_world - G
    thrust_norm = np.linalg.norm(thrust, axis=-1, keepdims=True)
    z_b = thrust / np.clip(thrust_norm, 1e-6, None)

    cy, sy = np.cos(yaw), np.sin(yaw)
    x_c = np.stack([cy, sy, np.zeros_like(cy)], axis=-1)

    y_b = np.cross(z_b, x_c)
    y_b = y_b / np.clip(np.linalg.norm(y_b, axis=-1, keepdims=True), 1e-6, None)
    x_b = np.cross(y_b, z_b)

    Rmat = np.stack([x_b, y_b, z_b], axis=-1)
    roll = np.arctan2(Rmat[..., 2, 1], Rmat[..., 2, 2])
    pitch = -np.arcsin(np.clip(Rmat[..., 2, 0], -1.0, 1.0))
    return roll, pitch, Rmat


def rpy_to_quat(roll, pitch, yaw):
    """ZYX intrinsic Euler -> quaternion [w, x, y, z]."""
    rot = R.from_euler('ZYX', np.stack([yaw, pitch, roll], axis=-1))
    q_xyzw = rot.as_quat()
    return np.concatenate([q_xyzw[..., 3:4], q_xyzw[..., :3]], axis=-1)


def quat_to_omega_body(q_wxyz, t):
    """Numerical body-frame angular velocity from quaternion derivative."""
    dt = np.gradient(t)
    qd = np.gradient(q_wxyz, axis=0) / dt[:, None]
    q_conj = q_wxyz.copy()
    q_conj[:, 1:] *= -1
    w1, x1, y1, z1 = q_conj[:, 0], q_conj[:, 1], q_conj[:, 2], q_conj[:, 3]
    w2, x2, y2, z2 = qd[:, 0], qd[:, 1], qd[:, 2], qd[:, 3]
    px = w1*x2 + x1*w2 + y1*z2 - z1*y2
    py = w1*y2 - x1*z2 + y1*w2 + z1*x2
    pz = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return 2.0 * np.stack([px, py, pz], axis=-1)


# -------------------------- IMU SIMULATION --------------------------

def compute_true_imu_body(acc_world, omega_body, quat_wxyz):
    """
    Convert world-frame kinematic accel + body angular velocity into
    'true' (noiseless) IMU readings:
      accel_true_body = R_wb^T * (acc_world - g_world)   [specific force]
      gyro_true_body  = omega_body                        [already body frame]

    R_wb maps body->world, so R_wb^T maps world->body.
    Build R_wb from quaternion [w, x, y, z].
    """
    # scipy expects [x, y, z, w]
    q_xyzw = np.concatenate([quat_wxyz[..., 1:], quat_wxyz[..., 0:1]], axis=-1)
    R_wb = R.from_quat(q_xyzw).as_matrix()        # (N, 3, 3), body->world
    R_bw = R_wb.transpose(0, 2, 1)                # world->body

    # specific force in world: f_world = a_world - g_world
    # (gravity is [0,0,-9.81], so a_world - g = a_world + [0,0,9.81])
    f_world = acc_world - G
    # Rotate to body: per-timestep matrix-vector product
    accel_true_body = np.einsum('nij,nj->ni', R_bw, f_world)

    gyro_true_body = omega_body
    return accel_true_body, gyro_true_body


def add_imu_noise(accel_true, gyro_true, fs, accel_err, gyro_err, profile_name='custom'):
    """
    Apply GNSS-INS-Sim IMU noise model with explicit error dicts.

    accel_err keys: 'b', 'b_drift', 'b_corr', 'vrw'
    gyro_err  keys: 'b', 'b_drift', 'b_corr', 'arw'
    """
    accel_meas = acc_gen(fs, accel_true, accel_err)
    gyro_meas = gyro_gen(fs, gyro_true, gyro_err)
    err_info = {
        'profile': profile_name,
        'accel_b': accel_err['b'],
        'accel_b_drift': accel_err['b_drift'],
        'accel_b_corr': accel_err['b_corr'],
        'accel_vrw': accel_err['vrw'],
        'gyro_b': gyro_err['b'],
        'gyro_b_drift': gyro_err['b_drift'],
        'gyro_b_corr': gyro_err['b_corr'],
        'gyro_arw': gyro_err['arw'],
    }
    return accel_meas, gyro_meas, err_info


# -------------------------- MAIN --------------------------

def generate_trajectory(cfg: TrajConfig, verbose=True, max_attempts=5):
    """Generate one trajectory. Retries if z overshoots workspace.

    Strategy: if z overshoots bounds, shrink the inner-CP z box and retry.
    Up to max_attempts; raises if all fail.
    """
    for attempt in range(max_attempts):
        traj = _generate_trajectory_once(cfg, verbose=verbose)
        z_min, z_max = traj['pos'][:, 2].min(), traj['pos'][:, 2].max()
        # Allow small numerical slack
        if z_min >= Z_MIN - 0.05 and z_max <= Z_MAX + 0.05:
            return traj
        if verbose:
            print(f"  z overshoot detected (z range [{z_min:.2f}, {z_max:.2f}]), "
                  f"retry {attempt+1}/{max_attempts}")
        # Bump seed slightly to get different CPs
        cfg = TrajConfig(**{**asdict(cfg), 'seed': cfg.seed + 100000 * (attempt + 1)})
    if verbose:
        print(f"  WARNING: returning trajectory with z overshoot after {max_attempts} attempts")
    return traj


def _generate_trajectory_once(cfg: TrajConfig, verbose=True):
    """Generate one trajectory from a TrajConfig (single attempt)."""
    np.random.seed(cfg.seed)

    # Step 1: sample CPs (positions + per-segment steps)
    pos_cps, steps = generate_position_cps(cfg)

    # Per-segment durations: t_seg = step_length * time_factor
    seg_durations = steps * cfg.time_factor          # shape (n_cps - 1,)
    times = np.concatenate([[0.0], np.cumsum(seg_durations)])
    T_design = float(times[-1])

    yaw_cps = generate_yaw_cps(cfg.n_cps, cfg, seg_durations=seg_durations)

    if verbose:
        cp_dists = np.linalg.norm(np.diff(pos_cps, axis=0), axis=1)
        print(f"CPs sampled: {cfg.n_cps}")
        print(f"  cone={cfg.cone_angle_deg:.1f}°, time_factor={cfg.time_factor:.3f} s/m, "
              f"yaw_step=[{cfg.yaw_step_min_deg:.0f},{cfg.yaw_step_max_deg:.0f}]°")
        print(f"  CP step: min={cp_dists.min():.2f}, max={cp_dists.max():.2f}, "
              f"mean={cp_dists.mean():.2f} m")
        print(f"  Seg duration: min={seg_durations.min():.2f}, "
              f"max={seg_durations.max():.2f}, mean={seg_durations.mean():.2f} s")
        print(f"  Total path length: {cp_dists.sum():.1f} m")
        print(f"  Design duration: {T_design:.1f} s")

    # Step 2: min-snap fit
    polys_xyz = fit_minsnap_position(pos_cps, times)
    polys_yaw = fit_minsnap_yaw(yaw_cps, times)

    # Step 3: dense eval
    n_dense = int(round(T_design * DENSE_HZ)) + 1
    t_dense = np.linspace(0, T_design, n_dense)
    derivs_xyz = eval_polys(polys_xyz, t_dense, order=4)
    pos = derivs_xyz[0]
    vel = derivs_xyz[1]
    acc = derivs_xyz[2]

    derivs_yaw = eval_polys(polys_yaw, t_dense, order=2)
    yaw = derivs_yaw[0][:, 0]

    # Step 4: peaks BEFORE rescale
    v_peak = np.linalg.norm(vel, axis=1).max()
    a_peak = np.linalg.norm(acc, axis=1).max()
    roll_pre, pitch_pre, _ = differential_flatness_rpy(acc, yaw)
    quat_pre = rpy_to_quat(roll_pre, pitch_pre, yaw)
    omega_pre = quat_to_omega_body(quat_pre, t_dense)
    omega_peak = np.linalg.norm(omega_pre[5:-5], axis=1).max()
    rp_peak_rad = max(np.abs(roll_pre).max(), np.abs(pitch_pre).max())

    # Step 5: time-rescale
    # v scales 1/α, a scales 1/α², ω scales 1/α
    # roll/pitch ≈ atan(|a_horiz| / g); after rescale, |a_horiz| -> |a_horiz|/α²
    # tan(rp_new) = tan(rp_old)/α². Need rp_new ≤ RP_MAX:
    #     α² ≥ tan(rp_peak) / tan(RP_MAX)  ->  α ≥ √( tan(rp_peak) / tan(RP_MAX) )
    alpha_v = v_peak / V_MAX
    alpha_a = np.sqrt(a_peak / A_MAX)
    alpha_w = omega_peak / OMEGA_MAX
    if rp_peak_rad > RP_MAX_RAD:
        alpha_rp = np.sqrt(np.tan(rp_peak_rad) / np.tan(RP_MAX_RAD))
    else:
        alpha_rp = 0.0
    alpha = max(alpha_v, alpha_a, alpha_w, alpha_rp, 1.0)
    if verbose:
        print(f"  Pre-rescale peaks: |v|={v_peak:.2f}, |a|={a_peak:.2f}, "
              f"|ω|={omega_peak:.2f}, rp={np.rad2deg(rp_peak_rad):.1f}°")
        print(f"  Rescale: α_v={alpha_v:.2f}, α_a={alpha_a:.2f}, "
              f"α_ω={alpha_w:.2f}, α_rp={alpha_rp:.2f} -> α={alpha:.2f}")

    T_final = alpha * T_design
    n_final = int(round(T_final * DENSE_HZ)) + 1
    t_phys = np.linspace(0, T_final, n_final)
    t_design_samples = t_phys / alpha

    derivs_xyz_final = eval_polys(polys_xyz, t_design_samples, order=4)
    pos = derivs_xyz_final[0]
    vel = derivs_xyz_final[1] / alpha
    acc = derivs_xyz_final[2] / (alpha ** 2)

    derivs_yaw_final = eval_polys(polys_yaw, t_design_samples, order=2)
    yaw = derivs_yaw_final[0][:, 0]

    # Differential flatness on rescaled accel
    roll, pitch, _ = differential_flatness_rpy(acc, yaw)
    quat = rpy_to_quat(roll, pitch, yaw)
    omega_body = quat_to_omega_body(quat, t_phys)

    # Downsample
    stride = DENSE_HZ // OUT_HZ
    sl = slice(None, None, stride)

    # Compute true (noiseless) body-frame IMU readings from downsampled GT
    accel_true_body, gyro_true_body = compute_true_imu_body(
        acc[sl], omega_body[sl], quat[sl]
    )

    # Apply GNSS-INS-Sim noise model
    accel_meas, gyro_meas, imu_err = add_imu_noise(
        accel_true_body, gyro_true_body, OUT_HZ,
        IMU_CUSTOM_ERR['accel'], IMU_CUSTOM_ERR['gyro'],
        profile_name=IMU_PROFILE_NAME,
    )

    out = {
        't': t_phys[sl],
        'pos': pos[sl],
        'vel': vel[sl],
        'acc_world': acc[sl],
        'roll': roll[sl],
        'pitch': pitch[sl],
        'yaw': yaw[sl],
        'quat_wxyz': quat[sl],
        'omega_body': omega_body[sl],
        # True (noiseless) IMU readings, body frame
        'accel_true_body': accel_true_body,
        'gyro_true_body': gyro_true_body,
        # Noisy IMU measurements
        'accel_meas': accel_meas,
        'gyro_meas': gyro_meas,
        'imu_err': imu_err,
        'cps_pos': pos_cps,
        'cps_yaw': yaw_cps,
        'cps_time': alpha * times,
        'cps_steps': steps,
        'alpha': alpha,
        'duration': T_final,
        'design_duration': T_design,
        'cfg': cfg,
    }

    if verbose:
        v_mag = np.linalg.norm(out['vel'], axis=1)
        a_mag = np.linalg.norm(out['acc_world'], axis=1)
        w_mag = np.linalg.norm(out['omega_body'][5:-5], axis=1)
        print(f"\nPost-rescale diagnostics:")
        print(f"  Final duration: {T_final:.1f} s, samples @ {OUT_HZ}Hz: {len(out['t'])}")
        print(f"  |v| max: {v_mag.max():.2f} m/s (bound {V_MAX})")
        print(f"  |a| max: {a_mag.max():.2f} m/s² (bound {A_MAX})")
        print(f"  |ω| max: {w_mag.max():.2f} rad/s (bound {OMEGA_MAX})")
        print(f"  roll range: [{np.rad2deg(out['roll']).min():.1f}, "
              f"{np.rad2deg(out['roll']).max():.1f}]°")
        print(f"  pitch range: [{np.rad2deg(out['pitch']).min():.1f}, "
              f"{np.rad2deg(out['pitch']).max():.1f}]°")
        print(f"  z range: [{out['pos'][:,2].min():.2f}, "
              f"{out['pos'][:,2].max():.2f}] m")
        print(f"  xy span: x[{out['pos'][:,0].min():.1f},{out['pos'][:,0].max():.1f}], "
              f"y[{out['pos'][:,1].min():.1f},{out['pos'][:,1].max():.1f}] m")
        a_resid = accel_meas - accel_true_body
        g_resid = gyro_meas - gyro_true_body
        print(f"  accel residual std: {a_resid.std(axis=0).round(4)}")
        print(f"  gyro residual std:  {g_resid.std(axis=0).round(5)} rad/s")

    return out


# -------------------------- PLOTTING --------------------------

def plot_trajectory(traj, save_path):
    fig = plt.figure(figsize=(16, 12))

    # 3D trajectory + orientation triads
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.plot(traj['pos'][:, 0], traj['pos'][:, 1], traj['pos'][:, 2],
              'b-', lw=1.0, alpha=0.7, label='trajectory')
    ax3d.scatter(traj['cps_pos'][:, 0], traj['cps_pos'][:, 1], traj['cps_pos'][:, 2],
                 c='r', s=20, label='control points')

    triad_stride = OUT_HZ
    triad_len = 1.0
    for i in range(0, len(traj['t']), triad_stride):
        rot = R.from_quat(np.concatenate([traj['quat_wxyz'][i, 1:],
                                           traj['quat_wxyz'][i, 0:1]]))
        Rm = rot.as_matrix()
        p = traj['pos'][i]
        for axis_idx, color in zip(range(3), ['r', 'g', 'b']):
            d = Rm[:, axis_idx] * triad_len
            ax3d.plot([p[0], p[0]+d[0]], [p[1], p[1]+d[1]], [p[2], p[2]+d[2]],
                      color=color, lw=1.5)

    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title(f'3D trajectory (N={len(traj["cps_pos"])} CPs, T={traj["duration"]:.1f}s, α={traj["alpha"]:.2f})')
    ax3d.legend()
    pad = 2
    ax3d.set_xlim(traj['pos'][:,0].min()-pad, traj['pos'][:,0].max()+pad)
    ax3d.set_ylim(traj['pos'][:,1].min()-pad, traj['pos'][:,1].max()+pad)
    ax3d.set_zlim(0, Z_MAX + 2)

    # Velocity panel
    ax_v = fig.add_subplot(2, 3, 2)
    ax_v.plot(traj['t'], traj['vel'][:, 0], label='vx', alpha=0.7)
    ax_v.plot(traj['t'], traj['vel'][:, 1], label='vy', alpha=0.7)
    ax_v.plot(traj['t'], traj['vel'][:, 2], label='vz', alpha=0.7)
    v_mag = np.linalg.norm(traj['vel'], axis=1)
    ax_v.plot(traj['t'], v_mag, 'k-', lw=1.2, label='|v|')
    ax_v.axhline(V_MAX, color='r', ls='--', alpha=0.5, label=f'bound ±{V_MAX}')
    ax_v.axhline(-V_MAX, color='r', ls='--', alpha=0.5)
    ax_v.set_xlabel('t (s)'); ax_v.set_ylabel('velocity (m/s)')
    ax_v.set_title('Linear velocity'); ax_v.legend(loc='best', fontsize=8); ax_v.grid(alpha=0.3)

    # Accel panel
    ax_a = fig.add_subplot(2, 3, 3)
    ax_a.plot(traj['t'], traj['acc_world'][:, 0], label='ax', alpha=0.7)
    ax_a.plot(traj['t'], traj['acc_world'][:, 1], label='ay', alpha=0.7)
    ax_a.plot(traj['t'], traj['acc_world'][:, 2], label='az', alpha=0.7)
    a_mag = np.linalg.norm(traj['acc_world'], axis=1)
    ax_a.plot(traj['t'], a_mag, 'k-', lw=1.2, label='|a|')
    ax_a.axhline(A_MAX, color='r', ls='--', alpha=0.5, label=f'bound ±{A_MAX}')
    ax_a.axhline(-A_MAX, color='r', ls='--', alpha=0.5)
    ax_a.set_xlabel('t (s)'); ax_a.set_ylabel('accel (m/s²)')
    ax_a.set_title('Linear acceleration (world)'); ax_a.legend(loc='best', fontsize=8); ax_a.grid(alpha=0.3)

    # Omega panel
    ax_w = fig.add_subplot(2, 3, 4)
    sl_inner = slice(5, -5)
    ax_w.plot(traj['t'][sl_inner], traj['omega_body'][sl_inner, 0], label='ωx', alpha=0.7)
    ax_w.plot(traj['t'][sl_inner], traj['omega_body'][sl_inner, 1], label='ωy', alpha=0.7)
    ax_w.plot(traj['t'][sl_inner], traj['omega_body'][sl_inner, 2], label='ωz', alpha=0.7)
    w_mag = np.linalg.norm(traj['omega_body'][sl_inner], axis=1)
    ax_w.plot(traj['t'][sl_inner], w_mag, 'k-', lw=1.2, label='|ω|')
    ax_w.axhline(OMEGA_MAX, color='r', ls='--', alpha=0.5, label=f'bound ±{OMEGA_MAX}')
    ax_w.axhline(-OMEGA_MAX, color='r', ls='--', alpha=0.5)
    ax_w.set_xlabel('t (s)'); ax_w.set_ylabel('omega (rad/s)')
    ax_w.set_title('Angular velocity (body)'); ax_w.legend(loc='best', fontsize=8); ax_w.grid(alpha=0.3)

    # RPY panel
    ax_rpy = fig.add_subplot(2, 3, 5)
    ax_rpy.plot(traj['t'], np.rad2deg(traj['roll']), label='roll', alpha=0.8, color='C0')
    ax_rpy.plot(traj['t'], np.rad2deg(traj['pitch']), label='pitch', alpha=0.8, color='C1')
    ax_rpy.axhline(30, color='r', ls='--', alpha=0.5, label='roll/pitch ±30°')
    ax_rpy.axhline(-30, color='r', ls='--', alpha=0.5)
    ax_rpy.set_xlabel('t (s)'); ax_rpy.set_ylabel('roll/pitch (deg)')
    ax_rpy.grid(alpha=0.3)
    ax_rpy_yaw = ax_rpy.twinx()
    ax_rpy_yaw.plot(traj['t'], np.rad2deg(traj['yaw']), label='yaw (raw)', alpha=0.6,
                     color='C2', lw=0.8)
    ax_rpy_yaw.set_ylabel('yaw (deg, unwrapped)', color='C2')
    ax_rpy_yaw.tick_params(axis='y', labelcolor='C2')
    ax_rpy.set_title('Roll / Pitch / Yaw')
    ax_rpy.legend(loc='upper left', fontsize=8)
    ax_rpy_yaw.legend(loc='upper right', fontsize=8)

    # Top-down
    ax_xy = fig.add_subplot(2, 3, 6)
    ax_xy.plot(traj['pos'][:, 0], traj['pos'][:, 1], 'b-', lw=1.0, alpha=0.7)
    ax_xy.scatter(traj['cps_pos'][:, 0], traj['cps_pos'][:, 1], c='r', s=15)
    ax_xy.scatter([traj['cps_pos'][0, 0]], [traj['cps_pos'][0, 1]], c='g', s=80,
                   marker='o', label='start', zorder=5)
    ax_xy.scatter([traj['cps_pos'][-1, 0]], [traj['cps_pos'][-1, 1]], c='m', s=80,
                   marker='X', label='end', zorder=5)
    ax_xy.add_patch(plt.Rectangle((-75, -75), 150, 150, fill=False,
                                    edgecolor='gray', ls=':', label='plane (150x150)'))
    ax_xy.add_patch(plt.Rectangle((-XY_BOUND, -XY_BOUND), 2*XY_BOUND, 2*XY_BOUND,
                                    fill=False, edgecolor='orange', ls='--',
                                    label=f'workspace (±{XY_BOUND})'))
    ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Top-down view'); ax_xy.set_aspect('equal')
    ax_xy.legend(fontsize=8); ax_xy.grid(alpha=0.3)
    pad = max(5, 0.2 * max(np.ptp(traj['pos'][:,0]), np.ptp(traj['pos'][:,1])))
    ax_xy.set_xlim(traj['pos'][:,0].min()-pad, traj['pos'][:,0].max()+pad)
    ax_xy.set_ylim(traj['pos'][:,1].min()-pad, traj['pos'][:,1].max()+pad)

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    print(f"\nSaved plot: {save_path}")


def plot_imu(traj, save_path):
    """
    Compare ground-truth body-frame IMU vs noisy measurements from
    GNSS-INS-Sim. Six panels: accel x/y/z and gyro x/y/z, true vs meas.
    Plus residual histograms.
    """
    fig = plt.figure(figsize=(16, 12))
    t = traj['t']
    a_true = traj['accel_true_body']
    a_meas = traj['accel_meas']
    g_true = traj['gyro_true_body']
    g_meas = traj['gyro_meas']
    err = traj['imu_err']

    axis_labels = ['x', 'y', 'z']
    # Top row: accel per axis
    for k in range(3):
        ax = fig.add_subplot(4, 3, k + 1)
        ax.plot(t, a_meas[:, k], color='C3', alpha=0.5, lw=0.6, label='meas (noisy)')
        ax.plot(t, a_true[:, k], color='C0', lw=1.2, label='true')
        ax.set_xlabel('t (s)'); ax.set_ylabel(f'accel_{axis_labels[k]} (m/s²)')
        ax.set_title(f'Accel body {axis_labels[k]} — true vs meas')
        ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)

    # Second row: gyro per axis
    for k in range(3):
        ax = fig.add_subplot(4, 3, 4 + k)
        ax.plot(t, g_meas[:, k], color='C3', alpha=0.5, lw=0.6, label='meas (noisy)')
        ax.plot(t, g_true[:, k], color='C0', lw=1.2, label='true')
        ax.set_xlabel('t (s)'); ax.set_ylabel(f'gyro_{axis_labels[k]} (rad/s)')
        ax.set_title(f'Gyro body {axis_labels[k]} — true vs meas')
        ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)

    # Third row: residual histograms (accel)
    a_resid = a_meas - a_true
    for k in range(3):
        ax = fig.add_subplot(4, 3, 7 + k)
        ax.hist(a_resid[:, k], bins=60, color='C3', alpha=0.7)
        sigma = a_resid[:, k].std()
        ax.axvline(sigma, color='k', ls='--', alpha=0.6, lw=0.8, label=f'±1σ ({sigma:.4f})')
        ax.axvline(-sigma, color='k', ls='--', alpha=0.6, lw=0.8)
        ax.set_xlabel(f'accel_{axis_labels[k]} residual (m/s²)')
        ax.set_ylabel('count')
        ax.set_title(f'Accel {axis_labels[k]} residual')
        ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)

    # Fourth row: residual histograms (gyro)
    g_resid = g_meas - g_true
    for k in range(3):
        ax = fig.add_subplot(4, 3, 10 + k)
        ax.hist(g_resid[:, k], bins=60, color='C3', alpha=0.7)
        sigma = g_resid[:, k].std()
        ax.axvline(sigma, color='k', ls='--', alpha=0.6, lw=0.8, label=f'±1σ ({sigma:.5f})')
        ax.axvline(-sigma, color='k', ls='--', alpha=0.6, lw=0.8)
        ax.set_xlabel(f'gyro_{axis_labels[k]} residual (rad/s)')
        ax.set_ylabel('count')
        ax.set_title(f'Gyro {axis_labels[k]} residual')
        ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"IMU noise model: {err['profile']} | "
        f"accel vrw={err['accel_vrw'][0]:.4g} m/s²/√Hz, "
        f"b_drift={err['accel_b_drift'][0]:.2g} m/s² | "
        f"gyro arw={np.rad2deg(err['gyro_arw'][0])*60:.3g} °/√hr, "
        f"b_drift={np.rad2deg(err['gyro_b_drift'][0])*3600:.3g} °/hr",
        fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    print(f"Saved IMU plot: {save_path}")


# ============================================================
# DATASET I/O
# ============================================================

# Column orders (used for both CSV and NPZ keys for clarity)
TRAJ_COLUMNS = ['t',
                'x', 'y', 'z',
                'vx', 'vy', 'vz',
                'ax', 'ay', 'az',
                'qw', 'qx', 'qy', 'qz',
                'roll', 'pitch', 'yaw',
                'omega_bx', 'omega_by', 'omega_bz']

IMU_COLUMNS = ['t',
               'accel_x', 'accel_y', 'accel_z',
               'gyro_x', 'gyro_y', 'gyro_z',
               'accel_true_x', 'accel_true_y', 'accel_true_z',
               'gyro_true_x', 'gyro_true_y', 'gyro_true_z']


def stack_trajectory_array(traj):
    """Pack trajectory arrays into (N, 20) numpy array in TRAJ_COLUMNS order."""
    return np.column_stack([
        traj['t'],
        traj['pos'][:, 0], traj['pos'][:, 1], traj['pos'][:, 2],
        traj['vel'][:, 0], traj['vel'][:, 1], traj['vel'][:, 2],
        traj['acc_world'][:, 0], traj['acc_world'][:, 1], traj['acc_world'][:, 2],
        traj['quat_wxyz'][:, 0], traj['quat_wxyz'][:, 1],
        traj['quat_wxyz'][:, 2], traj['quat_wxyz'][:, 3],
        traj['roll'], traj['pitch'], traj['yaw'],
        traj['omega_body'][:, 0], traj['omega_body'][:, 1], traj['omega_body'][:, 2],
    ])


def stack_imu_array(traj):
    """Pack IMU arrays into (N, 13) numpy array in IMU_COLUMNS order."""
    return np.column_stack([
        traj['t'],
        traj['accel_meas'][:, 0], traj['accel_meas'][:, 1], traj['accel_meas'][:, 2],
        traj['gyro_meas'][:, 0], traj['gyro_meas'][:, 1], traj['gyro_meas'][:, 2],
        traj['accel_true_body'][:, 0], traj['accel_true_body'][:, 1], traj['accel_true_body'][:, 2],
        traj['gyro_true_body'][:, 0], traj['gyro_true_body'][:, 1], traj['gyro_true_body'][:, 2],
    ])


def write_csv(path, columns, data):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(data)


def save_trajectory(traj, out_dir):
    """Save one trajectory to disk: CSV + NPZ + meta + plots."""
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Stack arrays
    traj_arr = stack_trajectory_array(traj)
    imu_arr = stack_imu_array(traj)

    # CSV
    write_csv(os.path.join(out_dir, 'trajectory.csv'), TRAJ_COLUMNS, traj_arr)
    write_csv(os.path.join(out_dir, 'imu.csv'), IMU_COLUMNS, imu_arr)

    # NPZ
    np.savez_compressed(
        os.path.join(out_dir, 'trajectory.npz'),
        columns=np.array(TRAJ_COLUMNS),
        data=traj_arr.astype(np.float64),
    )
    np.savez_compressed(
        os.path.join(out_dir, 'imu.npz'),
        columns=np.array(IMU_COLUMNS),
        data=imu_arr.astype(np.float64),
    )

    # Diagnostics for meta
    v_mag = np.linalg.norm(traj['vel'], axis=1)
    a_mag = np.linalg.norm(traj['acc_world'], axis=1)
    w_mag = np.linalg.norm(traj['omega_body'][5:-5], axis=1)
    a_resid = traj['accel_meas'] - traj['accel_true_body']
    g_resid = traj['gyro_meas'] - traj['gyro_true_body']

    meta = {
        # Trajectory config
        'cfg': asdict(traj['cfg']),
        # Fixed config used
        'fixed_config': {
            'xy_bound': XY_BOUND,
            'z_min': Z_MIN, 'z_max': Z_MAX,
            'plane_size': PLANE_SIZE,
            'step_min': STEP_MIN, 'step_max': STEP_MAX,
            'v_max': V_MAX, 'a_max': A_MAX, 'omega_max': OMEGA_MAX,
            'dense_hz': DENSE_HZ, 'out_hz': OUT_HZ,
            'imu_profile': IMU_PROFILE_NAME,
        },
        # Outcome
        'duration': float(traj['duration']),
        'design_duration': float(traj['design_duration']),
        'alpha': float(traj['alpha']),
        'n_samples': int(len(traj['t'])),
        'n_cps': int(len(traj['cps_pos'])),
        'cp_steps': {
            'min': float(np.min(traj['cps_steps'])),
            'max': float(np.max(traj['cps_steps'])),
            'mean': float(np.mean(traj['cps_steps'])),
        },
        'path_length': float(np.linalg.norm(np.diff(traj['cps_pos'], axis=0), axis=1).sum()),
        'peaks': {
            'v_max': float(v_mag.max()),
            'a_max': float(a_mag.max()),
            'omega_max': float(w_mag.max()),
            'roll_max_deg': float(np.abs(np.rad2deg(traj['roll'])).max()),
            'pitch_max_deg': float(np.abs(np.rad2deg(traj['pitch'])).max()),
        },
        'xy_span': {
            'x_min': float(traj['pos'][:, 0].min()),
            'x_max': float(traj['pos'][:, 0].max()),
            'y_min': float(traj['pos'][:, 1].min()),
            'y_max': float(traj['pos'][:, 1].max()),
            'z_min': float(traj['pos'][:, 2].min()),
            'z_max': float(traj['pos'][:, 2].max()),
        },
        'imu_residual_std': {
            'accel': a_resid.std(axis=0).tolist(),
            'gyro': g_resid.std(axis=0).tolist(),
        },
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Plots
    plot_trajectory(traj, os.path.join(plots_dir, 'traj.png'))
    plot_imu(traj, os.path.join(plots_dir, 'imu.png'))
    plt.close('all')


def generate_dataset(out_root, n_train=50, n_val=10, n_test=10, base_seed=1000):
    """Generate the full dataset."""
    splits = [('train', n_train), ('val', n_val), ('test', n_test)]
    rng_master = np.random.default_rng(base_seed)

    dataset_meta = {
        'imu_profile': IMU_PROFILE_NAME,
        'fixed_config': {
            'xy_bound': XY_BOUND, 'z_min': Z_MIN, 'z_max': Z_MAX,
            'plane_size': PLANE_SIZE,
            'step_min': STEP_MIN, 'step_max': STEP_MAX,
            'v_max': V_MAX, 'a_max': A_MAX, 'omega_max': OMEGA_MAX,
            'out_hz': OUT_HZ,
        },
        'splits': {},
    }

    for split, n in splits:
        split_dir = os.path.join(out_root, split)
        os.makedirs(split_dir, exist_ok=True)
        split_seeds = []
        print(f"\n========== {split.upper()} ({n} trajectories) ==========")
        for i in range(n):
            seed = int(rng_master.integers(0, 10**9))
            cfg = sample_traj_config(seed=seed, rng=rng_master)
            traj_dir = os.path.join(split_dir, f'traj_{i:03d}')
            print(f"\n[{split} {i:03d}/{n-1}] seed={seed}, n_cps={cfg.n_cps}, "
                  f"cone={cfg.cone_angle_deg:.1f}°, time_factor={cfg.time_factor:.3f}")
            traj = generate_trajectory(cfg, verbose=True)
            save_trajectory(traj, traj_dir)
            split_seeds.append({
                'index': i,
                'requested_seed': seed,
                'used_seed': traj['cfg'].seed,
                'duration': float(traj['duration']),
            })
        dataset_meta['splits'][split] = {
            'n_trajectories': n,
            'seeds': split_seeds,
        }

    with open(os.path.join(out_root, 'dataset_meta.json'), 'w') as f:
        json.dump(dataset_meta, f, indent=2)
    print(f"\nDataset saved to {out_root}")


# -------------------------- ENTRY --------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'dataset'], default='dataset')
    parser.add_argument('--out', type=str, default=DEFAULT_TRAJ_DATA_ROOT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--n_val', type=int, default=10)
    parser.add_argument('--n_test', type=int, default=10)
    args = parser.parse_args()

    print(f"v4 trajectory generator")
    print(f"Bounds: |v|≤{V_MAX}, |a|≤{A_MAX}, |ω|≤{OMEGA_MAX}")
    print(f"Workspace: xy∈±{XY_BOUND}, z∈[{Z_MIN},{Z_MAX}], plane {PLANE_SIZE}x{PLANE_SIZE}")
    print(f"Step ∈ [{STEP_MIN},{STEP_MAX}] m, IMU: {IMU_PROFILE_NAME}")

    if args.mode == 'single':
        rng = np.random.default_rng(args.seed)
        cfg = sample_traj_config(seed=args.seed, rng=rng)
        print(f"\nSingle trajectory test, cfg: {cfg}\n")
        traj = generate_trajectory(cfg, verbose=True)
        plot_trajectory(traj, 'trajectory/traj_v4.png')
        plot_imu(traj, 'imu/traj_v4_imu.png')
    else:
        generate_dataset(args.out, n_train=args.n_train,
                         n_val=args.n_val, n_test=args.n_test)
