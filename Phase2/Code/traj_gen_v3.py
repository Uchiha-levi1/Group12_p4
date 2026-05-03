"""
Trajectory generator v3 for Deep VIO project.

Adds: IMU noise model from GNSS-INS-Sim applied to ground truth.

Pipeline (unchanged from v2):
  1-10. Geometry-first sampling, min-snap fit, time-rescale, differential flatness
        for orientation, body-frame omega, downsample to 100 Hz.

New in v3:
  11. Compute true accelerometer specific force in body frame:
        f_body = R_wb^T · (acc_world - g)
      Compute true gyro: gyro_body = omega_body (already body-frame).
  12. Apply gnss_ins_sim noise model (acc_gen, gyro_gen) using a chosen
      IMU accuracy preset. Noise = constant bias + Gauss-Markov drift + white.
  13. Output adds: accel_meas, gyro_meas (the noisy IMU readings the net trains on).
  14. Extra plot: ground-truth IMU vs noisy IMU, residual histograms.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import minsnap_trajectories as ms
from scipy.spatial.transform import Rotation as R

# GNSS-INS-Sim noise model (applies bias + Gauss-Markov drift + white noise)
from gnss_ins_sim.pathgen.pathgen import acc_gen, gyro_gen
from gnss_ins_sim.sim.imu_model import IMU as GnssIMU

# -------------------------- CONFIG --------------------------
SEED = 42
N_CPS = 20

# Workspace (plane 150x150, centered at origin; margin 20m -> ±55)
XY_BOUND = 55.0
Z_MIN, Z_MAX = 5.0, 10.0
Z_CLAMP_MARGIN = 1.0    # CP z stays within [Z_MIN+1, Z_MAX-1]

# CP step
STEP_MIN, STEP_MAX = 2.0, 5.0
HEADING_COS_THRESHOLD = 0.0   # cos(90°) = 0; dot >= 0 means within 90°

# Yaw walk
YAW_STEP_MIN_DEG, YAW_STEP_MAX_DEG = 5.0, 30.0
YAW_SAME_SIGN_PROB = 0.7

# Bounds (conservative)
V_MAX = 3.0
A_MAX = 4.0
OMEGA_MAX = 1.0

# Sampling rates
DENSE_HZ = 1000
OUT_HZ = 100

# Design duration before rescale (drives initial vel/accel via min-snap)
T_DESIGN_PER_CP = 2.0   # seconds per segment as initial guess
T_DESIGN = T_DESIGN_PER_CP * (N_CPS - 1)

# Backtrack
N_RETRY_PER_CP = 100
N_BACKTRACK_BUDGET = 50

# Gravity
G = np.array([0.0, 0.0, -9.81])

# IMU noise model
# Profile: vibration-affected MPU-6050 (drone with prop noise dominating white noise channel)
# Per-sample σ at fs=100 Hz: accel 0.15 m/s², gyro 0.008 rad/s
# Bias drifts: accel 0.1 mg, gyro 20°/hr (chip-level, unaffected by vibration)
IMU_PROFILE_NAME = 'vibration_mpu6050'
IMU_CUSTOM_ERR = {
    'accel': {
        'b':       np.array([0.0, 0.0, 0.0]),                 # constant bias, m/s²
        'b_drift': np.array([9.81e-4, 9.81e-4, 9.81e-4]),     # 0.1 mg, m/s²
        'b_corr':  np.array([100.0, 100.0, 100.0]),           # correlation time, s
        'vrw':     np.array([0.015, 0.015, 0.015]),           # m/s/√Hz; σ_per_sample=vrw·√fs=0.15
    },
    'gyro': {
        'b':       np.array([0.0, 0.0, 0.0]),                 # constant bias, rad/s
        'b_drift': np.array([9.696e-5, 9.696e-5, 9.696e-5]),  # 20°/hr, rad/s
        'b_corr':  np.array([100.0, 100.0, 100.0]),
        'arw':     np.array([0.0008, 0.0008, 0.0008]),        # rad/s/√Hz; σ_per_sample=arw·√fs=0.008
    },
}

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
    """Sample CP[0] in inner box, CP[1] = CP[0] + step*d. Returns p0, p1, d_init."""
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
                return p0, p1, d
        # Couldn't extend from this p0, resample p0


def propose_next_cp(p_prev, d_prev):
    """Try one proposal of next CP via heading-cone sampling."""
    d = sample_unit_vector_3d()
    if np.dot(d, d_prev) < HEADING_COS_THRESHOLD:
        return None, None
    step = np.random.uniform(STEP_MIN, STEP_MAX)
    p_new = p_prev + step * d
    if not in_workspace(p_new):
        return None, None
    return p_new, d


def generate_position_cps():
    """Generate N_CPS position control points with backtracking."""
    backtracks_left = N_BACKTRACK_BUDGET
    while True:
        # Initialize
        p0, p1, d_init = sample_initial_cps()
        cps = [p0, p1]
        dirs = [d_init]   # dirs[i] = direction CP[i] -> CP[i+1]; len = len(cps)-1
        i = 2

        while i < N_CPS:
            success = False
            for _ in range(N_RETRY_PER_CP):
                p_new, d_new = propose_next_cp(cps[-1], dirs[-1])
                if p_new is not None:
                    cps.append(p_new)
                    dirs.append(d_new)
                    i += 1
                    success = True
                    break

            if not success:
                # Backtrack
                if backtracks_left <= 0 or len(cps) <= 2:
                    # Hard restart
                    break
                cps.pop()
                dirs.pop()
                i -= 1
                backtracks_left -= 1

        if len(cps) == N_CPS:
            return np.array(cps)
        # else: hard restart from sample_initial_cps


def generate_yaw_cps(n):
    """Generate n yaw waypoints via 1D random walk with smooth-turn bias."""
    yaws = [np.random.uniform(-np.pi, np.pi)]
    last_sign = np.random.choice([-1, 1])
    for _ in range(n - 1):
        mag = np.deg2rad(np.random.uniform(YAW_STEP_MIN_DEG, YAW_STEP_MAX_DEG))
        if np.random.random() < YAW_SAME_SIGN_PROB:
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

def generate_trajectory(seed=SEED, verbose=True):
    np.random.seed(seed)

    # Step 1: sample CPs
    pos_cps = generate_position_cps()
    yaw_cps = generate_yaw_cps(N_CPS)
    times = np.linspace(0, T_DESIGN, N_CPS)

    if verbose:
        cp_dists = np.linalg.norm(np.diff(pos_cps, axis=0), axis=1)
        print(f"CPs sampled: {N_CPS}")
        print(f"  CP step: min={cp_dists.min():.2f}, max={cp_dists.max():.2f}, "
              f"mean={cp_dists.mean():.2f} m")
        print(f"  Total path length (CPs): {cp_dists.sum():.1f} m")
        print(f"  Design duration: {T_DESIGN:.1f} s")

    # Step 2: min-snap fit
    polys_xyz = fit_minsnap_position(pos_cps, times)
    polys_yaw = fit_minsnap_yaw(yaw_cps, times)

    # Step 3: dense eval
    n_dense = int(round(T_DESIGN * DENSE_HZ)) + 1
    t_dense = np.linspace(0, T_DESIGN, n_dense)
    derivs_xyz = eval_polys(polys_xyz, t_dense, order=4)  # (5, N, 3)
    pos = derivs_xyz[0]
    vel = derivs_xyz[1]
    acc = derivs_xyz[2]

    derivs_yaw = eval_polys(polys_yaw, t_dense, order=2)  # (3, N, 1)
    yaw = derivs_yaw[0][:, 0]

    # Step 4: peaks BEFORE rescale
    v_peak = np.linalg.norm(vel, axis=1).max()
    a_peak = np.linalg.norm(acc, axis=1).max()

    # Build provisional orientation to estimate omega for rescale
    roll_pre, pitch_pre, _ = differential_flatness_rpy(acc, yaw)
    quat_pre = rpy_to_quat(roll_pre, pitch_pre, yaw)
    omega_pre = quat_to_omega_body(quat_pre, t_dense)
    # Trim ends for gradient noise
    omega_peak = np.linalg.norm(omega_pre[5:-5], axis=1).max()

    # Step 5: time-rescale
    alpha_v = v_peak / V_MAX
    alpha_a = np.sqrt(a_peak / A_MAX)
    alpha_w = omega_peak / OMEGA_MAX
    alpha = max(alpha_v, alpha_a, alpha_w, 1.0)
    if verbose:
        print(f"  Pre-rescale peaks: |v|={v_peak:.2f}, |a|={a_peak:.2f}, "
              f"|ω|={omega_peak:.2f}")
        print(f"  Rescale factors: α_v={alpha_v:.2f}, α_a={alpha_a:.2f}, "
              f"α_ω={alpha_w:.2f} -> α={alpha:.2f}")

    # Apply rescale: physical time t_phys = α * t_design.
    # New positions: p(t_phys) = p_design(t_phys / α).
    # Need to evaluate at design-time samples corresponding to physical-time grid.
    T_final = alpha * T_DESIGN
    n_final = int(round(T_final * DENSE_HZ)) + 1
    t_phys = np.linspace(0, T_final, n_final)
    t_design_samples = t_phys / alpha

    derivs_xyz_final = eval_polys(polys_xyz, t_design_samples, order=4)
    pos = derivs_xyz_final[0]
    # Velocity: dp/dt_phys = (dp/dt_design) * (1/α)
    vel = derivs_xyz_final[1] / alpha
    # Acceleration: d²p/dt_phys² = (d²p/dt_design²) * (1/α²)
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
        # Noisy IMU measurements (what the network sees)
        'accel_meas': accel_meas,
        'gyro_meas': gyro_meas,
        'imu_err': imu_err,
        'cps_pos': pos_cps,
        'cps_yaw': yaw_cps,
        'cps_time': alpha * times,
        'alpha': alpha,
        'duration': T_final,
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
        # IMU diagnostics
        a_resid = accel_meas - accel_true_body
        g_resid = gyro_meas - gyro_true_body
        print(f"\nIMU ({IMU_PROFILE_NAME}):")
        print(f"  accel true |body| range: [{np.linalg.norm(accel_true_body, axis=1).min():.2f}, "
              f"{np.linalg.norm(accel_true_body, axis=1).max():.2f}] m/s²")
        print(f"  accel residual std (per axis): {a_resid.std(axis=0).round(4)}")
        print(f"  gyro residual std (per axis): {g_resid.std(axis=0).round(5)} rad/s")
        # Sanity: at hover, accel z should be ~+9.81
        print(f"  accel_true_body z mean: {accel_true_body[:,2].mean():.2f} (≈9.81 if mostly hovering)")

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
    ax3d.set_title(f'3D trajectory (N={N_CPS} CPs, T={traj["duration"]:.0f}s, α={traj["alpha"]:.2f})')
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


# -------------------------- ENTRY --------------------------

if __name__ == '__main__':
    print(f"v3 trajectory: N_CPS={N_CPS}, step∈[{STEP_MIN},{STEP_MAX}]m, "
          f"heading cone ±90°")
    print(f"Bounds: |v|≤{V_MAX}, |a|≤{A_MAX}, |ω|≤{OMEGA_MAX}")
    print(f"Workspace: xy∈±{XY_BOUND}, z∈[{Z_MIN},{Z_MAX}], plane 150x150")
    print(f"IMU model: {IMU_PROFILE_NAME}\n")

    traj = generate_trajectory(seed=SEED, verbose=True)
    plot_trajectory(traj, 'trajectory/traj_v3.png')
    plot_imu(traj, 'imu/traj_v3_imu.png')
