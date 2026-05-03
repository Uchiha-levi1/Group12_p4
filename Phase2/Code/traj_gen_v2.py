"""
Trajectory generator v2 for Deep VIO project.

Algorithm (geometry first, dynamics via time-rescale):
  1. Sample CP[0] in inner box.
  2. Sample CP[1] = CP[0] + step·d, step in [2,5]m, d uniform on 3D sphere.
     Initial direction d0 = (CP[1]-CP[0])/|.|.
  3. For i in 2..N-1: sample d_i uniform on sphere, accept if d_i · d_{i-1} >= 0
     (90° cone) AND CP[i-1]+step·d_i is inside workspace. Step in [2,5]m.
     If 100 retries exhausted -> backtrack and re-extend.
  4. Yaw CPs: 1D random walk. Step magnitude in [5°,30°]. Sign 70% same as
     previous, 30% flipped.
  5. Min-snap fit via library:
     - Position: all CPs as waypoints, vel=accel=0 at endpoints, interior free.
     - Yaw: same structure in 1D.
  6. Eval at 1000 Hz. Compute v_peak, a_peak, omega_peak.
  7. Time-rescale by α = max(v_peak/V_MAX, sqrt(a_peak/A_MAX), ω_peak/Ω_MAX, 1).
  8. Re-evaluate. Differential flatness for roll/pitch from accel+yaw.
  9. Quaternion + body-frame omega.
  10. Downsample 1000 Hz -> 100 Hz output.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import minsnap_trajectories as ms
from scipy.spatial.transform import Rotation as R

# -------------------------- CONFIG --------------------------
SEED = 41
N_CPS = 20

# Workspace (plane 150x150, centered at origin; margin 20m -> ±55)
XY_BOUND = 55.0
Z_MIN, Z_MAX = 5.0, 10.0
Z_CLAMP_MARGIN = 1.0    # CP z stays within [Z_MIN+1, Z_MAX-1]

# CP step
STEP_MIN, STEP_MAX = 2.0, 4.0
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


# -------------------------- ENTRY --------------------------

if __name__ == '__main__':
    print(f"v2 trajectory: N_CPS={N_CPS}, step∈[{STEP_MIN},{STEP_MAX}]m, "
          f"heading cone ±90°")
    print(f"Bounds: |v|≤{V_MAX}, |a|≤{A_MAX}, |ω|≤{OMEGA_MAX}")
    print(f"Workspace: xy∈±{XY_BOUND}, z∈[{Z_MIN},{Z_MAX}], plane 150x150\n")

    traj = generate_trajectory(seed=SEED, verbose=True)
    plot_trajectory(traj, '../traj_v21.png')
