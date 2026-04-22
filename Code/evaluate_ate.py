import numpy as np
import argparse
import os

###
def load_estimates(path):
    """Load VIO estimates: timestamp tx ty tz qx qy qz qw"""
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            vals = [float(x) for x in line.strip().split()]
            data.append(vals)
    data = np.array(data)
    timestamps = data[:, 0]
    positions  = data[:, 1:4]
    return timestamps, positions


def load_groundtruth(path):
    """
    Load EuRoC ground truth CSV.
    Columns: timestamp(ns), px, py, pz, qw, qx, qy, qz, ...
    Returns timestamps in seconds and positions.
    """
    data = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            vals = [float(x) for x in line.strip().split(',')]
            data.append(vals)
    data = np.array(data)
    timestamps = data[:, 0] * 1e-9   # ns → s
    positions  = data[:, 1:4]
    return timestamps, positions


def associate(ts_est, ts_gt, max_diff=0.02):
    """
    Nearest-neighbour timestamp matching.
    Returns indices (idx_est, idx_gt) of matched pairs within max_diff seconds.
    """
    idx_est, idx_gt = [], []
    j = 0
    for i, t in enumerate(ts_est):
        # advance gt pointer to nearest
        while j + 1 < len(ts_gt) and abs(ts_gt[j+1] - t) < abs(ts_gt[j] - t):
            j += 1
        if abs(ts_gt[j] - t) < max_diff:
            idx_est.append(i)
            idx_gt.append(j)
    return np.array(idx_est), np.array(idx_gt)


def align_se3(p_est, p_gt):
    """
    Umeyama SE(3) alignment (scale fixed to 1).
    Finds R, t minimising sum ||R @ p_est[i] + t - p_gt[i]||^2.
    Returns R (3x3), t (3,), and aligned positions.
    """
    mu_est = p_est.mean(axis=0)
    mu_gt  = p_gt.mean(axis=0)

    p_est_c = p_est - mu_est
    p_gt_c  = p_gt  - mu_gt

    n = len(p_est)
    C = (p_gt_c.T @ p_est_c) / n

    U, D, Vt = np.linalg.svd(C)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    t = mu_gt - R @ mu_est

    p_aligned = (R @ p_est.T).T + t
    return R, t, p_aligned


def compute_rmse_ate(p_aligned, p_gt):
    errors = np.linalg.norm(p_gt - p_aligned, axis=1)
    rmse   = np.sqrt(np.mean(errors ** 2))
    return rmse, errors


def plot_trajectory(p_est_aligned, p_gt, errors, rmse):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        fig = plt.figure(figsize=(14, 5))

        # 3D trajectory
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(*p_gt.T,          'b-',  linewidth=1, label='Ground Truth')
        ax1.plot(*p_est_aligned.T, 'r--', linewidth=1, label='Estimated (aligned)')
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
        ax1.set_title(f'Trajectory  RMSE ATE = {rmse:.4f} m')
        ax1.legend()

        # Per-frame error
        ax2 = fig.add_subplot(122)
        ax2.plot(errors, 'g-', linewidth=1)
        ax2.axhline(rmse, color='r', linestyle='--', label=f'RMSE = {rmse:.4f} m')
        ax2.set_xlabel('Frame'); ax2.set_ylabel('ATE (m)')
        ax2.set_title('Per-frame Absolute Trajectory Error')
        ax2.legend()

        plt.tight_layout()
        out = '../results/ate_plot.png'
        plt.savefig(out, dpi=150)
        print(f'Plot saved to {out}')
        plt.show()
    except ImportError:
        print('matplotlib not available — skipping plot.')
###


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--est',  type=str,
        default='../results/stamped_traj_estimate.txt',
        help='Path to VIO estimate file.')
    parser.add_argument('--gt',   type=str,
        default='../Data/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv',
        help='Path to EuRoC ground truth CSV.')
    parser.add_argument('--max_diff', type=float, default=0.02,
        help='Max timestamp difference (s) for association.')
    args = parser.parse_args()

    ###
    ts_est, p_est = load_estimates(args.est)
    ts_gt,  p_gt  = load_groundtruth(args.gt)

    idx_est, idx_gt = associate(ts_est, ts_gt, args.max_diff)
    print(f'Matched {len(idx_est)} poses (est={len(ts_est)}, gt={len(ts_gt)})')

    p_est_matched = p_est[idx_est]
    p_gt_matched  = p_gt[idx_gt]

    R, t, p_aligned = align_se3(p_est_matched, p_gt_matched)
    rmse, errors    = compute_rmse_ate(p_aligned, p_gt_matched)

    print(f'\nSE(3) alignment:')
    print(f'  R =\n{R}')
    print(f'  t = {t}')
    print(f'\nRMSE ATE : {rmse:.4f} m')
    print(f'Mean  ATE: {np.mean(errors):.4f} m')
    print(f'Max   ATE: {np.max(errors):.4f} m')
    print(f'Min   ATE: {np.min(errors):.4f} m')

    plot_trajectory(p_aligned, p_gt_matched, errors, rmse)
    ###
