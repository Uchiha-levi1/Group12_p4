import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def load_estimate(path):
    ts, xyz = [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts.append(float(parts[0]))
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(ts), np.array(xyz)


def load_euroc_gt(path, offset=0.0):
    ts, xyz = [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            t = float(parts[0]) * 1e-9
            if t < offset:
                continue
            ts.append(t)
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(ts), np.array(xyz)


def associate(t_est, t_gt, max_diff=0.02):
    """For each estimate timestamp, find nearest GT index within max_diff seconds."""
    idx_gt = np.searchsorted(t_gt, t_est)
    idx_gt = np.clip(idx_gt, 1, len(t_gt) - 1)
    left = idx_gt - 1
    right = idx_gt
    pick = np.where(
        np.abs(t_gt[left] - t_est) < np.abs(t_gt[right] - t_est), left, right)
    diffs = np.abs(t_gt[pick] - t_est)
    keep = diffs < max_diff
    return np.where(keep)[0], pick[keep]


def umeyama(src, dst):
    """Return R, t minimizing ||dst - (R src + t)||. Src/dst: (N, 3)."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s0 = src - mu_s
    d0 = dst - mu_d
    H = s0.T @ d0
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R, t


def ate_rmse(est_aligned, gt):
    err = np.linalg.norm(est_aligned - gt, axis=1)
    return float(np.sqrt((err ** 2).mean())), err


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--est', type=str,
        default='../results/trajectory.txt')
    p.add_argument('--gt', type=str,
        default='../Data/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv')
    p.add_argument('--offset', type=float, default=40.0,
        help='GT start offset in seconds (matches dataset.set_starttime).')
    p.add_argument('--plot', type=str, default='../results/ate_plot.png',
        help='Output plot path (png). Pass empty string to show interactively.')
    p.add_argument('--max-diff', type=float, default=0.03)
    args = p.parse_args()

    t_est, xyz_est = load_estimate(args.est)
    t_gt, xyz_gt = load_euroc_gt(args.gt, offset=args.offset)

    i_est, i_gt = associate(t_est, t_gt, max_diff=args.max_diff)
    if len(i_est) < 10:
        raise RuntimeError(f'Too few matches: {len(i_est)}')

    est_m = xyz_est[i_est]
    gt_m = xyz_gt[i_gt]

    R, t = umeyama(est_m, gt_m)
    est_aligned = (R @ xyz_est.T).T + t
    est_m_aligned = est_aligned[i_est]

    rmse, err = ate_rmse(est_m_aligned, gt_m)
    print(f'matches: {len(i_est)}')
    print(f'ATE RMSE: {rmse:.4f} m')
    print(f'ATE mean: {err.mean():.4f} m   median: {np.median(err):.4f} m   max: {err.max():.4f} m')

    fig3d = plt.figure(figsize=(8, 7))
    ax = fig3d.add_subplot(111, projection='3d')
    ax.plot(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2], '-', color='red', label='GT (Vicon)', linewidth=1.5)
    ax.plot(est_aligned[:, 0], est_aligned[:, 1], est_aligned[:, 2], '-', color='blue', label='Estimate (aligned)', linewidth=1.0)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(f'Trajectory 3D  (ATE RMSE = {rmse:.3f} m)')
    ax.legend()
    fig3d.tight_layout()

    fig2d = plt.figure(figsize=(8, 7))
    ax2 = fig2d.add_subplot(111)
    ax2.plot(xyz_gt[:, 0], xyz_gt[:, 1], '-', color='red', label='GT (Vicon)', linewidth=1.5)
    ax2.plot(est_aligned[:, 0], est_aligned[:, 1], '-', color='blue', label='Estimate (aligned)', linewidth=1.0)
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('Top-down (x-y)')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend()
    fig2d.tight_layout()

    if args.plot:
        base, ext = args.plot.rsplit('.', 1) if '.' in args.plot else (args.plot, 'png')
        path_3d = f'{base}_3d.{ext}'
        path_xy = f'{base}_xy.{ext}'
        fig3d.savefig(path_3d, dpi=120)
        fig2d.savefig(path_xy, dpi=120)
        print(f'saved plots: {path_3d}, {path_xy}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
