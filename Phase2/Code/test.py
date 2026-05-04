"""
Test the best checkpoint on the test split.

Loads checkpoint, runs dead-reckoning on every test trajectory,
plots predicted vs GT (raw and SE(3)-aligned), writes per-trajectory
and aggregate ATE to a JSON + CSV.

Usage:
    python test.py --data_root ./phase2_data --run_dir ./io_lstm_run
    python test.py --data_root ./phase2_data --run_dir ./io_lstm_run \
                   --ckpt ./io_lstm_run/checkpoints/best.pt
"""

import os
import json
import argparse
from glob import glob

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from model import BiLSTM_IO
from eval import evaluate_trajectory


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_trajectory(result: dict, out_path: str, title: str = ''):
    """Plot 3D + per-axis comparison of predicted vs GT trajectory.

    Saves a single figure with 4 panels:
        - 3D view (raw pred vs GT)
        - 3D view (aligned pred vs GT)
        - XY top-down (aligned)
        - per-axis traces (aligned)
    """
    pred_raw = result['pred_positions']
    pred_aln = result['aligned_pred_positions']
    gt = result['gt_positions']
    rmse = result['rmse_ate']

    fig = plt.figure(figsize=(14, 10))

    # 3D raw
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', label='GT', linewidth=1.5)
    ax1.plot(pred_raw[:, 0], pred_raw[:, 1], pred_raw[:, 2],
             'r--', label='Pred (raw)', linewidth=1.0)
    ax1.scatter(gt[0, 0], gt[0, 1], gt[0, 2], c='g', s=40, label='start', zorder=5)
    ax1.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2], c='b', s=40, label='end', zorder=5)
    ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]'); ax1.set_zlabel('z [m]')
    ax1.set_title('3D — raw prediction')
    ax1.legend(loc='best', fontsize=8)

    # 3D aligned
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', label='GT', linewidth=1.5)
    ax2.plot(pred_aln[:, 0], pred_aln[:, 1], pred_aln[:, 2],
             'r--', label='Pred (SE(3) aligned)', linewidth=1.0)
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]'); ax2.set_zlabel('z [m]')
    ax2.set_title(f'3D — aligned (RMSE ATE = {rmse:.3f} m)')
    ax2.legend(loc='best', fontsize=8)

    # XY top-down (aligned)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(gt[:, 0], gt[:, 1], 'k-', label='GT', linewidth=1.5)
    ax3.plot(pred_aln[:, 0], pred_aln[:, 1], 'r--', label='Pred (aligned)', linewidth=1.0)
    ax3.scatter(gt[0, 0], gt[0, 1], c='g', s=40, label='start', zorder=5)
    ax3.scatter(gt[-1, 0], gt[-1, 1], c='b', s=40, label='end', zorder=5)
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]')
    ax3.set_title('Top-down XY (aligned)')
    ax3.set_aspect('equal', adjustable='datalim')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8)

    # Per-axis (aligned)
    ax4 = fig.add_subplot(2, 2, 4)
    idx = np.arange(len(gt))
    ax4.plot(idx, gt[:, 0], 'k-', label='gt x', linewidth=1.0)
    ax4.plot(idx, gt[:, 1], 'k--', label='gt y', linewidth=1.0)
    ax4.plot(idx, gt[:, 2], 'k:', label='gt z', linewidth=1.0)
    ax4.plot(idx, pred_aln[:, 0], 'r-', label='pred x', linewidth=1.0, alpha=0.8)
    ax4.plot(idx, pred_aln[:, 1], 'g-', label='pred y', linewidth=1.0, alpha=0.8)
    ax4.plot(idx, pred_aln[:, 2], 'b-', label='pred z', linewidth=1.0, alpha=0.8)
    ax4.set_xlabel('window index')
    ax4.set_ylabel('position [m]')
    ax4.set_title('Per-axis position (aligned)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=7, ncol=2)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_error_growth(result: dict, out_path: str, title: str = ''):
    """Plot per-window position error along the trajectory (aligned)."""
    pred_aln = result['aligned_pred_positions']
    gt = result['gt_positions']
    err = np.linalg.norm(pred_aln - gt, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(err, 'r-', linewidth=1.2)
    ax.set_xlabel('window index')
    ax.set_ylabel('position error (aligned) [m]')
    ax.set_title(title or 'Drift growth along trajectory')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_summary_bar(per_traj: list, out_path: str):
    """Bar chart of ATE per test trajectory + aggregate stats."""
    names = [os.path.basename(p['folder']) for p in per_traj]
    ates = [p['rmse_ate'] for p in per_traj]
    mean_ate = float(np.mean(ates))
    median_ate = float(np.median(ates))

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(names) + 4), 4))
    bars = ax.bar(range(len(names)), ates, color='steelblue', edgecolor='k')
    ax.axhline(mean_ate, color='r', linestyle='--', label=f'mean = {mean_ate:.2f} m')
    ax.axhline(median_ate, color='g', linestyle=':', label=f'median = {median_ate:.2f} m')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('RMSE ATE [m]')
    ax.set_title('Per-trajectory ATE on test split')
    for bar, val in zip(bars, ates):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./phase2_data')
    parser.add_argument('--run_dir', type=str, default='./io_lstm_run')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint path; defaults to <run_dir>/checkpoints/best.pt')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val', 'train'])
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--out_subdir', type=str, default='test_results')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load IMU normalization stats
    meta_path = os.path.join(args.data_root, 'dataset_meta.json')
    with open(meta_path, 'r') as f:
        dataset_meta = json.load(f)
    if 'imu_norm_stats' not in dataset_meta:
        raise RuntimeError(
            f"'imu_norm_stats' missing from {meta_path}. "
            "Run: python compute_imu_stats.py --data_root " + args.data_root
        )
    imu_mean = np.array(dataset_meta['imu_norm_stats']['mean'], dtype=np.float32)
    imu_std = np.array(dataset_meta['imu_norm_stats']['std'], dtype=np.float32)
    print(f"IMU norm stats loaded from {meta_path}")

    # Resolve checkpoint
    ckpt_path = args.ckpt or os.path.join(args.run_dir, 'checkpoints', 'best.pt')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    epoch = ckpt.get('epoch', '?')
    saved_val_ate = ckpt.get('val_mean_ate', None)
    print(f"Checkpoint epoch={epoch}, val_mean_ate={saved_val_ate}")

    # Build model and load weights
    model = BiLSTM_IO().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Output dir
    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Iterate over split
    split_dir = os.path.join(args.data_root, args.split)
    folders = sorted(glob(os.path.join(split_dir, 'traj_*')))
    if not folders:
        raise FileNotFoundError(f"No trajectories under {split_dir}")
    print(f"Evaluating {len(folders)} trajectories from {split_dir}")

    per_traj = []
    for folder in folders:
        result = evaluate_trajectory(
            model, folder, device,
            window_size=args.window, stride=args.stride,
            imu_mean=imu_mean, imu_std=imu_std,
        )
        name = os.path.basename(folder)
        ate = result['rmse_ate']
        n_steps = len(result['gt_positions'])
        print(f"  {name}: ATE = {ate:.4f} m  ({n_steps} dead-reckoning steps)")

        # Plot
        plot_trajectory(
            result,
            os.path.join(plots_dir, f'{name}_trajectory.png'),
            title=f'{name}  —  RMSE ATE = {ate:.3f} m',
        )
        plot_error_growth(
            result,
            os.path.join(plots_dir, f'{name}_error.png'),
            title=f'{name} — error growth',
        )

        per_traj.append({
            'folder': folder,
            'name': name,
            'rmse_ate': ate,
            'n_steps': int(n_steps),
            'pred_positions': result['pred_positions'],
            'aligned_pred_positions': result['aligned_pred_positions'],
            'gt_positions': result['gt_positions'],
        })

    # Aggregate
    ates = np.array([p['rmse_ate'] for p in per_traj])
    summary = {
        'checkpoint': ckpt_path,
        'epoch': int(epoch) if isinstance(epoch, int) else epoch,
        'split': args.split,
        'n_trajectories': int(len(per_traj)),
        'mean_ate': float(ates.mean()),
        'median_ate': float(np.median(ates)),
        'std_ate': float(ates.std()),
        'min_ate': float(ates.min()),
        'max_ate': float(ates.max()),
        'per_trajectory': [
            {'name': p['name'], 'rmse_ate': p['rmse_ate'], 'n_steps': p['n_steps']}
            for p in per_traj
        ],
    }

    print("\nAggregate ATE on", args.split, "split:")
    print(f"  mean   = {summary['mean_ate']:.4f} m")
    print(f"  median = {summary['median_ate']:.4f} m")
    print(f"  std    = {summary['std_ate']:.4f} m")
    print(f"  min    = {summary['min_ate']:.4f} m")
    print(f"  max    = {summary['max_ate']:.4f} m")

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, 'per_trajectory.csv'), 'w') as f:
        f.write('name,rmse_ate_m,n_steps\n')
        for p in per_traj:
            f.write(f"{p['name']},{p['rmse_ate']},{p['n_steps']}\n")

    plot_summary_bar(per_traj, os.path.join(out_dir, 'ate_summary.png'))

    # Save raw arrays for any post-hoc analysis
    np_save_path = os.path.join(out_dir, 'predictions.npz')
    np.savez(
        np_save_path,
        names=np.array([p['name'] for p in per_traj]),
        ates=ates,
        **{f'{p["name"]}_pred_raw': p['pred_positions'] for p in per_traj},
        **{f'{p["name"]}_pred_aligned': p['aligned_pred_positions'] for p in per_traj},
        **{f'{p["name"]}_gt': p['gt_positions'] for p in per_traj},
    )
    print(f"\nSaved: {os.path.join(out_dir, 'summary.json')}")
    print(f"Saved: {np_save_path}")
    print(f"Saved: {plots_dir}/")


if __name__ == '__main__':
    main()
