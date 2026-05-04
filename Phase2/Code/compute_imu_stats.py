"""
Compute per-channel mean and std from all train IMU data.

Run after traj_gen_v4.py, before train.py:
    python compute_imu_stats.py --data_root ../static/tmp/phase2_data

Writes 'imu_norm_stats' into dataset_meta.json.
"""

import os
import json
import argparse

import numpy as np
from glob import glob

_PHASE2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATA_ROOT = os.path.join(_PHASE2_ROOT, 'static', 'tmp', 'phase2_data')

IMU_ACCEL = slice(1, 4) # only values with noise 
IMU_GYRO = slice(4, 7) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, 'train')
    folders = sorted(glob(os.path.join(train_dir, 'traj_*')))
    if not folders:
        raise FileNotFoundError(f"No traj_* folders under {train_dir}")

    all_imu = []
    for folder in folders:
        imu = np.load(os.path.join(folder, 'imu.npz'))['data']
        accel = imu[:, IMU_ACCEL]
        gyro = imu[:, IMU_GYRO]
        all_imu.append(np.concatenate([accel, gyro], axis=1))

    all_imu = np.concatenate(all_imu, axis=0).astype(np.float64)  # (N_total, 6)
    mean = all_imu.mean(axis=0)
    std = all_imu.std(axis=0)
    std[std < 1e-8] = 1.0

    channels = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    print(f"Stats from {len(folders)} train trajectories, {len(all_imu):,} samples")
    for ch, m, s in zip(channels, mean, std):
        print(f"  {ch:10s}  mean={m:+.6f}  std={s:.6f}")

    meta_path = os.path.join(args.data_root, 'dataset_meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    meta['imu_norm_stats'] = {
        'channels': channels,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'n_train_trajectories': len(folders),
        'n_train_samples': int(len(all_imu)),
    }

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {meta_path}")


if __name__ == '__main__':
    main()
