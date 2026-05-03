"""
Evaluation: dead-reckoning + ATE with SE(3) (Umeyama) alignment.

Given a model and a trajectory, slide non-overlapping windows
(stride=window_size), predict (Δp, ΔR) per window, chain to a full
trajectory, align to GT, compute RMSE ATE.

Also provides an oracle eval (uses GT relative poses) for sanity
checking the dead-reckoning math.
"""

import numpy as np
import torch

from dataset import load_trajectory_for_eval, quat_wxyz_to_R, compute_relative_pose
from losses import gram_schmidt_6d_to_R


def dead_reckon(start_pos: np.ndarray, start_R: np.ndarray,
                deltas_p_body: np.ndarray, deltas_R_body: np.ndarray):
    """Chain body-frame relative poses into world-frame trajectory.

    Args:
        start_pos: (3,)
        start_R:   (3, 3) initial body->world rotation
        deltas_p_body: (M, 3)
        deltas_R_body: (M, 3, 3)

    Returns:
        positions:  (M+1, 3) including start
        rotations:  (M+1, 3, 3)
    """
    M = len(deltas_p_body)
    positions = np.zeros((M + 1, 3), dtype=np.float64)
    rotations = np.zeros((M + 1, 3, 3), dtype=np.float64)

    positions[0] = start_pos
    rotations[0] = start_R

    p = start_pos.copy()
    Rcurr = start_R.copy()
    for k in range(M):
        # delta_p is in body frame at sample k -> rotate to world before adding
        p = p + Rcurr @ deltas_p_body[k]
        Rcurr = Rcurr @ deltas_R_body[k]
        positions[k + 1] = p
        rotations[k + 1] = Rcurr

    return positions, rotations


def umeyama_alignment(pred: np.ndarray, gt: np.ndarray, with_scale: bool = False):
    """SE(3) alignment of predicted trajectory to GT (Umeyama 1991).

    Args:
        pred: (N, 3)
        gt:   (N, 3)
        with_scale: if True compute Sim(3) (scale + rot + trans), else SE(3).

    Returns:
        s, R, t such that aligned = s * R @ pred.T + t (column convention)
    """
    assert pred.shape == gt.shape and pred.shape[1] == 3
    N = pred.shape[0]

    mu_p = pred.mean(axis=0)
    mu_g = gt.mean(axis=0)
    p_c = pred - mu_p
    g_c = gt - mu_g

    # Cross-covariance
    H = (p_c.T @ g_c) / N

    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    Rmat = Vt.T @ S @ U.T

    if with_scale:
        var_p = (p_c ** 2).sum() / N
        s = (D * np.diag(S)).sum() / var_p
    else:
        s = 1.0

    t = mu_g - s * Rmat @ mu_p
    return s, Rmat, t


def apply_alignment(pred: np.ndarray, s: float, Rmat: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (Rmat @ pred.T)).T + t


def compute_ate_rmse(pred: np.ndarray, gt: np.ndarray, with_scale: bool = False):
    """Align pred -> gt with Umeyama, return RMSE ATE and aligned pred."""
    s, Rmat, t = umeyama_alignment(pred, gt, with_scale=with_scale)
    aligned = apply_alignment(pred, s, Rmat, t)
    err = np.linalg.norm(aligned - gt, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    return rmse, aligned, (s, Rmat, t)


@torch.no_grad()
def predict_windows_in_order(model, windows_imu: np.ndarray, device, batch_size: int = 256):
    """Run model on windows in order. windows_imu: (M, W, 6). Returns deltas."""
    model.eval()
    deltas_p = []
    deltas_R6 = []
    M = len(windows_imu)
    for i in range(0, M, batch_size):
        batch = torch.from_numpy(windows_imu[i:i + batch_size]).to(device)
        dp, d6 = model(batch)
        deltas_p.append(dp.cpu().numpy())
        deltas_R6.append(d6.cpu().numpy())
    deltas_p = np.concatenate(deltas_p, axis=0)
    deltas_R6 = np.concatenate(deltas_R6, axis=0)

    # Convert 6D rep -> 3x3 R via the same Gram-Schmidt (numpy version)
    deltas_R = np.zeros((M, 3, 3), dtype=np.float64)
    for k in range(M):
        d6 = deltas_R6[k]
        a1 = d6[:3]
        a2 = d6[3:]
        b1 = a1 / max(np.linalg.norm(a1), 1e-8)
        proj = (b1 @ a2) * b1
        b2 = a2 - proj
        b2 = b2 / max(np.linalg.norm(b2), 1e-8)
        b3 = np.cross(b1, b2)
        deltas_R[k] = np.stack([b1, b2, b3], axis=1)

    return deltas_p.astype(np.float64), deltas_R


def evaluate_trajectory(model, traj_folder: str, device,
                        window_size: int = 100, stride: int = 100):
    """Run model on a trajectory, dead-reckon, compute ATE.

    Returns dict with rmse_ate, predicted positions, GT positions, etc.
    """
    data = load_trajectory_for_eval(traj_folder, window_size=window_size, stride=stride)

    deltas_p_body, deltas_R_body = predict_windows_in_order(
        model, data['windows_imu'], device,
    )

    # Initial pose from GT at first window's start
    start_idx = int(data['window_starts'][0])
    start_pos = data['gt_pos'][start_idx]
    start_R = quat_wxyz_to_R(data['gt_quat_wxyz'][start_idx])

    pred_positions, pred_rotations = dead_reckon(
        start_pos, start_R, deltas_p_body, deltas_R_body,
    )

    # Build matched GT positions at the same sample indices the predictions
    # correspond to. Window k starts at window_starts[k] and ends at
    # window_starts[k] + W - 1. For non-overlapping stride==W, end of window k
    # equals start of window k+1.
    matched_indices = list(data['window_starts'])
    matched_indices.append(int(data['window_starts'][-1]) + window_size - 1)
    matched_indices = np.array(matched_indices)

    gt_matched = data['gt_pos'][matched_indices]
    rmse, aligned_pred, transform = compute_ate_rmse(
        pred_positions, gt_matched, with_scale=False,
    )

    return {
        'rmse_ate': rmse,
        'pred_positions': pred_positions,    # raw (unaligned)
        'aligned_pred_positions': aligned_pred,
        'gt_positions': gt_matched,
        'transform': transform,              # (s, R, t)
        'folder': traj_folder,
    }


def evaluate_dataset(model, split_dir: str, device,
                     window_size: int = 100, stride: int = 100):
    """Evaluate all trajectories under split_dir. Returns list of result dicts."""
    from glob import glob
    import os
    folders = sorted(glob(os.path.join(split_dir, 'traj_*')))
    return [evaluate_trajectory(model, f, device, window_size, stride) for f in folders]


# ---------------------- ORACLE (sanity check) ----------------------

def oracle_dead_reckon(traj_folder: str, window_size: int = 100, stride: int = 100):
    """Use ground-truth relative poses (no network) to dead-reckon.

    Output should match GT positions to within numerical precision.
    Use this on Day 1 to verify the dead-reckoning math.
    """
    data = load_trajectory_for_eval(traj_folder, window_size=window_size, stride=stride)
    pos = data['gt_pos']
    quat = data['gt_quat_wxyz']
    starts = data['window_starts']
    W = window_size

    deltas_p_body = []
    deltas_R_body = []
    for k_start in starts:
        k_end = int(k_start) + W - 1
        delta_p, delta_R_6d = compute_relative_pose(
            pos[int(k_start)], quat[int(k_start)],
            pos[k_end], quat[k_end],
        )
        # Recover full ΔR from 6D
        a1 = delta_R_6d[:3]
        a2 = delta_R_6d[3:]
        b1 = a1 / max(np.linalg.norm(a1), 1e-8)
        proj = (b1 @ a2) * b1
        b2 = a2 - proj
        b2 = b2 / max(np.linalg.norm(b2), 1e-8)
        b3 = np.cross(b1, b2)
        delta_R = np.stack([b1, b2, b3], axis=1)

        deltas_p_body.append(delta_p)
        deltas_R_body.append(delta_R)

    deltas_p_body = np.array(deltas_p_body, dtype=np.float64)
    deltas_R_body = np.array(deltas_R_body, dtype=np.float64)

    start_idx = int(starts[0])
    start_pos = pos[start_idx]
    start_R = quat_wxyz_to_R(quat[start_idx])

    pred_positions, _ = dead_reckon(start_pos, start_R, deltas_p_body, deltas_R_body)

    matched = list(starts)
    matched.append(int(starts[-1]) + W - 1)
    matched = np.array(matched)
    gt_matched = pos[matched]

    err = np.linalg.norm(pred_positions - gt_matched, axis=1)
    return {
        'pred_positions': pred_positions,
        'gt_positions': gt_matched,
        'max_err': float(err.max()),
        'mean_err': float(err.mean()),
    }
