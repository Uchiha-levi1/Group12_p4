"""
utils_pose.py
Pose math helpers for Visual Odometry pipeline.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.

    Args:
        q: array of shape (4,) in [qx, qy, qz, qw] order

    Returns:
        R: (3, 3) rotation matrix
    """
    qx, qy, qz, qw = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    return R


def rotmat_to_axisangle(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to axis-angle vector (Rodrigues).

    The magnitude of the returned vector encodes the rotation angle (radians).

    Args:
        R: (3, 3) rotation matrix

    Returns:
        r: (3,) axis-angle vector
    """
    # Clamp trace for numerical safety
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)

    if np.abs(angle) < 1e-6:
        # Near-zero rotation — axis is arbitrary, magnitude is 0
        return np.zeros(3, dtype=np.float64)

    # Skew-symmetric part gives axis direction
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.zeros(3, dtype=np.float64)

    axis = axis / axis_norm
    return axis * angle


def relative_transform(
    t1: np.ndarray, R1: np.ndarray,
    t2: np.ndarray, R2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute relative transform from pose 1 to pose 2 in world frame.

        T_rel = T1^{-1} * T2

    Args:
        t1: (3,) translation of pose 1
        R1: (3, 3) rotation of pose 1
        t2: (3,) translation of pose 2
        R2: (3, 3) rotation of pose 2

    Returns:
        dt: (3,) relative translation in frame 1
        dR: (3, 3) relative rotation matrix
    """
    dR = R1.T @ R2
    dt = R1.T @ (t2 - t1)
    return dt, dR


def pose_to_6d(t: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Pack relative pose into a 6D vector [dx, dy, dz, rx, ry, rz].

    Args:
        t: (3,) translation
        R: (3, 3) rotation matrix

    Returns:
        v: (6,) pose vector
    """
    r = rotmat_to_axisangle(R)
    return np.concatenate([t, r]).astype(np.float32)


# ---------------------------------------------------------------------------
# Torch helpers for testing / inference
# ---------------------------------------------------------------------------

def poses_to_trajectory(rel_poses: np.ndarray) -> np.ndarray:
    """
    Dead-reckon a trajectory from relative 6D poses.

    Args:
        rel_poses: (N, 6) array of [dx, dy, dz, rx, ry, rz]

    Returns:
        positions: (N+1, 3) absolute positions (starts at origin)
    """
    positions = [np.zeros(3)]
    R_acc = np.eye(3)
    t_acc = np.zeros(3)

    for pose in rel_poses:
        dt = pose[:3]
        r  = pose[3:]

        angle = np.linalg.norm(r)
        if angle < 1e-8:
            dR = np.eye(3)
        else:
            axis = r / angle
            K = np.array([
                [       0, -axis[2],  axis[1]],
                [ axis[2],        0, -axis[0]],
                [-axis[1],  axis[0],        0],
            ])
            dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        t_acc = t_acc + R_acc @ dt
        R_acc = R_acc @ dR
        positions.append(t_acc.copy())

    return np.array(positions)
