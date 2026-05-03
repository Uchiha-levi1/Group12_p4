"""
Losses for IO Bi-LSTM:
  - L1 on Δp
  - Geodesic on R recovered from 6D rotation rep via Gram-Schmidt
"""

import torch
import torch.nn.functional as F


def gram_schmidt_6d_to_R(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation rep -> 3x3 rotation matrix via Gram-Schmidt.

    d6 shape (..., 6) where d6[..., :3] is column 0 and d6[..., 3:] is column 1.
    Returns rotation matrix (..., 3, 3).
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]

    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - proj, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack as columns: shape (..., 3, 3) with columns [b1, b2, b3]
    R = torch.stack([b1, b2, b3], dim=-1)
    return R


def geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two rotation matrices, mean over batch.

    angle = arccos((trace(R_pred^T R_gt) - 1) / 2)
    """
    R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((trace - 1.0) / 2.0).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
    angle = torch.acos(cos)
    return angle.mean()


def io_loss(delta_p_pred, delta_R_6d_pred, delta_p_gt, delta_R_6d_gt,
            lambda_p: float = 1.0, lambda_r: float = 5.0):
    """Combined L1 + geodesic loss.

    Returns dict with components for logging.
    """
    l_trans = F.l1_loss(delta_p_pred, delta_p_gt)

    R_pred = gram_schmidt_6d_to_R(delta_R_6d_pred)
    R_gt = gram_schmidt_6d_to_R(delta_R_6d_gt)
    l_rot = geodesic_loss(R_pred, R_gt)

    l_total = lambda_p * l_trans + lambda_r * l_rot
    return {
        'total': l_total,
        'trans': l_trans.detach(),
        'rot': l_rot.detach(),
    }
