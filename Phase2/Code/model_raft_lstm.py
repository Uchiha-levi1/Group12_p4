"""
model_raft_lstm.py
Visual-only VO model: RAFT optical flow → LSTM → FC → 6D pose.

Tensor shapes through the network (for batch B, seq_len L, H x W image):
    Input images:    (B, L+1, 3, H, W)
    Per-pair flow:   (B*L, 2, H, W)     — RAFT output, last iteration
    Flow features:   (B*L, C)           — after spatial pooling + projection
    LSTM input:      (B, L, C)
    LSTM output:     (B, L, 128)
    FC output:       (B, L, 6)
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RAFT import with graceful fallback
# ---------------------------------------------------------------------------

def _load_raft(device: torch.device) -> tuple[nn.Module, object]:
    """
    Try to load torchvision RAFT. Returns (model, transforms_fn).
    Raises ImportError with a helpful message if unavailable.
    """
    try:
        import torchvision
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights)
        model = model.to(device).eval()
        return model, weights.transforms()
    except Exception as tv_err:
        raise ImportError(
            f"torchvision RAFT unavailable ({tv_err}).\n"
            "Install with: pip install torchvision>=0.15\n"
            "Or clone https://github.com/princeton-vl/RAFT and add to PYTHONPATH."
        ) from tv_err


# ---------------------------------------------------------------------------
# Flow encoder (wraps RAFT + spatial pooling + linear projection)
# ---------------------------------------------------------------------------

class FlowEncoder(nn.Module):
    """
    Extracts per-pair flow features using RAFT (frozen by default).

    Input:  img1 (B, 3, H, W), img2 (B, 3, H, W) — values in [0, 1] after
            dataset normalization, but RAFT internally expects [0, 255].
            We un-normalise before passing to RAFT.
    Output: features (B, feature_dim)
    """

    # ImageNet stats used by dataset transform
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        feature_dim: int = 256,
        freeze_raft: bool = True,
        num_flow_updates: int = 12,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_flow_updates = num_flow_updates

        self.raft, _ = _load_raft(device)

        if freeze_raft:
            for p in self.raft.parameters():
                p.requires_grad = False
            print("[FlowEncoder] RAFT weights frozen.")
        else:
            print("[FlowEncoder] RAFT weights trainable.")

        # Spatial pooling + linear projection of 2-channel flow → feature_dim
        self.pool = nn.AdaptiveAvgPool2d((8, 8))   # -> (B, 2, 8, 8)
        self.proj = nn.Sequential(
            nn.Flatten(),                           # -> (B, 128)
            nn.Linear(2 * 8 * 8, feature_dim),
            nn.ReLU(inplace=True),
        )

    def _unnorm(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse ImageNet normalisation and scale to [0, 255]."""
        mean = self._MEAN.to(x.device)
        std  = self._STD.to(x.device)
        return ((x * std + mean) * 255.0).clamp(0, 255)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1: (B, 3, H, W)  normalised float32
            img2: (B, 3, H, W)  normalised float32

        Returns:
            feat: (B, feature_dim)
        """
        # RAFT expects uint8-range float [0, 255]
        i1 = self._unnorm(img1)
        i2 = self._unnorm(img2)

        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.raft.parameters())):
            flow_preds = self.raft(i1, i2, num_flow_updates=self.num_flow_updates)

        flow = flow_preds[-1]           # (B, 2, H, W)
        pooled = self.pool(flow)        # (B, 2, 8, 8)
        feat = self.proj(pooled)        # (B, feature_dim)
        return feat


# ---------------------------------------------------------------------------
# Full VO model
# ---------------------------------------------------------------------------

class RAFTLSTMVOModel(nn.Module):
    """
    Visual odometry model:
        images (B, L+1, 3, H, W)
        → FlowEncoder on each consecutive pair
        → (B, L, feature_dim)
        → LSTM stack (128, 128 with dropout)
        → FC
        → (B, L, 6)

    Args:
        feature_dim:       output dim of FlowEncoder projection
        lstm_hidden:       LSTM hidden size (both layers)
        lstm_dropout:      dropout between LSTM layers
        freeze_raft:       whether to freeze RAFT weights
        num_flow_updates:  RAFT refinement iterations (trade speed vs accuracy)
        device:            torch device (needed to load RAFT)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        lstm_hidden: int = 128,
        lstm_dropout: float = 0.3,
        freeze_raft: bool = True,
        num_flow_updates: int = 12,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.flow_encoder = FlowEncoder(
            feature_dim=feature_dim,
            freeze_raft=freeze_raft,
            num_flow_updates=num_flow_updates,
            device=device,
        )

        # Two-layer LSTM
        # Layer 1: input→hidden
        self.lstm1 = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        # Layer 2: hidden→hidden with dropout on input
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(lstm_dropout)

        # FC head: hidden → 6D pose
        self.fc = nn.Linear(lstm_hidden, 6)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, L+1, 3, H, W)

        Returns:
            poses: (B, L, 6)

        Intermediate shapes:
            per-pair flow feat:  (B*L, feature_dim)
            after reshape:       (B, L, feature_dim)
            after lstm1:         (B, L, 128)
            after dropout+lstm2: (B, L, 128)
            after fc:            (B, L, 6)
        """
        B, Lp1, C, H, W = images.shape
        L = Lp1 - 1

        # Build consecutive pairs and flatten batch+time dims for RAFT
        img1 = images[:, :L,  :, :, :].reshape(B * L, C, H, W)  # (B*L, 3, H, W)
        img2 = images[:, 1:,  :, :, :].reshape(B * L, C, H, W)  # (B*L, 3, H, W)

        feats = self.flow_encoder(img1, img2)   # (B*L, feature_dim)
        feats = feats.view(B, L, -1)            # (B, L, feature_dim)

        # LSTM stack
        out1, _ = self.lstm1(feats)             # (B, L, 128)
        out1 = self.dropout(out1)
        out2, _ = self.lstm2(out1)              # (B, L, 128)

        poses = self.fc(out2)                   # (B, L, 6)
        return poses


# ---------------------------------------------------------------------------
# Quick shape check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = torch.device("cpu")
    print(f"Running shape check on {device}...")

    B, L, H, W = 2, 4, 256, 256
    dummy_images = torch.randn(B, L + 1, 3, H, W, device=device)

    model = RAFTLSTMVOModel(
        feature_dim=256,
        lstm_hidden=128,
        freeze_raft=True,
        num_flow_updates=4,   # fast for test
        device=device,
    )
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(dummy_images)

    print(f"Input:  {tuple(dummy_images.shape)}")
    print(f"Output: {tuple(out.shape)}  — expected ({B}, {L}, 6)")
    assert out.shape == (B, L, 6), "Shape mismatch!"
    print("Shape check passed ✓")
