"""
dataset_raft_vo.py
PyTorch Dataset for visual-only RAFT VO pipeline.

Layout expected:
    <data_root>/
        train/traj_000/images/*.png
        train/traj_000/groundtruth.csv
        val/...
        test/...

groundtruth.csv format:
    # timestamp tx ty tz qx qy qz qw
    1000000 0.1 0.2 0.3 0.0 0.0 0.0 1.0
    ...

Each dataset sample is a sequence of L consecutive image pairs
and the corresponding L relative 6D pose targets.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils_pose import quat_to_rotmat, relative_transform, pose_to_6d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_groundtruth(csv_path: Path) -> dict[int, np.ndarray]:
    """
    Parse groundtruth.csv -> {timestamp_ns: array([tx,ty,tz,qx,qy,qz,qw])}.
    Lines beginning with '#' are treated as comments.
    """
    poses: dict[int, np.ndarray] = {}
    with open(csv_path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",") if "," in line else line.split()
            if len(parts) < 8:
                continue
            ts = int(parts[0])
            vals = np.array([float(v) for v in parts[1:8]], dtype=np.float64)
            poses[ts] = vals
    return poses


def _nearest_timestamp(query: int, keys: list[int]) -> int:
    """Return key in sorted keys list closest to query."""
    idx = np.searchsorted(keys, query)
    if idx == 0:
        return keys[0]
    if idx >= len(keys):
        return keys[-1]
    before, after = keys[idx - 1], keys[idx]
    return before if abs(query - before) <= abs(query - after) else after


# ---------------------------------------------------------------------------
# Trajectory parser
# ---------------------------------------------------------------------------

class Trajectory:
    """
    Represents a single trajectory folder.
    Aligns image timestamps with GT and exposes (image_path, pose_6d) pairs.
    """

    def __init__(self, traj_dir: Path, image_size: tuple[int, int] = (256, 256)):
        self.traj_dir = traj_dir
        self.image_size = image_size

        # Collect sorted image paths
        img_dir = traj_dir / "images"
        img_paths = sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem))
        assert len(img_paths) >= 2, f"Need >= 2 images in {img_dir}"

        # Load GT poses
        gt_path = traj_dir / "groundtruth.csv"
        gt_dict = _load_groundtruth(gt_path)
        gt_keys = sorted(gt_dict.keys())

        # Align each image timestamp to nearest GT
        self.frames: list[tuple[Path, np.ndarray]] = []
        for img_path in img_paths:
            ts = int(img_path.stem)
            nearest = _nearest_timestamp(ts, gt_keys)
            vals = gt_dict[nearest]  # [tx, ty, tz, qx, qy, qz, qw]
            t = vals[:3]
            R = quat_to_rotmat(vals[3:])
            self.frames.append((img_path, t, R))

        # Pre-compute relative 6D targets for consecutive pairs
        # targets[i] = relative pose from frame i to frame i+1
        self.targets: list[np.ndarray] = []
        for i in range(len(self.frames) - 1):
            _, t1, R1 = self.frames[i]
            _, t2, R2 = self.frames[i + 1]
            dt, dR = relative_transform(t1, R1, t2, R2)
            self.targets.append(pose_to_6d(dt, dR))

    def __len__(self) -> int:
        # Number of valid sequence start indices for seq_len L
        return len(self.targets)

    def get_img_path(self, idx: int) -> Path:
        return self.frames[idx][0]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RAFTVODataset(Dataset):
    """
    Loads sequences of length `seq_len` from trajectory folders.

    Each sample:
        images: (seq_len+1, 3, H, W)  — seq_len+1 frames needed for seq_len pairs
        targets: (seq_len, 6)          — relative 6D poses

    RAFT will process pairs (images[i], images[i+1]) for i in 0..seq_len-1.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        seq_len: int = 8,
        image_size: tuple[int, int] = (256, 256),
        stride: int = 1,
    ):
        self.seq_len = seq_len
        self.image_size = image_size
        self.stride = stride

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),          # -> [0, 1] float32
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        split_dir = Path(data_root) / split
        assert split_dir.exists(), f"Split directory not found: {split_dir}"

        traj_dirs = sorted(split_dir.glob("traj_*"))
        assert len(traj_dirs) > 0, f"No traj_* folders found in {split_dir}"

        # Build per-trajectory Trajectory objects
        self.trajectories: list[Trajectory] = []
        for td in traj_dirs:
            try:
                traj = Trajectory(td, image_size=image_size)
                if len(traj) >= seq_len:
                    self.trajectories.append(traj)
                else:
                    print(f"[WARN] Skipping {td.name}: only {len(traj)} pairs < seq_len={seq_len}")
            except Exception as e:
                print(f"[WARN] Skipping {td.name}: {e}")

        # Build flat index: (traj_idx, start_frame_idx)
        self.index: list[tuple[int, int]] = []
        for ti, traj in enumerate(self.trajectories):
            max_start = len(traj) - seq_len + 1
            for si in range(0, max_start, stride):
                self.index.append((ti, si))

        print(f"[Dataset] split={split} | trajectories={len(self.trajectories)} "
              f"| samples={len(self.index)} | seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ti, si = self.index[idx]
        traj = self.trajectories[ti]

        # Load seq_len+1 frames (for seq_len pairs)
        imgs = []
        for fi in range(si, si + self.seq_len + 1):
            img_path = traj.get_img_path(fi)
            img = Image.open(img_path).convert("RGB")
            imgs.append(self.transform(img))

        images = torch.stack(imgs, dim=0)  # (seq_len+1, 3, H, W)

        # Stack targets
        target_list = [
            torch.tensor(traj.targets[si + k], dtype=torch.float32)
            for k in range(self.seq_len)
        ]
        targets = torch.stack(target_list, dim=0)  # (seq_len, 6)

        return {"images": images, "targets": targets}
