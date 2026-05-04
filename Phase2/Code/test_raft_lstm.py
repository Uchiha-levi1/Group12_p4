"""
test_raft_lstm.py
Evaluation + visualization for RAFT-LSTM Visual Odometry model.

Usage:
    python test_raft_lstm.py \
        --checkpoint runs/exp01/checkpoints/best.pt \
        --data_root static/phase2_data \
        --output_dir runs/exp01/test_outputs

Generates under output_dir:
    metrics.json               — per-axis and overall MSE
    metrics_per_traj.csv       — per-trajectory RMSE
    traj_XXX_topdown.png       — top-down XY trajectory plots
    traj_XXX_3d.png            — 3D trajectory plots
    traj_XXX_axes.png          — per-axis prediction vs GT curves
    traj_XXX_error.png         — per-step error over time
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset_raft_vo import RAFTVODataset
from model_raft_lstm import RAFTLSTMVOModel
from utils_pose import poses_to_trajectory


AXIS_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz"]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        all_preds:   (N_samples, seq_len, 6)
        all_targets: (N_samples, seq_len, 6)
    """
    model.eval()
    preds_list, tgts_list = [], []

    for batch in loader:
        images  = batch["images"].to(device)
        targets = batch["targets"].to(device)
        preds   = model(images)

        preds_list.append(preds.cpu().numpy())
        tgts_list.append(targets.cpu().numpy())

    all_preds   = np.concatenate(preds_list,   axis=0)  # (N, L, 6)
    all_targets = np.concatenate(tgts_list, axis=0)     # (N, L, 6)
    return all_preds, all_targets


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """
    Args:
        preds, targets: (N, L, 6)

    Returns dict with overall and per-axis MSE/RMSE.
    """
    flat_p = preds.reshape(-1, 6)
    flat_t = targets.reshape(-1, 6)

    overall_mse  = float(np.mean((flat_p - flat_t) ** 2))
    overall_rmse = float(np.sqrt(overall_mse))

    per_axis_mse  = [float(np.mean((flat_p[:, i] - flat_t[:, i]) ** 2)) for i in range(6)]
    per_axis_rmse = [float(np.sqrt(v)) for v in per_axis_mse]

    return {
        "overall_mse":   overall_mse,
        "overall_rmse":  overall_rmse,
        "per_axis_mse":  dict(zip(AXIS_LABELS, per_axis_mse)),
        "per_axis_rmse": dict(zip(AXIS_LABELS, per_axis_rmse)),
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_trajectory_topdown(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """Top-down XY plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_traj[:, 0],   gt_traj[:, 1],   "r-o", ms=3, lw=1.5, label="GT")
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], "b--s", ms=3, lw=1.5, label="Pred")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Top-down trajectory {title}")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_trajectory_3d(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """3D trajectory plot."""
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(gt_traj[:, 0],   gt_traj[:, 1],   gt_traj[:, 2],   "r-", lw=1.5, label="GT")
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], "b--", lw=1.5, label="Pred")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D trajectory {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_per_axis(
    preds: np.ndarray,
    targets: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """
    Per-axis prediction vs GT curves over time.
    preds/targets: (T, 6) flattened sequence
    """
    T = preds.shape[0]
    steps = np.arange(T)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for i, (ax, label) in enumerate(zip(axes.flat, AXIS_LABELS)):
        ax.plot(steps, targets[:, i], "r-",  lw=1, label="GT",   alpha=0.8)
        ax.plot(steps, preds[:, i],   "b--", lw=1, label="Pred", alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Per-axis GT vs Pred {title}", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_error_over_time(
    preds: np.ndarray,
    targets: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """
    L2 error over time steps.
    preds/targets: (T, 6)
    """
    errors = np.sqrt(np.sum((preds - targets) ** 2, axis=-1))  # (T,)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors, "k-", lw=1.2)
    ax.fill_between(range(len(errors)), errors, alpha=0.15, color="steelblue")
    ax.set_xlabel("step")
    ax.set_ylabel("L2 pose error")
    ax.set_title(f"Error over time {title}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-trajectory evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_trajectory(
    model: torch.nn.Module,
    traj_ds: RAFTVODataset,
    traj_idx: int,
    device: torch.device,
    output_dir: Path,
    seq_len: int,
) -> dict:
    """Run inference and generate plots for a single trajectory."""
    traj = traj_ds.trajectories[traj_idx]
    traj_name = traj.traj_dir.name

    # Collect all samples from this trajectory
    indices = [j for j, (ti, _) in enumerate(traj_ds.index) if ti == traj_idx]
    if not indices:
        return {}

    sub_ds = Subset(traj_ds, indices)
    loader = DataLoader(sub_ds, batch_size=4, shuffle=False, num_workers=0)

    preds_arr, tgts_arr = run_inference(model, loader, device)

    # Flatten (N, L, 6) -> (N*L, 6)
    flat_preds = preds_arr.reshape(-1, 6)
    flat_tgts  = tgts_arr.reshape(-1, 6)

    # Dead-reckon trajectories
    gt_traj   = poses_to_trajectory(flat_tgts)
    pred_traj = poses_to_trajectory(flat_preds)

    # Compute metrics
    mse  = float(np.mean((flat_preds - flat_tgts) ** 2))
    rmse = float(np.sqrt(mse))

    # Plots
    plot_trajectory_topdown(
        gt_traj, pred_traj,
        output_dir / f"{traj_name}_topdown.png",
        title=traj_name,
    )
    plot_trajectory_3d(
        gt_traj, pred_traj,
        output_dir / f"{traj_name}_3d.png",
        title=traj_name,
    )
    plot_per_axis(
        flat_preds, flat_tgts,
        output_dir / f"{traj_name}_axes.png",
        title=traj_name,
    )
    plot_error_over_time(
        flat_preds, flat_tgts,
        output_dir / f"{traj_name}_error.png",
        title=traj_name,
    )

    return {"trajectory": traj_name, "mse": mse, "rmse": rmse}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test RAFT-LSTM Visual Odometry")
    p.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    p.add_argument("--data_root",   default="static/phase2_data")
    p.add_argument("--output_dir",  default=None,
                   help="Output directory (default: sibling of checkpoint dir)")
    p.add_argument("--device",      default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--batch_size",  type=int, default=4)
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    device = select_device(args.device)
    print(f"[Test] Device: {device}")

    ckpt_path  = Path(args.checkpoint)
    ckpt       = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})

    output_dir = Path(args.output_dir) if args.output_dir \
        else ckpt_path.parent.parent / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct model from saved config
    seq_len    = saved_args.get("seq_len",     8)
    image_size = tuple(saved_args.get("image_size", [256, 256]))

    model = RAFTLSTMVOModel(
        feature_dim=saved_args.get("feature_dim",  256),
        lstm_hidden=saved_args.get("lstm_hidden",  128),
        freeze_raft=True,
        num_flow_updates=saved_args.get("flow_iters", 12),
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Test] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")

    # ---- Overall metrics ----
    test_ds = RAFTVODataset(
        args.data_root, split="test",
        seq_len=seq_len, image_size=image_size,
    )
    nw = 0 if str(device) == "mps" else args.num_workers
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=nw)

    all_preds, all_tgts = run_inference(model, test_loader, device)
    metrics = compute_metrics(all_preds, all_tgts)

    print("\n[Metrics]")
    print(f"  Overall MSE:  {metrics['overall_mse']:.6f}")
    print(f"  Overall RMSE: {metrics['overall_rmse']:.6f}")
    print("  Per-axis RMSE:")
    for ax, v in metrics["per_axis_rmse"].items():
        print(f"    {ax}: {v:.6f}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Per-trajectory plots ----
    import csv
    per_traj_rows = []
    for ti in range(len(test_ds.trajectories)):
        row = evaluate_trajectory(
            model, test_ds, ti, device, output_dir, seq_len
        )
        if row:
            per_traj_rows.append(row)
            print(f"  {row['trajectory']}: RMSE={row['rmse']:.6f}")

    with open(output_dir / "metrics_per_traj.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trajectory", "mse", "rmse"])
        writer.writeheader()
        writer.writerows(per_traj_rows)

    print(f"\n[Test] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
