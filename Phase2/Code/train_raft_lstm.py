"""
train_raft_lstm.py
Training script for RAFT-LSTM Visual Odometry model.

Usage:
    python train_raft_lstm.py \
        --data_root static/phase2_data \
        --run_dir runs/exp01 \
        --epochs 50 \
        --batch_size 4 \
        --seq_len 8

MPS example (Apple Silicon):
    python train_raft_lstm.py \
        --data_root static/phase2_data \
        --run_dir runs/mps_exp \
        --device mps \
        --batch_size 2 \
        --num_workers 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_raft_vo import RAFTVODataset
from model_raft_lstm import RAFTLSTMVOModel


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(requested: str | None) -> torch.device:
    """
    Priority: CUDA → MPS → CPU (unless overridden by --device flag).
    """
    if requested and requested != "auto":
        dev = torch.device(requested)
        if requested == "mps" and not torch.backends.mps.is_available():
            print("[WARN] MPS requested but not available. Falling back to CPU.")
            dev = torch.device("cpu")
        elif requested == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            dev = torch.device("cpu")
        return dev

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] Apple Silicon MPS backend selected.")
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> float:
    model.train(is_train)
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images  = batch["images"].to(device)   # (B, L+1, 3, H, W)
            targets = batch["targets"].to(device)  # (B, L, 6)

            preds = model(images)                  # (B, L, 6)
            loss  = criterion(preds, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def save_checkpoint(
    state: dict,
    ckpt_dir: Path,
    filename: str,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RAFT-LSTM Visual Odometry")

    p.add_argument("--data_root",    default="static/phase2_data")
    p.add_argument("--run_dir",      default="runs/exp01",
                   help="Directory for logs, checkpoints, plots")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--seq_len",      type=int,   default=8,
                   help="Number of consecutive frame pairs per sample")
    p.add_argument("--image_size",   type=int,   nargs=2, default=[256, 256],
                   metavar=("H", "W"))
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--device",       default="auto",
                   help="cuda | mps | cpu | auto")
    p.add_argument("--feature_dim",  type=int,   default=256)
    p.add_argument("--lstm_hidden",  type=int,   default=128)
    p.add_argument("--freeze_raft",  action="store_true", default=True)
    p.add_argument("--flow_iters",   type=int,   default=12,
                   help="RAFT refinement iterations (lower = faster)")
    p.add_argument("--save_every",   type=int,   default=10,
                   help="Save periodic epoch checkpoint every N epochs (0=off)")
    p.add_argument("--resume",       default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"[Train] Using device: {device}")

    # MPS safety note
    if str(device) == "mps":
        print("[MPS] Note: some RAFT ops may fall back to CPU on MPS. "
              "If you hit errors, set --device cpu or --num_workers 0.")

    run_dir   = Path(args.run_dir)
    ckpt_dir  = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["device_resolved"] = str(device)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Datasets & loaders ----
    image_size = tuple(args.image_size)
    train_ds = RAFTVODataset(args.data_root, split="train",
                             seq_len=args.seq_len, image_size=image_size)
    val_ds   = RAFTVODataset(args.data_root, split="val",
                             seq_len=args.seq_len, image_size=image_size)

    # num_workers=0 recommended for MPS to avoid multiprocessing issues
    nw = 0 if str(device) == "mps" else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=nw, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=False)

    # ---- Model ----
    model = RAFTLSTMVOModel(
        feature_dim=args.feature_dim,
        lstm_hidden=args.lstm_hidden,
        freeze_raft=args.freeze_raft,
        num_flow_updates=args.flow_iters,
        device=device,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    start_epoch = 0
    best_val_loss = float("inf")

    # ---- Resume ----
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Train] Resumed from epoch {start_epoch - 1}")

    # ---- Log file ----
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,elapsed_s\n")

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"  Training for {args.epochs} epochs")
    print(f"  Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"  Checkpoints → {ckpt_dir}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion,
                               optimizer, device, is_train=True)
        val_loss   = run_epoch(model, val_loader,   criterion,
                               None,      device, is_train=False)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"[Epoch {epoch:04d}] train={train_loss:.6f}  "
              f"val={val_loss:.6f}  ({elapsed:.1f}s)")

        # Checkpoint state
        state = {
            "epoch":          epoch,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "train_loss":     train_loss,
            "val_loss":       val_loss,
            "best_val_loss":  best_val_loss,
            "args":           vars(args),
        }

        # Always save latest
        save_checkpoint(state, ckpt_dir, "last.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state["best_val_loss"] = best_val_loss
            save_checkpoint(state, ckpt_dir, "best.pt")
            print(f"  ↳ New best val loss: {best_val_loss:.6f} — saved best.pt")

        # Periodic checkpoint
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(state, ckpt_dir, f"epoch_{epoch:04d}.pt")

        # Append to log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{elapsed:.2f}\n")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
