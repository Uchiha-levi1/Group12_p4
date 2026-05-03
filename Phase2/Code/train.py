"""
Train Bi-LSTM IO model.

Usage:
    python train.py --data_root ./phase2_data --out_dir ./io_lstm_run

Notes:
    - Window 100, stride 50 (train) / 100 (eval)
    - Body-frame relative pose target
    - Loss: 1 * L1(Δp) + 5 * geodesic(R)
    - Adam, lr=1e-3, wd=1e-5, step LR ÷10 at epochs 30 and 45
    - Gradient clip norm 1.0
    - Batch 256, epochs 80
    - No augmentation
    - Save every 10 epochs + best-val-ATE
    - TensorBoard logging
    - Val ATE every 10 epochs on val split
    - SE(3) alignment before ATE
"""

import os
import json
import argparse
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import IODataset
from model import BiLSTM_IO
from losses import io_loss
from eval import evaluate_dataset, oracle_dead_reckon


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_one_epoch(model, loader, optimizer, device, lambda_p, lambda_r, grad_clip):
    model.train()
    sums = {'total': 0.0, 'trans': 0.0, 'rot': 0.0}
    n = 0
    for batch in loader:
        imu = batch['imu'].to(device, non_blocking=True)
        dp_gt = batch['delta_p'].to(device, non_blocking=True)
        dr_gt = batch['delta_R_6d'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        dp_pred, dr_pred = model(imu)
        losses = io_loss(dp_pred, dr_pred, dp_gt, dr_gt,
                         lambda_p=lambda_p, lambda_r=lambda_r)
        losses['total'].backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = imu.size(0)
        sums['total'] += losses['total'].item() * bs
        sums['trans'] += losses['trans'].item() * bs
        sums['rot'] += losses['rot'].item() * bs
        n += bs

    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def validate_per_window(model, loader, device, lambda_p, lambda_r):
    model.eval()
    sums = {'total': 0.0, 'trans': 0.0, 'rot': 0.0}
    n = 0
    for batch in loader:
        imu = batch['imu'].to(device, non_blocking=True)
        dp_gt = batch['delta_p'].to(device, non_blocking=True)
        dr_gt = batch['delta_R_6d'].to(device, non_blocking=True)
        dp_pred, dr_pred = model(imu)
        losses = io_loss(dp_pred, dr_pred, dp_gt, dr_gt,
                         lambda_p=lambda_p, lambda_r=lambda_r)
        bs = imu.size(0)
        sums['total'] += losses['total'].item() * bs
        sums['trans'] += losses['trans'].item() * bs
        sums['rot'] += losses['rot'].item() * bs
        n += bs
    return {k: v / max(n, 1) for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./phase2_data')
    parser.add_argument('--out_dir', type=str, default='./io_lstm_run')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--lambda_p', type=float, default=1.0)
    parser.add_argument('--lambda_r', type=float, default=5.0)
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--train_stride', type=int, default=50)
    parser.add_argument('--eval_stride', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[30, 45])
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--ckpt_every', type=int, default=10)
    parser.add_argument('--val_ate_every', type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    tb_dir = os.path.join(args.out_dir, 'tensorboard')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = get_device()
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # ---------- Sanity check oracle on first val trajectory ----------
    val_dir = os.path.join(args.data_root, 'val')
    from glob import glob
    val_folders = sorted(glob(os.path.join(val_dir, 'traj_*')))
    if val_folders:
        sanity = oracle_dead_reckon(val_folders[0],
                                    window_size=args.window,
                                    stride=args.eval_stride)
        print(f"Oracle dead-reckoning sanity (val/traj_000):")
        print(f"  max position error  = {sanity['max_err']:.6f} m")
        print(f"  mean position error = {sanity['mean_err']:.6f} m")
        print(f"  (should be < 1e-3)")

    # ---------- Datasets ----------
    train_set = IODataset(
        os.path.join(args.data_root, 'train'),
        window_size=args.window, stride=args.train_stride,
    )
    val_set_window = IODataset(
        os.path.join(args.data_root, 'val'),
        window_size=args.window, stride=args.eval_stride,
    )
    print(f"Train windows: {len(train_set)}")
    print(f"Val windows  : {len(val_set_window)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_set_window, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ---------- Model + optimizer ----------
    model = BiLSTM_IO().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_factor,
    )

    writer = SummaryWriter(log_dir=tb_dir)
    csv_log = open(os.path.join(args.out_dir, 'training.csv'), 'w')
    csv_log.write('epoch,lr,train_total,train_trans,train_rot,'
                  'val_total,val_trans,val_rot,val_mean_ate,epoch_time\n')

    best_val_ate = float('inf')
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            args.lambda_p, args.lambda_r, args.grad_clip,
        )
        val_metrics = validate_per_window(
            model, val_loader, device, args.lambda_p, args.lambda_r,
        )

        scheduler.step()

        # Per-window logging
        writer.add_scalar('lr', lr_now, epoch)
        writer.add_scalar('train/total', train_metrics['total'], epoch)
        writer.add_scalar('train/trans_l1', train_metrics['trans'], epoch)
        writer.add_scalar('train/rot_geodesic', train_metrics['rot'], epoch)
        writer.add_scalar('val/total', val_metrics['total'], epoch)
        writer.add_scalar('val/trans_l1', val_metrics['trans'], epoch)
        writer.add_scalar('val/rot_geodesic', val_metrics['rot'], epoch)

        # Trajectory-level val ATE every N epochs
        val_mean_ate = float('nan')
        if epoch % args.val_ate_every == 0 or epoch == args.epochs:
            results = evaluate_dataset(
                model, os.path.join(args.data_root, 'val'), device,
                window_size=args.window, stride=args.eval_stride,
            )
            ates = [r['rmse_ate'] for r in results]
            val_mean_ate = float(np.mean(ates))
            val_median_ate = float(np.median(ates))
            writer.add_scalar('val/mean_ate', val_mean_ate, epoch)
            writer.add_scalar('val/median_ate', val_median_ate, epoch)
            print(f"  val ATE: mean={val_mean_ate:.3f} m, median={val_median_ate:.3f} m, "
                  f"per-traj=[{', '.join(f'{a:.2f}' for a in ates)}]")

            if val_mean_ate < best_val_ate:
                best_val_ate = val_mean_ate
                best_epoch = epoch
                ckpt = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_mean_ate': val_mean_ate,
                    'config': vars(args),
                }
                torch.save(ckpt, os.path.join(ckpt_dir, 'best.pt'))
                print(f"  new best val ATE = {val_mean_ate:.3f} m -> saved best.pt")

        # Periodic checkpoint
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': vars(args),
            }
            torch.save(ckpt, os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt'))

        epoch_time = time.time() - t0
        print(f"[{epoch:3d}/{args.epochs}] "
              f"lr={lr_now:.1e}  "
              f"train: total={train_metrics['total']:.4f} "
              f"(L1={train_metrics['trans']:.4f}, geo={train_metrics['rot']:.4f})  "
              f"val: total={val_metrics['total']:.4f} "
              f"(L1={val_metrics['trans']:.4f}, geo={val_metrics['rot']:.4f})  "
              f"[{epoch_time:.1f}s]")

        csv_log.write(f"{epoch},{lr_now},"
                      f"{train_metrics['total']},{train_metrics['trans']},{train_metrics['rot']},"
                      f"{val_metrics['total']},{val_metrics['trans']},{val_metrics['rot']},"
                      f"{val_mean_ate},{epoch_time}\n")
        csv_log.flush()

    csv_log.close()
    writer.close()

    # Final summary
    print(f"\nDone. Best val ATE = {best_val_ate:.3f} m at epoch {best_epoch}.")


if __name__ == '__main__':
    main()
