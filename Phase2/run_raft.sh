#!/bin/bash
#SBATCH --job-name "raft_lstm_vo"
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=32g
#SBATCH --partition academic           # RBE549 class → must use academic
#SBATCH --time 0-12:00:00
#SBATCH --gres=gpu:1                   # 1 GPU (academic partition limit)
#SBATCH --constraint="A100|V100"
#SBATCH --output logs/%j_train.out     # stdout → logs/<jobid>_train.out
#SBATCH --error  logs/%j_train.err     # stderr → logs/<jobid>_train.err

# ── Environment ──────────────────────────────────────────────────────────────
module load python
module load cuda/12.2

# Activate your virtualenv (create once with: python -m venv ~/venvs/vo_env)
source ~/venvs/vo_env/bin/activate

# ── Paths (edit PROJECT_ROOT to match your Turing home layout) ───────────────
PROJECT_ROOT=~/Phase2
DATA_ROOT=$PROJECT_ROOT/static/phase2_data
CODE_DIR=$PROJECT_ROOT/Code
RUN_DIR=$PROJECT_ROOT/runs/exp_$(date +%Y%m%d_%H%M%S)

mkdir -p "$PROJECT_ROOT/logs"   # for SLURM output files
mkdir -p "$RUN_DIR"

echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Run dir:      $RUN_DIR"
echo "Data root:    $DATA_ROOT"
echo "========================================"

# ── Install deps if not already in venv ─────────────────────────────────────
pip install -q -r "$PROJECT_ROOT/requirements.txt"

# ── Train ────────────────────────────────────────────────────────────────────
cd "$CODE_DIR"

python train_raft_lstm.py \
    --data_root   "$DATA_ROOT"  \
    --run_dir     "$RUN_DIR"    \
    --epochs      50            \
    --batch_size  8             \
    --seq_len     8             \
    --image_size  256 256       \
    --lr          1e-4          \
    --num_workers 8             \
    --device      cuda          \
    --flow_iters  12            \
    --save_every  10

echo "Training complete. Checkpoints in $RUN_DIR/checkpoints/"
