#!/usr/bin/env bash
#SBATCH --job-name=glmamba-train
#SBATCH --partition=gpu
#SBATCH --account=iitr
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=glmamba-train.%j.out
#SBATCH --error=glmamba-train.%j.err
#SBATCH --export=ALL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

module purge
module load gcc/11.1.0
module load cuda/11.6
module load python/python3.10

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate glmamba

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Multi-node DDP: all nodes must agree on the same MASTER_ADDR and MASTER_PORT
# SLURM_NODELIST gives e.g. "node[01-04]" — extract the first node as master
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun --export=ALL python -m glmamba.train_lightning \
  --data-root "/scratch/$USER/brats2021" \
  --out-dir "/scratch/$USER/glmamba_runs/run_multinode" \
  --scale 2 \
  --epochs 50 \
  --batch-size 4 \
  --num-workers 8 \
  --lr 2e-4 \
  --amp \
  --num-nodes 4 \
  --gpus-per-node 2