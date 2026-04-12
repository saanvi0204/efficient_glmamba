#!/usr/bin/env bash
#SBATCH --job-name=glmamba-train
#SBATCH --partition=gpu
#SBATCH --account=iitr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=glmamba-train.%j.out
#SBATCH --error=glmamba-train.%j.err
#SBATCH --export=ALL

set -euo pipefail

# Always run from the repo root so `python -m glmamba...` works even if the job
# was submitted from inside `scripts/`.
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

# Ensure the runtime picks a new-enough libstdc++ (GLIBCXX_3.4.29) from the conda env / gcc module
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

srun --export=ALL python -m glmamba.train_lightning \
  --data-root "/scratch/$USER/brats2021" \
  --out-dir "/scratch/$USER/glmamba_runs/run_001" \
  --scale 2 \
  --epochs 50 \
  --batch-size 2 \
  --num-workers 8 \
  --lr 2e-4 \
  --amp

