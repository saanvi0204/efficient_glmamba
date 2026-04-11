#!/usr/bin/env bash
#SBATCH --job-name=glmamba-eval
#SBATCH --partition=gpu
#SBATCH --account=iitr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/glmamba-eval.%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load cuda/11.6
module load python/python3.10
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate glmamba

srun python -m glmamba.eval \
  --data-root "/scratch/$USER/brats2021" \
  --test-subjects "/scratch/$USER/glmamba_runs/run_001/val_subjects.txt" \
  --scale 2 \
  --checkpoint "/scratch/$USER/glmamba_runs/run_001/best_psnr.pt"

