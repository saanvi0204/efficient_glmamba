#!/usr/bin/env bash
#SBATCH --job-name=glmamba-eval
#SBATCH --partition=gpu
#SBATCH --account=iitr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=glmamba-eval.%j.out
#SBATCH --error=glmamba-eval.%j.err
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

# Configuration matching training setup
DATA_ROOT="/scratch/$USER/brats2021"
CHECKPOINT_DIR="/scratch/$USER/glmamba_runs/run_multinode"
SCALE=2
NORMALIZE="minmax"

# Use last checkpoint only
CHECKPOINT="$CHECKPOINT_DIR/last.ckpt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi
echo "Using last checkpoint: $CHECKPOINT"

# Use validation subjects from training
VAL_SUBJECTS="$CHECKPOINT_DIR/val_subjects.txt"
if [ ! -f "$VAL_SUBJECTS" ]; then
    echo "ERROR: Validation subjects file not found: $VAL_SUBJECTS"
    exit 1
fi

echo "=========================================="
echo "GL-Mamba Evaluation"
echo "=========================================="
echo "Data root:    $DATA_ROOT"
echo "Checkpoint:   $CHECKPOINT"
echo "Val subjects: $VAL_SUBJECTS"
echo "Scale:        $SCALE"
echo "Normalize:    $NORMALIZE"
echo "=========================================="

# Run evaluation
python -m glmamba.eval \
  --data-root "$DATA_ROOT" \
  --test-subjects "$VAL_SUBJECTS" \
  --checkpoint "$CHECKPOINT" \
  --scale $SCALE \
  --normalize "$NORMALIZE" \
  --batch-size 8 \
  --num-workers 16 \
  --device cuda

echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
