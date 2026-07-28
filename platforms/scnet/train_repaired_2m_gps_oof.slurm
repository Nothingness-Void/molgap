#!/bin/bash
# Non-executable OOF design placeholder. This is not a Fusion dependency.
#SBATCH --job-name=molgap-oof
#SBATCH --array=0-9
#SBATCH --output=logs/oof_%A_%a.out
#SBATCH --error=logs/oof_%A_%a.err
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

FOLD=$((SLURM_ARRAY_TASK_ID / 2))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
  KIND=gps7
  LAYERS=7
else
  KIND=gps9
  LAYERS=9
fi

echo "Design only: fold=${FOLD} kind=${KIND} layers=${LAYERS}"
echo "Do not submit: genuine OOF requires leakage-free scratch training, held-out prediction export, acceptance, and an explicit Router-research authorization."
exit 64
