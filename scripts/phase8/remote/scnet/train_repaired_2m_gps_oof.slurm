#!/bin/bash
# Prepared OOF array only. Do not submit before the frozen cache is staged.
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

echo "Prepared only: fold=${FOLD} kind=${KIND} layers=${LAYERS}"
echo "Submission requires an accepted immutable graph cache and explicit launch record."
exit 64
