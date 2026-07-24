#!/usr/bin/env bash
#SBATCH --job-name=molgap-storage-migrate
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=/work1/share/%u/molgap_storage/logs/%x-%j.out
#SBATCH --error=/work1/share/%u/molgap_storage/logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
DEST_ROOT="${DEST_ROOT:-/work1/share/$USER/molgap_storage}"
export PROJECT_ROOT DEST_ROOT

mkdir -p "$DEST_ROOT/logs"
cd "$PROJECT_ROOT"

paths=(
  results/phase8/repair_v3_1p5m_scnet
  results/phase8/residual_target_40k_scnet
  results/phase8/broad_residual98k_scnet
)
if [[ $# -gt 0 ]]; then
  paths=("$@")
fi

bash scripts/phase8/remote/scnet/migrate_project_storage.sh "${paths[@]}"
