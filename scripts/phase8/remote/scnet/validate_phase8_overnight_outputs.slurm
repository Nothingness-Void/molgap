#!/usr/bin/env bash
#SBATCH --job-name=molgap-overnight-accept
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SHARE_ROOT="${SHARE_ROOT:-/work1/share/$USER}"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs results/phase8/remote

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/validation/accept_overnight_outputs.py \
  --project "$PROJECT_DIR" \
  --pcqm-root "$SHARE_ROOT/molgap-results/phase8/pcqm4mv2_train_residual_scan" \
  --out results/phase8/remote/overnight_20260723_acceptance.json
