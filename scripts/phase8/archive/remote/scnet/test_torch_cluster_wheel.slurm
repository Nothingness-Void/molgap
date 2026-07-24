#!/usr/bin/env bash
# Isolated one-DCU compatibility probe for a user-supplied torch-cluster wheel.
# Submit with:
#   sbatch --export=ALL,CANDIDATE_WHEEL=/path/to/torch_cluster.whl \
#     scripts/phase8/archive/remote/scnet/test_torch_cluster_wheel.slurm
# This does not alter the platform Conda environment or the existing .scnet/site.
#SBATCH --job-name=molgap-torchcluster-probe
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CANDIDATE_WHEEL="${CANDIDATE_WHEEL:?Set CANDIDATE_WHEEL to the uploaded wheel path}"
CANDIDATE_SITE="${CANDIDATE_SITE:-$PROJECT_DIR/.scnet/site-torchcluster-candidate}"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$CANDIDATE_SITE"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10

sha256sum "$CANDIDATE_WHEEL"
python -m pip install --no-deps --no-index --target "$CANDIDATE_SITE" "$CANDIDATE_WHEEL"
export PYTHONPATH="$PROJECT_DIR/src:$CANDIDATE_SITE:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import torch
import torch_cluster

print(f"torch={torch.__version__}")
print(f"hip={torch.version.hip}")
print(f"torch_cluster={torch_cluster.__version__}")
PY

python scripts/phase8/remote/scnet/check_dcu_environment.py \
  --output results/scnet/torchcluster_candidate_environment.json
