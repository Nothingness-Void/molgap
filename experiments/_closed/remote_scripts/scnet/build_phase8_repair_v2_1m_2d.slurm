#!/usr/bin/env bash
# Build the repair-v2 1M 2D cache in restartable 50K graph shards.
# Submit with: sbatch scripts/phase8/archive/remote/scnet/build_phase8_repair_v2_1m_2d.slurm
# The output is intentionally separate from the rejected 1M continuation.
#SBATCH --job-name=molgap-repairv2-1m-2d
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=03:00:00
# BW-1 account policy exposes one DCU with 8 CPU cores and about 28 GiB RAM.
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs results/phase8/repair_v2_scnet

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/data/build_graphs.py \
  --csv data/raw/phase8_repair_v2_1m.csv \
  --which 2d \
  --tag repair_v2_1m \
  --resume \
  --out-2d results/phase8/repair_v2_scnet/pyg_2d_graphs_bond_repair_v2_1m.pt

python - <<'PY'
from pathlib import Path
import torch

path = Path("results/phase8/repair_v2_scnet/pyg_2d_graphs_bond_repair_v2_1m.pt")
graphs = torch.load(path, weights_only=False)
indices = torch.tensor([int(graph.source_idx.item()) for graph in graphs])
assert len(graphs) == 1_000_000, len(graphs)
assert indices.unique().numel() == 1_000_000
assert int(indices.min()) == 0 and int(indices.max()) == 999_999
print(f"validated_2d_graphs={len(graphs)} path={path}", flush=True)
PY
