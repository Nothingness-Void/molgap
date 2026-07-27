#!/usr/bin/env bash
# Build the additive repair-v3 1.5M 2D cache in restartable 50K shards.
#SBATCH --job-name=molgap-repairv3-1p5m-2d
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_DIR="results/phase8/repair_v3_1p5m_scnet"
mkdir -p results/scnet/logs "$RUN_DIR"
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

TOPUP_CACHE="$RUN_DIR/pyg_2d_graphs_bond_repair_v3_topup_500k.pt"
FULL_CACHE="$RUN_DIR/pyg_2d_graphs_bond_repair_v3_1p5m.pt"

python scripts/phase8/data/build_graphs.py \
  --csv data/raw/phase8_repair_v3_1p5m.csv \
  --which 2d --tag repair_v3_topup_500k --start-row 1000000 --max-rows 500000 --resume \
  --out-2d "$TOPUP_CACHE"

python scripts/phase8/data/merge_2d_graph_caches.py \
  --base results/phase8/pyg_2d_graphs_bond_expansion_1m.pt \
  --append "$TOPUP_CACHE" --out "$FULL_CACHE" \
  --report "$RUN_DIR/graph_merge_report.json" \
  --expected-base 1000000 --expected-append 500000 \
  --delete-append-after-success

python - <<'PY'
from pathlib import Path
import torch

path = Path("results/phase8/repair_v3_1p5m_scnet/pyg_2d_graphs_bond_repair_v3_1p5m.pt")
graphs = torch.load(path, weights_only=False)
indices = torch.tensor([int(graph.source_idx.item()) for graph in graphs])
assert len(graphs) == 1_500_000, len(graphs)
assert indices.unique().numel() == 1_500_000
assert int(indices.min()) == 0 and int(indices.max()) == 1_499_999
print(f"validated_2d_graphs={len(graphs)} path={path}", flush=True)
PY
