#!/usr/bin/env bash
#SBATCH --job-name=molgap-br98k-build
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_DIR="results/phase8/broad_residual98k_scnet"
mkdir -p "$RUN_DIR" results/scnet/logs
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/archive/scaleup/data/assemble_repair_v2_1m.py \
  --base-csv data/raw/phase8_expansion_1m.csv \
  --topup-csv data/raw/phase8_broad_residual_topup_97798.csv \
  --out-csv data/raw/phase8_expansion_1m_broad_residual.csv \
  --report-json "$RUN_DIR/assembly_report.json" \
  --dataset-name phase8_expansion_1m_broad_residual \
  --expected-base-rows 1000000 --expected-topup-rows 97798

TOPUP="$RUN_DIR/pyg_2d_graphs_broad_residual98k_topup.pt"
FULL="$RUN_DIR/pyg_2d_graphs_expansion_1m_broad_residual.pt"
python scripts/phase8/data/build_graphs.py \
  --csv data/raw/phase8_expansion_1m_broad_residual.csv \
  --which 2d --tag broad_residual98k_topup --start-row 1000000 --max-rows 97798 --resume \
  --out-2d "$TOPUP"
python scripts/phase8/data/merge_2d_graph_caches.py \
  --base results/phase8/pyg_2d_graphs_bond_expansion_1m.pt \
  --append "$TOPUP" --out "$FULL" --report "$RUN_DIR/graph_merge_report.json" \
  --expected-base 1000000 --expected-append 97798 --delete-append-after-success

python - <<'PY'
import torch
p = "results/phase8/broad_residual98k_scnet/pyg_2d_graphs_expansion_1m_broad_residual.pt"
graphs = torch.load(p, weights_only=False)
indices = torch.tensor([int(graph.source_idx.item()) for graph in graphs])
assert len(graphs) == 1_097_798
assert torch.equal(indices, torch.arange(1_097_798))
print(f"validated graphs={len(graphs):,}", flush=True)
PY
