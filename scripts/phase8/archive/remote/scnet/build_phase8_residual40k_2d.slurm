#!/usr/bin/env bash
#SBATCH --job-name=molgap-res40k-build
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_DIR="results/phase8/residual_target_40k_scnet"
mkdir -p "$RUN_DIR" results/scnet/logs
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/archive/scaleup/data/assemble_repair_v2_1m.py \
  --base-csv data/raw/phase8_expansion_1m.csv \
  --topup-csv data/raw/residual_target_round01_development_traincols.csv \
  --out-csv data/raw/phase8_expansion_1m_residual40k.csv \
  --report-json "$RUN_DIR/assembly_report.json" \
  --dataset-name phase8_expansion_1m_residual40k \
  --expected-base-rows 1000000 --expected-topup-rows 40457

TOPUP="$RUN_DIR/pyg_2d_graphs_residual40k_topup.pt"
FULL="$RUN_DIR/pyg_2d_graphs_expansion_1m_residual40k.pt"
python scripts/phase8/data/build_graphs.py \
  --csv data/raw/phase8_expansion_1m_residual40k.csv \
  --which 2d --tag residual40k_topup --start-row 1000000 --max-rows 40457 --resume \
  --out-2d "$TOPUP"
python scripts/phase8/data/merge_2d_graph_caches.py \
  --base results/phase8/pyg_2d_graphs_bond_expansion_1m.pt \
  --append "$TOPUP" --out "$FULL" --report "$RUN_DIR/graph_merge_report.json" \
  --expected-base 1000000 --expected-append 40457 --delete-append-after-success

python - <<'PY'
import torch
p="results/phase8/residual_target_40k_scnet/pyg_2d_graphs_expansion_1m_residual40k.pt"
g=torch.load(p,weights_only=False)
idx=torch.tensor([int(x.source_idx.item()) for x in g])
assert len(g)==1_040_457 and torch.equal(idx,torch.arange(1_040_457))
print(f"validated graphs={len(g):,}",flush=True)
PY
