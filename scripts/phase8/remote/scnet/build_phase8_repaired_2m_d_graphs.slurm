#!/usr/bin/env bash
#SBATCH --job-name=molgap-rep2m-D-graph
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
TEAM_ROOT="${TEAM_ROOT:-/work1/share/$USER}"
DATA_CSV="${DATA_CSV:-$TEAM_ROOT/molgap-data/phase8_repaired_2m.csv}"
STAGE_ROOT="${STAGE_ROOT:-$TEAM_ROOT/molgap-results/phase8/repaired_2m_d}"
GRAPH="$STAGE_ROOT/pyg_2d_graphs_repaired_2m.pt"
WORK_DIR="$STAGE_ROOT/graph_work"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$STAGE_ROOT" "$WORK_DIR"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/data/build_graphs.py \
  --csv "$DATA_CSV" --which 2d \
  --out-2d "$GRAPH" --tag repaired_2m_d \
  --work-dir "$WORK_DIR" --resume

GRAPH="$GRAPH" python - <<'PY'
import json
import os
from pathlib import Path

import torch

path = Path(os.environ["GRAPH"])
graphs = torch.load(path, map_location="cpu", weights_only=False)
source_idx = torch.tensor([int(graph.source_idx.item()) for graph in graphs])
labels = torch.cat([graph.y for graph in graphs])
report = {
    "state": "accepted",
    "path": str(path),
    "rows": len(graphs),
    "source_idx_min": int(source_idx.min()),
    "source_idx_max": int(source_idx.max()),
    "source_idx_unique": int(source_idx.unique().numel()),
    "finite_labels": bool(torch.isfinite(labels).all()),
    "size_bytes": path.stat().st_size,
}
if report["rows"] != 2_000_000 or report["source_idx_unique"] != 2_000_000:
    raise RuntimeError(report)
target = path.with_name("graph_acceptance.json")
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(json.dumps(report, indent=2))
os.replace(temporary, target)
print(report)
PY
