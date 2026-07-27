#!/usr/bin/env bash
# Resume the accepted GPS11 run only to finalize metrics and export embeddings.
#SBATCH --job-name=molgap-rep2m-D-gps11-final
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
TEAM_ROOT="${TEAM_ROOT:-/work1/share/$USER}"
STAGE_ROOT="${STAGE_ROOT:-$TEAM_ROOT/molgap-results/phase8/repaired_2m_d}"
RUN_SEED="${RUN_SEED:-42}"
GRAPH="$STAGE_ROOT/pyg_2d_graphs_repaired_2m.pt"
RUN_ROOT="$STAGE_ROOT/gps11_160_seed${RUN_SEED}"
MODEL="$STAGE_ROOT/phase8_repaired_2m_d_gps11_160_seed${RUN_SEED}.pt"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$RUN_ROOT"
test -s "$RUN_ROOT/training_state.pt"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/training/train_encoder.py \
  --kind gps \
  --graphs "$GRAPH" \
  --hidden-channels 160 --num-layers 11 --num-heads 4 --dropout 0.05 \
  --seed "$RUN_SEED" --split-seed 42 \
  --epochs 40 --patience 12 --lr 0.00008 \
  --batch-size 256 --eval-batch-size 256 --embedding-batch-size 256 \
  --replay-boundary 500000 --replay-weight 3.0 \
  --checkpoint-out "$RUN_ROOT/training_state.pt" --checkpoint-every 1 \
  --resume-from "$RUN_ROOT/training_state.pt" \
  --model-out "$MODEL" \
  --metrics-out "$RUN_ROOT/metrics.json" \
  --embeddings-out "$RUN_ROOT/embeddings.pt"

python - "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = [root / "metrics.json", root / "training_state.pt", root / "embeddings.pt"]
for path in paths:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing final artifact: {path}")
manifest = {
    "status": "complete",
    "artifacts": {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    },
}
temporary = root / ".completion_manifest.json.tmp"
temporary.write_text(json.dumps(manifest, indent=2) + "\n")
temporary.replace(root / "completion_manifest.json")
PY
