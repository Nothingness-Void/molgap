#!/usr/bin/env bash
# Compute-node shape and one-step backward preflight only; no training result.
#SBATCH --job-name=molgap-pc100k-edgegps-pre
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=/work1/share/acf9jvb3sm/molgap-results/phase8/pubchemqc100k_architecture/dynamic_edge_gps_seed42/preflight-%j.out
#SBATCH --error=/work1/share/acf9jvb3sm/molgap-results/phase8/pubchemqc100k_architecture/dynamic_edge_gps_seed42/preflight-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/molgap}"
RUN_ROOT="${RUN_ROOT:-/work1/share/acf9jvb3sm/molgap-results/phase8/pubchemqc100k_architecture}"
cd "$PROJECT_DIR"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python "$PROJECT_DIR/experiments/pubchemqc100k_architecture/preflight_dynamic_edge_gps.py" \
  --graphs "$RUN_ROOT/pyg_2d_graphs_pubchemqc100k_architecture.pt"

