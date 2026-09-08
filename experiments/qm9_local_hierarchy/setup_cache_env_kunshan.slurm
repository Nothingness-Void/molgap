#!/usr/bin/env bash
#SBATCH --job-name=molgap-qm9-cache-env
#SBATCH --partition=kshctest02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
: "${ROOT:?}"
case "$(realpath "$ROOT")" in /public/home/scnaqkfcy3/*) ;; *) exit 2 ;; esac
module purge
export PYTHONNOUSERSITE=1
PYTHON=/public/home/scnaqkfcy3/molgap-envs/cpu-cache-py311/bin/python
DEPS="$ROOT/cpu-deps"
mkdir -p "$DEPS"
"$PYTHON" -m pip install --target "$DEPS" --upgrade \
  'numpy==1.26.4' 'scipy==1.14.1' 'pandas==2.2.3' \
  'scikit-learn==1.5.2' 'rdkit==2023.9.6' 'ogb==1.3.6'
PYTHONPATH="$DEPS" "$PYTHON" -c \
  'import numpy,ogb,pandas,rdkit,scipy,sklearn,torch,torch_geometric; print({"numpy":numpy.__version__,"ogb":ogb.__version__,"pandas":pandas.__version__,"rdkit":rdkit.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"torch":torch.__version__,"pyg":torch_geometric.__version__})'
