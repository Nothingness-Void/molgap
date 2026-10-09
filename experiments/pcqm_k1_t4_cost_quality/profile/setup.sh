#!/usr/bin/env bash
set -euo pipefail
python -m pip install --quiet 'numpy<2' 'torch-geometric==2.6.1' 'ogb==1.3.6'
