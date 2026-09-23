"""Select the frozen Kaggle1 profile of the existing GPTrans T4x2 runner."""
from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["MOLGAP_PAIR_PROFILE"] = "reference-centered-logits-kaggle1-v1"
matches = list(Path("/kaggle/input").rglob("experiments/pcqm_gptrans_pair_norm_100k/run_candidates.py"))
if len(matches) != 1:
    raise FileNotFoundError(f"Expected one GPTrans pair runner, found {len(matches)}")
runpy.run_path(str(matches[0]), run_name="__main__")
