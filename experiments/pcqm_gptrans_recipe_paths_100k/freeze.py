from pathlib import Path
from molgap.constants import REPO_ROOT
import json
from molgap.gptrans_followup_release import freeze_followup

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT,
        base="experiments/pcqm_gptrans_recipe_paths_100k",
        modes=("degree_group_decay_ema999", "degree_path_endpoints_ema999"),
        run="gptrans-g1-group-path-dual-s42", terminal_reference=True), indent=2))
