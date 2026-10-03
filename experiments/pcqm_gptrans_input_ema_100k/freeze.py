from pathlib import Path
from molgap.constants import REPO_ROOT
import json
from molgap.gptrans_followup_release import freeze_followup

if __name__ == "__main__":
    print(json.dumps(freeze_followup(REPO_ROOT), indent=2))
