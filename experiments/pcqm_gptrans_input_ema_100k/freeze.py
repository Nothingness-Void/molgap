from pathlib import Path
import json
from molgap.gptrans_followup_release import freeze_followup

if __name__ == "__main__":
    print(json.dumps(freeze_followup(Path(__file__).resolve().parents[2]), indent=2))
