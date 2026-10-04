"""Close saved profiling metadata; no local model construction or inference."""
import argparse
import json
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.gptrans_scale_profile_acceptance import accept_and_close

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(accept_and_close(REPO_ROOT, a.records, a.package), indent=2))
