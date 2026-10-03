"""No-training, no-inference qualification of retained G1/G2 evidence."""
import json
from pathlib import Path
from molgap.constants import REPO_ROOT

from molgap.gptrans_candidate_qualification import qualify_author_candidates

if __name__ == "__main__":
    print(json.dumps(qualify_author_candidates(REPO_ROOT), indent=2))
