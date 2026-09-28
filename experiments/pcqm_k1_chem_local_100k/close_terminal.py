"""Close both accepted arms through the shared idempotent RML transaction."""
import argparse
import json

from molgap.constants import REPO_ROOT
from molgap.k1_chem_local import MODES
from molgap.k1_chem_local_terminal import prepare
from molgap.research_memory.terminal_wiring import close_terminal_multi_arm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalized-at", required=True)
    args = parser.parse_args()
    inputs = [prepare(mode, args.finalized_at) for mode in MODES]
    print(json.dumps(close_terminal_multi_arm(REPO_ROOT, inputs), default=str))
