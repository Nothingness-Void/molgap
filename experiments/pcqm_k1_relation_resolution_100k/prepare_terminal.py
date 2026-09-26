"""Close a controller-authored terminal decision through shared strict RML.

No scientific outcome is inferred from COMPLETE. Supply the real acceptance,
paired analysis, seven-dimensional evidence, observed costs/roles, decision,
and comparison readiness in the shared terminal format after results exist.
"""
import argparse
import json
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.research_memory.terminal_wiring import close_terminal_arm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(close_terminal_arm(REPO_ROOT, args.trajectory, args.terminal,
                                       trace=args.trace), indent=2))
