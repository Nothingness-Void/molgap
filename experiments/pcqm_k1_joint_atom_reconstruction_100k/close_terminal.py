"""Saved-artifact analysis and existing shared terminal pipeline only."""
import argparse
import json
from molgap.k1_joint_terminal import analyze, prepare_training, prepare_audit, RECIPES
from molgap.constants import REPO_ROOT
from molgap.research_memory.terminal_wiring import close_terminal_multi_arm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("analyze", "close"))
    parser.add_argument("--finalized-at")
    args = parser.parse_args()
    if args.operation == "analyze":
        report = analyze()
        print(json.dumps({"roles": report["roles"], "native_cost_allocation": report["native_cost_allocation"]}, indent=2))
    else:
        if not args.finalized_at:
            parser.error("close requires an explicit retained terminal timestamp")
        inputs = [prepare_training(r, args.finalized_at) for r in RECIPES]
        inputs.append(prepare_audit(args.finalized_at))
        print(json.dumps(close_terminal_multi_arm(REPO_ROOT, inputs), default=str))
