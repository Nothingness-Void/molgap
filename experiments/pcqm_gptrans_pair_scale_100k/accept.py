"""Thin saved-output and RML closure entry; never executes a model."""
from pathlib import Path
from molgap.constants import REPO_ROOT
import argparse
from molgap.gptrans_author_acceptance import accept_training_outputs
from molgap.gptrans_author_terminal import close_author_outputs
from molgap.training_reproducibility import atomic_json

BASE = "experiments/pcqm_gptrans_pair_scale_100k"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--close-rml", action="store_true")
    parser.add_argument("--arm", action="append")
    args = parser.parse_args()
    root = REPO_ROOT
    result = accept_training_outputs(root, args.records, args.package,
        experiment_ref=BASE + "/gpu", selected_arms=args.arm)
    atomic_json(args.output, result)
    if args.close_rml:
        print(close_author_outputs(root, args.records, args.output, experiment_ref=BASE))
    print("Saved-artifact acceptance only; require actual candidate/reference replay-pool inclusion.")
