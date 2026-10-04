"""Accept the two distinct scopes independently, never supply one arm's evidence to another."""
import argparse
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.gptrans_author_acceptance import accept_training_outputs
from molgap.gptrans_author_terminal import close_author_outputs
from molgap.gptrans_scale_ema_acceptance import accept_outputs, close_outputs
from molgap.training_reproducibility import atomic_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True, type=Path)
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--arm", choices=("degree_bond_local_ema999", "scale_ema"), required=True)
    p.add_argument("--close-rml", action="store_true")
    args = p.parse_args()
    base = "experiments/pcqm_gptrans_capacity_relations_100k"
    if args.arm == "scale_ema":
        atomic_json(args.output, accept_outputs(REPO_ROOT, args.records, args.package))
        if args.close_rml:
            print(close_outputs(REPO_ROOT, args.records, args.output))
    else:
        atomic_json(args.output, accept_training_outputs(REPO_ROOT, args.records, args.package,
            experiment_ref=base + "/gpu", selected_arms=[args.arm]))
        if args.close_rml:
            print(close_author_outputs(REPO_ROOT, args.records, args.output, experiment_ref=base))
