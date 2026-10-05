"""Thin saved-artifact and native per-arm terminal/RML acceptance entrypoint."""
import argparse
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.gptrans_author_acceptance import accept_training_outputs
from molgap.gptrans_author_terminal import close_author_outputs
from molgap.training_reproducibility import atomic_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True, type=Path)
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--close-rml", action="store_true")
    args = p.parse_args()
    base = "experiments/pcqm_gptrans_pair_transition_100k"
    atomic_json(args.output, accept_training_outputs(REPO_ROOT, args.records, args.package, experiment_ref=base + "/gpu"))
    if args.close_rml:
        print(close_author_outputs(REPO_ROOT, args.records, args.output, experiment_ref=base))
