"""Independent saved-artifact verification and RML closure through existing owners."""
import argparse
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.gptrans_author_acceptance import accept_training_outputs
from molgap.gptrans_author_terminal import close_author_outputs
from molgap.training_reproducibility import atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--close-rml", action="store_true")
    args = parser.parse_args()
    base = "experiments/pcqm_gptrans_decay_clock_100k"
    result = accept_training_outputs(REPO_ROOT, args.records, args.package, experiment_ref=base + "/gpu")
    atomic_json(args.output, result)
    if args.close_rml:
        print(close_author_outputs(REPO_ROOT, args.records, args.output, experiment_ref=base))
