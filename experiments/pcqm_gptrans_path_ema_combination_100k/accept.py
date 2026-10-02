"""Saved artifacts and existing RML closure only; no model inference."""
from pathlib import Path
import argparse
from molgap.gptrans_author_acceptance import accept_training_outputs
from molgap.gptrans_author_terminal import close_author_outputs
from molgap.training_reproducibility import atomic_json

BASE = "experiments/pcqm_gptrans_path_ema_combination_100k"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--close-rml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    result = accept_training_outputs(root, args.records, args.package, experiment_ref=BASE + "/gpu")
    atomic_json(args.output, result)
    if args.close_rml:
        print(close_author_outputs(root, args.records, args.output, experiment_ref=BASE))
    print("Require actual candidate/reference Replay admission after closure.")
