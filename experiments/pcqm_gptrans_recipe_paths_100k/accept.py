from pathlib import Path
import argparse
from molgap.gptrans_author_acceptance import accept_training_outputs
from molgap.training_reproducibility import atomic_json
from molgap.gptrans_author_terminal import close_author_outputs

BASE = "experiments/pcqm_gptrans_recipe_paths_100k"

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
        closures = close_author_outputs(root, args.records, args.output, experiment_ref=BASE)
        print("Saved-artifact acceptance and terminal transactions passed:", len(closures))
        print("Replay-Ready remains contingent on verified replay-pool inclusion for both pairs.")
    else:
        print("Saved-artifact acceptance passed; terminal RML/Replay closure remains separate.")
