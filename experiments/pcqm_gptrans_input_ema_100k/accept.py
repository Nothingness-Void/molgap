from pathlib import Path
from molgap.constants import REPO_ROOT
import argparse
from molgap.gptrans_author_acceptance import accept_training_outputs
from molgap.training_reproducibility import atomic_json
from molgap.gptrans_author_terminal import close_author_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--close-rml", action="store_true")
    args = parser.parse_args()
    result = accept_training_outputs(REPO_ROOT, args.records, args.package,
        experiment_ref="experiments/pcqm_gptrans_input_ema_100k/gpu")
    atomic_json(args.output, result)
    if args.close_rml:
        closures = close_author_outputs(REPO_ROOT, args.records, args.output,
            experiment_ref="experiments/pcqm_gptrans_input_ema_100k")
        print("Saved-artifact acceptance and terminal transactions passed:", len(closures))
    else:
        print("Saved-artifact acceptance passed; terminal RML/Replay closure remains separate.")
