"""No-inference acceptance delegated to the retained native family owners."""
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
    a = p.parse_args()
    base = "experiments/pcqm_gptrans_local_control_100k"
    atomic_json(a.output, accept_training_outputs(REPO_ROOT, a.records, a.package, experiment_ref=base + "/gpu"))
    if a.close_rml:
        print(close_author_outputs(REPO_ROOT, a.records, a.output, experiment_ref=base))
