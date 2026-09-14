"""Reuse frozen saved-tensor acceptance; never construct/infer a model."""
import argparse
import importlib.util
from pathlib import Path
from molgap.constants import REPO_ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "reference", "submission", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("accept_relation", REPO_ROOT / "experiments/pcqm_gptrans_relation_flow/accept.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(module.accept(args.root, args.reference, args.submission, args.output, modes=("memory_value", "memory_message")))
