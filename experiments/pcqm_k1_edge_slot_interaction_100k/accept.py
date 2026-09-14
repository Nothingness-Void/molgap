"""Saved-artifact acceptance for K1 edge/slot pairwise interactions."""
import argparse
import importlib.util
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.k1_edge_slot_interaction import MODES, PARAMETERS
from molgap.training_reproducibility import atomic_json


def accept(reference_root, candidate_root):
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_frozen_acceptance", path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    result = shared.accept(
        reference_root,
        candidate_root,
        modes=MODES,
        expected_parameters=PARAMETERS,
        initialization_policy="identical-tensors-altered-edge-dataflow",
    )
    for mode in MODES:
        checks = result["candidates"][mode]["record"]["preflight"]["mechanism_checks"]
        expected = "no-slot-attention" if mode == MODES[0] else "uniform-return"
        if checks.get("slot_mode") != expected or len(checks.get("slot_layers", [])) != 3:
            raise RuntimeError(f"Slot interaction checks changed: {mode}")
        if not all(row.get("invariant") is True for row in checks["slot_layers"]):
            raise RuntimeError(f"Slot invariant failed: {mode}")
    result["format"] = "molgap-k1-edge-slot-interaction-acceptance-v1"
    result["scientific_round"] = 2
    result["model_inference_executed"] = False
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, accept(args.reference_root, args.candidate_root))
