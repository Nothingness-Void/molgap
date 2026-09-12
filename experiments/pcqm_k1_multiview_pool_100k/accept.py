"""No-inference acceptance for K1 multi-view allocation attempt 2."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.pcqm_k1_variants_runner import MINIMUM_GAIN_EV, STOCHASTICITY_FLOOR_EV
from molgap.screen_policy import evaluate_reference_gain, validate_reference_screen_contract


def _load_shared_acceptance_helpers():
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("molgap_k1_v4_accept", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared acceptance helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._bootstrap_upper, module._load_arm


_bootstrap_upper, _load_arm = _load_shared_acceptance_helpers()
REFERENCE = "neural_atom_k1_v4"
CANDIDATE = "neural_atom_k1_h4"


def accept(reference_root: Path, candidate_root: Path) -> dict:
    reference, reference_payload = _load_arm(reference_root, REFERENCE)
    candidate, candidate_payload = _load_arm(candidate_root, CANDIDATE)
    if not torch.equal(reference_payload["target"], candidate_payload["target"]):
        raise RuntimeError("Development targets differ")
    if not torch.equal(reference_payload["source_idx"], candidate_payload["source_idx"]):
        raise RuntimeError("Development row order differs")
    certificates = {
        reference["contract"]["runtime_certificate_id"]: reference["runtime_certificate"],
        candidate["contract"]["runtime_certificate_id"]: candidate["runtime_certificate"],
    }
    comparability = validate_reference_screen_contract(
        reference=reference["contract"],
        candidate=candidate["contract"],
        runtime_certificates=certificates,
    )
    error_delta = (
        (candidate_payload["prediction"] - candidate_payload["target"]).abs()
        - (reference_payload["prediction"] - reference_payload["target"]).abs()
    ).numpy()
    bootstrap_upper, bootstrap_interval = _bootstrap_upper(error_delta)
    gate = evaluate_reference_gain(
        reference_mae_eV=reference_payload["mae"],
        candidate_mae_eV=candidate_payload["mae"],
        stochasticity_floor_eV=STOCHASTICITY_FLOOR_EV,
        minimum_material_gain_eV=MINIMUM_GAIN_EV,
        paired_row_bootstrap_upper_eV=bootstrap_upper,
    )
    reserve = 1.0 - candidate["training"]["peak_reserved_mib"] / candidate["training"]["total_memory_mib"]
    gate["memory_reserve_fraction"] = reserve
    gate["memory_gate_passed"] = reserve >= 0.15
    gate["passed"] = bool(gate["passed"] and gate["memory_gate_passed"])
    return {
        "format": "molgap-pcqm-k1-multiview-pool-acceptance-v1",
        "accepted": True,
        "reference": reference,
        "candidate": candidate,
        "reference_development_gap_mae_eV": reference_payload["mae"],
        "candidate_development_gap_mae_eV": candidate_payload["mae"],
        "candidate_gain_eV": reference_payload["mae"] - candidate_payload["mae"],
        "paired_error_delta_bootstrap_95_eV": bootstrap_interval,
        "comparability": comparability,
        "gate": gate,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = accept(args.reference_root, args.candidate_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
