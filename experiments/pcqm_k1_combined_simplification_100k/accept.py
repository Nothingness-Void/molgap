"""No-inference acceptance for the K1 combined simplification."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from molgap.constants import REPO_ROOT
from molgap.pcqm_k1_variants_runner import MINIMUM_GAIN_EV, STOCHASTICITY_FLOOR_EV
from molgap.screen_policy import evaluate_reference_gain, validate_reference_screen_contract


REFERENCE = "neural_atom_k1_v4"
CANDIDATE = "neural_atom_k1_no_attention_uniform_return"
NO_SLOT = "neural_atom_k1_no_slot_attention"
UNIFORM = "neural_atom_k1_uniform_return"


def _helpers():
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("molgap_k1_v4_accept", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._bootstrap_upper, module._load_arm


_bootstrap_upper, _load_arm = _helpers()


def _paired(candidate_payload, other_payload):
    if not torch.equal(candidate_payload["target"], other_payload["target"]):
        raise RuntimeError("Development targets differ")
    if not torch.equal(candidate_payload["source_idx"], other_payload["source_idx"]):
        raise RuntimeError("Development row order differs")
    delta = (
        (candidate_payload["prediction"] - candidate_payload["target"]).abs()
        - (other_payload["prediction"] - other_payload["target"]).abs()
    ).numpy()
    _, interval = _bootstrap_upper(delta)
    return {
        "gain_eV": other_payload["mae"] - candidate_payload["mae"],
        "candidate_minus_other_error_bootstrap_95_eV": interval,
        "fraction_rows_improved": float(np.mean(delta < 0)),
    }


def accept(reference_root: Path, no_slot_root: Path, uniform_root: Path, candidate_root: Path):
    reference, reference_payload = _load_arm(reference_root, REFERENCE)
    no_slot, no_slot_payload = _load_arm(
        no_slot_root, NO_SLOT, expected_parameters={NO_SLOT: 3_608_897}
    )
    uniform, uniform_payload = _load_arm(
        uniform_root, UNIFORM, expected_parameters={UNIFORM: 3_658_817}
    )
    candidate, candidate_payload = _load_arm(
        candidate_root, CANDIDATE, expected_parameters={CANDIDATE: 3_608_897}
    )
    certificates = {
        reference["contract"]["runtime_certificate_id"]: reference["runtime_certificate"],
        candidate["contract"]["runtime_certificate_id"]: candidate["runtime_certificate"],
    }
    comparability = validate_reference_screen_contract(
        reference=reference["contract"], candidate=candidate["contract"],
        runtime_certificates=certificates,
    )
    versus_reference = _paired(candidate_payload, reference_payload)
    gate = evaluate_reference_gain(
        reference_mae_eV=reference_payload["mae"],
        candidate_mae_eV=candidate_payload["mae"],
        stochasticity_floor_eV=STOCHASTICITY_FLOOR_EV,
        minimum_material_gain_eV=MINIMUM_GAIN_EV,
        paired_row_bootstrap_upper_eV=versus_reference[
            "candidate_minus_other_error_bootstrap_95_eV"
        ][1],
    )
    reserve = 1.0 - candidate["training"]["peak_reserved_mib"] / candidate["training"]["total_memory_mib"]
    gate["memory_reserve_fraction"] = reserve
    gate["memory_gate_passed"] = reserve >= 0.15
    gate["passed"] = bool(gate["passed"] and gate["memory_gate_passed"])
    return {
        "format": "molgap-pcqm-k1-combined-simplification-acceptance-v1",
        "accepted": True,
        "reference_development_gap_mae_eV": reference_payload["mae"],
        "candidate_development_gap_mae_eV": candidate_payload["mae"],
        "candidate_record": candidate,
        "comparability": comparability,
        "versus_reference": versus_reference,
        "versus_no_slot_attention": _paired(candidate_payload, no_slot_payload),
        "versus_uniform_return": _paired(candidate_payload, uniform_payload),
        "gate": gate,
        "selected_candidate": CANDIDATE if gate["passed"] else None,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--no-slot-root", type=Path, required=True)
    parser.add_argument("--uniform-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = accept(args.reference_root, args.no_slot_root, args.uniform_root, args.candidate_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
