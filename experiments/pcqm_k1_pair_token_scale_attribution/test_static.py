from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def test_contract_matches_pairtoken_scale_bridge() -> None:
    reference = json.loads((ROOT / "training_contract.json").read_text())
    candidate = json.loads(
        (REPO / "experiments/pcqm_k1_pair_token_500k/training_contract.json").read_text()
    )
    for key in (
        "benchmark_id",
        "manifest_sha256",
        "roles",
        "seed",
        "precision",
        "tf32_enabled",
        "physical_batch_per_device",
        "gradient_accumulation_steps",
        "tail_batch_policy",
        "epochs",
        "optimizer_steps_per_epoch",
        "total_optimizer_steps",
        "total_sample_presentations",
        "optimizer",
        "schedule",
        "selection",
    ):
        assert reference[key] == candidate[key]


def test_runner_is_thin_and_no_protected_role_literal() -> None:
    tree = ast.parse((ROOT / "run_reference.py").read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert [node.name for node in functions] == ["main"]
    source = (ROOT / "run_reference.py").read_text()
    assert "official_validation" not in source
    assert "test_dev" not in source


def test_reference_wrapper_has_frozen_identity() -> None:
    source = (
        REPO / "src/molgap/pcqm_k1_matched60_500k.py"
    ).read_text()
    assert 'MODE = "neural_atom_k1"' in source
    assert "PARAMETERS = 3_658_817" in source
    assert "reference_mae_eV=None" in source
