"""Static/synthetic-only scope and release tests; no model or graph execution."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import pytest

from molgap.gptrans_triplet_portability import BASE, validate_release_contract
from molgap.experiment_allocation import AllocationLedger

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    contract = json.loads((ROOT / BASE / "contract.json").read_text())
    release = {"files": {name: spec["sha256"] for name, spec in {**contract["model_assets"], **contract["reference_payloads"]}.items()}}
    inventory = {name: {"sha256": digest} for name, digest in contract["source_identities"].items()}
    metadata = json.loads((ROOT / BASE / "kernel-metadata.json").read_text())
    return contract, release, inventory, metadata


def test_scope_accepts_exact_retained_assets():
    validate_release_contract(*fixture())


@pytest.mark.parametrize("field,value", [("optimizer_steps",1), ("protected_roles_read",True), ("portability_rows",50000), ("workers",["portability", "new_candidate"])])
def test_scope_rejects_unreleased_change(field, value):
    contract, release, inventory, metadata = fixture()
    contract = deepcopy(contract)
    contract[field] = value
    with pytest.raises(ValueError):
        validate_release_contract(contract, release, inventory, metadata)


def test_retained_source_and_mounts_fail_closed():
    contract, release, inventory, metadata = fixture()
    inventory = deepcopy(inventory)
    inventory["src/molgap/gptrans_triplet_communication.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_release_contract(contract, release, inventory, metadata)
    contract, release, inventory, metadata = fixture()
    metadata["dataset_sources"].append("unreleased/official-validation")
    with pytest.raises(ValueError):
        validate_release_contract(contract, release, inventory, metadata)


def test_idle_second_device_is_not_free():
    ledger = AllocationLedger(spec_identity="test", hardware=["T4", "T4"], assignments={"portability":0}, started=0, clock=lambda:10)
    snapshot = ledger.snapshot("complete")
    assert snapshot["allocated_device_seconds"] == 20
    assert snapshot["unassigned_device_seconds"] == 10


def test_no_training_or_geometry_calls():
    for name in ("gptrans_triplet_portability.py", "gptrans_triplet_portability_prepare.py"):
        tree = ast.parse((ROOT / "src/molgap" / name).read_text())
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"step", "backward", "train"} for n in ast.walk(tree))
        assert not any(isinstance(n, ast.Attribute) and n.attr in {"pos", "edge_distance", "wedge_angle_cos"} for n in ast.walk(tree))
