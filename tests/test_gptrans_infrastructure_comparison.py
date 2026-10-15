"""CPU-only declaration and release-gate checks for the GPTrans execution pair."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from molgap.constants import REPO_ROOT
from molgap.experiment_spec import ExperimentSpec
from molgap.gptrans_infrastructure_comparison import (
    BASE,
    INITIAL_STATE_PATH,
    TARGET_MANIFEST,
    check_plan,
    comparison_base,
    freeze_comparison,
    is_comparison,
)
from molgap.training_reproducibility import atomic_json, sha256_file


@pytest.fixture
def declaration(tmp_path, monkeypatch):
    """Freeze into an isolated fixture root, never into the working tree."""
    for relative in (
        "authority.json", "role_plan.json", "protocol.md", "decision.md",
    ):
        source = REPO_ROOT / BASE / relative
        destination = tmp_path / BASE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    target = tmp_path / TARGET_MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / TARGET_MANIFEST, target)
    initial = tmp_path / INITIAL_STATE_PATH
    initial.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / INITIAL_STATE_PATH, initial)

    import molgap.experiment_source_inventory as inventory
    import molgap.gptrans_infrastructure_comparison as owner

    monkeypatch.setattr(inventory, "registered_source_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: "0" * 40 + "\n")
    result = freeze_comparison(tmp_path)
    spec = ExperimentSpec.from_json((tmp_path / BASE / "experiment_spec.json").read_text())
    plan = json.loads((tmp_path / BASE / "acceptance_plan.json").read_text())
    return tmp_path, spec, plan, result


def test_freeze_emits_two_reference_arms_and_paired_profiler(declaration):
    root, spec, plan, result = declaration
    assert is_comparison(spec)
    assert comparison_base(spec) == BASE
    assert result["kernel"] == "nvoid912/molgap-gptrans-infrastructure-dual-100k-v1"
    assert [arm["family"] for arm in spec.to_dict()["arms"]] == [
        {"name": "gptrans_t", "version": "1"},
        {"name": "gptrans_t", "version": "2"},
    ]
    assert json.loads((root / plan["arms"][0]["contract"]["path"]).read_text())[
        "execution_only_chunking"] is True
    assert "execution_only_chunking" not in json.loads(
        (root / plan["arms"][1]["contract"]["path"]).read_text())
    assert check_plan(spec, root, plan)["status"] == "ACCEPTANCE_INPUTS_AVAILABLE"
    staged = root / INITIAL_STATE_PATH
    assert check_plan(spec, root, plan,
                      initial_states={arm: staged for arm in ("original", "optimized")})[
                          "status"] == "ACCEPTANCE_INPUTS_AVAILABLE"


def test_gate_rejects_missing_native_execution_chunk_binding(declaration):
    root, spec, plan, _ = declaration
    entry = plan["arms"][0]
    path = root / entry["contract"]["path"]
    record = json.loads(path.read_text())
    record.pop("execution_only_chunking")
    atomic_json(path, record)
    entry["contract"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="execution-only chunks"):
        check_plan(spec, root, plan)


def test_gate_rejects_authority_expansion(declaration):
    root, spec, plan, _ = declaration
    path = root / plan["authority"]["path"]
    authority = json.loads(path.read_text())
    authority["official_validation_allowed"] = True
    atomic_json(path, authority)
    plan["authority"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="scope|authority"):
        check_plan(spec, root, plan)


def test_gate_rejects_profile_expansion(declaration):
    root, spec, plan, _ = declaration
    entry = plan["arms"][0]
    path = root / entry["contract"]["path"]
    record = json.loads(path.read_text())
    record["profiling"]["active_steps"] = 17
    atomic_json(path, record)
    entry["contract"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="profile|window"):
        check_plan(spec, root, plan)


def test_gate_rejects_shared_scientific_drift(declaration):
    root, spec, plan, _ = declaration
    entry = plan["arms"][1]
    path = root / entry["contract"]["path"]
    record = json.loads(path.read_text())
    record["ema_decay"] = 0.999
    atomic_json(path, record)
    entry["contract"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="recipe|scientific|execution policy"):
        check_plan(spec, root, plan)


def test_gate_rejects_original_sampler_identity_drift(declaration):
    root, spec, plan, _ = declaration
    entry = plan["arms"][0]
    path = root / entry["contract"]["path"]
    record = json.loads(path.read_text())
    record["row_order_fingerprint"] = "0" * 64
    atomic_json(path, record)
    entry["contract"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="scientific|sampler|row order"):
        check_plan(spec, root, plan)


def test_comparison_base_rejects_cross_round_plan_alias(declaration):
    _, spec, _, _ = declaration
    payload = deepcopy(spec.to_dict())
    payload["prospective"]["arms"][0]["plan_spec_ref"] = BASE + "/original/other.json"
    malformed = ExperimentSpec(payload)
    with pytest.raises(ValueError, match="suffix"):
        comparison_base(malformed)
