"""Shared declarations are pure data construction, not research authorization."""
import json
import pytest

from molgap.execution_comparison import (
    ExecutionQuestion, execution_proposal, freeze_comparison_declarations,
    safe_comparison_base, run_version,
)


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
def test_proposal_preserves_explicit_desktop_or_server_ownership(family):
    question = ExecutionQuestion("policy", "DIAGNOSTIC", "question", "gap", "mechanism",
                                 (), (), (), "retain", "reject")
    arguments = dict(question=question, arm={"arm_id": family}, trajectory_id="T-synthetic",
        source_commit="a" * 40, run="synthetic", version="v1", contract_ref="contract.json",
        contract_refs=["contract.json"], authority_ref="authority.json", role_ref="roles.json",
        decision_ref="decision.md", timestamp="2026-10-15T00:00:00Z")
    for owner, track, platform in (("desktop", "B", "kaggle1"), ("server", "C", "kaggle3")):
        result = execution_proposal(**arguments, owner=owner, track=track, platform=platform)
        assert result["trajectory"]["owner"] == owner
        assert result["trajectory"]["track"] == track
        assert result["costs"][0]["platform"] == platform
        assert result["costs"][0]["measurement"]["device_hours"]["status"] == "estimated"
        assert result["trajectory"]["record_mode"] == "prospective"
    with pytest.raises(TypeError):
        execution_proposal(**arguments)


@pytest.mark.parametrize("path", ["../outside", "experiments/../outside", "C:/outside", "experiments//x", "experiments/CON"])
def test_unsafe_comparison_paths_are_rejected(path):
    with pytest.raises(ValueError):
        safe_comparison_base(path, base="experiments")


@pytest.mark.parametrize("version", ["v0", "v01", "1", "../v1"])
def test_run_version_requires_positive_version(version):
    with pytest.raises(ValueError):
        run_version(version)


def test_shared_envelope_never_overwrites_existing_release(tmp_path):
    base = "experiments/synthetic"
    existing = tmp_path / base / "experiment_spec.json"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"immutable")
    with pytest.raises(ValueError, match="already frozen"):
        freeze_comparison_declarations(tmp_path, base=base, experiment_id="synthetic", logical_run_id="synthetic",
            arms=[], bindings=[], jobs=[], entries=[], plan_format="synthetic", authority_ref="authority.json",
            source_extras=[], account="owner", kernel="owner/kernel", title="Synthetic",
            source_dataset="owner/source", data_datasets=["owner/fixed-data"])
    assert existing.read_bytes() == b"immutable"


@pytest.mark.parametrize("family,version", [("neural_atom_k1", "3"), ("gptrans_t", "2")])
def test_envelopes_retain_desktop_acceptance_schema_and_freeze_authority(tmp_path, monkeypatch, family, version):
    from test_experiment_workflow import _arm, _spec_payload
    from molgap import experiment_source_inventory as inventory
    monkeypatch.setattr(inventory, "registered_source_files", lambda *a, **k: [])
    arms = [_arm(family, version, name) for name in ("original", "optimized")]
    bindings = _spec_payload(arms)["prospective"]["arms"]
    authority_ref = "experiments/synthetic/authority.json"
    authority = tmp_path / authority_ref
    authority.parent.mkdir(parents=True)
    authority.write_text("{}")
    result = freeze_comparison_declarations(tmp_path, base="experiments/synthetic", experiment_id="synthetic",
        logical_run_id="synthetic", arms=arms, bindings=bindings, jobs=[], entries=[],
        plan_format="molgap-family-acceptance-plan-v1", authority_ref=authority_ref,
        source_extras=[], account="owner", kernel="owner/kernel", title="Synthetic",
        source_dataset="owner/source", data_datasets=["owner/data"],
        dataset_manifests={name: "manifest.json" for name in ("original", "optimized")})
    acceptance = json.loads((authority.parent / "acceptance_plan.json").read_bytes())
    plan = json.loads((authority.parent / "workflow_plan.json").read_bytes())
    assert set(acceptance) == {"format", "spec_identity", "arms"}
    assert acceptance["spec_identity"] == result["spec_identity"]
    assert authority_ref in plan["source_files"]
    assert plan["dataset_manifests"] == {"original": "manifest.json", "optimized": "manifest.json"}
