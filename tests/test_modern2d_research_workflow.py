"""Fresh research releases preserve history and explicitly bind the fast path."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from molgap.family_training_recipe import FamilyRecipe
from molgap.modern2d_training import Modern2DFormalTrainingProfile
from molgap.experiment_input import _read_scientific_recipe, resolve_experiment_input
from molgap.family_training_runner import read_recipe
from test_modern2d_formal_training import case
from test_modern2d_scientific_input import configured
from test_experiment_workflow import _write_json, _git


def test_original_execution_default_is_wire_compatible(case):
    wire = case.recipe.to_dict()
    assert "execution_mode" not in wire["formal_training"]
    assert FamilyRecipe.from_dict(wire).identity == case.recipe.identity


def test_batched_execution_is_explicit_ppgt_only(case):
    base = case.recipe
    profile = Modern2DFormalTrainingProfile(purpose="configured-screen",
        horizon_passes=60, horizon_steps=base.optimizer_steps, execution_mode="batched-v1")
    schedule = replace(base.schedule, total_units=base.optimizer_steps)
    if base.family != "ppgt_small":
        with pytest.raises(ValueError, match="unrelated family"):
            replace(base, formal_training=profile, schedule=schedule)
    else:
        recipe = replace(base, formal_training=profile, schedule=schedule)
        assert recipe.to_dict()["formal_training"]["execution_mode"] == "batched-v1"
        assert FamilyRecipe.from_dict(recipe.to_dict()).identity == recipe.identity
    with pytest.raises(ValueError, match="explicit new configured"):
        Modern2DFormalTrainingProfile(execution_mode="batched-v1")


def test_prepared_v2_execution_is_explicit_for_both_families(case):
    base = case.recipe
    profile = Modern2DFormalTrainingProfile(purpose="configured-screen",
        horizon_passes=60, horizon_steps=base.optimizer_steps, execution_mode="prepared-v2")
    schedule = replace(base.schedule, total_units=base.optimizer_steps)
    recipe = replace(base, formal_training=profile, schedule=schedule)

    wire = recipe.to_dict()
    assert wire["formal_training"]["execution_mode"] == "prepared-v2"
    assert FamilyRecipe.from_dict(wire).identity == recipe.identity
    with pytest.raises(ValueError, match="explicit new configured"):
        Modern2DFormalTrainingProfile(execution_mode="prepared-v2")


def test_scientific_source_reissue_needs_new_authority(configured):
    c = configured
    path = c.repo / c.value["arms"][0]["recipe_ref"]
    record = json.loads(path.read_bytes())
    family = (record["family"], record["family_version"])
    record["execution_policy"]["feature_transform_source_sha256"] = "0" * 64
    _write_json(path, record)
    authority = dict(purpose="CONFIGURED_SCREEN", scientific_training_authorized=True,
                     source_reissue_authorized=True)
    retained, current = _read_scientific_recipe(path, family, authority)
    assert retained == record
    assert current.execution_policy["feature_transform_source_sha256"] != "0" * 64
    for delta in (dict(source_reissue_authorized=False), dict(scientific_training_authorized=False)):
        with pytest.raises(ValueError, match="Execution policy differs"):
            _read_scientific_recipe(path, family, {**authority, **delta})


def test_prepared_v2_configured_authority_requires_fresh_native_gate(configured):
    from molgap.modern2d_training_release import validate_configured_authority

    c = configured
    _, spec, workflow, _ = resolve_experiment_input(c.path, c.repo)
    recipe = read_recipe(c.repo / workflow["arms"][0]["recipe"])[1]
    recipe = replace(recipe, formal_training=replace(
        recipe.formal_training, execution_mode="prepared-v2"))
    authority = {**c.authority, "execution_mode": "prepared-v2",
        "fresh_native_execution_gate": True}

    assert validate_configured_authority(authority, spec.to_dict(),
        {"candidate": recipe}) == 1800
    missing_gate = {key: value for key, value in authority.items()
        if key != "fresh_native_execution_gate"}
    for rejected in (missing_gate,
            {**authority, "fresh_native_execution_gate": False},
            {**authority, "execution_mode": "original"}):
        with pytest.raises(ValueError, match="Prepared execution requires fresh native"):
            validate_configured_authority(rejected, spec.to_dict(), {"candidate": recipe})


@pytest.mark.parametrize("with_repair_proof,repair_report", [
    (False, None),
    (True, {"reference_mode": "graphwise", "update_equivalent": True}),
], ids=["missing-reference-proof", "wrong-reference-proof"])
def test_prepared_preflight_requires_fresh_accepted_native_reference(
        case, tmp_path, monkeypatch, with_repair_proof, repair_report):
    import torch
    from types import SimpleNamespace

    from molgap import family_training_runner as runner
    from molgap import family_training_session as session_owner
    from molgap import k1_execution_qualification as qualification
    from molgap import modern2d_execution_proof as execution_proof
    from molgap.research_memory.trace import file_digest
    from molgap.screen_policy import canonical_fingerprint

    steps = case.recipe.optimizer_steps
    recipe = replace(case.recipe,
        formal_training=Modern2DFormalTrainingProfile(purpose="configured-screen",
            horizon_passes=60, horizon_steps=steps, execution_mode="prepared-v2"),
        schedule=replace(case.recipe.schedule, total_units=steps))
    probe = runner._probe_recipe(recipe)
    provenance = {"source": "synthetic-preflight-fixture"}
    (tmp_path / "runtime_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    def write_state(name, kind):
        path = tmp_path / name
        torch.save({"kind": kind, "rng_state": {"cuda": [0]}}, path)
        return {"path": name, "sha256": file_digest(path),
            "state_fingerprint": "synthetic-complete-state"}

    proofs = [write_state(name, "complete") for name in
        ("repeat-a.pt", "repeat-b.pt", "resumed.pt")]
    origin_path = tmp_path / "resumed-first.pt"
    torch.save({"kind": "origin", "rng_state": {"cuda": [0]}}, origin_path)
    certificate = {"format": "molgap-runtime-certificate-v1", "status": "accepted",
        "provenance_sha256": canonical_fingerprint(provenance),
        "probe_recipe_identity": probe.identity, "formal_training_exposure": 0,
        "proofs": proofs, "resume_origin": {"path": origin_path.name,
            "sha256": file_digest(origin_path)}, "execution_proof": None, "profile_proof": None}
    if with_repair_proof:
        report_path = tmp_path / "execution_repair.json"
        report_path.write_text(json.dumps({"fixture": True}), encoding="utf-8")
        certificate["execution_repair_proof"] = {"path": report_path.name,
            "sha256": file_digest(report_path)}
    (tmp_path / "runtime_certificate.json").write_text(json.dumps(certificate), encoding="utf-8")

    monkeypatch.setattr(session_owner, "validate_session_resume", lambda _probe, state:
        SimpleNamespace(optimizer_step=1 if state["kind"] == "origin" else 2,
            status="paused" if state["kind"] == "origin" else "complete"))
    monkeypatch.setattr(runner, "_state_fingerprint", lambda _state: "synthetic-complete-state")
    monkeypatch.setattr(qualification, "_verify_execution_proof", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_profile_proof", lambda *args, **kwargs: None)
    verified = []

    def verify_repair(_root, proof, observed_probe):
        verified.append(proof)
        assert proof == certificate["execution_repair_proof"]
        assert observed_probe.identity == probe.identity
        return repair_report

    monkeypatch.setattr(execution_proof, "verify_repair_proof", verify_repair)
    with pytest.raises(ValueError, match="fresh accepted-native update equivalence"):
        runner._verify_preflight(tmp_path, recipe, provenance)
    assert bool(verified) is with_repair_proof


def test_batched_scientific_input_preserves_mode(configured):
    c = configured
    if c.value["arms"][0]["model"]["family"] != "ppgt_small":
        return
    value, authority = deepcopy(c.value), deepcopy(c.authority)
    value["arms"][0]["training"]["execution_mode"] = "batched-v1"
    from molgap.research_memory.trace import file_digest
    qualification = "experiments/question/native_qualification.json"
    _write_json(c.repo / qualification, dict(native_certificate_status="accepted",
        execution_repair=dict(family="ppgt_small", update_equivalent=True, scientific_training_released=False)))
    authority.update(execution_mode="batched-v1", execution_qualification_ref=dict(
        path=qualification, sha256=file_digest(c.repo / qualification)))
    _write_json(c.path, value)
    _write_json(c.repo / value["authority_ref"], authority)
    _git(c.repo, "add", ".")
    _git(c.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Explicit new batched screen")
    _, _, workflow, _ = resolve_experiment_input(c.path, c.repo)
    record = json.loads((c.repo / workflow["arms"][0]["recipe"]).read_bytes())
    assert record["formal_training"]["execution_mode"] == "batched-v1"
    assert qualification in workflow["source_records"]
