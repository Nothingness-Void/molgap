"""Formal admission is additive; execution uses bounded CPU fixtures.

The journeys exercise actual FamilyTrainingSession state and output recovery;
cloud bootstrap, real data and native runtime remain untested here.
"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from molgap import family_training_runner as runner
from molgap import modern2d_training as formal
from molgap.experiment_execution import (
    MODERN2D_FORMAL_INTERFACES, MODERN2D_FORMAL_PROFILE, TRAINING_ADAPTERS,
    validate_training_registry,
)
from molgap.experiment_family_artifacts import artifact_adapter
from molgap.experiment_family_workflow import RunContext
from molgap.family_training_kernel import FamilyKernel
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_session import FamilyTrainingSession, validate_session_resume
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.modern2d_feature_binding import indexed_transform_identity
from molgap.training_reproducibility import configure_fp32_determinism, sha256_file
from molgap.v4_runtime import normalized_source_sha256
from test_family_training_session import assert_equal
from test_modern2d_integration_session import contract_file, make_preview_session
from test_modern2d_qualification import permit_record


@pytest.fixture(params=list(zip(formal.MODEL_INTERFACES, formal.INTERFACES)),
                ids=lambda pair: "/".join(pair[1]))
def case(request, tmp_path):
    old, new = request.param
    session, preview, data, initial = make_preview_session(tmp_path, *old, [])
    session.close()
    recipe = replace(preview, family_version=new[1],
        formal_training=formal.Modern2DFormalTrainingProfile(),
        schedule=(ScheduleSpec("four_stage_warmup_cosine_step", formal.FORMAL_STEPS, warmup_units=4)
                  if new[0] == "esa_small" else ScheduleSpec("cosine_step", formal.FORMAL_STEPS)))
    return SimpleNamespace(recipe=recipe, preview=preview, data=data, initial=initial)


def fixture_authority(recipe):
    return {"scope": "CPU_SYNTHETIC_ONLY", "user_authorized": True, "recipe_identity": recipe.identity}


def actual_session(case, checkpoints):
    configure_fp32_determinism(42)
    admission = formal.admit_synthetic_formal_training(case.recipe, case.data,
        authority=fixture_authority(case.recipe))
    kernel = FamilyKernel(case.recipe, case.initial, device="cpu", formal_training=admission)
    return FamilyTrainingSession(kernel,
        FamilyLoaders(case.recipe, case.data, ExecutionTuning(training_workers=0, development_workers=0)),
        checkpoint=lambda progress, state: checkpoints.append(deepcopy(state)),
        selected=lambda *args: None, pass_finished=lambda *args: None)


def bind_formal_outputs(tmp_path, case, session, name):
    contract, record = contract_file(tmp_path, session)
    context = RunContext("synthetic-question", "synthetic-run", case.recipe.family,
        "a" * 64, "b" * 64, case.recipe.family, case.recipe.family_version,
        "c" * 40, "d" * 64, "e" * 64, sha256_file(contract),
        "local", "fixture", "fixture/run", None)
    outputs = SessionOutputs(tmp_path / name, context, contract=contract,
        adapter=MODERN2D_FORMAL_PROFILE, trajectory_id="synthetic-formal-trajectory",
        train_role_identity=record["train_role_identity"])
    initial_audit = json.loads((outputs.root / "role_audit.json").read_bytes())
    assert all(initial_audit["role_use"][role] == "unknown"
               for role in ("official_validation", "test_dev", "test_challenge"))
    session.loaders.attach_role_audit(outputs.role_audit)
    session.checkpoint_builder = outputs.checkpoint_from_builder
    session.selected = outputs.selected
    session.pass_finished = outputs.pass_finished
    session.batch_finished = outputs.batch_finished
    return outputs, context


def test_formal_interfaces_reuse_exact_factories_transform_and_distinct_artifacts(case):
    validate_training_registry()
    recipe, preview = case.recipe, case.preview
    adapter, old = TRAINING_ADAPTERS[recipe.interface], TRAINING_ADAPTERS[preview.interface]
    assert adapter.formal_training and not adapter.integration_preview
    assert adapter.target_identity_in_completion
    assert old.integration_preview and not old.formal_training
    assert adapter.model_factory == old.model_factory
    assert MODERN2D_FORMAL_INTERFACES[recipe.interface][0] == old.model_factory
    assert indexed_transform_identity(recipe.interface) == indexed_transform_identity(preview.interface)
    assert artifact_adapter(MODERN2D_FORMAL_PROFILE, recipe.interface).protocol == "family-session"
    with pytest.raises(ValueError, match="Formal artifact"):
        artifact_adapter(MODERN2D_FORMAL_PROFILE, preview.interface)
    assert FamilyRecipe.from_dict(json.loads(json.dumps(recipe.to_dict()))).identity == recipe.identity
    assert "formal_training" not in preview.to_dict()


def test_formal_actual_session_resume_full_state_and_readonly_dev(case, tmp_path):
    full = actual_session(case, [])
    full_outputs, context = bind_formal_outputs(tmp_path, case, full, "formal-full")
    runtime = {"platform": context.platform, "account": context.account, "precision": "fp32",
        "source_commit": context.source_commit, "source_archive_sha256": context.source_archive_sha256}
    costs = [{"metric": "wall_seconds", "unit": "seconds", "value": 1.,
        "status": "measured", "semantics": "process_wall", "hardware": "synthetic-cpu"},
        {"metric": "device_seconds", "unit": "seconds", "value": None,
        "status": "missing", "semantics": "allocated_device", "hardware": "synthetic-cpu"}]
    (full_outputs.root / "runtime_certificate.json").write_text(
        json.dumps({"fixture": "runtime-certificate"}), encoding="utf-8")
    (full_outputs.root / "runtime_provenance.json").write_text(
        json.dumps({"fixture": "runtime-provenance"}), encoding="utf-8")
    full_outputs.logger.start(full.progress)
    endpoint = full.run()
    full_report = full_outputs.complete(endpoint, runtime=runtime, costs=costs)
    assert full_report["status"] == "MECHANICALLY_VERIFIED"
    role_audit = json.loads((full_outputs.root / "role_audit.json").read_bytes())
    assert role_audit["context"] == context.to_dict()
    assert all(role_audit["role_use"][role] == "untouched"
               for role in ("official_validation", "test_dev", "test_challenge"))
    assert role_audit["role_use"]["train"] == "consumed"
    assert role_audit["role_use"]["development"] == "consumed"
    assert {event["access_kind"] for event in role_audit["roles"]} >= {
        "training_membership", "prediction_input", "labels_read", "metric_computed", "selection_used"}
    assert full_report["observed"]["role_audit"]["path"] == "role_audit.json"
    manifest = json.loads((full_outputs.root / "output_manifest.json").read_bytes())
    assert manifest["artifacts"]["role_audit"]["path"] == "role_audit.json"
    assert manifest["artifacts"]["role_audit"]["sha256"] == sha256_file(
        full_outputs.root / "role_audit.json")
    for name in ("runtime_certificate", "runtime_provenance"):
        assert full_report["observed"]["artifacts"][name]["path"] == name + ".json"
        assert full_report["observed"]["artifacts"][name]["sha256"] == sha256_file(
            full_outputs.root / (name + ".json"))
    expected = deepcopy(full.state_dict())
    checkpoints = []
    first = actual_session(case, checkpoints)
    split_outputs, split_context = bind_formal_outputs(tmp_path, case, first, "formal-split")
    split_outputs.logger.start(first.progress)
    assert first.run(should_pause=lambda: first.progress.optimizer_step >= 2).optimizer_step == 2
    retained = split_outputs.retained_state()
    assert retained["cursor"]["next_batch"] == 0
    split_trace = json.loads((split_outputs.root / "canonical_trace.json").read_bytes())
    split_rows = [row for row in split_trace["observations"] if row["event"] == "observation"]
    assert [(row["optimizer_step"], row["sample_presentations"]) for row in split_rows] == [(2, 256)]
    retained_before_resume = deepcopy(retained)
    resumed = actual_session(case, [])
    resumed_outputs, resumed_context = bind_formal_outputs(tmp_path, case, resumed, "formal-split")
    retained = resumed_outputs.retained_state()
    resumed.load_state_dict(deepcopy(retained))
    resumed_outputs.logger.start(resumed.progress)
    assert resumed.run() == endpoint
    assert_equal(expected, resumed.state_dict())
    assert_equal(retained_before_resume, retained)
    assert endpoint.optimizer_step == 4 and endpoint.sample_presentations == 512
    if case.recipe.schedule.kind == "cosine_step":
        assert expected["scheduler"]["native"]["T_max"] == 46860
    else:
        assert expected["scheduler"]["native"] is None
        assert expected["scheduler"]["completed_units"] == 4
    assert expected["ema"] is None and expected["optimizer"]["state"]
    assert expected["rng_state"]["cuda"] == []
    assert validate_session_resume(case.recipe, retained).next_batch == 0
    resumed_report = resumed_outputs.complete(resumed.progress, runtime=runtime, costs=costs)
    assert resumed_report["status"] == "MECHANICALLY_VERIFIED"
    resumed_audit = json.loads((resumed_outputs.root / "role_audit.json").read_bytes())
    assert len(resumed_audit["roles"]) == len(role_audit["roles"])
    assert resumed_audit["context"] == resumed_context.to_dict()
    assert resumed_report["observed"]["selected_weights"] == "live"
    trace = json.loads((resumed_outputs.root / "canonical_trace.json").read_bytes())
    rows = [row for row in trace["observations"] if row["event"] == "observation"]
    assert [(row["optimizer_step"], row["sample_presentations"]) for row in rows] == [(2, 256), (4, 512)]
    assert resumed_context == split_context
    readonly = actual_session(case, [])
    readonly.load_state_dict(expected)
    before = deepcopy(readonly.state_dict())
    result = readonly.kernel.evaluate(readonly.loaders)
    assert tuple(result["source_idx"].tolist()) == case.data.development_source_indices
    assert_equal(before, readonly.state_dict())
    readonly.close()


def test_native_admission_and_preview_fail_before_payload(case, tmp_path, monkeypatch):
    from molgap import family_training_data, family_training_kernel, v4_runtime
    def forbidden(*args, **kwargs):
        raise AssertionError("Unadmitted source/model/data/CUDA access")
    monkeypatch.setattr(family_training_data, "load_training_data", forbidden)
    monkeypatch.setattr(family_training_kernel, "load_family_model", forbidden)
    monkeypatch.setattr(v4_runtime, "inspect_frozen_state_artifact", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(runner, "_loaded_source", forbidden)
    options = dict(spec=None, package_dir=tmp_path, expected_package_identity="a" * 64,
        arm_id="reference", mode="reference", recipe_path=tmp_path / "absent.json",
        initial_state_path=case.initial, input_root=tmp_path, account="fixture",
        run_reference="fixture/run", trajectory_id="fixture")
    monkeypatch.setattr(runner, "read_recipe", lambda path: ({}, case.recipe))
    with pytest.raises(ValueError, match="bound authority before payload"):
        runner._bootstrap(**options)
    for device in ("cpu", "cuda:0"):
        with pytest.raises(ValueError, match="bound authority before model payload"):
            FamilyKernel(case.recipe, case.initial, device=device)
    monkeypatch.setattr(runner, "read_recipe", lambda path: ({}, case.preview))
    with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
        runner._bootstrap(**options)
    with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
        FamilyKernel(case.preview, case.initial, device="cuda:0")


def test_formal_bootstrap_sets_threads_before_data_read_without_payload(tmp_path, monkeypatch):
    """The formal thread policy must cover CPU label/statistics admission too."""
    from molgap import experiment_execution, family_training_data, v4_runtime
    from molgap.family_training_data import DatasetSelection
    from molgap.modern2d_training import Modern2DFormalTrainingContext

    # Keep this seam entirely synthetic: no model, graph, or CUDA payload is
    # needed to prove the ordering of the formal CPU policy and data read.
    recipe = SimpleNamespace(
        integration_preview=False, cpu_synthetic_only=False, formal_interface=True,
        mode="reference", family="ppgt_small", identity="a" * 64,
        initialization_sha256="b" * 64, input_sidecar=None, train_rows=256,
        development_rows=0, target_mean_eV=3.0, target_std_eV=2.0,
        dataset=DatasetSelection("binding.json", "c" * 64, False),
    )
    spec = SimpleNamespace(to_dict=lambda: {
        "arms": [{"arm_id": "ppgt", "data": {"roles": []}},
                 {"arm_id": "esa", "data": {"roles": []}}],
        "platform": {"cpu_cores": 4, "accelerator": "NvidiaTeslaT4"},
    })
    context = SimpleNamespace(
        training_recipe_sha256="d" * 64, source_commit="e" * 40,
        to_dict=lambda: {},
    )
    formal_context = object.__new__(Modern2DFormalTrainingContext)
    reads = []

    def read_data(root, selection):
        reads.append(torch.get_num_threads())
        return SimpleNamespace(
            train_rows=recipe.train_rows, development_rows=0,
            train_source_start=0, development_source_start=None,
            train_source_indices=None, development_source_indices=None,
            indexed=False, target_mean_eV=recipe.target_mean_eV,
            target_std_eV=recipe.target_std_eV,
        )

    monkeypatch.setattr(runner, "read_recipe", lambda path: ({}, recipe))
    monkeypatch.setattr(runner, "validate_screen_recipe", lambda *args: None)
    monkeypatch.setattr(runner, "RunContext",
                        SimpleNamespace(for_training=lambda *args, **kwargs: context))
    monkeypatch.setattr(runner, "normalized_source_sha256",
                        lambda path: context.training_recipe_sha256)
    monkeypatch.setattr(runner, "_loaded_source", lambda path: tmp_path)
    monkeypatch.setattr(experiment_execution, "validate_staged_trajectory",
                        lambda *args: {"trajectory_id": "fixture",
                                       "state_at_start": {"source_commit": context.source_commit}})
    monkeypatch.setattr(v4_runtime, "inspect_frozen_state_artifact",
                        lambda *args, **kwargs: {"sha256": "f" * 64})
    monkeypatch.setattr(Modern2DFormalTrainingContext, "validate_before_payload",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(Modern2DFormalTrainingContext, "validate_loaded_data",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(family_training_data, "load_training_data", read_data)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        with pytest.raises(ValueError, match="isolated observed accelerator"):
            runner._bootstrap(
                spec=spec, package_dir=tmp_path, expected_package_identity="1" * 64,
                arm_id="ppgt", mode="reference", recipe_path=tmp_path / "recipe.json",
                initial_state_path=tmp_path / "initial.pt", input_root=tmp_path,
                account="fixture", run_reference="fixture/run", trajectory_id="fixture",
                training_only=True, formal_training=formal_context,
            )
    finally:
        torch.set_num_threads(previous_threads)
    assert reads == [1]


def test_formal_target_statistics_keep_exact_frozen_comparison():
    recipe = SimpleNamespace(
        family="ppgt_small", target_mean_eV=1.275090206323082, target_std_eV=2.0,
    )
    data = SimpleNamespace(target_mean_eV=1.2750902063230574, target_std_eV=2.0)
    with pytest.raises(ValueError, match="Decoded train target statistics differ"):
        runner._assert_target_statistics(recipe, data)


@pytest.mark.parametrize("change", [
    {"seed": 43}, {"ema_decay": .999}, {"formal_training": None},
    {"checkpoint_every": 782}, {"qualification": True}, {"family_version": "1"},
    {"schedule": ScheduleSpec("cosine_step", 2)},
])
def test_formal_recipe_cannot_broaden_scope(case, change):
    with pytest.raises((ValueError, TypeError)):
        replace(case.recipe, **change)


def test_old_qualification_certificate_cannot_be_relabelled_formal(case, tmp_path):
    certificate = dict(format="molgap-modern2d-qualification-certificate-v1", status="accepted",
        scope="QUALIFICATION_ONLY", formal_training_exposure=0, scratch_optimizer_updates=6)
    provenance = {"recipe_identity": case.recipe.identity}
    from molgap.screen_policy import canonical_fingerprint
    certificate["provenance_sha256"] = canonical_fingerprint(provenance)
    (tmp_path / "runtime_certificate.json").write_text(json.dumps(certificate), encoding="utf-8")
    (tmp_path / "runtime_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="Preflight probe identity differs"):
        runner._verify_preflight(tmp_path, case.recipe, provenance)
    with pytest.raises(ValueError, match="cannot admit formal training"):
        runner._verify_preflight(tmp_path, case.preview, provenance)


def test_formal_probe_retains_prospectively_frozen_warmup_and_horizon(case):
    recipe = replace(case.recipe, schedule=ScheduleSpec("four_stage_warmup_cosine_step",
        46860, warmup_units=4, reference_epochs=60))
    probe = runner._probe_recipe(recipe)
    assert probe.schedule == recipe.schedule
    assert probe.optimizer_steps == 2 and probe.checkpoint_every == 1
    assert probe.formal_training.phase == "native-preflight"
    assert probe.qualification is None and not probe.dataset.development
    assert probe.development_rows == 0 and probe.selection == "final-live"
    assert probe.initialization_sha256 == recipe.initialization_sha256
    assert probe.learning_rate == recipe.learning_rate and probe.weight_decay == recipe.weight_decay
    assert recipe.execution_policy["precision"] == "fp32" and recipe.execution_policy["tf32"] is False


@pytest.fixture
def native_metadata(case, tmp_path):
    train, train_record = permit_record("train_100k")
    dev, dev_record = permit_record("common_dev")
    binding = dict(format="molgap-indexed-training-input-v1", data_root="absent",
        scope="ACCEPTED_INDEXED_INPUT_V1", train=train_record, development=dev_record)
    path = tmp_path / "formal-binding.json"
    path.write_text(json.dumps(binding), encoding="utf-8")
    recipe = replace(case.recipe, train_rows=100000, development_rows=50000,
        optimizer_steps=46860, dataset=replace(case.recipe.dataset,
            manifest=path.name, manifest_sha256=sha256_file(path)))
    recipe_path = tmp_path / "formal-recipe.json"
    recipe_path.write_text(json.dumps(recipe.to_dict()), encoding="utf-8")
    authority = dict(format=formal.AUTHORITY_FORMAT, owner="server", user_authorized=True,
        logical_run_id="synthetic-run", run_reference="fixture/run",
        purpose="FORMAL_100K_TRAINING", roles=["train_100k", "common_dev"],
        interfaces=[list(item) for item in formal.INTERFACES], seeds=[42], physical_batch_size=128,
        horizon_passes=60, horizon_steps=46860, arms=["ppgt", "esa"], scientific_training_authorized=True,
        protected_role_access_authorized=False, extra_seed_authorized=False,
        automatic_successor_authorized=False, scale_500k_authorized=False, full_training_authorized=False,
        gpu=dict(owner="fixture", accelerator="NvidiaTeslaT4", allocated_devices=2,
            wall_ceiling_seconds=16200, allocated_t4_hours_ceiling=9))
    authority_path = tmp_path / "formal-authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    run = RunContext("synthetic-question", "synthetic-run", recipe.family, "a" * 64, "b" * 64,
        recipe.family, recipe.family_version, "c" * 40, "d" * 64, "e" * 64,
        normalized_source_sha256(recipe_path), "kaggle", "fixture", "fixture/run", None)
    context = formal.Modern2DFormalTrainingContext(run, recipe.identity, sha256_file(authority_path),
        train, dev, "f" * 64, "1" * 64)
    declaration = dict(platform=dict(name="kaggle", accelerator="NvidiaTeslaT4", device_count=2),
        arms=[dict(arm_id=arm, family=dict(zip(("name", "version"), interface)),
            initialization=dict(seed=42), addons=[], data=dict(roles=[dict(role=r) for r in ("train", "development")]))
            for arm, interface in zip(authority["arms"], formal.INTERFACES)])
    spec = SimpleNamespace(identity=run.spec_identity, to_dict=lambda: declaration)
    return SimpleNamespace(recipe=recipe, context=context, spec=spec, path=authority_path,
        authority=authority, binding=binding, binding_path=path, root=tmp_path)


def validate_metadata(case):
    case.context.validate_before_payload(recipe=case.recipe, context=case.context.run_context,
        spec=case.spec, authority_path=case.path, input_root=case.root)


def test_native_metadata_exact_binding_roundtrip_and_full_roles(native_metadata):
    validate_metadata(native_metadata)
    context = native_metadata.context
    assert formal.Modern2DFormalTrainingContext.from_dict(json.loads(json.dumps(context.to_dict()))) == context
    assert native_metadata.recipe.optimizer_steps == 60 * (100000 // 128)
    assert native_metadata.recipe.development_rows == 50000
    with pytest.raises(ValueError, match="ROLE_BLOCKED"):
        replace(context, train_permit=replace(context.train_permit, role="train_500k"))
    with pytest.raises(ValueError, match="source/package"):
        context.validate_before_payload(recipe=native_metadata.recipe,
            context=replace(context.run_context, source_archive_sha256="9" * 64),
            spec=native_metadata.spec, authority_path=native_metadata.path, input_root=native_metadata.root)


@pytest.mark.parametrize("change", [
    {"purpose": "EXECUTION_QUALIFICATION_ONLY"}, {"scientific_training_authorized": False},
    {"roles": ["train_500k", "common_dev"]}, {"protected_role_access_authorized": True},
    {"extra_seed_authorized": True}, {"scale_500k_authorized": True}, {"full_training_authorized": True},
    {"horizon_steps": 2}, {"seeds": [43]},
])
def test_bound_authority_cannot_broaden_or_substitute_qualification(native_metadata, change):
    case = native_metadata
    case.authority.update(change)
    case.path.write_text(json.dumps(case.authority), encoding="utf-8")
    case.context = replace(case.context, authority_sha256=sha256_file(case.path))
    with pytest.raises(ValueError, match="authority does not cover"):
        validate_metadata(case)


def test_role_history_and_authority_bytes_are_independently_bound(native_metadata):
    case = native_metadata
    case.path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="authority bytes changed"):
        validate_metadata(case)


def test_native_metadata_resolves_nested_unique_pinned_mount(native_metadata):
    case = native_metadata
    nested = case.root / "datasets" / "fixture" / "accepted-inputs" / case.binding_path.name
    nested.parent.mkdir(parents=True)
    case.binding_path.replace(nested)
    validate_metadata(case)
    duplicate = case.root / "another-mount" / nested.name
    duplicate.parent.mkdir()
    duplicate.write_bytes(nested.read_bytes())
    with pytest.raises(ValueError, match="one uniquely mounted pinned manifest"):
        validate_metadata(case)


def test_common_dev_role_history_cannot_be_rebound_to_training(native_metadata):
    case = native_metadata
    case.binding["development"]["role_history"]["usage"] = "training"
    case.binding_path.write_text(json.dumps(case.binding), encoding="utf-8")
    case.recipe = replace(case.recipe, dataset=replace(case.recipe.dataset,
        manifest_sha256=sha256_file(case.binding_path)))
    case.context = replace(case.context, recipe_identity=case.recipe.identity)
    with pytest.raises(ValueError, match="ROLE_BLOCKED"):
        validate_metadata(case)


@pytest.mark.parametrize("value", [None, "", "NaN", "Infinity", "-Infinity", "0", "-1", "invalid"])
def test_formal_allocation_deadline_rejects_missing_or_nonfinite_values(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MOLGAP_ALLOCATION_PAUSE_AT", raising=False)
    else:
        monkeypatch.setenv("MOLGAP_ALLOCATION_PAUSE_AT", value)
    with pytest.raises(ValueError, match="finite"):
        runner._formal_pause_deadline()
