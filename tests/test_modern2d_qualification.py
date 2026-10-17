"""Generated CPU qualification replay and fail-closed admission boundaries."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import time
from types import SimpleNamespace

import pytest
import torch

from molgap.data.indexed_inputs import AcceptedIndexedInputPermit
from molgap.experiment_execution import execute_training_phase, validate_execution_plan
from molgap.experiment_family_workflow import RunContext
from molgap.family_training_kernel import FamilyKernel
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_session import FamilyTrainingSession, validate_session_resume
from molgap import family_training_runner as runner
from molgap import modern2d_qualification as owner
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import normalized_source_sha256
from test_experiment_spec import payload
from test_family_training_session import assert_equal
from test_modern2d_integration_session import (
    INTERFACES, make_preview_session, preview_spec,
)


@pytest.fixture(scope="module", params=INTERFACES, ids=lambda item: "/".join(item))
def qualification_case(request, tmp_path_factory):
    family, version = request.param
    root = tmp_path_factory.mktemp("qualification-" + family)
    session, preview, data, initial = make_preview_session(root, family, version, [])
    session.close()
    recipe = replace(preview, dataset=replace(preview.dataset, development=False),
        development_rows=0, optimizer_steps=2, schedule=ScheduleSpec("cosine_step", 2),
        selection="final-live", qualification=owner.QualificationProbe())
    data = replace(data, development=None, development_rows=0,
        development_source_indices=None, development_source_start=None)
    return SimpleNamespace(recipe=recipe, preview=preview, data=data, initial=initial)


def forbidden(*args, **kwargs):
    raise AssertionError("Admission reached source/payload/model/CUDA work")


@pytest.fixture
def before_payload(monkeypatch):
    from molgap import family_training_data, family_training_kernel, v4_runtime
    monkeypatch.setattr(family_training_data, "load_training_data", forbidden)
    monkeypatch.setattr(family_training_kernel, "load_family_model", forbidden)
    monkeypatch.setattr(v4_runtime, "inspect_frozen_state_artifact", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(runner, "_loaded_source", forbidden)


@pytest.fixture(scope="module")
def synthetic_result(qualification_case, tmp_path_factory):
    case = qualification_case
    output = tmp_path_factory.mktemp("qualification-proof-" + case.recipe.family) / "scratch"
    updates, runs, restores = [], [], []
    real_step, real_run = FamilyKernel.step, FamilyTrainingSession.run
    real_restore = FamilyTrainingSession.load_state_dict

    def step(kernel, batch, **options):
        assert kernel.device.type == "cpu" and batch.num_graphs == 128
        result = real_step(kernel, batch, **options)
        updates.append((kernel.recipe.interface, batch.num_graphs))
        return result

    def run(session, *args, **options):
        result = real_run(session, *args, **options)
        runs.append((result.optimizer_step, result.status))
        return result

    def restore(session, state):
        restores.append(deepcopy(state))
        return real_restore(session, state)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(FamilyKernel, "step", step)
        patch.setattr(FamilyTrainingSession, "run", run)
        patch.setattr(FamilyTrainingSession, "load_state_dict", restore)
        patch.setattr(torch.cuda, "_lazy_init", forbidden)
        certificate = owner.run_synthetic_modern2d_qualification(recipe=case.recipe,
            data=case.data, initial_state_path=case.initial, output=output)
    return SimpleNamespace(output=output, certificate=certificate,
        updates=updates, runs=runs, restores=restores)


def test_synthetic_actual_session_six_updates_exact_repeat_and_serialized_resume(
        qualification_case, synthetic_result):
    case, result = qualification_case, synthetic_result
    assert result.updates == [(case.recipe.interface, 128)] * 6
    assert result.runs == [(2, "complete"), (2, "complete"), (1, "paused"), (2, "complete")]
    assert len(result.restores) == 1
    states = [torch.load(result.output / name, map_location="cpu", weights_only=True)
        for name in ("repeat-a.pt", "repeat-b.pt", "resumed.pt")]
    origin = torch.load(result.output / "resumed-first.pt", map_location="cpu", weights_only=True)
    assert_equal(origin, result.restores[0])
    progress = validate_session_resume(case.recipe, origin)
    assert (progress.optimizer_step, progress.sample_presentations, progress.status) == (1, 128, "paused")
    for state in states:
        progress = validate_session_resume(case.recipe, state)
        assert (progress.optimizer_step, progress.sample_presentations, progress.status) == (2, 256, "complete")
        assert state["scheduler"]["completed_units"] == 2
        assert state["optimizer"]["state"]
        assert all(entry["step"].item() == 2 for entry in state["optimizer"]["state"].values())
        assert state["rng_state"]["cuda"] == []
        for component in ("model", "optimizer", "scheduler", "rng_state", "cursor"):
            assert_equal(states[0][component], state[component])
        assert_equal(states[0], state)
    initial = torch.load(case.initial, map_location="cpu", weights_only=True)
    assert any(not torch.equal(value, states[0]["model"][key]) for key, value in initial.items())


def test_synthetic_certificate_has_zero_formal_exposure_and_scratch_accounting(
        qualification_case, synthetic_result):
    case, result = qualification_case, synthetic_result
    certificate = result.certificate
    assert certificate == json.loads((result.output / "runtime_certificate.json").read_text())
    assert certificate["format"] == owner.CERTIFICATE_FORMAT
    assert certificate["status"] == "CPU_SYNTHETIC_VERIFIED"
    assert certificate["scope"] == "CPU_SYNTHETIC_ONLY"
    assert certificate["native_execution_released"] is False
    assert certificate["formal_training_exposure"] == 0
    assert certificate["scientific_acceptance"] == "NOT_EVALUATED"
    assert certificate["scratch_optimizer_updates"] == 6
    assert certificate["qualification_context_identity"] is None
    assert certificate["probe_recipe_identity"] == case.recipe.identity
    assert len({proof["state_fingerprint"] for proof in certificate["proofs"]}) == 1
    for proof in [*certificate["proofs"], certificate["resume_origin"]]:
        assert sha256_file(result.output / proof["path"]) == proof["sha256"]
    resource = json.loads((result.output / "resource_qualification.json").read_text())
    assert sha256_file(result.output / "resource_qualification.json") == certificate["resource_qualification_sha256"]
    assert resource["physical_batch_size"] == 128
    assert resource["scratch_optimizer_updates"] == 6
    assert resource["scratch_sample_presentations"] == 768
    assert resource["scope"] == "CPU_SYNTHETIC_ONLY"
    assert resource["allocated_accelerators"] == []
    assert resource["peak_allocated_bytes"] is resource["peak_reserved_bytes"] is None
    assert resource["allocated_device_seconds"] is None
    assert resource["process_wall_seconds"] > 0
    assert resource["end_to_end_scratch_samples_per_second"] > 0
    provenance = json.loads((result.output / "runtime_provenance.json").read_text())
    assert provenance["scope"] == "CPU_SYNTHETIC_ONLY"
    with pytest.raises(ValueError, match="cannot admit formal training"):
        runner._verify_preflight(result.output, case.recipe, provenance)
    # Even a forged accepted status must not bypass the recipe/interface gate.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "_json", lambda path: {"status": "accepted"})
        with pytest.raises(ValueError, match="cannot admit formal training"):
            runner._verify_preflight(result.output, case.preview, provenance)


def test_qualification_recipe_roundtrip_is_explicit_and_frozen(qualification_case):
    recipe = qualification_case.recipe
    restored = FamilyRecipe.from_dict(json.loads(json.dumps(recipe.to_dict())))
    assert restored.identity == recipe.identity
    assert type(restored.qualification) is owner.QualificationProbe
    assert restored.execution_policy["scope"] == "QUALIFICATION_ONLY"
    assert restored.execution_policy["native_qualification"] == "NOT_EVALUATED"
    assert replace(recipe, qualification=None).identity != recipe.identity


@pytest.mark.parametrize("change", [
    {"seed": 43}, {"batch_size": 64}, {"batch_size": True},
    {"optimizer_steps": 1}, {"checkpoint_every": 2}, {"finite_check_every": 2},
    {"train_rows": 129}, {"drop_last": False}, {"qualification": True},
    {"family_version": "1"}, {"family_version": "99"},
])
def test_invalid_qualification_recipe_rejected(qualification_case, change, before_payload):
    with pytest.raises((TypeError, ValueError)):
        replace(qualification_case.recipe, **change)


@pytest.mark.parametrize("change", [
    {"purpose": "scientific-training"}, {"steps_per_path": 3},
    {"steps_per_path": True}, {"scratch_updates_per_model": 7},
    {"scratch_updates_per_model": True},
])
def test_probe_cannot_expand_scratch_budget(change):
    with pytest.raises(ValueError, match="three two-update scratch paths"):
        owner.QualificationProbe(**change)


def permit_record(role):
    authority = dict(scope="ACCEPTED_INDEXED_INPUT_V1", role=role,
        acceptance_sha256="1" * 64, manifest_sha256=("2" if role == "train_100k" else "3") * 64,
        version_identity="4" * 64, operations=["prediction_input", "labels_read"],
        protected=False, max_chunk_bytes=1024)
    history = dict(role=role, manifest_sha256=authority["manifest_sha256"],
        usage="training" if role.startswith("train_") else "selection-used")
    permit = AcceptedIndexedInputPermit(role, authority["acceptance_sha256"],
        authority["manifest_sha256"], authority["version_identity"],
        canonical_fingerprint(authority), canonical_fingerprint(history), 1024)
    return permit, dict(permit=asdict(permit), authority=authority, role_history=history)


@pytest.fixture
def native_metadata(qualification_case, tmp_path, monkeypatch, before_payload):
    """Only metadata is generated; no real dataset, native model or device exists."""
    train, train_record = permit_record("train_100k")
    dev, dev_record = permit_record("common_dev")
    binding = dict(format="molgap-indexed-training-input-v1", data_root="absent",
        scope="ACCEPTED_INDEXED_INPUT_V1", train=train_record, development=dev_record)
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    recipe = replace(qualification_case.recipe, train_rows=100000,
        dataset=replace(qualification_case.recipe.dataset, manifest="binding.json",
            manifest_sha256=sha256_file(binding_path)))
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe.to_dict()), encoding="utf-8")
    authority = dict(format="molgap-modern2d-bounded-qualification-authority-v1",
        owner="server", user_authorized=True, purpose="EXECUTION_QUALIFICATION_ONLY",
        roles=["train_100k", "common_dev"], interfaces=[list(item) for item in INTERFACES],
        seeds=[42], physical_batch_size=128, arms=["ppgt_author_v2", "esa_native_v3"],
        scientific_training_authorized=False, protected_role_access_authorized=False,
        extra_seed_authorized=False, automatic_successor_authorized=False,
        gpu=dict(platform="kaggle2", owner="fixture", accelerator="NvidiaTeslaT4",
            allocated_devices=2, wall_ceiling_seconds=1800, allocated_t4_hours_ceiling=1))
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    arm_id = "ppgt_author_v2" if recipe.family == "ppgt_small" else "esa_native_v3"
    run_context = RunContext("synthetic-question", "synthetic-run", arm_id, "a" * 64,
        "b" * 64, recipe.family, recipe.family_version, "c" * 40, "d" * 64,
        "e" * 64, normalized_source_sha256(recipe_path), "kaggle", "fixture", "fixture/run", None)
    context = owner.Modern2DQualificationContext(run_context, recipe.identity,
        sha256_file(authority_path), train, dev, "f" * 64)
    declaration = dict(platform=dict(name="kaggle", accelerator="NvidiaTeslaT4",
        device_count=2, cpu_cores=4), arms=[dict(arm_id=arm,
            family=dict(zip(("name", "version"), interface)), initialization=dict(seed=42),
            addons=[], data=dict(roles=[dict(role="train")]))
        for arm, interface in zip(authority["arms"], INTERFACES)])
    spec = SimpleNamespace(identity=run_context.spec_identity, to_dict=lambda: declaration)
    options = dict(spec=spec, package_dir=tmp_path / "absent-package",
        expected_package_identity=run_context.package_identity, arm_id=arm_id,
        mode="reference", recipe_path=recipe_path, initial_state_path=tmp_path / "absent.pt",
        input_root=tmp_path, account="fixture", run_reference="fixture/run",
        authority_path=authority_path)
    result = SimpleNamespace(recipe=recipe, context=context, options=options,
        authority=authority, binding=binding, binding_path=binding_path, declaration=declaration)
    monkeypatch.setattr(runner, "read_recipe", lambda path: ({}, result.recipe))
    monkeypatch.setattr(runner, "validate_screen_recipe", lambda *args: None)
    monkeypatch.setattr(RunContext, "for_training", classmethod(lambda cls, *args, **kwargs: run_context))
    return result


def bootstrap(metadata):
    return owner.bootstrap_qualification(metadata.context, began=time.perf_counter(), **metadata.options)


def test_valid_metadata_reaches_only_deliberate_source_boundary(native_metadata):
    # Proves negative cases below are not rejected by unrelated malformed setup.
    with pytest.raises(AssertionError, match="source/payload/model/CUDA"):
        bootstrap(native_metadata)


@pytest.mark.parametrize("change", [
    {"max_wall_seconds": 1801}, {"max_wall_seconds": 0}, {"max_wall_seconds": True},
    {"allocated_accelerators": ("NvidiaTeslaP100",)},
    {"allocated_accelerators": ["NvidiaTeslaT4", "NvidiaTeslaT4"]},
    {"purpose": "formal-training"}, {"authority_sha256": "invalid"},
    {"recipe_identity": "invalid"}, {"run_context": True},
])
def test_invalid_reviewed_context_rejected(native_metadata, change):
    with pytest.raises((ValueError, TypeError)):
        replace(native_metadata.context, **change)


@pytest.mark.parametrize("field,value", [
    ("family_name", "gptrans_t"), ("family_version", "1"),
    ("family_version", "99"), ("platform", "local"),
])
def test_context_interface_version_and_platform_are_exact(native_metadata, field, value):
    context = native_metadata.context
    with pytest.raises(ValueError, match="Only Kaggle"):
        replace(context, run_context=replace(context.run_context, **{field: value}))


@pytest.mark.parametrize("change", ["500k", "swapped", "version", "acceptance", "boolean"])
def test_context_requires_exact_100k_and_common_dev_permits(native_metadata, change):
    context = native_metadata.context
    changes = {"train_permit": replace(context.train_permit, role="train_500k")}
    if change == "swapped":
        changes = dict(train_permit=context.development_permit, development_permit=context.train_permit)
    elif change in ("version", "acceptance"):
        field = "version_identity" if change == "version" else "acceptance_sha256"
        changes = dict(development_permit=replace(context.development_permit, **{field: "9" * 64}))
    elif change == "boolean":
        changes = dict(train_permit=True)
    with pytest.raises(ValueError, match="ROLE_BLOCKED"):
        replace(context, **changes)


@pytest.mark.parametrize("rows", [128, 256, 500000, 3378606])
def test_real_qualification_is_100k_only(native_metadata, rows):
    with pytest.raises(ValueError, match="train-only 100K"):
        replace(native_metadata.recipe, train_rows=rows)


@pytest.mark.parametrize("change", [
    {"user_authorized": False}, {"owner": "desktop"}, {"seeds": [43]},
    {"roles": ["train_500k", "common_dev"]}, {"physical_batch_size": 64},
    {"scientific_training_authorized": True}, {"protected_role_access_authorized": True},
    {"extra_seed_authorized": True}, {"automatic_successor_authorized": True},
    {"interfaces": [["ppgt_small", "1"], ["esa_small", "3"]]},
    {"gpu": dict(platform="kaggle2", owner="fixture", accelerator="NvidiaTeslaT4",
        allocated_devices=2, wall_ceiling_seconds=1801, allocated_t4_hours_ceiling=1)},
])
def test_authority_scope_rejected_before_payload_or_gpu(native_metadata, change):
    metadata = native_metadata
    metadata.authority.update(change)
    path = metadata.options["authority_path"]
    path.write_text(json.dumps(metadata.authority), encoding="utf-8")
    metadata.context = replace(metadata.context, authority_sha256=sha256_file(path))
    with pytest.raises(ValueError, match="authority does not cover"):
        bootstrap(metadata)


def test_authority_bytes_must_match_independent_pin(native_metadata):
    native_metadata.options["authority_path"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="authority bytes changed"):
        bootstrap(native_metadata)


@pytest.mark.parametrize("field,value", [
    ("source_commit", "9" * 40), ("source_archive_sha256", "9" * 64),
    ("package_identity", "9" * 64), ("training_recipe_sha256", "9" * 64),
    ("spec_identity", "9" * 64), ("logical_run_id", "changed-run"),
    ("account", "changed"), ("run_reference", "fixture/changed"),
])
def test_observed_source_context_must_match_reviewed_pins(native_metadata, monkeypatch, field, value):
    changes = {field: value}
    if field == "account":
        changes["run_reference"] = "changed/run"
    observed = replace(native_metadata.context.run_context, **changes)
    monkeypatch.setattr(RunContext, "for_training", classmethod(lambda cls, *args, **kwargs: observed))
    with pytest.raises(ValueError, match="source/package/recipe/account/run differs"):
        bootstrap(native_metadata)


def test_recipe_source_bytes_must_match_reviewed_context(native_metadata):
    native_metadata.options["recipe_path"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source/package/recipe/account/run differs"):
        bootstrap(native_metadata)


@pytest.mark.parametrize("change", ["permit", "authority", "history", "synthetic"])
def test_input_role_bindings_cannot_self_authorize(native_metadata, change):
    metadata = native_metadata
    if change == "synthetic":
        metadata.binding["scope"] = "CPU_SYNTHETIC_ONLY"
    elif change == "permit":
        metadata.binding["train"]["permit"]["version_identity"] = "9" * 64
    elif change == "authority":
        metadata.binding["train"]["authority"]["protected"] = True
    else:
        metadata.binding["development"]["role_history"]["usage"] = "unused"
    metadata.binding_path.write_text(json.dumps(metadata.binding), encoding="utf-8")
    metadata.recipe = replace(metadata.recipe, dataset=replace(metadata.recipe.dataset,
        manifest_sha256=sha256_file(metadata.binding_path)))
    metadata.context = replace(metadata.context, recipe_identity=metadata.recipe.identity)
    with pytest.raises(ValueError, match="ROLE_BLOCKED|actual accepted indexed"):
        bootstrap(metadata)


def test_synthetic_recipe_cannot_enter_native_path(qualification_case, native_metadata):
    native_metadata.recipe = qualification_case.recipe
    native_metadata.context = replace(native_metadata.context, recipe_identity=qualification_case.recipe.identity)
    with pytest.raises(ValueError, match="exact real-input recipe"):
        bootstrap(native_metadata)


def test_reviewed_context_cannot_switch_admitted_interfaces(native_metadata):
    metadata = native_metadata
    other = next(item for item in INTERFACES if item != metadata.recipe.interface)
    metadata.context = replace(metadata.context, run_context=replace(metadata.context.run_context,
        family_name=other[0], family_version=other[1]))
    with pytest.raises(ValueError, match="interface differs from reviewed context"):
        bootstrap(metadata)


def test_native_path_rejects_decoded_synthetic_data_before_gpu(
        native_metadata, qualification_case, monkeypatch):
    from molgap import family_training_data, v4_runtime
    calls = []
    def load(*args):
        calls.append("generated-cpu-fixture")
        return qualification_case.data
    monkeypatch.setattr(runner, "_loaded_source", lambda *args: "generated-source-placeholder")
    monkeypatch.setattr(v4_runtime, "inspect_frozen_state_artifact", lambda *args, **kwargs: {})
    monkeypatch.setattr(family_training_data, "load_training_data", load)
    with pytest.raises(ValueError, match="decoded input differs from actual accepted train-only 100K"):
        bootstrap(native_metadata)
    assert calls == ["generated-cpu-fixture"]


@pytest.mark.parametrize("violation", ["identity", "deadline", "ceiling"])
def test_issued_admission_enforces_recipe_deadline_and_shared_six_update_ceiling(
        native_metadata, monkeypatch, violation):
    metadata = native_metadata
    began = time.perf_counter()
    admission = owner._QualificationAdmission(metadata.context, metadata.recipe,
        began=began, issuer=owner._ISSUER)
    recipe = metadata.recipe
    if violation == "identity":
        recipe = replace(recipe, learning_rate=recipe.learning_rate * 2)
    elif violation == "deadline":
        monkeypatch.setattr(owner.time, "perf_counter", lambda: began + metadata.context.max_wall_seconds)
    else:
        for _ in range(6):
            admission.before_update(recipe)
    before = admission.updates
    with pytest.raises((ValueError, TimeoutError), match="different recipe|wall budget|ceiling"):
        admission.before_update(recipe)
    assert admission.updates == before == (6 if violation == "ceiling" else 0)


@pytest.mark.parametrize("admission", [True, False, {}, SimpleNamespace(updates=0)])
def test_forged_kernel_admission_rejected_before_model_or_gpu(
        qualification_case, before_payload, admission):
    with pytest.raises(TypeError, match="process-local bounded qualification admission"):
        FamilyKernel(qualification_case.recipe, qualification_case.initial,
            device="cuda:0", qualification=admission)


def test_forged_issuer_and_native_admission_cannot_be_deserialized(native_metadata):
    with pytest.raises(TypeError, match="issued by the bound adapter"):
        owner._QualificationAdmission(native_metadata.context, native_metadata.recipe,
            began=time.perf_counter(), issuer=True)
    with pytest.raises(TypeError, match="Reviewed Modern2DQualificationContext"):
        owner.bootstrap_qualification(True, began=time.perf_counter(), **native_metadata.options)
    restored = owner.Modern2DQualificationContext.from_dict(
        json.loads(json.dumps(native_metadata.context.to_dict())))
    assert restored == native_metadata.context
    assert restored.identity == native_metadata.context.identity


def test_ordinary_preflight_training_and_execute_remain_preview_blocked(
        qualification_case, before_payload, monkeypatch, tmp_path, payload):
    case = qualification_case
    options = dict(recipe_path=tmp_path / "absent.json", spec=None, package_dir=None,
        expected_package_identity=None, arm_id=None, mode="reference", initial_state_path=None,
        input_root=None, account=None, run_reference=None, trajectory_id=None)
    for recipe in (case.preview, case.recipe):
        monkeypatch.setattr(runner, "read_recipe", lambda path: ({}, recipe))
        with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
            runner.run_screen_preflight(output=tmp_path / "ordinary-preflight", **options)
        with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
            runner.run_screen_arm(output=tmp_path / "ordinary-training", **options)
        spec = preview_spec(deepcopy(payload), recipe)
        with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
            validate_execution_plan(spec, [])
        for phase in ("preflight", "train"):
            with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
                execute_training_phase(spec=spec, job={"arm_id": recipe.family}, phase=phase,
                    source_root=tmp_path, package_dir=tmp_path,
                    expected_package_identity="e" * 64, input_root=tmp_path,
                    output=tmp_path / phase, account="fixture", run_reference="fixture/run")
        with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
            FamilyKernel(recipe, case.initial, device="cuda:0")
    with pytest.raises(ValueError, match="cannot enter formal training"):
        runner.run_screen_arm(output=tmp_path / "qualification-training", qualification=True)


@pytest.mark.parametrize("change", ["real_data", "development", "untyped_train", "real_recipe"])
def test_direct_synthetic_route_rejects_mixed_inputs(
        qualification_case, before_payload, tmp_path, change):
    case = qualification_case
    recipe, data = case.recipe, case.data
    if change == "real_data":
        data = replace(data, synthetic=False)
    elif change == "development":
        data = replace(data, development=[data.train[0]])
    elif change == "untyped_train":
        data = replace(data, train=iter(data.train))
    else:
        recipe = replace(recipe, train_rows=100000,
            dataset=replace(recipe.dataset, manifest="accepted.json"))
    with pytest.raises(ValueError, match="bounded generated train-only CPU fixtures"):
        owner.run_synthetic_modern2d_qualification(recipe=recipe, data=data,
            initial_state_path=case.initial, output=tmp_path / "mixed")


def test_synthetic_output_requires_fresh_scratch(qualification_case, before_payload, tmp_path):
    (tmp_path / "sentinel").write_text("preserve", encoding="utf-8")
    case = qualification_case
    with pytest.raises(ValueError, match="fresh scratch"):
        owner.run_synthetic_modern2d_qualification(recipe=case.recipe, data=case.data,
            initial_state_path=case.initial, output=tmp_path)
    assert (tmp_path / "sentinel").read_text() == "preserve"
