"""Explicit Modern2D previews reuse the actual Session; generated CPU data only."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from importlib import import_module
import json

import pytest
import torch

from molgap.experiment_execution import (
    MODERN2D_PREVIEW_PROFILE, SESSION_INTERFACES, TRAINING_ADAPTERS,
    build_family_recipe, graph_training_adapter, validate_execution_plan,
    validate_training_registry, workflow_capabilities,
)
from molgap.experiment_family_artifacts import artifact_adapter
from molgap.experiment_family_workflow import RunContext, inspect_output, tensor_digest
from molgap.experiment_spec import ExperimentSpec, FAMILIES
from molgap.family_execution_profile import FamilyExecutionProfiler, ProfileConfig
from molgap.family_training_data import DatasetSelection, TrainingData
from molgap.family_training_kernel import FamilyKernel
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_runner import family_arm_bindings, read_recipe, validate_screen_recipe
from molgap.family_training_session import FamilyTrainingSession
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.training_reproducibility import atomic_torch_save, configure_fp32_determinism, sha256_file
from molgap.v4_runtime import model_state_sha256
from modern2d_fixtures import graph
from test_experiment_spec import arm, payload
from test_family_training_session import assert_equal


INTERFACES = (("ppgt_small", "2"), ("esa_small", "3"))


def preview_config(family):
    if family == "ppgt_small":
        return dict(hidden=8, heads=2, blocks=1, pair_dim=4, pe_mlp_hidden=8,
            stem_blocks=1, rrwp_steps=2, spe_bases=1, attention_dropout=.1,
            stochastic_depth=.1, max_nodes=8, max_edges=16)
    return dict(hidden=8, heads=2, layer_types=("M", "S", "P"), graph_dim=8,
        mlp_hidden=16, seeds=2, max_nodes=8, max_edges=16)


def make_preview_session(tmp_path, family, version, checkpoints, *, manifest_sha256="a" * 64):
    configure_fp32_determinism(42)
    torch.set_num_threads(1)
    module = import_module(TRAINING_ADAPTERS[(family, version)].model_factory.split(":")[0])
    config = preview_config(family)
    initial = module.make_model({"model_config": config})
    path = tmp_path / (family + "-" + version + "-initial.pt")
    atomic_torch_save(path, initial.state_dict())
    selection = DatasetSelection("synthetic/modern2d_preview.json", manifest_sha256, True)
    recipe = FamilyRecipe(family, "reference", 42, model_state_sha256(initial), selection,
        256, 3, 4, .001, .01, ScheduleSpec("cosine_step", 4), "best-development-live",
        "torch-pass-randperm", checkpoint_every=1, target_mean_eV=3., target_std_eV=2.,
        finite_check_every=1, model_config=config, family_version=version,
        torch_version=str(torch.__version__).split("+", 1)[0])
    train_ids = tuple(2 * index + 1 for index in range(256))
    dev_ids = (2001, 2005, 2011)
    train = [graph(1 if index % 5 == 0 else 2, source) for index, source in enumerate(train_ids)]
    development = [graph(nodes, source) for nodes, source in zip((1, 2, 3), dev_ids)]
    data = TrainingData(train, development, selection.manifest_sha256, 256, 3, None, None,
        3., 2., synthetic=True, train_source_indices=train_ids,
        development_source_indices=dev_ids, indexed=True)
    loaders = FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0, development_workers=0))
    kernel = FamilyKernel(recipe, path, device="cpu")
    session = FamilyTrainingSession(kernel, loaders,
        checkpoint=lambda progress, state: checkpoints.append(deepcopy(state)),
        selected=lambda *args: None, pass_finished=lambda *args: None)
    return session, recipe, data, path


def contract_file(tmp_path, session):
    data, recipe = session.loaders.data, session.recipe
    record = build_family_recipe(recipe.interface, configuration=recipe,
        train_role_identity="3" * 64, development_role_identity="4" * 64,
        source_idx_sha256=tensor_digest(torch.tensor(data.development_source_indices), role="source_idx"),
        target_sha256=tensor_digest(torch.cat([item.y for item in data.development]), role="target"))
    path = tmp_path / (recipe.family + "-contract.json")
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return path, record


def preview_spec(payload, recipe):
    declaration = arm(recipe.family)
    contract = FAMILIES[recipe.interface]
    declaration["family"]["version"] = recipe.interface[1]
    declaration["data"]["feature_schema"] = contract.feature_schema
    declaration["data"]["dataset"]["sha256"] = recipe.dataset.manifest_sha256
    declaration["initialization"]["state_sha256"] = recipe.initialization_sha256
    bindings = family_arm_bindings(recipe)
    declaration["base"] = bindings["base"]
    declaration["training"]["recipe"]["name"] = contract.recipe
    for key in ("sampler", "transform", "objective"):
        declaration["training"][key] = bindings[key]
    for role in declaration["data"]["roles"]:
        role["usage_sha256"] = ("3" if role["role"] == "train" else "4") * 64
    payload["arms"] = [declaration]
    return ExperimentSpec(payload)


@pytest.mark.parametrize("family,version", INTERFACES)
def test_preview_actual_session_four_steps_exact_serialized_resume_and_indexed_eval(tmp_path, monkeypatch, family, version):
    from molgap import gptrans_execution_backend
    def forbidden(*args, **kwargs):
        raise AssertionError("Modern2D preview must not bind or prepare GPTrans layout")
    monkeypatch.setattr(gptrans_execution_backend, "bind_model_layout", forbidden)
    monkeypatch.setattr(gptrans_execution_backend, "prepare_layout", forbidden)
    complete, recipe, data, _ = make_preview_session(tmp_path, family, version, [])
    assert recipe.integration_preview and not recipe.cpu_synthetic_only
    assert recipe.batch_size == 128 and not recipe.execution_policy["cpu_layout"]
    assert FamilyRecipe.from_dict(json.loads(json.dumps(recipe.to_dict()))).identity == recipe.identity
    final = complete.run()
    expected = deepcopy(complete.state_dict())
    checkpoints = []
    first, _, _, _ = make_preview_session(tmp_path, family, version, checkpoints)
    assert first.run(should_pause=lambda: True).optimizer_step == 1
    checkpoint = tmp_path / (family + "-paused.pt")
    atomic_torch_save(checkpoint, checkpoints[-1])
    serialized = torch.load(checkpoint, map_location="cpu", weights_only=True)
    resumed, _, _, _ = make_preview_session(tmp_path, family, version, [])
    resumed.load_state_dict(serialized)
    assert resumed.run() == final
    assert final.optimizer_step == 4 and final.sample_presentations == 512
    assert expected["cursor"]["selected_step"] in (2, 4)
    assert_equal(expected, resumed.state_dict())
    readonly, _, _, _ = make_preview_session(tmp_path, family, version, [])
    readonly.load_state_dict(expected)
    before = deepcopy(readonly.state_dict())
    evaluation = readonly.kernel.evaluate(readonly.loaders)
    assert tuple(evaluation["source_idx"].tolist()) == data.development_source_indices
    assert_equal(before, readonly.state_dict())
    readonly.close()


@pytest.mark.parametrize("family,version", INTERFACES)
def test_preview_registry_bindings_and_native_routes_fail_before_io(tmp_path, monkeypatch, payload, family, version):
    from molgap import family_training_runner as runner
    from molgap import family_training_data as data_owner
    validate_training_registry()
    session, recipe, _, initial = make_preview_session(tmp_path, family, version, [])
    try:
        spec = preview_spec(payload, recipe)
        adapter = TRAINING_ADAPTERS[recipe.interface]
        assert adapter.integration_preview and adapter.module == "molgap.family_training_runner"
        assert artifact_adapter(MODERN2D_PREVIEW_PROFILE, recipe.interface).protocol == "family-session"
        capabilities = workflow_capabilities(spec)
        assert not capabilities["registered_training_arms"]
        assert capabilities["execution_qualification"] == "INTEGRATION_PENDING"
        assert "INTEGRATION_PENDING" in capabilities["unsupported_training_arms"][family]
        assert FAMILIES[recipe.interface].feature_schema != FAMILIES[(family, "1")].feature_schema
        with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
            validate_execution_plan(spec, [])
        with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
            FamilyKernel(recipe, initial, device="cuda:0")
        monkeypatch.setattr(runner, "read_recipe", lambda path: ({}, recipe))
        def forbidden(*args, **kwargs):
            raise AssertionError("Native input/runtime work was reached")
        monkeypatch.setattr(data_owner, "load_training_data", forbidden)
        monkeypatch.setattr(torch.cuda, "is_available", forbidden)
        with pytest.raises(ValueError, match="INTEGRATION_PENDING"):
            runner._bootstrap(recipe_path=tmp_path / "absent.json", spec=None,
                expected_package_identity=None, mode="reference", account=None,
                run_reference=None, arm_id=None, package_dir=None, input_root=None,
                initial_state_path=None, trajectory_id=None)
    finally:
        session.close()


@pytest.mark.parametrize("family,version", INTERFACES)
def test_preview_can_declare_real_indexed_contract_without_native_release(tmp_path, family, version):
    session, recipe, data, _ = make_preview_session(tmp_path, family, version, [])
    try:
        for rows in (100000, 500000):
            real = replace(recipe, dataset=DatasetSelection("bindings/accepted_train.json", "b" * 64, True),
                train_rows=rows, development_rows=50000, optimizer_steps=(rows // 128) * 2,
                schedule=ScheduleSpec("cosine_epoch", 2))
            assert real.integration_preview and not real.cpu_synthetic_only
            assert FamilyRecipe.from_dict(json.loads(json.dumps(real.to_dict()))).identity == real.identity
            assert real.execution_policy["native_qualification"] == "INTEGRATION_PENDING"
            fake_real = replace(data, synthetic=False, manifest_sha256=real.dataset.manifest_sha256,
                train_rows=rows, development_rows=50000)
            with pytest.raises(ValueError, match="accepted indexed role datasets"):
                FamilyLoaders(real, fake_real, ExecutionTuning(training_workers=0, development_workers=0))
        with pytest.raises(ValueError, match="synthetic fixtures"):
            replace(recipe, optimizer_steps=6, schedule=ScheduleSpec("cosine_step", 6))
        with pytest.raises(ValueError, match="accepted indexed"):
            replace(recipe, dataset=DatasetSelection("bindings/unqualified.json", "b" * 64, True))
        with pytest.raises(FrozenInstanceError):
            recipe.model_config.hidden = 16
    finally:
        session.close()


@pytest.mark.parametrize("family,version", INTERFACES)
def test_preview_resume_rejects_changed_config_data_and_optimizer_without_mutation(tmp_path, family, version):
    checkpoints = []
    first, recipe, data, initial = make_preview_session(tmp_path, family, version, checkpoints)
    first.run(should_pause=lambda: True)
    retained = checkpoints[-1]
    changed_configs = [replace(recipe, model_config={**asdict(recipe.model_config), "max_nodes": 16}),
        replace(recipe, dataset=replace(recipe.dataset, manifest_sha256="b" * 64))]
    for changed in changed_configs:
        changed_data = replace(data, manifest_sha256=changed.dataset.manifest_sha256)
        target = FamilyTrainingSession(FamilyKernel(changed, initial, device="cpu"),
            FamilyLoaders(changed, changed_data, ExecutionTuning(training_workers=0, development_workers=0)),
            checkpoint=lambda *args: None, selected=lambda *args: None, pass_finished=lambda *args: None)
        before = deepcopy(target.state_dict())
        with pytest.raises(ValueError, match="identity"):
            target.load_state_dict(retained)
        assert_equal(before, target.state_dict())
        target.close()
    target, _, _, _ = make_preview_session(tmp_path, family, version, [])
    before = deepcopy(target.state_dict())
    changed = deepcopy(retained)
    changed["optimizer"]["param_groups"][0]["foreach"] = False
    with pytest.raises(ValueError, match="AdamW"):
        target.load_state_dict(changed)
    assert_equal(before, target.state_dict())
    target.close()


@pytest.mark.parametrize("family,version", INTERFACES)
def test_preview_output_is_mechanical_and_explicitly_integration_pending(tmp_path, payload, family, version):
    session, recipe, _, _ = make_preview_session(tmp_path, family, version, [])
    path, record = contract_file(tmp_path, session)
    spec = preview_spec(payload, recipe)
    validate_screen_recipe(spec, family, record)
    assert read_recipe(path)[1].identity == recipe.identity
    context = RunContext("synthetic-question", "synthetic-run", family, "a" * 64, "b" * 64,
        family, version, "c" * 40, "d" * 64, "e" * 64, sha256_file(path),
        "local", "fixture", "fixture/run", None)
    outputs = SessionOutputs(tmp_path / "output", context, contract=path,
        adapter=MODERN2D_PREVIEW_PROFILE, trajectory_id="synthetic-trajectory",
        train_role_identity=record["train_role_identity"])
    session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    final = session.run()
    runtime = {"platform": context.platform, "account": context.account, "precision": "fp32",
        "source_commit": context.source_commit, "source_archive_sha256": context.source_archive_sha256}
    costs = [{"metric": "wall_seconds", "unit": "seconds", "status": "measured", "value": 1.,
        "semantics": "process_wall", "hardware": "synthetic-cpu"},
        {"metric": "device_seconds", "unit": "seconds", "status": "missing", "value": None,
         "semantics": "allocated_device", "hardware": "synthetic-cpu"}]
    report = outputs.complete(final, runtime=runtime, costs=costs)
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert report["observed"]["integration_status"] == "INTEGRATION_PENDING"
    assert report["observed"]["native_execution_released"] is False
    assert report["scientific_acceptance"] == report["runtime_qualification"] == "NOT_EVALUATED"
    assert inspect_output(outputs.root, context=context, expected=record["acceptance_requirements"])["status"] == report["status"]


@pytest.mark.parametrize("family,version", INTERFACES)
def test_preview_profiler_observes_session_without_qualifying_native(tmp_path, family, version):
    session, _, _, _ = make_preview_session(tmp_path, family, version, [])
    with FamilyExecutionProfiler(session, tmp_path / "profile", ProfileConfig(0, 1)) as profiler:
        session.run(should_pause=lambda: True)
    assert profiler.report["captured_updates"] == 1
    assert profiler.report["integration_status"] == "INTEGRATION_PENDING"
    assert profiler.report["native_execution_released"] is False
    assert not profiler.report["device_events_observed"]


@pytest.mark.parametrize("family,version", INTERFACES)
def test_preview_factory_dispatch_rejects_wrong_versions_and_native_mixture(family, version):
    adapter = TRAINING_ADAPTERS[(family, version)]
    for key in ((family, "99"), (family, "1")):
        with pytest.raises(ValueError, match="Unreviewed"):
            graph_training_adapter(key, model_factory=adapter.model_factory, integration_preview=True)
    with pytest.raises(ValueError, match="Unreviewed"):
        graph_training_adapter((family, version), model_factory=adapter.model_factory,
            integration_preview=True, cpu_synthetic_only=True)
    assert SESSION_INTERFACES[family] == (family, "1")
    assert TRAINING_ADAPTERS[(family, "1")].cpu_synthetic_only
    with pytest.raises(ValueError, match="Preview artifact profile"):
        artifact_adapter(MODERN2D_PREVIEW_PROFILE, (family, "1"))


@pytest.mark.parametrize("family,version", INTERFACES)
def test_explicit_preview_sandbox_never_uses_legacy_family_dispatch(tmp_path, monkeypatch, payload, family, version):
    from molgap import fast_sandbox
    session, recipe, _, _ = make_preview_session(tmp_path, family, version, [])
    spec = preview_spec(payload, recipe)
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview was routed through the historical baseline family sandbox")
    monkeypatch.setattr(fast_sandbox, "run_family_sandbox", forbidden)
    try:
        with pytest.raises(ValueError, match="explicit bounded configuration"):
            fast_sandbox.run_arm_sandbox(spec, family)
    finally:
        session.close()
