"""Synthetic CPU contracts/output fixtures; not native or scientific qualification."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest
import torch

from molgap.experiment_execution import build_family_recipe, training_adapter
from molgap.experiment_family_workflow import RunContext, inspect_output, tensor_digest
from molgap.experiment_package import _name
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_runner import INTERFACES, PROFILES, family_arm_bindings, read_recipe
from molgap.research_memory.trace import load_canonical_trace
from molgap.training_reproducibility import sha256_file, validate_accelerator_request
from test_family_training_session import events, make_session
from test_experiment_spec import payload


def contract_file(tmp_path, session):
    data, recipe = session.loaders.data, session.recipe
    source = torch.cat([graph.source_idx for graph in data.development]) if data.development else None
    target = torch.cat([graph.y for graph in data.development]) if data.development else None
    record = build_family_recipe(INTERFACES[recipe.family], configuration=recipe,
        train_role_identity="3" * 64, development_role_identity="4" * 64 if data.development else None,
        source_idx_sha256=tensor_digest(source, role="source_idx") if source is not None else None,
        target_sha256=tensor_digest(target, role="target") if target is not None else None)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return path, record


def context(family, path):
    return RunContext("synthetic-question", "synthetic-run", "reference", "a" * 64, "b" * 64,
        *INTERFACES[family], "c" * 40, "d" * 64, "e" * 64, sha256_file(path),
        "local", "fixture", "fixture/run", None)


@pytest.mark.parametrize("platform", ["local", "kaggle", "ims", "scnet", "colab"])
def test_platform_declaration_does_not_change_the_training_owner(payload, platform):
    payload["platform"]["name"] = platform
    assert ExperimentSpec(payload).to_dict()["platform"]["name"] == platform
    for family in INTERFACES:
        adapter = training_adapter({"family": {"name": family, "version": INTERFACES[family][1]}, "addons": []})
        assert adapter.module == "molgap.family_training_runner"


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("rows,development", [(100000, True), (500000, True), (3378606, False)])
def test_one_recipe_builder_and_training_adapter_for_each_scale(family, rows, development):
    session = make_session(family, development=development, schedule_kind="cosine_epoch", events=events())
    try:
        steps = rows // 128
        recipe = replace(session.recipe, train_rows=rows, optimizer_steps=steps * 2)
        record = build_family_recipe(INTERFACES[family], configuration=recipe,
            train_role_identity="3" * 64, development_role_identity="4" * 64 if development else None,
            source_idx_sha256="5" * 64 if development else None, target_sha256="6" * 64 if development else None)
        assert record["acceptance_requirements"]["sample_presentations"] == steps * 128 * 2
        adapter = training_adapter({"family": {"name": family, "version": INTERFACES[family][1]}, "addons": []})
        assert adapter.module == "molgap.family_training_runner"
        assert adapter.artifact_adapter == PROFILES[family]
        assert "platform" not in recipe.to_dict()
        assert family_arm_bindings(recipe)["transform"]["sha256"] != family_arm_bindings(
            replace(recipe, target_mean_eV=recipe.target_mean_eV + .01))["transform"]["sha256"]
    finally:
        session.close()


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("development", [False, True])
@pytest.mark.parametrize("process_timing", [False, True])
def test_actual_session_output_translation_and_cpu_mechanical_inspection(
        tmp_path, family, development, process_timing, monkeypatch):
    import molgap.family_training_policy as family_policy
    base_mode_policy = family_policy.mode_policy

    def timing_policy(policy_family, mode):
        return replace(base_mode_policy(policy_family, mode), process_timing=process_timing)

    monkeypatch.setattr(family_policy, "mode_policy", timing_policy)
    session = make_session(family, development=development, schedule_kind="cosine_step", events=events())
    path, record = contract_file(tmp_path, session)
    ctx = context(family, path)
    outputs = SessionOutputs(tmp_path / "output", ctx, contract=path, adapter=PROFILES[family],
        trajectory_id="synthetic-trajectory", train_role_identity=record["train_role_identity"])
    session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    session.batch_finished = outputs.batch_finished
    published_checkpoints = {}

    def publish_checkpoint(progress, state_builder):
        outputs.checkpoint_from_builder(progress, state_builder)
        published_checkpoints[progress.optimizer_step] = sha256_file(outputs.root / "last_checkpoint.pt")

    session.checkpoint_builder = publish_checkpoint
    update_learning_rates = []
    kernel_step = session.kernel.step

    def observe_update_rate(batch, *, check_finite):
        update_learning_rates.append(tuple(group["lr"] for group in session.kernel.optimizer.param_groups))
        return kernel_step(batch, check_finite=check_finite)

    session.kernel.step = observe_update_rate
    outputs.logger.start(session.progress)
    progress = session.run()
    schedule = session.recipe.schedule
    expected_update_lrs = [schedule.minimum_lr +
        (session.recipe.learning_rate - schedule.minimum_lr) * .5 *
        (1 + math.cos(math.pi * unit / schedule.total_units))
        for unit in range(session.recipe.optimizer_steps)]
    assert all(len(groups) == 1 for groups in update_learning_rates)
    assert [groups[0] for groups in update_learning_rates] == pytest.approx(expected_update_lrs)
    runtime = {"platform": ctx.platform, "account": ctx.account, "precision": "fp32",
        "source_commit": ctx.source_commit, "source_archive_sha256": ctx.source_archive_sha256}
    costs = [{"metric": "wall_seconds", "unit": "seconds", "status": "measured", "value": 1.,
        "semantics": "process_wall", "hardware": "synthetic-cpu"},
        {"metric": "device_seconds", "unit": "seconds", "status": "missing", "value": None,
        "semantics": "allocated_device", "hardware": "synthetic-cpu"}]
    report = outputs.complete(progress, runtime=runtime, costs=costs)
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert report["scientific_acceptance"] == report["replay_readiness"] == "NOT_EVALUATED"
    assert (outputs.root / "development_predictions.pt").is_file() == development
    assert ("development_mae_eV" in report["observed"]) == development
    telemetry = json.loads((outputs.root / "training_progress.json").read_text(encoding="utf-8"))
    assert telemetry["optimizer_steps"] == telemetry["checkpoint_optimizer_steps"] == progress.optimizer_step
    assert telemetry["status"] == progress.status
    if development:
        assert telemetry["best_development_mae_eV"] == pytest.approx(report["observed"]["development_mae_eV"])
    else:
        assert telemetry["development_mae_eV"] is telemetry["best_development_mae_eV"] is None

    trace = load_canonical_trace(outputs.root / "canonical_trace.json")
    observations = [row for row in trace["observations"] if row["event"] == "observation"]
    steps_per_pass = session.recipe.loop_plan.steps_per_pass
    full_pass_rows = [row for row in observations if row["optimizer_step"] % steps_per_pass == 0]
    assert len(full_pass_rows) == progress.pass_index
    assert trace["device_time_semantics"] == ("sum_over_devices" if process_timing else None)
    for row in full_pass_rows:
        step = row["optimizer_step"]
        assert row["learning_rate"] == pytest.approx(update_learning_rates[step - 1][0])
        assert row["checkpoint_identity"] == published_checkpoints[step]
        assert len(row["checkpoint_identity"]) == 64
        assert row["wall_time_seconds"] > 0
        assert row["cumulative_wall_time_seconds"] > 0
        if process_timing:
            assert row["device_time_seconds"] > 0
            assert row["cumulative_device_time_seconds"] > 0
        else:
            assert row["device_time_seconds"] is None
            assert row["cumulative_device_time_seconds"] is None

    final_row = observations[-1]
    if (final_row["optimizer_step"] == progress.optimizer_step and
            final_row["optimizer_step"] % steps_per_pass == 0):
        assert final_row["checkpoint_identity"] == sha256_file(outputs.root / "last_checkpoint.pt")
    else:
        assert not development and progress.next_batch != 0
        assert final_row["epoch_or_pass"] == pytest.approx(progress.optimizer_step / steps_per_pass)
    manifest_path = outputs.root / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resume_path = outputs.root / manifest["artifacts"]["resume"]["path"]
    state = torch.load(resume_path, map_location="cpu", weights_only=True)
    state["sample_presentations"] += 1
    torch.save(state, resume_path)
    manifest["artifacts"]["resume"]["sha256"] = sha256_file(resume_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    blocked = inspect_output(outputs.root, context=ctx, expected=record["acceptance_requirements"])
    assert blocked["status"] == "BLOCKED"
    assert any("exposure" in error for error in blocked["blockers"])


def test_contract_rejects_counter_changes_and_unknown_fields(tmp_path):
    session = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=events())
    try:
        path, original = contract_file(tmp_path, session)
        assert read_recipe(path)[1] == session.recipe
        for change in ("counter", "unknown", "development"):
            record = deepcopy(original)
            if change == "counter":
                record["acceptance_requirements"]["sample_presentations"] += 1
            elif change == "unknown":
                record["callback"] = "external:execute"
            else:
                record["development_role_identity"] = "fake"
            path.write_text(json.dumps(record), encoding="utf-8")
            with pytest.raises(ValueError):
                read_recipe(path)
    finally:
        session.close()


def test_reviewed_python_io_source_does_not_allow_artifact_directories():
    assert _name("src/molgap/family_training_outputs.py") == "src/molgap/family_training_outputs.py"
    assert _name("src/molgap/other_outputs.py") == "src/molgap/other_outputs.py"
    for path in ("experiments/example/outputs/model.py", "data/family_training_outputs.py"):
        with pytest.raises(ValueError):
            _name(path)


@pytest.mark.parametrize("requested,observed,hip", [
    ("T4", "Tesla T4", False), ("NvidiaTeslaT4", "NVIDIA T4", False),
    ("A100", "NVIDIA A100-SXM4-40GB", False),
    ("RTX 5060", "NVIDIA GeForce RTX 5060", False),
    ("DCU", "opaque-device-name", True),
])
def test_resource_matching_is_metadata_only(requested, observed, hip):
    validate_accelerator_request(requested, observed, hip_runtime=hip)


@pytest.mark.parametrize("requested,observed,hip", [
    ("T4", "NVIDIA A100-SXM4-40GB", False), ("A100", "Tesla T4", False),
    ("RTX 5060", "NVIDIA GeForce RTX 5060 Ti", False),
    ("GPU", "GPU", False), ("DCU", "NVIDIA A100-SXM4-40GB", False),
    ("NVIDIA GPU", "NVIDIA GPU", False), ("NVIDIA", "NVIDIA", False),
    ("---", "---", False), ("CUDA", "CUDA", False),
    ("ACCELERATOR", "ACCELERATOR", False), ("T4", "GPU", False),
    ("DCU", "GPU", True), ("DCU", "NVIDIA", True),
])
def test_wrong_allocation_cannot_be_qualified(requested, observed, hip):
    with pytest.raises(ValueError):
        validate_accelerator_request(requested, observed, hip_runtime=hip)
