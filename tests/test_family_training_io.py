"""Synthetic CPU checks for observed curves, checkpoint IO and capture teardown."""
from dataclasses import replace
import json
import hashlib
from pathlib import Path

import pytest
import torch

from molgap import family_training_outputs as output_owner
from molgap.experiment_family_workflow import _json
from molgap.family_execution_profile import FamilyExecutionProfiler, ProfileConfig
from molgap.family_training_runner import PROFILES
from molgap.training_reproducibility import sha256_file
from test_family_training_runner import contract_file, context
from test_family_training_session import events, make_session
from test_shared_resume_transport import bundle_case
from test_family_training_data import pinned_data


@pytest.mark.parametrize("value", [None, 1, 3, True, 0, -1, "1"])
def test_resume_retention_contract_is_explicit_and_default_bytes_unchanged(tmp_path, value):
    from molgap.experiment_execution import build_family_recipe
    from molgap.family_training_runner import INTERFACES, read_recipe
    session = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=events())
    try:
        options = {"configuration": session.recipe, "train_role_identity": "3" * 64}
        if value is not None and (type(value) is not int or value < 1):
            with pytest.raises(ValueError, match="resume_keep_count"):
                build_family_recipe(INTERFACES[session.recipe.family], **options, resume_keep_count=value)
            return
        baseline = build_family_recipe(INTERFACES[session.recipe.family], **options)
        record = build_family_recipe(INTERFACES[session.recipe.family], **options, resume_keep_count=value)
        if value is None:
            assert record == baseline and "resume_keep_count" not in record
        else:
            assert hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() != hashlib.sha256(
                json.dumps(baseline, sort_keys=True).encode()).hexdigest()
            assert record.pop("resume_keep_count") == value
            assert record == baseline
            record["resume_keep_count"] = value
        path = tmp_path / "retention_contract.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        assert read_recipe(path)[1] == session.recipe
        ctx = context(session.recipe.family, path)
        out = output_owner.SessionOutputs(tmp_path / "out", ctx, contract=path,
            adapter=PROFILES[session.recipe.family], trajectory_id="synthetic-trajectory", train_role_identity="3" * 64)
        assert out.resume_keep_count == value
    finally:
        session.close()


@pytest.mark.parametrize("schedule", ["cosine_step", "cosine_epoch", "warmup_cosine_epoch"])
def test_pass_metrics_separate_used_lr_from_scheduler_next(schedule):
    log = events()
    session = make_session("neural_atom_k1", development=True, schedule_kind=schedule, events=log)
    step, used = session.kernel.step, []
    def record(*args, **kwargs):
        used.append(session.kernel.optimizer.param_groups[0]["lr"])
        return step(*args, **kwargs)
    session.kernel.step = record
    session.run()
    for index, (_, metrics) in enumerate(log["passes"]):
        assert metrics["learning_rates_used"] == [used[index * 2 + 1]]
    assert log["passes"][-1][1]["scheduler_next_learning_rates"] == [session.kernel.optimizer.param_groups[0]["lr"]]


def test_pass_curve_binds_actual_update_lr_and_checkpoint_once(tmp_path, monkeypatch):
    session = make_session("neural_atom_k1", development=True, schedule_kind="cosine_step", events=events())
    used = []
    step = session.kernel.step
    def observed_step(*args, **kwargs):
        used.append(session.kernel.optimizer.param_groups[0]["lr"])
        return step(*args, **kwargs)
    session.kernel.step = observed_step
    contract, record = contract_file(tmp_path, session)
    ctx = context(session.recipe.family, contract)
    outputs = output_owner.SessionOutputs(tmp_path / "output", ctx, contract=contract,
        adapter=PROFILES[session.recipe.family], trajectory_id="synthetic-trajectory",
        train_role_identity=record["train_role_identity"])
    session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    digests = []
    real_digest = output_owner.file_digest
    def digest(path):
        digests.append(Path(path))
        return real_digest(path)
    monkeypatch.setattr(output_owner, "file_digest", digest)
    session.run()
    rows = _json(outputs.root / "canonical_trace.json")["observations"]
    assert [row["learning_rate"] for row in rows] == [used[1], used[3]]
    assert rows[-1]["learning_rate"] != session.kernel.optimizer.param_groups[0]["lr"]
    assert rows[-1]["checkpoint_identity"] == sha256_file(outputs.root / "last_checkpoint.pt")
    assert [(row["optimizer_step"], row["sample_presentations"]) for row in rows] == [(2, 256), (4, 512)]
    assert len([path for path in digests if path.parent == outputs.pending and path.name == "last_checkpoint.pt"]) == 2
    # One hash of the old durable generation when building the second journal;
    # no extra full checkpoint rehash during synchronous commit finalization.
    assert len([path for path in digests if path == outputs.root / "last_checkpoint.pt"]) == 1


def test_profile_removes_capture_hooks_before_remaining_updates(tmp_path, monkeypatch):
    log = events()
    session = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=log)
    original_forward = session.kernel.model.forward
    original_loader = session.loaders.train
    original_step = session.kernel.step
    seen = []
    def step(*args, **kwargs):
        if profile._finished:
            assert session.kernel.model.forward == original_forward
            assert session.loaders.train == original_loader
            assert getattr(session.kernel, "update_observer", None) is None
            seen.append(True)
        return original_step(*args, **kwargs)
    session.kernel.step = step
    with FamilyExecutionProfiler(session, tmp_path / "capture", ProfileConfig(0, 1)) as profile:
        session.run()
    assert len(seen) == 4
    assert profile.report["capture_off_training_spans"] == []
    assert not (tmp_path / "capture/training_spans.json").exists()


def test_resume_payload_publication_does_not_read_whole_checkpoint(bundle_case, tmp_path, monkeypatch):
    from molgap import experiment_resume as transport
    source, _, ctx, trajectory, _ = bundle_case
    original = Path.read_bytes
    def read_bytes(path):
        if path.suffix == ".pt":
            raise AssertionError("Checkpoint payload must stream")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    second = tmp_path / "streamed"
    transport.create_resume_bundle(source, ctx, trajectory, second)
    assert transport.validate_resume_bundle(second, context=ctx)["status"] == "TRANSPORT_VERIFIED"
    transport.create_resume_bundle(source, ctx, trajectory, second)


def test_chunk_retention_deletes_only_proven_duplicate_bundles(bundle_case, tmp_path):
    from molgap import experiment_resume as transport
    source, _, ctx, trajectory, _ = bundle_case
    parent = tmp_path / "chunks"
    for index in (1, 2):
        transport.create_resume_bundle(source, ctx, trajectory, parent / f"pass_{index:02d}")
    report = transport.prune_duplicate_resume_bundles(parent, parent / "pass_02", keep_count=1, context=ctx)
    assert report["removed_duplicates"] == ["pass_01"]
    torch.save({"context": ctx, "model": {"weight": torch.zeros(1)},
                "cursor": {"epoch": 2, "next_batch": 0}, "optimizer_step": 3,
                "sample_presentations": 6}, source / "last_checkpoint.pt")
    transport.create_resume_bundle(source, ctx, trajectory, parent / "pass_03")
    report = transport.prune_duplicate_resume_bundles(parent, parent / "pass_03", keep_count=1, context=ctx)
    assert report["retained_unique"] == ["pass_02"]
    assert (parent / "pass_02/files/last_checkpoint.pt").is_file()


def _chunk_session(tmp_path):
    session = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=events())
    contract, record = contract_file(tmp_path, session)
    record["resume_keep_count"] = 1
    contract.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    ctx = context(session.recipe.family, contract)
    trajectory = {"trajectory_id": "synthetic-trajectory", "record_mode": "prospective",
        "state_at_start": {"source_commit": ctx.source_commit, "source_config_identity": ctx.arm_identity},
        "actions": [{"action_id": "A1", "run_ids": ["synthetic-run:reference"], "attempt_ids": ["attempt1"]}],
        "decision": {"outcome": "ACTIVE"}}
    outputs = output_owner.SessionOutputs(tmp_path / "output", ctx, contract=contract,
        adapter=PROFILES[session.recipe.family], trajectory_id="synthetic-trajectory",
        train_role_identity=record["train_role_identity"], resume_trajectory=trajectory)
    outputs.policy = replace(outputs.policy, chunk_passes=1)
    session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    return session, outputs


@pytest.mark.parametrize("prune_fails", [False, True])
def test_normal_chunk_checkpoint_reuses_previous_bundle_and_isolates_prune(tmp_path, monkeypatch, prune_fails):
    from molgap import experiment_resume as transport
    session, outputs = _chunk_session(tmp_path)
    original_create, calls = transport.create_resume_bundle, []
    def create(source, ctx, trajectory, destination, **options):
        calls.append((destination, options.get("reuse_bundle")))
        return original_create(source, ctx, trajectory, destination, **options)
    monkeypatch.setattr(transport, "create_resume_bundle", create)
    if prune_fails:
        def fail(*args, **kwargs):
            raise PermissionError("synthetic optional cleanup failure")
        monkeypatch.setattr(transport, "prune_duplicate_resume_bundles", fail)
    result = session.run()
    assert result.status == "complete"
    assert [(path.name, previous.name if previous else None) for path, previous in calls] == [
        ("pass_01", None), ("pass_02", "pass_01")]
    first, second = [path for path, _ in calls]
    assert (first / "files/training_contract.json").samefile(second / "files/training_contract.json")
    assert not (outputs.root / "training_contract.json").samefile(second / "files/training_contract.json")
    assert transport.validate_resume_bundle(second, context=outputs.context)["status"] == "TRANSPORT_VERIFIED"
    assert outputs._previous_resume_bundle == second
    if prune_fails:
        assert outputs.resume_retention_report["retention_target_met"] is False
        assert "cleanup failure" in outputs.resume_retention_report["retained_unverified"][0]["reason"]
    else:
        assert outputs.resume_retention_report["retained_unique"] == ["pass_01"]


@pytest.mark.parametrize("operation", ["atomic_write", "sync_directory"])
def test_chunk_cleanup_isolation_does_not_hide_checkpoint_commit_failure(tmp_path, monkeypatch, operation):
    session, outputs = _chunk_session(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("synthetic durable commit failure")
    monkeypatch.setattr(output_owner, operation, fail)
    with pytest.raises(OSError, match="durable commit failure"):
        session.run()
    assert outputs._previous_resume_bundle is None


def test_prune_skips_loading_unique_historical_checkpoints(bundle_case, tmp_path, monkeypatch):
    from molgap import experiment_resume as transport
    source, _, ctx, trajectory, _ = bundle_case
    parent = tmp_path / "chunks"
    for index in (1, 2, 3):
        torch.save({"context": ctx, "model": {"weight": torch.tensor([index])},
                    "cursor": {"epoch": index, "next_batch": 0}, "optimizer_step": index,
                    "sample_presentations": index * 2}, source / "last_checkpoint.pt")
        (source / "canonical_trace.json").write_text(json.dumps({"observations": [
            {"event": "observation", "optimizer_step": index, "sample_presentations": index * 2}]}))
        transport.create_resume_bundle(source, ctx, trajectory, parent / f"pass_{index:02d}")
    loaded, original = [], transport.safe_cpu_torch_load
    def load(path):
        loaded.append(Path(path))
        return original(path)
    monkeypatch.setattr(transport, "safe_cpu_torch_load", load)
    report = transport.prune_duplicate_resume_bundles(parent, parent / "pass_03", keep_count=1, context=ctx)
    assert report["retained_unique"] == ["pass_02", "pass_01"]
    assert [path.parent.parent.name for path in loaded] == ["pass_03"]


def test_synthetic_cpu_data_phase_measurements(pinned_data, monkeypatch):
    from molgap import family_training_data as owner
    from molgap.packed_graph_dataset import PackedGraphDataset
    root, selection, _ = pinned_data
    counts = {}
    def counted(label, function):
        def call(*args, **kwargs):
            counts[label] = counts.get(label, 0) + 1
            return function(*args, **kwargs)
        return call
    monkeypatch.setattr(owner, "sha256_file", counted("sha", owner.sha256_file))
    monkeypatch.setattr(PackedGraphDataset, "__init__", counted("decode", PackedGraphDataset.__init__))
    monkeypatch.setattr(owner, "sample_std_compat", counted("stats_std", owner.sample_std_compat))
    data = owner.load_training_data(root, selection)
    assert data.train_rows == 4 and data.development_rows == 2
    assert counts == {"sha": 4, "decode": 2, "stats_std": 1}
