"""Synthetic/metadata checks only; no molecular forward or cloud qualification."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import time

import pytest
import torch

from molgap.family_pair_release import resource_estimate, check_same_run_plan
from molgap.family_training_data import DatasetSelection, resolve_pinned_manifest
from molgap.training_reproducibility import atomic_json, sha256_file


def test_resource_profile_rejects_memory_time_and_invalid_exposure():
    args = dict(train_step_seconds=.01, forward_seconds=.005, reserved_bytes=10, total_bytes=100)
    first = resource_estimate(**args)
    assert resource_estimate(**args, completed_passes=30)["estimated_remaining_seconds"] == first["estimated_remaining_seconds"] / 2
    for update in ({"reserved_bytes": 86}, {"train_step_seconds": 1}, {"forward_seconds": float("nan")},
                   {"completed_passes": -1}, {"completed_passes": 60}):
        with pytest.raises(ValueError):
            resource_estimate(**{**args, **update})


def test_shared_source_inventory_retains_fixed_pickle_dependency():
    from molgap.experiment_source_inventory import COMMON_SOURCE_FILES
    assert "src/molgap/pcqm_wedge.py" in COMMON_SOURCE_FILES


def test_mount_resolution_is_hash_bound_unique_and_does_not_shadow_direct_path(tmp_path):
    directory = tmp_path / "actual-mount"
    directory.mkdir()
    atomic_json(directory / "manifest.json", {"fixture": 1})
    selection = DatasetSelection("logical/manifest.json", sha256_file(directory / "manifest.json"), True)
    assert resolve_pinned_manifest(tmp_path, selection) == directory / "manifest.json"
    (tmp_path / "other").mkdir()
    (tmp_path / "other/manifest.json").write_bytes((directory / "manifest.json").read_bytes())
    with pytest.raises(ValueError, match="uniquely"):
        resolve_pinned_manifest(tmp_path, selection)
    (tmp_path / "logical").mkdir()
    atomic_json(tmp_path / "logical/manifest.json", {"fixture": 2})
    with pytest.raises(ValueError, match="SHA"):
        resolve_pinned_manifest(tmp_path, selection)


def test_two_live_ema_metrics_ev_trace_and_real_ten_pass_transport(tmp_path):
    from test_family_training_session import make_session, events
    from test_family_training_runner import context
    from molgap.family_training_runner import build_screen_recipe
    from molgap.experiment_family_workflow import tensor_digest
    from molgap.family_training_session import ContractSchedule
    from molgap.family_training_outputs import SessionOutputs
    from molgap.experiment_resume import restore_resume_bundle, validate_resume_bundle
    session = make_session("gptrans_t", development=True, schedule_kind="cosine_epoch", events=events())
    recipe = replace(session.recipe, mode="degree_bond_local_ema999", optimizer_steps=24,
                     schedule=replace(session.recipe.schedule, total_units=12))
    session.recipe = session.kernel.recipe = session.loaders.recipe = recipe
    session.schedule = ContractSchedule(session.kernel)
    record = build_screen_recipe(recipe.mode, configuration=recipe, family=("gptrans_t", "2"),
        train_role_identity="3" * 64, development_role_identity="4" * 64,
        source_idx_sha256=tensor_digest(torch.arange(257, 260), role="source_idx"),
        target_sha256=tensor_digest(torch.arange(3).float(), role="target"))
    contract = tmp_path / "contract.json"
    atomic_json(contract, record)
    ctx = context("gptrans_t", contract)
    trajectory = {"trajectory_id": "synthetic-cycle", "record_mode": "prospective",
        "state_at_start": {"source_commit": ctx.source_commit, "source_config_identity": ctx.arm_identity},
        "actions": [{"action_id": "A001", "run_ids": [ctx.run_reference], "attempt_ids": ["v1"]}],
        "decision": {"outcome": "ACTIVE"}}
    outputs = SessionOutputs(tmp_path / "execution/reference", ctx, contract=contract,
        adapter="gptrans-session-v1", trajectory_id=trajectory["trajectory_id"],
        train_role_identity=record["train_role_identity"], resume_trajectory=json.dumps(trajectory).encode())
    session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    progress = session.run(should_pause=lambda: session.progress.pass_index == 10)
    assert progress.status == "paused" and progress.optimizer_step == 20
    assert not (outputs.root / "output_manifest.json").exists()
    rows = outputs.stage.recorder.record["observations"]
    assert len(rows) == 10 and all(r["live_dev_metric"] is not None and r["ema_dev_metric"] is not None for r in rows)
    assert outputs.stage.recorder.record["metric_semantics"]["live_train_metric"]["unit"] == "eV"
    assert all(r["checkpoint_identity"] and r["cumulative_device_time_seconds"] > 0 for r in rows)
    bundle = tmp_path / "execution/checkpoint_chunks/reference/pass_10"
    assert validate_resume_bundle(bundle)["status"] == "TRANSPORT_VERIFIED"
    restored = tmp_path / "restored"
    restore_resume_bundle(bundle, restored)
    assert sha256_file(restored / "last_checkpoint.pt") == sha256_file(outputs.root / "last_checkpoint.pt")


def test_shared_environment_profile_is_explicit_and_keeps_legacy_default(tmp_path, monkeypatch):
    from molgap import kaggle_python_environment as owner
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps({"packages": {"torch": "2.7.1+cu126"}}))
    monkeypatch.setattr(owner.subprocess, "run", run)
    _, result = owner.prepare_python(tmp_path / "env", deadline=time.time() + 30,
                                     source_root=tmp_path, profile="gptrans-shared")
    assert "torch==2.7.1+cu126" in calls[2]
    assert "https://download.pytorch.org/whl/cu126" in calls[2]
    assert "molgap.family_training_runner" in calls[3][2]
    assert "torch==2.4.1+cu121" in owner.REQUIREMENTS
    assert result["status"] == "IMPORTS_QUALIFIED_NOT_GPU_CALIBRATED"


@pytest.fixture
def frozen_local_pair(tmp_path):
    from molgap.experiment_spec import ExperimentSpec
    root = Path(__file__).resolve().parents[1]
    base = Path("experiments/pcqm_gptrans_cycle_pair_100k")
    if not (root / base / "gpu/acceptance_plan.json").exists():
        pytest.skip("Local prospective pair metadata is not present")
    plan = json.loads((root / base / "gpu/acceptance_plan.json").read_bytes())
    initials = {a["arm_id"]: root / a["initial_state"]["path"] for a in plan["arms"]}
    if not all(p.exists() for p in initials.values()):
        pytest.skip("Ignored local initial artifacts are not present")
    pointers = [plan[k] for k in ("authority", "audit", "cache_acceptance")]
    for arm in plan["arms"]:
        pointers.extend([arm["contract"], arm["target_manifest"]])
    for pin in pointers:
        path = tmp_path / pin["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((root / pin["path"]).read_bytes())
    spec = ExperimentSpec.from_json((root / base / "gpu/experiment_spec.json").read_text())
    return spec, tmp_path, plan, initials


def test_same_run_plan_never_labels_future_reference_accepted(frozen_local_pair):
    spec, root, plan, initials = frozen_local_pair
    result = check_same_run_plan(spec, root, plan, initial_states=initials)
    assert result["reference_status"] == "PLANNED_NOT_ACCEPTED"
    assert result["scientific_acceptance"] == "NOT_EVALUATED"


@pytest.mark.parametrize("change", ["authority", "extra_arm", "unaccepted_cache", "nonzero_cycle"])
def test_same_run_release_fails_closed(frozen_local_pair, monkeypatch, change):
    import molgap.family_pair_release as owner
    spec, root, plan, initials = frozen_local_pair
    if change == "extra_arm":
        plan["arms"].append(deepcopy(plan["arms"][0]))
    elif change in {"authority", "unaccepted_cache"}:
        pin = plan["authority" if change == "authority" else "cache_acceptance"]
        path = root / pin["path"]
        data = json.loads(path.read_bytes())
        data["user_authorized" if change == "authority" else "accepted"] = False
        atomic_json(path, data)
        pin["sha256"] = sha256_file(path)
    else:
        real = owner.read_frozen_state_artifact
        def mutate(path, **kwargs):
            state, metadata = real(path, **kwargs)
            if "cycle_embedding.weight" in state:
                state["cycle_embedding.weight"][1, 0] = 1
            return state, metadata
        monkeypatch.setattr(owner, "read_frozen_state_artifact", mutate)
    with pytest.raises(ValueError):
        check_same_run_plan(spec, root, plan, initial_states=initials)


def test_deferred_audit_refuses_unaccepted_training(tmp_path):
    from molgap.family_pair_acceptance import bind_deferred_audit
    path = tmp_path / "acceptance.json"
    atomic_json(path, {"accepted": False})
    with pytest.raises(ValueError, match="independently accepted"):
        bind_deferred_audit(acceptance_path=path, recipe_path=tmp_path / "absent", arm_id="x")


def test_audit_budget_stops_before_any_prediction(tmp_path, monkeypatch):
    from molgap.family_cycle_audit import retain_predictions
    from torch_geometric.data import Data, Batch
    from torch_geometric import loader
    batch = Batch.from_data_list([Data(source_idx=torch.tensor([100000]))])
    monkeypatch.setattr(loader, "DataLoader", lambda *_args, **_kw: [batch])
    model = SimpleNamespace(eval=lambda: None)
    with pytest.raises(TimeoutError, match="budget"):
        retain_predictions(model, [], start=100000, mean=0, std=1, output=tmp_path,
                           device="cpu", deadline=time.perf_counter() - 1)
    assert not list(tmp_path.iterdir())


def test_audit_loader_opens_only_development_shards(tmp_path, monkeypatch):
    import molgap.family_cycle_audit as audit
    import molgap.pcqm_fixed_records as records
    import molgap.packed_graph_dataset as packed
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {"roles": {"development": {
        "source_idx_start": 500000, "source_idx_stop": 550000, "rows": 50000}}})
    dev = tmp_path / "dev.pt"
    dev.write_bytes(b"synthetic-non-pickle")
    monkeypatch.setattr(audit, "resolve_pinned_manifest", lambda *args: manifest)
    monkeypatch.setattr(audit, "graph_records", lambda *args: [
        {"role": "train", "file": "DO_NOT_OPEN_TRAIN.pt"},
        {"role": "development", "file": "dev.pt", "sha256": sha256_file(dev),
         "source_idx_min": 500000, "rows": 50000}])
    opened = []
    class SyntheticPacked:
        def __init__(self, path, **kw):
            opened.append(Path(path).name)
            self._data = SimpleNamespace(source_idx=torch.arange(500000, 550000))
        def __len__(self):
            return 50000
    monkeypatch.setattr(packed, "PackedGraphDataset", SyntheticPacked)
    dataset, start = audit.development_view(tmp_path, audit.FIXED_500K_SHA)
    assert len(dataset) == 50000 and start == 500000 and opened == ["dev.pt"]


def test_audit_reproduction_failure_leaves_500k_unopened_and_labels_unscored(tmp_path, monkeypatch):
    import molgap.family_cycle_audit as audit
    import molgap.gptrans_cycle_pair as models
    import molgap.gptrans_execution_backend as backend
    checkpoint, original = tmp_path / "selected.pt", tmp_path / "original.pt"
    state = {"fixture": torch.tensor([1.])}
    torch.save({"model": state}, checkpoint)
    torch.save({"source_idx": torch.arange(100000, 150000), "prediction_eV": torch.ones(50000)}, original)
    class NoForwardModel:
        def load_state_dict(self, value, strict):
            assert strict
        def state_dict(self):
            return state
        def to(self, *args):
            return self
        def eval(self):
            return self
    monkeypatch.setattr(models, "construct", lambda *args: NoForwardModel())
    monkeypatch.setattr(backend, "bind_model_layout", lambda *args: None)
    monkeypatch.setattr(audit, "configure_fp32_determinism", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "Tesla T4")
    roles = []
    def development(*args):
        roles.append(args[1])
        assert len(roles) == 1, "500K must stay unopened"
        return object(), 100000
    monkeypatch.setattr(audit, "development_view", development)
    monkeypatch.setattr(audit, "retain_predictions", lambda *args, **kw: [])
    monkeypatch.setattr(audit, "read_retained_predictions", lambda *args: torch.zeros(50000))
    binding = {"training_independently_accepted": True, "mode": models.PARENT,
        "audit_kind": "fixed500k-full50k-NO_TRAIN", "batch_size": 128, "tf32": False,
        "maximum_wall_seconds": 1800, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "original_predictions": str(original), "original_predictions_sha256": sha256_file(original),
        "sidecar_manifest": "manifest.json", "sidecar_manifest_sha256": "1" * 64,
        "target_mean_eV": 0., "target_std_eV": 1.}
    with pytest.raises(ValueError, match="reproduction failed"):
        audit.run_bound_arm(binding=binding, input_root=tmp_path, output=tmp_path / "audit")
    assert roles == [audit.FIXED_100K_SHA]
