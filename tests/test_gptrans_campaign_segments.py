"""Tiny saved-tensor fault injection; never native GPU/scientific qualification."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molgap import gptrans_campaign_segments as owner
from molgap.experiment_family_workflow import RunContext, tensor_digest
from molgap.family_training_runner import read_recipe, _probe_recipe, _state_fingerprint
from molgap.gptrans_campaign_release import BASE
from molgap.research_memory.trace import canonicalize_trace, file_digest
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import capture_rng_state


REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def retained_state(recipe, step):
    """One parameter with genuine AdamW/session schema and synthetic CUDA bytes."""
    passes, offset = divmod(step, recipe.loop_plan.steps_per_pass)
    rng = capture_rng_state()
    rng["numpy"] = (rng["numpy"][0], rng["numpy"][1].tolist(), *rng["numpy"][2:])
    rng["cuda"] = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    group = {"params": [0], "betas": (.9, .999), "eps": 1e-8,
        "weight_decay": recipe.weight_decay, "amsgrad": False, "maximize": False,
        "capturable": False, "differentiable": False, "foreach": True, "fused": False,
        "lr": recipe.schedule.pre_update_lr(passes if offset else passes - 1, recipe.learning_rate)}
    selected = passes * recipe.loop_plan.steps_per_pass if recipe.dataset.development and passes else None
    if not recipe.dataset.development and step == recipe.optimizer_steps:
        selected = step
    return {"model": {"weight": torch.tensor([1.])}, "ema": {"weight": torch.tensor([1.])},
        "optimizer": {"param_groups": [group], "state": {0: {"step": torch.tensor(float(step)),
            "exp_avg": torch.tensor([.1]), "exp_avg_sq": torch.tensor([.01])}}},
        "scheduler": {"completed_units": passes, "native": None}, "rng_state": rng,
        "optimizer_step": step, "sample_presentations": step * 128,
        "cursor": {"recipe_identity": recipe.identity, "pass_index": passes, "next_batch": offset,
            "status": "complete" if step == recipe.optimizer_steps else "paused",
            "pending_absolute": [torch.tensor(1.) for _ in range(offset)], "pending_rows": offset * 128,
            "best_development_mae_eV": .25 if recipe.dataset.development and selected is not None else None,
            "selected_step": selected, "optimizer_state_ids": [0]}}


@pytest.fixture
def prefix(tmp_path):
    record, recipe = read_recipe(REPO / BASE / "reference/contract.json")
    contract = tmp_path / "contract.json"
    source, target = torch.tensor([100000, 100001, 100002]), torch.tensor([1., 2., 3.])
    record["acceptance_requirements"].update(development_rows=3,
        source_idx_sha256=tensor_digest(source, role="source_idx"), target_sha256=tensor_digest(target, role="target"))
    write_json(contract, record)
    context = RunContext("gptrans-research-slot1", "synthetic-run", "reference", "a" * 64, "b" * 64,
        "gptrans_t", "2", "c" * 40, "d" * 64, "e" * 64, file_digest(contract),
        "kaggle", "fixture", "fixture/kernel", "3")
    root = tmp_path / "output"
    root.mkdir()
    (root / "training_contract.json").write_bytes(contract.read_bytes())
    step = recipe.loop_plan.steps_per_pass * 10
    state = retained_state(recipe, step)
    torch.save({**state, "context": context.to_dict()}, root / "last_checkpoint.pt")
    checkpoint_sha = file_digest(root / "last_checkpoint.pt")
    selected = {"context": context.to_dict(), "model": state["ema"], "optimizer_step": step,
        "epoch_or_pass": 10., "selection": recipe.selection, "weights": "ema", "development_mae_eV": .25}
    torch.save(selected, root / "selected_model.pt")
    torch.save({"context": context.to_dict(), "source_idx": source, "target_eV": target,
        "prediction_eV": target + .25}, root / "development_predictions.pt")
    semantics = {name: {"metric": "MAE", "unit": "eV", "target": "Gap",
        "role_identity": record["train_role_identity" if name == "live_train_metric" else "development_role_identity"],
        "weights": "ema" if name == "ema_dev_metric" else "live", "direction": "minimize"}
        for name in ("live_train_metric", "live_dev_metric", "ema_dev_metric")}
    rows = [{"optimizer_step": (p + 1) * recipe.loop_plan.steps_per_pass,
        "sample_presentations": (p + 1) * recipe.loop_plan.steps_per_pass * 128,
        "epoch_or_pass": p + 1, "live_train_metric": 1., "live_dev_metric": .5,
        "ema_dev_metric": .25 if p == 9 else .5,
        "checkpoint_identity": checkpoint_sha if p == 9 else str(p + 1) * 64}
        for p in range(10)]
    trace = canonicalize_trace({"trajectory_id": "synthetic-trajectory",
        "run_id": context.logical_run_id + ":reference:downstream", "metric_semantics": semantics,
        "device_time_semantics": "sum_over_devices", "observations": rows})
    write_json(root / "canonical_trace.json", trace)
    write_json(root / "canonical_trace.json.context.json", {"context": context.to_dict(), "stage_id": "downstream"})
    write_json(root / "pause_receipt.json", {"context": context.to_dict(), "status": "PAUSED",
        "optimizer_steps": step, "sample_presentations": step * 128, "checkpoint_sha256": checkpoint_sha})
    write_json(root / "allocation_costs.json", {"context": context.to_dict(), "wall_seconds": 12.,
        "current_invocation_seconds": 12., "checkpoint_sha256": checkpoint_sha})
    write_json(root / "invocation_cost_state.json", {"context": context.to_dict(), "status": "closed",
        "prior_wall_seconds": 0., "current_invocation_seconds": 12.})
    return SimpleNamespace(root=root, context=context, record=record, recipe=recipe, state=state)


def inspect(case):
    return owner._inspect_prefix(case.root, case.context, case.record, case.recipe,
        trajectory_id="synthetic-trajectory")


def test_prefix_reuses_real_session_trace_target_and_selection_checks(prefix):
    before = {p.name: p.read_bytes() for p in prefix.root.iterdir()}
    report, rows = inspect(prefix)
    assert report["progress"]["completed_passes"] == 10
    assert report["observed"]["development_mae_eV"] == .25 and len(rows) == 10
    assert before == {p.name: p.read_bytes() for p in prefix.root.iterdir()}
    assert not (prefix.root / "output_manifest.json").exists()


@pytest.mark.parametrize("fault", ["manifest", "journal", "rng", "fp64", "optimizer_fp64", "selection",
    "prediction", "target", "row", "cost_unknown", "cost_open", "cost_sum", "pause", "missing_trace", "role", "trace_checkpoint"])
def test_prefix_refuses_incomplete_or_mismatching_retained_evidence(prefix, fault):
    if fault in {"manifest", "journal"}:
        write_json(prefix.root / ("output_manifest.json" if fault == "manifest" else "session_commit.json"), {})
    elif fault in {"rng", "fp64", "optimizer_fp64"}:
        path = prefix.root / "last_checkpoint.pt"
        state = torch.load(path, weights_only=True)
        if fault == "rng":
            state["rng_state"]["cuda"] = []
        elif fault == "optimizer_fp64":
            state["optimizer"]["state"][0]["exp_avg"] = state["optimizer"]["state"][0]["exp_avg"].double()
            state["optimizer"]["state"][0]["exp_avg_sq"] = state["optimizer"]["state"][0]["exp_avg_sq"].double()
        else:
            state["model"]["weight"] = state["model"]["weight"].double()
            state["ema"]["weight"] = state["ema"]["weight"].double()
        torch.save(state, path)
    elif fault == "selection":
        path = prefix.root / "selected_model.pt"
        state = torch.load(path, weights_only=True)
        state["weights"] = "live"
        torch.save(state, path)
    elif fault in {"prediction", "target", "row"}:
        path = prefix.root / "development_predictions.pt"
        state = torch.load(path, weights_only=True)
        state[{"prediction": "prediction_eV", "target": "target_eV", "row": "source_idx"}[fault]][0] += 1
        torch.save(state, path)
    else:
        name = {"cost_unknown": "allocation_costs.json", "cost_open": "invocation_cost_state.json",
            "cost_sum": "invocation_cost_state.json",
            "pause": "pause_receipt.json"}.get(fault, "canonical_trace.json")
        path = prefix.root / name
        value = json.loads(path.read_bytes())
        if fault == "cost_unknown":
            value["wall_seconds"] = None
        elif fault == "cost_open":
            value["status"] = "running"
        elif fault == "cost_sum":
            value["prior_wall_seconds"] = 5.
        elif fault == "pause":
            value["optimizer_steps"] -= 1
        elif fault == "missing_trace":
            value["observations"].pop()
        elif fault == "role":
            value["metric_semantics"]["live_train_metric"]["role_identity"] = "f" * 64
        else:
            value["observations"][-1]["checkpoint_identity"] = "f" * 64
        write_json(path, value)
    with pytest.raises(ValueError):
        inspect(prefix)


def test_prefix_preserves_strict_improvement_selection_on_equal_mae(prefix):
    path = prefix.root / "canonical_trace.json"
    trace = json.loads(path.read_bytes())
    trace["observations"][0]["ema_dev_metric"] = .25
    write_json(path, trace)
    with pytest.raises(ValueError, match="first strict best-development"):
        inspect(prefix)


@pytest.mark.parametrize("passes", [9, 11, 60])
def test_prefix_requires_incomplete_ten_pass_boundary(prefix, passes):
    state = retained_state(prefix.recipe, passes * prefix.recipe.loop_plan.steps_per_pass)
    torch.save({**state, "context": prefix.context.to_dict()}, prefix.root / "last_checkpoint.pt")
    with pytest.raises(ValueError, match="ten-pass boundaries"):
        inspect(prefix)


def test_scope_rejects_shortened_scientific_horizon(prefix):
    owner._scope(prefix.recipe)
    shorter = replace(prefix.recipe, optimizer_steps=prefix.recipe.loop_plan.steps_per_pass * 10,
        schedule=replace(prefix.recipe.schedule, total_units=10, reference_epochs=10))
    with pytest.raises(ValueError, match="60pass"):
        owner._scope(shorter)


def test_resume_proof_must_extend_exact_previous_segment(prefix):
    step = prefix.recipe.loop_plan.steps_per_pass * 10
    proofs = []
    for label in ("actual-resume-a.pt", "actual-resume-b.pt"):
        state = retained_state(prefix.recipe, step + 1)
        torch.save(state, prefix.root / label)
        proofs.append({"path": label, "sha256": file_digest(prefix.root / label),
            "state_fingerprint": _state_fingerprint(state)})
    cert = {"status": "accepted", "formal_training_exposure": 0, "resume_step": step,
        "checkpoint_sha256": "a" * 64, "proofs": proofs}
    write_json(prefix.root / "actual_resume_certificate.json", cert)
    observations = [{"optimizer_step": step, "checkpoint_identity": "a" * 64}]
    paths = owner._verify_actual_resume(prefix.root, prefix.recipe, observations, completed_step=2 * step)
    assert len(paths) == 3
    with pytest.raises(ValueError, match="retained-state"):
        owner._verify_actual_resume(prefix.root, prefix.recipe, observations, completed_step=3 * step)
    state["model"]["weight"][0] += 1
    torch.save(state, prefix.root / proofs[1]["path"])
    with pytest.raises(ValueError, match="bytes changed"):
        owner._verify_actual_resume(prefix.root, prefix.recipe, observations, completed_step=2 * step)


def test_retained_native_proof_payloads_are_rechecked_without_model_execution(prefix):
    from molgap.family_training_runner import _verify_preflight
    probe = _probe_recipe(prefix.recipe)
    provenance = {"fixture": "synthetic saved states, not an actual GPU qualification"}
    resource = prefix.root / "resource_qualification.json"
    write_json(resource, {"status": "accepted"})
    proofs = []
    for name in ("repeat-a.pt", "repeat-b.pt", "resumed.pt"):
        state = retained_state(probe, 2)
        torch.save(state, prefix.root / name)
        proofs.append({"path": name, "sha256": file_digest(prefix.root / name),
            "state_fingerprint": _state_fingerprint(state)})
    torch.save(retained_state(probe, 1), prefix.root / "resumed-first.pt")
    certificate = {"format": "molgap-runtime-certificate-v1", "status": "accepted",
        "provenance_sha256": canonical_fingerprint(provenance), "probe_recipe_identity": probe.identity,
        "formal_training_exposure": 0, "proofs": proofs,
        "resource_qualification_sha256": file_digest(resource),
        "resume_origin": {"path": "resumed-first.pt", "sha256": file_digest(prefix.root / "resumed-first.pt")}}
    write_json(prefix.root / "runtime_provenance.json", provenance)
    write_json(prefix.root / "runtime_certificate.json", certificate)
    assert _verify_preflight(prefix.root, prefix.recipe, provenance) == certificate
    state["ema"]["weight"][0] += 1
    torch.save(state, prefix.root / "resumed.pt")
    with pytest.raises(ValueError, match="proof artifact hash mismatch"):
        _verify_preflight(prefix.root, prefix.recipe, provenance)


def test_transport_is_called_only_after_slot_gate_and_read_bytes_remain_stable(tmp_path, monkeypatch):
    import molgap.experiment_workflow_resume as transport
    path = tmp_path / "observation.json"
    path.write_bytes(b"retained")
    checked = {"artifacts": {str(path): file_digest(path)}, "arms": {"reference": {"artifacts": {}}}}
    monkeypatch.setattr(owner, "accept_campaign_segment", lambda *args, **kwargs: deepcopy(checked))
    calls = []
    def export(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "RESUME_VERIFIED", "submitted": False}
    monkeypatch.setattr(transport, "build_workflow_resume", export)
    result = owner.build_campaign_segment_resume(None, None, None, None, None, None, None,
        arm_id="reference", output_root=tmp_path, contracts={})
    assert result["scientific_acceptance"] == result["replay_readiness"] == "NOT_EVALUATED"
    assert len(calls) == 1 and calls[0]["source_output"] == tmp_path / "reference"
    def changed(*args, **kwargs):
        path.write_bytes(b"changed")
        return deepcopy(checked)
    monkeypatch.setattr(owner, "accept_campaign_segment", changed)
    with pytest.raises(ValueError, match="before recovery"):
        owner.build_campaign_segment_resume(None, None, None, None, None, None, None,
            arm_id="reference", output_root=tmp_path, contracts={})
    assert len(calls) == 1


def test_transport_mutation_is_reported_after_export_without_rewriting_evidence(tmp_path, monkeypatch):
    import molgap.experiment_workflow_resume as transport
    path = tmp_path / "observation.json"
    path.write_bytes(b"retained")
    report = {"artifacts": {str(path): file_digest(path)}, "arms": {"reference": {"artifacts": {}}}}
    monkeypatch.setattr(owner, "accept_campaign_segment", lambda *args, **kwargs: report)
    def export(*args, **kwargs):
        path.write_bytes(b"producer changed")
        return {"status": "RESUME_VERIFIED"}
    monkeypatch.setattr(transport, "build_workflow_resume", export)
    with pytest.raises(ValueError, match="reconciliation"):
        owner.build_campaign_segment_resume(None, None, None, None, None, None, None,
            arm_id="reference", output_root=tmp_path, contracts={})
    assert path.read_bytes() == b"producer changed"


@pytest.fixture
def slot_gate(tmp_path, monkeypatch):
    """Orchestration boundaries only; native/tensor checks have separate tests."""
    from molgap.experiment_spec import ExperimentSpec
    spec = ExperimentSpec.from_json((REPO / BASE / "slot1/scientific_spec.json").read_text())
    root, package_dir, receipt = tmp_path / "slot", tmp_path / "package", tmp_path / "receipt.json"
    root.mkdir()
    package_dir.mkdir()
    receipt.write_bytes(b"synthetic-launch")
    for name in owner.SIDECARS:
        (package_dir / name).write_bytes(b"synthetic-package")
    tracked_names = [owner.RETENTION_FILENAME, "allocation_ledger.json", "execution_report.json",
        "reference/allocation_ledger.json", "cycle/allocation_ledger.json"]
    for name in tracked_names:
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"synthetic-retention")
    package = {"spec_identity": spec.identity, "source_commit": "c" * 40,
        "archive_sha256": "d" * 64, "package_identity": "e" * 64}
    campaign, contracts, contexts, reports = {"arms": {}}, {}, {}, {}
    for arm_id in ("reference", "cycle"):
        contract = tmp_path / (arm_id + ".json")
        contract.write_text((REPO / BASE / arm_id / "contract.json").read_text(), encoding="utf-8")
        record, recipe = read_recipe(contract)
        contracts[arm_id] = contract
        arm = next(a for a in spec.to_dict()["arms"] if a["arm_id"] == arm_id)
        ctx = RunContext(spec.to_dict()["experiment_id"], spec.to_dict()["logical_run_id"],
            arm_id, canonical_fingerprint(arm), spec.identity, "gptrans_t", "2", "c" * 40,
            "d" * 64, "e" * 64, file_digest(contract), "kaggle", "fixture", "fixture/kernel", "3")
        contexts[arm_id] = ctx
        campaign["arms"][arm_id] = {"contract": {"sha256": ctx.training_recipe_sha256},
            "initial_state": {"sha256": "f" * 64}}
        native = root / arm_id
        proofs = [{"path": name} for name in ("repeat-a.pt", "repeat-b.pt", "resumed.pt")]
        write_json(native / "runtime_certificate.json", {"status": "accepted", "proofs": proofs})
        write_json(native / "runtime_provenance.json", {"context": ctx.to_dict(),
            "recipe_identity": recipe.identity, "dataset_manifest_sha256": owner.FIXED_100K_SHA,
            "accelerator": "synthetic Tesla T4 observation", "initialization": {
                "state_sha256": recipe.initialization_sha256, "file_sha256": "f" * 64}})
        write_json(native / "resource_qualification.json", {"status": "accepted", "zero_start": {"accepted": True}})
        for name in ("repeat-a.pt", "repeat-b.pt", "resumed.pt", "resumed-first.pt"):
            (native / name).write_bytes(b"synthetic-proof")
        saved = native / "last_checkpoint.pt"
        saved.write_bytes(b"synthetic-prefix")
        reports[arm_id] = {"status": "MECHANICALLY_VERIFIED", "progress": {
            "completed_passes": 10, "optimizer_steps": recipe.loop_plan.steps_per_pass * 10},
            "artifacts": {str(saved): file_digest(saved)}}
    ledger = {"format": "molgap-allocation-ledger-v1", "spec_identity": spec.identity,
        "status": "complete", "wall_seconds": 100., "allocated_device_seconds": 200.,
        "prior_segments": [], "prior_unobserved_intervals": False}
    execution = {"status": "complete", "spec_identity": spec.identity, "arms": {
        arm: {"training_started": True, "terminal_status": "interrupted"} for arm in contexts}}
    retained = {"ledgers": {"root": ledger, "execution_report": execution}}
    monkeypatch.setattr(owner, "verify_experiment_source_package", lambda path: package)
    monkeypatch.setattr(owner, "validate_execution_retention", lambda *args, **kwargs: retained)
    monkeypatch.setattr(owner, "retired_bootstrap_cost", lambda *args, **kwargs: {
        "wall_seconds": 5., "allocated_device_seconds": 10.})
    monkeypatch.setattr(owner, "read_packaged_text", lambda *args: json.dumps(campaign))
    monkeypatch.setattr(owner.RunContext, "from_launch", lambda *args, **kwargs: contexts[kwargs["arm_id"]])
    monkeypatch.setattr(owner, "validate_screen_recipe", lambda *args: None)
    monkeypatch.setattr(owner, "_verify_preflight", lambda *args: None)
    monkeypatch.setattr(owner, "_inspect_prefix", lambda directory, *args, **kwargs: (deepcopy(reports[directory.name]), []))
    return SimpleNamespace(spec=spec, root=root, package=package_dir, receipt=receipt, contracts=contracts,
        retained=retained, ledger=ledger, execution=execution, reports=reports)


def accept_slot(case):
    return owner.accept_campaign_segment(case.spec, package_dir=case.package, receipt_path=case.receipt,
        output_root=case.root, contracts=case.contracts)


def test_slot_gate_preserves_two_arm_prefix_and_retired_full_allocation_cost(slot_gate):
    report = accept_slot(slot_gate)
    assert set(report["arms"]) == {"reference", "cycle"}
    assert report["full_allocation"] == {"wall_seconds": 105., "allocated_device_seconds": 210.}
    assert report["submitted"] is report["resume_published"] is False
    assert report["scientific_acceptance"] == report["replay_readiness"] == "NOT_EVALUATED"
    assert str(slot_gate.receipt) in report["artifacts"]
    assert all(str(path) in report["artifacts"] for path in slot_gate.contracts.values())


@pytest.mark.parametrize("fault", ["missing_arm", "not_started", "unknown_cost", "exhausted",
    "native_rejected", "resume_generation", "changed_proof", "changed_receipt"])
def test_slot_gate_blocks_bad_coverage_cost_qualification_and_mutation(slot_gate, monkeypatch, fault):
    if fault == "missing_arm":
        slot_gate.execution["arms"].pop("cycle")
    elif fault == "not_started":
        slot_gate.execution["arms"]["cycle"]["training_started"] = False
    elif fault == "unknown_cost":
        slot_gate.ledger["prior_unobserved_intervals"] = True
    elif fault == "exhausted":
        slot_gate.ledger.update(wall_seconds=28800., allocated_device_seconds=57600.)
    elif fault == "native_rejected":
        path = slot_gate.root / "cycle/runtime_certificate.json"
        value = json.loads(path.read_bytes())
        value["status"] = "failed"
        write_json(path, value)
    elif fault == "resume_generation":
        slot_gate.reports["cycle"]["progress"]["completed_passes"] = 20
    else:
        def mutate(*args):
            path = slot_gate.receipt if fault == "changed_receipt" else Path(args[0]) / "repeat-a.pt"
            path.write_bytes(b"changed-during-acceptance")
        monkeypatch.setattr(owner, "_verify_preflight", mutate)
    with pytest.raises(ValueError):
        accept_slot(slot_gate)


def test_nested_prior_invocations_are_counted_once_before_budget(slot_gate):
    old = {**deepcopy(slot_gate.ledger), "wall_seconds": 20., "allocated_device_seconds": 40.}
    older = {**deepcopy(slot_gate.ledger), "wall_seconds": 10., "allocated_device_seconds": 20.}
    old["prior_segments"] = [older]
    slot_gate.ledger["prior_segments"] = [old, older]
    assert accept_slot(slot_gate)["full_allocation"] == {"wall_seconds": 135., "allocated_device_seconds": 270.}


def test_second_arm_cannot_change_already_verified_first_arm(slot_gate, monkeypatch):
    def mutate(directory, *args):
        if Path(directory).name == "cycle":
            (slot_gate.root / "reference/last_checkpoint.pt").write_bytes(b"changed-after-first-arm")
    monkeypatch.setattr(owner, "_verify_preflight", mutate)
    with pytest.raises(ValueError, match="other assigned arm"):
        accept_slot(slot_gate)
