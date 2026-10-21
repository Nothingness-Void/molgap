"""Run diagnosis binds retained numerical evidence to independent frozen inputs."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molgap.experiment_attribution import diagnose_run
from molgap.experiment_family_workflow import RunContext
from molgap.training_reproducibility import sha256_file
from test_comparison_readiness import _reference_bundle
from test_experiment_attribution import _write_native_arm


def _resolved(tmp_path, arms, *, same_run=True):
    outputs, contexts = {}, {}
    for path in arms:
        manifest = json.loads((path / "output_manifest.json").read_text())
        context = RunContext(**manifest["context"])
        contract = json.loads((path / manifest["artifacts"]["contract"]["path"]).read_text())
        contexts[context.arm_id] = context
        outputs[context.arm_id] = {"output_dir": str(path), "expected": contract["acceptance_requirements"]}
    first = next(iter(contexts.values()))
    pair = {"reference_arm_id": first.arm_id, "candidate_arm_ids": list(contexts)[1:]}
    declaration = {"logical_run_id": first.logical_run_id,
                   "prospective": {"same_run_replay": pair} if same_run else {}}
    return {"spec": SimpleNamespace(identity=first.spec_identity, to_dict=lambda: declaration),
            "outputs": outputs, "contexts": contexts, "target_identities": {},
            "acceptance_plan": {"arms": [{"arm_id": key} for key in contexts]},
            "run_directory": str(tmp_path / "run")}


def _bind(monkeypatch, resolved):
    import molgap.experiment_run as owner
    monkeypatch.setattr(owner, "resolve_run_acceptance_inputs", lambda *args, **kwargs: resolved)


def _diagnose(resolved, tmp_path, **kwargs):
    context = next(iter(resolved["contexts"].values()))
    return diagnose_run(None, tmp_path, context.logical_run_id, "v1", **kwargs)


def test_run_diagnosis_compares_every_candidate_and_writes_json(monkeypatch, tmp_path):
    arms = [_write_native_arm(tmp_path / name, arm_id=name) for name in ("reference", "candidate-a", "candidate-b")]
    resolved = _resolved(tmp_path, arms)
    _bind(monkeypatch, resolved)
    output = tmp_path / "reports" / "diagnosis.json"
    report = _diagnose(resolved, tmp_path, output=output)
    assert report["status"] == "DIAGNOSED"
    assert set(report["comparisons"]) == {"candidate-a", "candidate-b"}
    for pair in report["comparisons"].values():
        assert pair["primary_comparison"]["n_bootstrap"] == 0
        assert pair["primary_comparison"]["upper_95_eV"] is None
        assert pair["primary_comparison"]["threshold_eV"] is None
        assert pair["primary_comparison"]["passes_material_gate"] is None
        assert pair["scientific_conclusion"] == "NOT_EVALUATED"
        assert pair["signals"]["interpretation"] == "POSTHOC_DESCRIPTIVE_ONLY"
    assert json.loads(output.read_text())["comparisons"].keys() == report["comparisons"].keys()


def test_run_diagnosis_rejects_self_consistent_output_against_frozen_launch(monkeypatch, tmp_path):
    arms = [_write_native_arm(tmp_path / name, arm_id=name) for name in ("reference", "candidate")]
    resolved = _resolved(tmp_path, arms)
    # The output is internally valid; its launch source differs from independent authority.
    resolved["contexts"]["candidate"] = replace(resolved["contexts"]["candidate"], source_commit="f" * 40)
    _bind(monkeypatch, resolved)
    report = _diagnose(resolved, tmp_path)
    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["comparisons"]["candidate"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("context/source/arm mismatch" in message for message in report["blockers"])


def test_run_diagnosis_rejects_output_contract_as_acceptance_authority(monkeypatch, tmp_path):
    arms = [_write_native_arm(tmp_path / name, arm_id=name) for name in ("reference", "candidate")]
    resolved = _resolved(tmp_path, arms)
    resolved["outputs"]["candidate"]["expected"] = {
        **resolved["outputs"]["candidate"]["expected"], "optimizer_steps": 1000}
    _bind(monkeypatch, resolved)
    report = _diagnose(resolved, tmp_path)
    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["arms"]["candidate"]["status"] == "BLOCKED"


def test_run_diagnosis_preserves_mechanical_result_when_legacy_analysis_is_unsupported(monkeypatch, tmp_path):
    arms = [_write_native_arm(tmp_path / name, arm_id=name) for name in ("reference", "candidate")]
    resolved = _resolved(tmp_path, arms)
    _bind(monkeypatch, resolved)

    def unsupported_legacy_contract(*args, **kwargs):
        raise ValueError("Unsupported legacy family contract")

    import molgap.experiment_attribution as attribution
    monkeypatch.setattr(attribution, "_load_arm_artifacts", unsupported_legacy_contract)
    report = _diagnose(resolved, tmp_path)

    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["mechanical_acceptance"] == "MECHANICALLY_VERIFIED"
    for arm_id in ("reference", "candidate"):
        assert report["arms"][arm_id]["status"] == "MECHANICALLY_VERIFIED"
        assert report["arms"][arm_id]["analysis_blockers"] == ["Unsupported legacy family contract"]
    pair = report["comparisons"]["candidate"]
    assert pair["status"] == "INSUFFICIENT_EVIDENCE"
    assert pair["blockers"] == [
        "reference: Unsupported legacy family contract",
        "candidate: Unsupported legacy family contract",
    ]


def _retained_case(tmp_path, candidate, *, supported=True):
    resolved = _resolved(tmp_path, [candidate], same_run=False)
    item = resolved["outputs"]["candidate"]
    expected = item["expected"]
    manifest = json.loads((candidate / "output_manifest.json").read_text())
    contract = json.loads((candidate / manifest["artifacts"]["contract"]["path"]).read_text())
    predictions = tmp_path / "retained_predictions.pt"
    payload = torch.load(candidate / "development_predictions.pt", map_location="cpu", weights_only=True)
    # A retained prediction bundle has no authority to invent a family context.
    payload.pop("context")
    torch.save(payload if supported else {"scalar_mae_eV": .1}, predictions)
    bundle = _reference_bundle()
    bundle["prediction_manifest"].update(
        prediction_sha256=sha256_file(predictions), source_idx_sha256=expected["source_idx_sha256"],
        target_sha256=expected["target_sha256"], row_count=expected["development_rows"],
        unique_source_idx=expected["development_rows"], evaluation_role_identity=contract["development_role_identity"])
    bundle["comparison_identity"].update(seed=contract["seed"], evaluation_weight_source="live")
    bundle_path = tmp_path / "reference_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    pointer = lambda path: {"path": path.name, "sha256": sha256_file(path)}
    resolved["acceptance_plan"]["arms"][0].update(
        reference_bundle=pointer(bundle_path), reference_artifacts={"predictions": pointer(predictions)})
    return resolved


def _attach_retained_curve(tmp_path, resolved, *, values=(0.9, 0.7)):
    """Pin a canonical reference curve to the retained numerical bundle."""
    from molgap.research_memory.trace import canonicalize_trace, json_bytes

    entry = resolved["acceptance_plan"]["arms"][0]
    reference_bundle_path = tmp_path / entry["reference_bundle"]["path"]
    bundle = json.loads(reference_bundle_path.read_text(encoding="utf-8"))
    candidate = resolved["outputs"]["candidate"]
    output_manifest = json.loads((Path(candidate["output_dir"]) / "output_manifest.json").read_text())
    candidate_trace = json.loads((Path(candidate["output_dir"]) /
        output_manifest["artifacts"]["trace"]["path"]).read_text())
    candidate_rows = [row for row in candidate_trace["observations"] if row["event"] == "observation"]
    role = bundle["prediction_manifest"]["evaluation_role_identity"]
    trace = canonicalize_trace({
        "trajectory_id": "retained-reference-trajectory",
        "run_id": "retained-reference-run",
        "metric_semantics": {
            "live_train_metric": {"metric": "MAE", "unit": "normalized-Gap", "target": "gap",
                                  "role_identity": role, "weights": "live", "direction": "minimize"},
            "live_dev_metric": {"metric": "MAE", "unit": "eV", "target": "gap",
                                 "role_identity": role, "weights": "live", "direction": "minimize"},
            "ema_dev_metric": None,
        },
        "observations": [{
            "event": "observation", "epoch_or_pass": row["epoch_or_pass"],
            "optimizer_step": row["optimizer_step"],
            "sample_presentations": row["sample_presentations"],
            "learning_rate": row["learning_rate"], "live_dev_metric": value,
            "live_train_metric": 0.25 + 0.05 * index,
        } for index, (row, value) in enumerate(zip(candidate_rows, values))],
    })
    trace_path = tmp_path / "retained_reference_trace.json"
    trace_path.write_bytes(json_bytes(trace))
    trace_manifest = {
        "schema": "molgap-trace-manifest-v1",
        "trajectory_id": trace["trajectory_id"], "reference_id": bundle["reference_id"],
        "comparison_role": "reference", "run_id": trace["run_id"],
        "contract_ref": "retained_reference_contract.json", "model_identity": "fixture-reference-model",
        "x_axis": "optimizer_steps", "presentation_semantics_ref": "retained_presentation_semantics.json",
        "weight_semantics": "live", "metric_semantics": "Gap MAE in eV",
        "evaluation_role_identity": role, "selection_semantics": "selected live weights",
        "trace_artifact_ref": trace_path.name, "trace_artifact_sha256": sha256_file(trace_path),
        "terminal_evidence_ref": "retained_reference_terminal.json",
        "backtest_eligibility": {"eligible": False, "exclusion_reasons": ["synthetic test fixture"]},
        "comparability_identity": {
            "scientific_contract": "fixture-contract", "dataset_identity": "fixture-dataset",
            "row_split_identity": "fixture-row-split", "architecture_identity": "fixture-model",
            "optimizer_identity": "fixture-optimizer", "lr_schedule_identity": "fixture-schedule",
            "target_transform_identity": "fixture-transform", "precision_identity": "fp32",
            "ema_semantics": "disabled", "evaluation_role_identity": role,
            "selection_role_identity": role, "x_axis_semantics": "optimizer_steps",
            "terminal_endpoint_identity": "fixture-terminal", "matched_architecture_required": False,
        },
        "exposure": {
            "optimizer_steps": max(row["optimizer_step"] for row in candidate_rows),
            "sample_presentations": max(row["sample_presentations"] for row in candidate_rows),
        },
    }
    manifest_path = tmp_path / "retained_reference_trace_manifest.json"
    manifest_path.write_text(json.dumps(trace_manifest), encoding="utf-8")
    bundle["trace_manifest_ref"] = manifest_path.name
    reference_bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    pointer = lambda path: {"path": path.name, "sha256": sha256_file(path)}
    entry["reference_bundle"] = pointer(reference_bundle_path)
    entry["reference_artifacts"]["trace_manifest"] = pointer(manifest_path)
    return trace_path, trace


def test_run_diagnosis_retained_predictions_preserve_unknown_reference_history(monkeypatch, tmp_path):
    candidate = _write_native_arm(tmp_path / "candidate", arm_id="candidate")
    resolved = _retained_case(tmp_path, candidate)
    _bind(monkeypatch, resolved)
    report = _diagnose(resolved, tmp_path)
    assert report["status"] == "DIAGNOSED"
    pair = report["comparisons"]["candidate"]
    assert pair["primary_comparison"]["mean_delta_eV"] == pytest.approx(0.)
    assert pair["trace_dynamics"]["control"] == {"evaluations": 0}
    assert pair["trace_dynamics"]["paired_curve"] == []
    assert pair["cost"]["control"]["process_wall_seconds"] is None
    assert pair["cost"]["candidate_to_control_wall_ratio"] is None
    assert "reference_run_context" in pair["unknowns"]
    assert "reference_trace_dynamics" in pair["unknowns"]


def test_run_diagnosis_uses_pinned_retained_reference_curve(monkeypatch, tmp_path):
    candidate = _write_native_arm(tmp_path / "candidate", arm_id="candidate")
    resolved = _retained_case(tmp_path, candidate)
    _, trace = _attach_retained_curve(tmp_path, resolved)
    _bind(monkeypatch, resolved)

    report = _diagnose(resolved, tmp_path)

    assert report["status"] == "DIAGNOSED"
    pair = report["comparisons"]["candidate"]
    control = pair["trace_dynamics"]["control"]
    assert control["evaluations"] == len(trace["observations"])
    assert control["final_mae_eV"] == pytest.approx(0.7)
    assert control["online_train_mae_at_selected_eV"] is None
    assert control["online_train_mae_at_final_eV"] is None
    assert [row["control_mae_eV"] for row in pair["trace_dynamics"]["paired_curve"]] == [0.9, 0.7]
    assert "canonical_trace_sha256" in pair["artifact_hashes"]["control"]
    assert "reference_trace_dynamics" not in pair["unknowns"]


def test_run_diagnosis_tampered_retained_reference_curve_is_insufficient(monkeypatch, tmp_path):
    candidate = _write_native_arm(tmp_path / "candidate", arm_id="candidate")
    resolved = _retained_case(tmp_path, candidate)
    trace_path, trace = _attach_retained_curve(tmp_path, resolved)
    trace["observations"][0]["live_dev_metric"] += 0.1
    from molgap.research_memory.trace import json_bytes
    trace_path.write_bytes(json_bytes(trace))
    _bind(monkeypatch, resolved)

    report = _diagnose(resolved, tmp_path)

    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    pair = report["comparisons"]["candidate"]
    assert pair["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("SHA mismatch" in blocker for blocker in pair["blockers"])


def test_run_diagnosis_missing_declared_retained_reference_curve_is_insufficient(monkeypatch, tmp_path):
    candidate = _write_native_arm(tmp_path / "candidate", arm_id="candidate")
    resolved = _retained_case(tmp_path, candidate)
    trace_path, _ = _attach_retained_curve(tmp_path, resolved)
    trace_path.unlink()
    _bind(monkeypatch, resolved)

    report = _diagnose(resolved, tmp_path)

    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    pair = report["comparisons"]["candidate"]
    assert pair["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("Declared retained reference curve is missing" in blocker
               for blocker in pair["blockers"])


def test_run_diagnosis_does_not_overwrite_unrelated_json(monkeypatch, tmp_path):
    candidate = _write_native_arm(tmp_path / "candidate", arm_id="candidate")
    resolved = _retained_case(tmp_path, candidate)
    _bind(monkeypatch, resolved)
    output = tmp_path / "caller-report.json"
    original = b'{"owner":"caller","keep":true}\n'
    output.write_bytes(original)

    with pytest.raises(ValueError, match="another record or run"):
        _diagnose(resolved, tmp_path, output=output)

    assert output.read_bytes() == original


def test_run_diagnosis_can_repeat_the_same_pinned_json_output(monkeypatch, tmp_path):
    candidate = _write_native_arm(tmp_path / "candidate", arm_id="candidate")
    resolved = _retained_case(tmp_path, candidate)
    _bind(monkeypatch, resolved)
    output = tmp_path / "diagnosis.json"

    first = _diagnose(resolved, tmp_path, output=output)
    first_bytes = output.read_bytes()
    second = _diagnose(resolved, tmp_path, output=output)

    assert second["status"] == first["status"] == "DIAGNOSED"
    assert output.read_bytes() == first_bytes


def test_run_diagnosis_missing_retained_rows_is_insufficient_evidence(monkeypatch, tmp_path):
    candidate = _write_native_arm(tmp_path / "candidate", arm_id="candidate")
    resolved = _retained_case(tmp_path, candidate, supported=False)
    _bind(monkeypatch, resolved)
    report = _diagnose(resolved, tmp_path)
    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("aligned tensor predictions" in message for message in report["blockers"])


def test_run_diagnosis_never_overwrites_inspected_artifacts(monkeypatch, tmp_path):
    arms = [_write_native_arm(tmp_path / name, arm_id=name) for name in ("reference", "candidate")]
    resolved = _resolved(tmp_path, arms)
    _bind(monkeypatch, resolved)
    with pytest.raises(ValueError, match="inspected artifacts"):
        _diagnose(resolved, tmp_path, output=arms[0] / "output_manifest.json")
