"""A planned control can enter replay only through its accepted same-run result."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from molgap.comparison_readiness import (
    REQUIRED_CANDIDATE_OBSERVED_BINDINGS,
    REQUIRED_OBSERVED_BINDINGS,
    assess_comparison_readiness,
    reference_bundle_digest,
)
from molgap.research_memory.compiler import compile_research_memory
from molgap.research_memory.paired import PAIR_OBSERVATION_SCHEMA, PAIR_SCHEMA, accepted_reference_evidence
from molgap.research_memory.terminal_wiring import (
    build_default_trace_manifest,
    close_terminal_arm,
    resolve_trace_for_terminal_arm,
)
from molgap.research_memory.trace import file_digest, json_bytes, load_canonical_trace

from test_comparison_readiness import _reference_bundle, _reference_side, _side, _target_transform_asset
from test_terminal_trace_closure import create_candidate_arm, setup_mock_repo


def _put(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def _binding(arm: dict, reference: dict, role: str) -> dict:
    return {
        "schema": PAIR_SCHEMA,
        "spec_identity": "a" * 64,
        "logical_run_id": "one-job",
        "arm_id": arm["exp_dir"].name,
        "comparison_role": role,
        "reference_arm_id": reference["exp_dir"].name,
        "reference_trajectory_id": reference["traj_id"],
        "reference_trajectory_ref": reference["traj_path"].relative_to(reference["exp_dir"].parents[1]).as_posix(),
    }


def _prepare_arm(root: Path, arm: dict, reference: dict, role: str) -> Path:
    trajectory = json.loads(arm["traj_path"].read_text())
    trajectory["state_at_start"]["same_run_replay"] = _binding(arm, reference, role)
    _put(arm["traj_path"], trajectory)
    _, canonical_path = resolve_trace_for_terminal_arm(root, arm["traj_path"], arm["terminal_path"])
    terminal = json.loads(arm["terminal_path"].read_text())
    manifest = build_default_trace_manifest(root, trajectory, terminal, load_canonical_trace(canonical_path))
    manifest["comparison_role"] = role
    manifest["reference_id"] = reference["ev_id"]
    manifest["comparability_identity"]["scientific_contract"] = "paired-test-contract"
    manifest["backtest_eligibility"] = {"eligible": True, "exclusion_reasons": []}
    terminal["trace_manifest"] = manifest
    terminal["same_run_observation"] = {
        "schema": PAIR_OBSERVATION_SCHEMA,
        "spec_identity": "a" * 64,
        "logical_run_id": "one-job",
        "platform_name": "kaggle",
        "platform_run_reference": "one-platform-job",
        "attempt_id": "att-1",
        "source_commit": "1" * 40,
        "source_package_sha256": "f" * 64,
    }
    _put(arm["terminal_path"], terminal)
    return canonical_path


def _strict_comparison(root: Path, reference: dict, candidate: dict,
                       purpose: str = "architecture_comparison") -> tuple[str, str]:
    reference_root = reference["exp_dir"]
    candidate_root = candidate["exp_dir"]
    transform = _target_transform_asset()
    transform_path = reference_root / "target_transform.json"
    _put(transform_path, transform)
    bundle = _reference_bundle()
    bundle["reference_id"] = reference["ev_id"]
    bundle["contract_ref"] = (reference_root / "contract.json").relative_to(root).as_posix()
    bundle["trace_manifest_ref"] = (reference_root / "rml_finalized/trace_manifest.json").relative_to(root).as_posix()
    bundle["target_transform_asset_ref"] = transform_path.relative_to(root).as_posix()
    bundle["acceptance_ref"] = (reference_root / "acceptance.json").relative_to(root).as_posix()
    bundle["decision_ref"] = (reference_root / "decision.md").relative_to(root).as_posix()
    for field in ("runtime_certificate_ref", "row_manifest_ref", "target_manifest_ref", "role_history_ref", "cost_records_ref"):
        pointer = (reference_root / (field + ".json")).relative_to(root).as_posix()
        _put(root / pointer, {"fixture": field})
        bundle[field] = pointer
    bundle_path = reference_root / "reference_bundle.json"
    _put(bundle_path, bundle)

    bundle_refs = {
        "runtime_certificate": "runtime_certificate_ref", "row_manifest": "row_manifest_ref",
        "target_manifest": "target_manifest_ref", "trace_manifest": "trace_manifest_ref",
        "role_history": "role_history_ref", "target_transform_asset": "target_transform_asset_ref",
        "cost_records": "cost_records_ref", "acceptance": "acceptance_ref", "decision": "decision_ref",
    }
    reference_side = _reference_side()
    reference_side["reference_bundle_id"] = bundle["reference_bundle_id"]
    reference_side["reference_bundle_sha256"] = reference_bundle_digest(bundle)
    candidate_side = _side("candidate-arch")
    field = {"architecture_comparison": "architecture_config_identity",
             "optimizer_comparison": "optimizer_identity",
             "initialization_comparison": "initialization_identity"}[purpose]
    if purpose != "architecture_comparison":
        candidate_side["comparison_identity"]["architecture_config_identity"] = "reference-arch"
        for side, value in ((candidate_side, "a" * 64), (reference_side, "b" * 64)):
            side["comparison_identity"][field] = value
    for side, directory, names in (
        (reference_side, reference_root, REQUIRED_OBSERVED_BINDINGS),
        (candidate_side, candidate_root, REQUIRED_CANDIDATE_OBSERVED_BINDINGS),
    ):
        bindings = {}
        for name in names:
            if side is reference_side and name in bundle_refs:
                pointer = bundle[bundle_refs[name]]
            else:
                path = directory / ("binding_" + name + ".json")
                _put(path, {"fixture": name})
                pointer = path.relative_to(root).as_posix()
            bindings[name] = {"ref": pointer, "sha256": file_digest(root / pointer)}
        side["artifact_bindings"] = bindings
    comparison = assess_comparison_readiness(
        candidate_id=candidate["ev_id"], candidate=candidate_side,
        reference_id=reference["ev_id"], reference=reference_side,
        declared_intervention_fields=[field], experiment_purpose=purpose,
        intervention_group_id=field,
    )
    assert comparison["strict_ready"] is True
    path = candidate_root / "comparison_readiness.json"
    _put(path, comparison)
    return path.relative_to(root).as_posix(), file_digest(path)


@pytest.mark.parametrize("purpose", ["architecture_comparison", "optimizer_comparison", "initialization_comparison"])
def test_same_run_pair_reaches_replay_only_after_reference_acceptance(tmp_path, purpose):
    setup_mock_repo(tmp_path)
    reference = create_candidate_arm(tmp_path, "pair_ref", "T-pair-ref", "run-ref", "ev-pair-ref")
    candidate = create_candidate_arm(tmp_path, "pair_candidate", "T-pair-candidate", "run-candidate", "ev-pair-candidate")
    ref_trace = _prepare_arm(tmp_path, reference, reference, "reference")
    candidate_trace = _prepare_arm(tmp_path, candidate, reference, "candidate")

    with pytest.raises(ValueError, match="not yet terminally accepted"):
        accepted_reference_evidence(tmp_path, json.loads(candidate["traj_path"].read_text()))
    assert close_terminal_arm(tmp_path, reference["traj_path"], reference["terminal_path"], trace=ref_trace)["pipeline_status"] == "COMPLETE"
    with pytest.raises(ValueError, match="requires strict V5 comparison"):
        close_terminal_arm(tmp_path, candidate["traj_path"], candidate["terminal_path"], trace=candidate_trace)
    assert not (candidate["exp_dir"] / "rml_finalized").exists()
    comparison_ref, digest = _strict_comparison(tmp_path, reference, candidate, purpose)
    terminal = json.loads(candidate["terminal_path"].read_text())
    terminal["comparison_readiness_ref"] = comparison_ref
    terminal["artifact_hashes"][comparison_ref] = digest
    terminal["same_run_observation"]["platform_run_reference"] = "different-job"
    _put(candidate["terminal_path"], terminal)
    with pytest.raises(ValueError, match="different observed platform jobs"):
        close_terminal_arm(tmp_path, candidate["traj_path"], candidate["terminal_path"], trace=candidate_trace)
    terminal["same_run_observation"]["platform_run_reference"] = "one-platform-job"
    _put(candidate["terminal_path"], terminal)
    assert close_terminal_arm(tmp_path, candidate["traj_path"], candidate["terminal_path"], trace=candidate_trace)["pipeline_status"] == "COMPLETE"
    pool = compile_research_memory(tmp_path)["replay_pool.json"]
    replay = json.loads(pool)
    entries = [entry for entry in replay["entries"] if entry["trajectory_id"] in {reference["traj_id"], candidate["traj_id"]}]
    assert {entry["comparison_role"] for entry in entries} == {"reference", "candidate"}
    assert {entry["reference_id"] for entry in entries} == {reference["ev_id"]}
    # Derive identities from the actually finalized comparison, not the generic
    # manifest's missing top-level recipe fields; frozen records stay unchanged.
    original = (candidate["exp_dir"] / "rml_finalized/trace_manifest.json").read_bytes()
    comparison = json.loads((tmp_path / comparison_ref).read_bytes())
    for entry in entries:
        identity = entry["comparability_identity"]
        assert identity["dataset_identity"] == comparison["matched_fields"]["dataset_identity"]
        assert json.loads(identity["ema_semantics"])["ema_decay"] == comparison["matched_fields"]["ema_decay"]
        assert "unspecified_in_contract" not in entry["comparability_key"]
    assert compile_research_memory(tmp_path)["replay_pool.json"] == pool
    assert (candidate["exp_dir"] / "rml_finalized/trace_manifest.json").read_bytes() == original
    from molgap.research_memory.backtest import build_screening_backtest
    from molgap.research_memory.replay import replay_manifests
    from molgap.research_memory.validate import validate_repository_records
    records = validate_repository_records(tmp_path)["records"]
    manifests = replay_manifests(tmp_path, records)
    # Batch/EMA/runtime differences in strict identities must survive projection,
    # even when the older coarse labels and reference ID happen to match.
    for field, changed in (("physical_batch_per_device", 129), ("loss_identity", "different-loss"),
                           ("schedule_identity", "different-schedule"), ("seed", 43),
                           ("feature_identity", "different-features")):
        altered = copy.deepcopy(manifests)
        for manifest in altered:
            if manifest["comparison_role"] == "candidate":
                manifest["comparability_identity"]["strict_identity"][field] = changed
        assert build_screening_backtest(altered)["included_comparable_groups"] == []
    if purpose != "architecture_comparison":
        field = comparison["declared_intervention_fields"][0]
        scope_name = "bound_" + field.removesuffix("_identity") + "_intervention"
        assert all(scope_name in entry["comparability_identity"] for entry in entries)
        altered = copy.deepcopy(manifests)
        altered[0]["comparability_identity"]["strict_identity"][field] = "unexpected"
        with pytest.raises(ValueError, match="differs from its bound comparison"):
            build_screening_backtest(altered)
        other_scope = ("bound_optimizer_intervention" if purpose == "initialization_comparison"
                       else "bound_initialization_intervention")
        for extra_scope in ("bound_unknown_intervention", other_scope):
            altered = copy.deepcopy(manifests)
            altered[0]["comparability_identity"][extra_scope] = {}
            with pytest.raises(ValueError, match="one supported bound intervention"):
                build_screening_backtest(altered)
        duplicated = {**records, "traces": [*records["traces"], next(
            item for item in records["traces"] if item[1]["comparison_role"] == "candidate")]}
        with pytest.raises(ValueError, match="one exact accepted candidate/reference trace pair"):
            replay_manifests(tmp_path, duplicated)
        # Retained manifests never grant exemptions themselves, including when
        # their purported scope names are unknown to the supported projection.
        forged = copy.deepcopy(records)
        forged["comparison_readiness"] = []
        for _, manifest in forged["traces"]:
            manifest["comparability_identity"][scope_name] = {"expected": "forged"}
            manifest["comparability_identity"]["bound_unknown_intervention"] = {}
        projected = replay_manifests(tmp_path, forged)
        assert all(not any(key.startswith("bound_") for key in manifest["comparability_identity"])
                   for manifest in projected)
        assert build_screening_backtest(projected)["included_comparable_groups"] == []
        if purpose == "initialization_comparison":
            altered = copy.deepcopy(manifests)
            for manifest in altered:
                if manifest["comparison_role"] == "candidate":
                    manifest["comparability_identity"]["strict_identity"]["architecture_config_identity"] = "different-architecture"
            assert build_screening_backtest(altered)["included_comparable_groups"] == []

    # A fresh checkout inherits accepted metadata without ignored checkpoints.
    # Index checking must match rebuild; strict artifact validation stays strict.
    from molgap.research_memory.compiler import rebuild_research_memory
    from molgap.research_memory.cli import main
    checkpoint = tmp_path / comparison["reference_artifact_bindings"]["checkpoint"]["ref"]
    checkpoint.unlink()
    with pytest.raises(ValueError, match="required repository pointer is missing"):
        validate_repository_records(tmp_path)
    assert rebuild_research_memory(tmp_path)["replay_pool.json"] == pool
    main(["--repo-root", str(tmp_path), "check", "--frozen"])
    with pytest.raises(ValueError, match="required repository pointer is missing"):
        main(["--repo-root", str(tmp_path), "check", "--frozen", "--portable"])
    (tmp_path / "research_memory/derived/replay_pool.json").write_bytes(b"stale")
    with pytest.raises(SystemExit, match="stale derived outputs: replay_pool.json"):
        main(["--repo-root", str(tmp_path), "check", "--frozen"])


def test_eligible_continuation_must_match_current_pair_attempt(tmp_path, monkeypatch):
    setup_mock_repo(tmp_path)
    reference = create_candidate_arm(
        tmp_path, "continued_ref", "T-continued-ref", "run-continued-ref", "ev-continued-ref"
    )
    candidate = create_candidate_arm(
        tmp_path, "continued_candidate", "T-continued-candidate", "run-continued-candidate",
        "ev-continued-candidate",
    )
    ref_trace = _prepare_arm(tmp_path, reference, reference, "reference")
    candidate_trace = _prepare_arm(tmp_path, candidate, reference, "candidate")
    assert close_terminal_arm(
        tmp_path, reference["traj_path"], reference["terminal_path"], trace=ref_trace
    )["pipeline_status"] == "COMPLETE"
    comparison_ref, digest = _strict_comparison(tmp_path, reference, candidate)

    terminal = json.loads(candidate["terminal_path"].read_bytes())
    terminal["comparison_readiness_ref"] = comparison_ref
    terminal["artifact_hashes"][comparison_ref] = digest
    terminal["same_run_observation"]["attempt_id"] = "one-job-v3"
    continuation_pointer = "experiments/continued_candidate/continuation.json"
    continuation_path = tmp_path / continuation_pointer
    _put(continuation_path, {"synthetic": "the test substitutes only proof validation"})
    terminal["continuation_ref"] = continuation_pointer
    terminal["artifact_hashes"][continuation_pointer] = file_digest(continuation_path)
    _put(candidate["terminal_path"], terminal)

    # The strict proof validator is an external boundary here. Its origin is
    # the reference's v1 attempt, while the candidate claims current v3.
    import importlib
    finalize_module = importlib.import_module("molgap.research_memory.finalize")
    monkeypatch.setattr(finalize_module, "validate_terminal_continuation",
        lambda _root, _terminal, **_kwargs: json.loads(
            reference["terminal_path"].read_bytes())["same_run_observation"])
    with pytest.raises(ValueError, match="different observed platform jobs"):
        close_terminal_arm(
            tmp_path, candidate["traj_path"], candidate["terminal_path"], trace=candidate_trace
        )


def test_replay_eligible_same_run_reference_rejects_a_cross_job_continuation(tmp_path):
    setup_mock_repo(tmp_path)
    reference = create_candidate_arm(
        tmp_path, "continued_ref", "T-continued-ref", "run-continued-ref", "ev-continued-ref"
    )
    trace_path = _prepare_arm(tmp_path, reference, reference, "reference")
    terminal = json.loads(reference["terminal_path"].read_bytes())
    terminal["trace_manifest"]["backtest_eligibility"] = {
        "eligible": True,
        "exclusion_reasons": [],
    }
    continuation_pointer = "experiments/continued_ref/continuation.json"
    continuation_path = tmp_path / continuation_pointer
    _put(continuation_path, {"synthetic": "must not authorize replay"})
    terminal["continuation_ref"] = continuation_pointer
    terminal["artifact_hashes"][continuation_pointer] = file_digest(continuation_path)
    _put(reference["terminal_path"], terminal)

    # Eligible continuations must pass the strict continuation owner
    # (research_memory/paired.py validate_terminal_continuation); an unbound
    # synthetic record is rejected before it can authorize replay.
    with pytest.raises(ValueError, match="invalid terminal continuation record"):
        close_terminal_arm(tmp_path, reference["traj_path"], reference["terminal_path"],
                           trace=trace_path)
