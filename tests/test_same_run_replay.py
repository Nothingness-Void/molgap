"""A planned control can enter replay only through its accepted same-run result."""

from __future__ import annotations

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


def _strict_comparison(root: Path, reference: dict, candidate: dict) -> tuple[str, str]:
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
        declared_intervention_fields=["architecture_config_identity"],
    )
    assert comparison["strict_ready"] is True
    path = candidate_root / "comparison_readiness.json"
    _put(path, comparison)
    return path.relative_to(root).as_posix(), file_digest(path)


def test_same_run_pair_reaches_replay_only_after_reference_acceptance(tmp_path):
    setup_mock_repo(tmp_path)
    reference = create_candidate_arm(tmp_path, "pair_ref", "T-pair-ref", "run-ref", "ev-pair-ref")
    candidate = create_candidate_arm(tmp_path, "pair_candidate", "T-pair-candidate", "run-candidate", "ev-pair-candidate")
    ref_trace = _prepare_arm(tmp_path, reference, reference, "reference")
    candidate_trace = _prepare_arm(tmp_path, candidate, reference, "candidate")

    with pytest.raises(ValueError, match="not yet terminally accepted"):
        accepted_reference_evidence(tmp_path, json.loads(candidate["traj_path"].read_text()))
    assert close_terminal_arm(tmp_path, reference["traj_path"], reference["terminal_path"], trace=ref_trace)["pipeline_status"] == "COMPLETE"
    with pytest.raises(ValueError, match="requires accepted control and strict V5 comparison"):
        close_terminal_arm(tmp_path, candidate["traj_path"], candidate["terminal_path"], trace=candidate_trace)
    assert not (candidate["exp_dir"] / "rml_finalized").exists()
    comparison_ref, digest = _strict_comparison(tmp_path, reference, candidate)
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
