"""Prospectively frozen same-run reference bindings for trace replay."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from molgap.evidence_pointers import load_json_object

from .paths import resolve_repo_pointer


PAIR_SCHEMA = "molgap-same-run-replay-binding-v1"
PAIR_OBSERVATION_SCHEMA = "molgap-same-run-observation-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def validate_pair_binding(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "spec_identity", "logical_run_id", "arm_id", "comparison_role",
        "reference_arm_id", "reference_trajectory_id", "reference_trajectory_ref",
    }:
        raise ValueError("same-run replay binding fields are incomplete or unknown")
    binding = dict(value)
    if binding["schema"] != PAIR_SCHEMA or not isinstance(binding["spec_identity"], str) or not _SHA256.fullmatch(binding["spec_identity"]):
        raise ValueError("invalid same-run replay binding identity")
    for field in ("logical_run_id", "arm_id", "reference_arm_id", "reference_trajectory_id", "reference_trajectory_ref"):
        if not isinstance(binding[field], str) or not binding[field].strip():
            raise ValueError(f"missing same-run replay binding {field}")
    if binding["comparison_role"] not in {"candidate", "reference"}:
        raise ValueError("same-run replay binding role must be candidate or reference")
    if binding["comparison_role"] == "reference" and (
        binding["arm_id"] != binding["reference_arm_id"]
    ):
        raise ValueError("same-run reference arm identity mismatch")
    pointer = binding["reference_trajectory_ref"]
    parts = pointer.split("/")
    if (not pointer.startswith("experiments/") or not pointer.endswith("/trajectory.json")
            or "\\" in pointer or ":" in pointer or any(part in {"", ".", ".."} for part in parts)):
        raise ValueError("same-run reference must name a prospective experiment trajectory")
    return binding


def pair_binding(trajectory: Mapping[str, Any]) -> dict[str, str] | None:
    value = trajectory["state_at_start"].get("same_run_replay")
    return validate_pair_binding(value) if value is not None else None


def validate_pair_observation(value: Any) -> dict[str, str]:
    required = {
        "schema", "spec_identity", "logical_run_id", "platform_name", "platform_run_reference",
        "attempt_id", "source_commit", "source_package_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("same-run observation fields are incomplete or unknown")
    observation = dict(value)
    if observation["schema"] != PAIR_OBSERVATION_SCHEMA:
        raise ValueError("unsupported same-run observation schema")
    for field in ("spec_identity", "source_package_sha256"):
        if not isinstance(observation[field], str) or not _SHA256.fullmatch(observation[field]):
            raise ValueError(f"invalid same-run observation {field}")
    if not isinstance(observation["source_commit"], str) or not _COMMIT.fullmatch(observation["source_commit"]):
        raise ValueError("invalid same-run observation source_commit")
    for field in ("logical_run_id", "platform_name", "platform_run_reference", "attempt_id"):
        if not isinstance(observation[field], str) or not observation[field].strip():
            raise ValueError(f"missing same-run observation {field}")
    return observation


def validate_ineligible_continuation(root: Path, terminal: Mapping[str, Any]) -> dict[str, str]:
    """Verify a retained control continuation without granting replay eligibility."""
    from .paths import verify_bound_artifact
    from .trace import load_canonical_trace

    pointer = terminal.get("continuation_ref")
    bindings = terminal.get("artifact_hashes", {})
    if not isinstance(pointer, str) or pointer not in bindings:
        raise ValueError("different platform jobs require a bound continuation record")
    verify_bound_artifact(root, pointer, bindings[pointer])
    path = resolve_repo_pointer(root, pointer)
    record = load_json_object(path)
    if set(record) != {"format", "origin", "continued", "source_trace", "continued_trace",
                       "source_context", "resume_binding", "continued_manifest", "checkpoint", "authority"} or record["format"] != "molgap-terminal-continuation-v1":
        raise ValueError("invalid terminal continuation record")
    origin = validate_pair_observation(record["origin"])
    continued = validate_pair_observation(record["continued"])
    if continued != validate_pair_observation(terminal.get("same_run_observation")):
        raise ValueError("continuation terminal job mismatch")
    for key in ("spec_identity", "logical_run_id", "source_commit", "source_package_sha256", "platform_name"):
        if origin[key] != continued[key]:
            raise ValueError("continuation changed frozen training identity")
    retained = {}
    for key in ("source_trace", "continued_trace", "source_context", "resume_binding", "continued_manifest", "checkpoint", "authority"):
        item = record[key]
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"} or bindings.get(item["path"]) != item["sha256"]:
            raise ValueError("continuation artifact is not bound")
        verify_bound_artifact(root, item["path"], item["sha256"])
        retained[key] = resolve_repo_pointer(root, item["path"])
    source = load_canonical_trace(retained["source_trace"])
    final = load_canonical_trace(retained["continued_trace"])
    for key in ("trajectory_id", "run_id", "metric_semantics"):
        if source[key] != final[key]:
            raise ValueError("continuation trace identity changed")
    before, after = source["observations"], final["observations"]
    if not before or len(after) <= len(before) or after[:len(before)] != before:
        raise ValueError("continuation must preserve an exact retained trace prefix")
    if final["run_id"] != terminal["run_id"] or final["trajectory_id"] != terminal["trajectory_id"]:
        raise ValueError("continuation trace is not the terminal run")
    manifest = load_json_object(retained["continued_manifest"])
    if manifest.get("format") != "molgap-family-output-v1" or manifest["artifacts"]["trace"]["sha256"] != record["continued_trace"]["sha256"]:
        raise ValueError("continuation output/trace binding mismatch")
    context = load_json_object(retained["source_context"])["context"]
    if context != manifest["context"]:
        raise ValueError("continuation output context changed")
    for field, observed_field in (("spec_identity", "spec_identity"),
                                  ("logical_run_id", "logical_run_id"),
                                  ("source_commit", "source_commit"),
                                  ("source_archive_sha256", "source_package_sha256"),
                                  ("platform", "platform_name")):
        if context[field] != origin[observed_field]:
            raise ValueError("continuation observation/output identity mismatch")
    for observation in (origin, continued):
        if observation["platform_run_reference"].partition("@")[0] != context["run_reference"]:
            raise ValueError("continuation observation/output run mismatch")
    resume = load_json_object(retained["resume_binding"])
    if (resume["spec_identity"], resume["package_identity"], resume["arm_id"],
            resume["start_epoch"], resume["end_epoch"]) != (
            context["spec_identity"], context["package_identity"], context["arm_id"],
            before[-1]["epoch_or_pass"], after[-1]["epoch_or_pass"]):
        raise ValueError("continuation resume identity/progress mismatch")
    if (resume["files"].get(retained["checkpoint"].name) != record["checkpoint"]["sha256"]
            or resume["files"].get(retained["source_trace"].name) != record["source_trace"]["sha256"]):
        raise ValueError("continuation checkpoint is not the retained source endpoint")
    if not any(a.get("locator") == record["continued_trace"]["path"]
               and a.get("sha256") == record["continued_trace"]["sha256"]
               for a in terminal["evidence"]["artifacts"]):
        raise ValueError("continuation trace is absent from accepted evidence")
    if terminal["trace_manifest"]["trace_artifact_ref"] != record["continued_trace"]["path"]:
        raise ValueError("continuation manifest selects a different trace")
    return origin


def reference_trajectory(root: Path, trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the frozen peer without using a later, unbound reference name."""
    binding = pair_binding(trajectory)
    if binding is None:
        raise ValueError("trajectory has no same-run replay binding")
    from .schemas import validate_trajectory

    path = resolve_repo_pointer(root, binding["reference_trajectory_ref"])
    if path is None or not path.is_file():
        raise ValueError("same-run reference trajectory is not locally retained")
    reference = validate_trajectory(load_json_object(path))
    other = pair_binding(reference)
    if other is None or other["comparison_role"] != "reference":
        raise ValueError("same-run peer is not a frozen reference arm")
    if reference["trajectory_id"] != binding["reference_trajectory_id"] or (
        other["spec_identity"], other["logical_run_id"], other["reference_trajectory_ref"], other["arm_id"]
    ) != (binding["spec_identity"], binding["logical_run_id"], binding["reference_trajectory_ref"], binding["reference_arm_id"]):
        raise ValueError("same-run peer binding differs from frozen candidate binding")
    if reference["state_at_start"]["source_commit"] != trajectory["state_at_start"]["source_commit"]:
        raise ValueError("same-run peer source commit differs")
    return reference


def terminal_reference_evidence(root: Path, trajectory: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Bind a completed same-run control, including one excluded from trace replay."""
    binding = pair_binding(trajectory)
    if binding is None:
        raise ValueError("trajectory has no same-run replay binding")
    reference = reference_trajectory(root, trajectory)
    reference_path = resolve_repo_pointer(root, binding["reference_trajectory_ref"])
    assert reference_path is not None
    destination = reference_path.parent / "rml_finalized"
    from .finalize import verified_receipt

    if not (destination / "finalization.json").is_file():
        raise ValueError("same-run reference is not yet terminally accepted")
    verified_receipt(destination)
    if (destination / "prospective_snapshot.json").read_bytes() != reference_path.read_bytes():
        raise ValueError("same-run reference prospective snapshot changed")
    finalized = load_json_object(destination / "trajectory.json")
    if (finalized["trajectory_id"] != reference["trajectory_id"]
            or finalized["state_at_start"] != reference["state_at_start"]
            or len(finalized["result"]["evidence_ids"]) != 1):
        raise ValueError("same-run reference has no unique accepted result")
    evidence = load_json_object(destination / "v5_evidence.json")
    evidence_id = finalized["result"]["evidence_ids"][0]
    if evidence["evidence_id"] != evidence_id or evidence["outcome"]["execution_status"] != "complete":
        raise ValueError("same-run reference evidence is incomplete")
    observation = validate_pair_observation(load_json_object(destination / "terminal_input.json").get("same_run_observation"))
    if (observation["spec_identity"], observation["logical_run_id"], observation["source_commit"]) != (
        binding["spec_identity"], binding["logical_run_id"], reference["state_at_start"]["source_commit"]
    ):
        raise ValueError("same-run reference observation differs from frozen plan")
    return evidence_id, finalized


def accepted_reference_evidence(root: Path, trajectory: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a replay-eligible control after its immutable finalization."""
    evidence_id, finalized = terminal_reference_evidence(root, trajectory)
    binding = pair_binding(trajectory)
    assert binding is not None
    reference_path = resolve_repo_pointer(root, binding["reference_trajectory_ref"])
    assert reference_path is not None
    destination = reference_path.parent / "rml_finalized"
    manifest_path = destination / "trace_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("same-run reference has no canonical trace manifest")
    manifest = load_json_object(manifest_path)
    if manifest["comparison_role"] != "reference" or manifest["reference_id"] != evidence_id or (
        not manifest["backtest_eligibility"]["eligible"] or manifest["backtest_eligibility"]["exclusion_reasons"]
    ):
        raise ValueError("same-run reference trace is not eligible")
    return evidence_id, finalized
