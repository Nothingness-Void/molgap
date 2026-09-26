"""Versioned descriptor translation into existing RML closure inputs.

Validation reads metadata and bound artifact bytes only. Execution delegates
explicitly to terminal_wiring; this module owns neither acceptance nor finalization.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from .experiment_spec import (
    ExperimentSpec, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL, _canonical, _digest, _identifier,
    _text, _unique_object,
)
from .screen_policy import canonical_fingerprint
from .research_memory.paths import repo_local_path, verify_bound_artifact
from .research_memory.schemas import validate_id, validate_trajectory
from .research_memory.trace import FIELDS, validate_canonical_trace


_TOP = frozenset({"schema_version", "spec_identity", "experiment_id", "logical_run_id", "arms"})
_ARM = frozenset({"arm_id", "arm_identity", "trajectory_id", "run_id", "trajectory", "terminal", "observed"})
_OPTIONAL = frozenset({"trace", "trace_source", "canonical_trace_output", "recovery_spec"})


def _fields(value: Any, required: frozenset, optional: frozenset, label: str) -> None:
    if type(value) is not dict or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError(f"{label}: missing or unknown fields")


def _path(value: Any) -> None:
    _text(value, "descriptor path")
    # Portable repository-relative spelling avoids host-dependent drive/ADS and
    # separator interpretation; repo_local_path additionally checks symlinks.
    if "\\" in value or ":" in value or value.startswith("/"):
        raise ValueError("descriptor paths must be repository-relative POSIX paths")
    for part in value.split("/"):
        if (part in {"", ".", ".."} or part.endswith((" ", "."))
                or any(ord(c) < 32 or c in '<>"|?*' for c in part)
                or PureWindowsPath(part).is_reserved()):
            raise ValueError("invalid descriptor path or path traversal")


def _fact(value: Any, label: str, validator) -> None:
    """Unknown observations have a reason, never a fabricated default."""
    _fields(value, frozenset({"value", "missing_reason"}), frozenset(), label)
    if value["value"] is None:
        _text(value["missing_reason"], label + ".missing_reason")
    else:
        if value["missing_reason"] is not None:
            raise ValueError(f"{label}: observed value cannot have missing_reason")
        validator(value["value"], label)


def _matches(value: Any, expected: Any, label: str) -> None:
    if _canonical(value) != _canonical(expected):
        raise ValueError(f"{label}: identity mismatch")


def _commit(value: Any, label: str) -> None:
    if type(value) is not str or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value):
        raise ValueError(f"{label}: expected lowercase full source commit")


def _progress(value: Any, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label}: expected nonnegative integer")


def _status(value: Any, label: str) -> None:
    if type(value) is not str or value not in {"complete", "failed", "cancelled", "interrupted"}:
        raise ValueError(f"{label}: unsupported terminal status")


def _observed(arm: dict, expected: dict, declaration: dict, spec_identity: str) -> None:
    observed = arm["observed"]
    _fields(observed, frozenset({"identity", "terminal", "artifacts", "progress", "costs", "missing_evidence"}),
            frozenset(), "observed")
    identity = observed["identity"]
    bindings = {
        "experiment_id": declaration["experiment_id"], "logical_run_id": declaration["logical_run_id"],
        "arm_id": arm["arm_id"], "spec_identity": spec_identity,
    }
    references = {
        "family": expected["family"],
        "recipe_identity": canonical_fingerprint(expected["training"]["recipe"]),
        "data_identity": canonical_fingerprint(expected["data"]),
        "split_identity": canonical_fingerprint({"split": expected["data"]["split"], "roles": expected["data"]["roles"]}),
        "feature_identity": expected["data"]["feature_sha256"], "target": expected["data"]["target"],
        "initialization_identity": canonical_fingerprint(expected["initialization"]),
    }
    _fields(identity, frozenset(bindings.keys() | references.keys() | {
        "attempt_id", "platform", "source_commit", "source_package_sha256",
    }), frozenset(), "observed.identity")
    for field, value in bindings.items():
        _matches(identity[field], value, "observed.identity." + field)
    for field, value in references.items():
        _fact(identity[field], "observed.identity." + field,
              lambda observed_value, label, expected_value=value: _matches(observed_value, expected_value, label))
    for field, validator in (("attempt_id", _identifier), ("source_commit", _commit), ("source_package_sha256", _digest)):
        _fact(identity[field], "observed.identity." + field, validator)
    platform = identity["platform"]
    _fields(platform, frozenset({"name", "run_reference"}), frozenset(), "observed.identity.platform")
    _matches(platform["name"], declaration["platform"]["name"], "observed.identity.platform.name")
    _fact(platform["run_reference"], "observed.identity.platform.run_reference", _text)
    terminal = observed["terminal"]
    _fields(terminal, frozenset({"status", "exit_reason"}), frozenset(), "observed.terminal")
    _fact(terminal["status"], "observed.terminal.status", _status)
    _fact(terminal["exit_reason"], "observed.terminal.exit_reason", _text)
    artifacts = observed["artifacts"]
    _fields(artifacts, frozenset({"metrics", "predictions", "checkpoint", "trace"}), frozenset(), "observed.artifacts")
    for name, artifact in artifacts.items():
        label = "observed.artifacts." + name
        _fields(artifact, frozenset({"status", "locator", "sha256", "missing_reason"}), frozenset(), label)
        if artifact["locator"] is not None:
            _path(artifact["locator"])
        if artifact["status"] == "available":
            _path(artifact["locator"])
            _digest(artifact["sha256"], label + ".sha256")
            if artifact["missing_reason"] is not None:
                raise ValueError(f"{label}: available artifact cannot have missing_reason")
        elif artifact["status"] == "missing":
            if artifact["sha256"] is not None:
                raise ValueError(f"{label}: missing artifact must have null sha256")
            _text(artifact["missing_reason"], label + ".missing_reason")
        else:
            raise ValueError(f"{label}: unsupported artifact status")
    progress = observed["progress"]
    _fields(progress, frozenset({"epoch", "step", "samples"}), frozenset(), "observed.progress")
    for field, value in progress.items():
        _fact(value, "observed.progress." + field, _progress)
    costs = observed["costs"]
    if type(costs) is not list or not costs:
        raise ValueError("observed.costs: expected nonempty array; declare missing measurements")
    seen = set()
    for cost in costs:
        _fields(cost, frozenset({"metric", "unit", "value", "status", "reason"}), frozenset(), "observed.cost")
        _identifier(cost["metric"], "observed.cost.metric")
        if cost["metric"] in seen:
            raise ValueError("duplicate cost metric")
        seen.add(cost["metric"])
        if cost["unit"] is not None:
            _text(cost["unit"], "observed.cost.unit")
        if cost["status"] == "measurement_missing":
            if cost["value"] is not None:
                raise ValueError("missing cost measurement must have null value")
            _text(cost["reason"], "observed.cost.reason")
        elif cost["status"] in ("measured", "estimated"):
            _text(cost["unit"], "observed.cost.unit")
            value = cost["value"]
            if type(value) not in (int, float) or value < 0 or (type(value) is float and not math.isfinite(value)):
                raise ValueError("cost value must be finite and nonnegative")
            if cost["status"] == "estimated":
                _text(cost["reason"], "observed.cost.estimate_basis")
            elif cost["reason"] is not None:
                raise ValueError("measured cost must have null reason")
        else:
            raise ValueError("unsupported cost status")
    reasons = observed["missing_evidence"]
    if type(reasons) is not list:
        raise ValueError("missing_evidence must be an array of reasons")
    for reason in reasons:
        _text(reason, "missing_evidence reason")


def _validate(spec: ExperimentSpec, payload: dict) -> dict:
    if type(spec) is not ExperimentSpec:
        raise ValueError("a validated ExperimentSpec is required")
    _fields(payload, _TOP, frozenset(), "descriptor")
    if payload["schema_version"] != TERMINAL_PROTOCOL:
        raise ValueError("unsupported terminal descriptor protocol")
    declaration = spec.to_dict()
    _digest(payload["spec_identity"], "spec_identity")
    if payload["spec_identity"] != spec.identity:
        raise ValueError("descriptor spec identity mismatch")
    for field in ("experiment_id", "logical_run_id"):
        if payload[field] != declaration[field]:
            raise ValueError(f"descriptor {field} mismatch")
    arms = payload["arms"]
    if type(arms) is not list or not arms:
        raise ValueError("descriptor arms must be a nonempty array")
    expected = {arm["arm_id"]: arm for arm in declaration["arms"]}
    prospective = ({entry["arm_id"]: entry for entry in declaration["prospective"]["arms"]}
                   if declaration["schema_version"] == SCHEMA_VERSION_V2 else None)
    seen: set[str] = set()
    trajectories: set[str] = set()
    inputs: set[str] = set()
    for arm in arms:
        _fields(arm, _ARM, _OPTIONAL, "descriptor arm")
        _identifier(arm["arm_id"], "arm_id")
        key = arm["arm_id"]
        if key not in expected or key in seen:
            raise ValueError("unknown or duplicate descriptor arm")
        seen.add(key)
        _digest(arm["arm_identity"], "arm_identity")
        if arm["arm_identity"] != canonical_fingerprint(expected[key]):
            raise ValueError("descriptor arm identity mismatch")
        _observed(arm, expected[key], declaration, spec.identity)
        for field in ("trajectory_id", "run_id"):
            validate_id(arm[field], field)
        if prospective is not None and arm["trajectory_id"] != prospective[key]["trajectory_id"]:
            raise ValueError("descriptor prospective trajectory identity mismatch")
        if arm["trajectory_id"] in trajectories:
            raise ValueError("duplicate trajectory identity; expected 1:1 mapping")
        trajectories.add(arm["trajectory_id"])
        for field in ("trajectory", "terminal", *sorted(_OPTIONAL & arm.keys())):
            _path(arm[field])
        for field in ("trajectory", "terminal"):
            if arm[field] in inputs:
                raise ValueError("duplicate trajectory/terminal path; expected 1:1 mapping")
            inputs.add(arm[field])
    if seen != expected.keys():
        raise ValueError("missing descriptor arms")
    return payload


@dataclass(frozen=True, init=False)
class TerminalDescriptor:
    """Immutable declaration snapshot; construction does not read or write files."""

    _canonical_json: str

    def __init__(self, spec: ExperimentSpec, payload: dict):
        object.__setattr__(self, "_canonical_json", _canonical(_validate(spec, payload)))

    @classmethod
    def from_json(cls, spec: ExperimentSpec, text: str) -> TerminalDescriptor:
        return cls(spec, json.loads(text, object_pairs_hook=_unique_object))

    def to_json(self) -> str:
        return self._canonical_json

    def to_dict(self) -> dict:
        return json.loads(self._canonical_json)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if type(value) is not dict:
        raise ValueError(f"metadata must be a JSON object: {path}")
    # Reject non-standard NaN/Infinity even in fields owned by downstream RML.
    _canonical(value)
    return value


def _recovery(value: dict, arm: dict) -> dict:
    _fields(value, frozenset({"metric_semantics"}), frozenset({
        "trajectory_id", "run_id", "rows_key", "field_mapping", "device_time_semantics",
    }), "recovery_spec")
    for field in ("trajectory_id", "run_id"):
        if field in value and value[field] != arm[field]:
            raise ValueError(f"recovery {field} mismatch")
    if "rows_key" in value:
        _text(value["rows_key"], "recovery rows_key")
    if "field_mapping" in value:
        mapping = value["field_mapping"]
        if type(mapping) is not dict or mapping.keys() - set(FIELDS):
            raise ValueError("recovery mapping uses unknown canonical fields")
        for column in mapping.values():
            _text(column, "recovery source column")
    semantics = value["metric_semantics"]
    if type(semantics) is dict:
        for definition in semantics.values():
            if definition is not None:
                _fields(definition, frozenset({
                    "metric", "unit", "target", "role_identity", "weights", "direction",
                }), frozenset(), "recovery metric semantics")
    if value.get("device_time_semantics") not in (None, "sum_over_devices"):
        raise ValueError("unsupported recovery device time semantics")
    # Reuse the trace validator for semantics without recovering observations or
    # supplying this temporary validation envelope as evidence to the closure.
    validate_canonical_trace({
        "schema": "molgap-trace-v1", "trajectory_id": arm["trajectory_id"],
        "run_id": arm["run_id"], "metric_semantics": semantics, "observations": [],
    })
    return value


def translate_terminal_descriptor(
    repo_root: str | Path, spec: ExperimentSpec, descriptor: TerminalDescriptor,
) -> list[dict[str, Any]]:
    """Read-only binding checks; return exactly the close_terminal_multi_arm arms.

    The descriptor keeps recovery_spec's path; the closure API requires its parsed
    object. All other explicit paths retain their original spelling. Omitted
    optional fields stay omitted. This is not a terminal acceptance dry run.
    """
    if type(descriptor) is not TerminalDescriptor:
        raise ValueError("a validated TerminalDescriptor is required")
    payload = _validate(spec, descriptor.to_dict())
    root = Path(repo_root).resolve()
    declaration = spec.to_dict()
    expected = {arm["arm_id"]: arm for arm in declaration["arms"]}
    from .research_memory.paired import pair_binding, validate_pair_observation
    v2 = declaration["schema_version"] == SCHEMA_VERSION_V2
    resolved = []
    paired_observations = []
    inputs: set[Path] = set()
    destinations: set[Path] = set()
    outputs: set[Path] = set()
    # Resolve every declared path before opening metadata, including outputs that
    # do not yet exist; aliases must not collapse independent RML transactions.
    for arm in payload["arms"]:
        paths = {field: repo_local_path(root, arm[field])
                 for field in ("trajectory", "terminal", *sorted(_OPTIONAL & arm.keys()))}
        for field, path in paths.items():
            if field == "canonical_trace_output":
                if path in outputs:
                    raise ValueError("duplicate canonical trace output")
                outputs.add(path)
            else:
                if not path.is_file():
                    raise ValueError(f"missing descriptor input file: {arm[field]}")
                if field in {"trajectory", "terminal"} and path in inputs:
                    raise ValueError("aliased trajectory/terminal mapping")
                inputs.add(path)
        destination = repo_local_path(root, paths["trajectory"].parent / "rml_finalized")
        if destination in destinations:
            raise ValueError("trajectories share an RML finalization directory")
        destinations.add(destination)
        resolved.append(paths)
        for artifact in arm["observed"]["artifacts"].values():
            # Even a known locator for missing evidence must stay inside the repo.
            if artifact["locator"] is not None:
                artifact_path = repo_local_path(root, artifact["locator"])
                inputs.add(artifact_path)
            if artifact["status"] == "available":
                if not artifact_path.is_file():
                    raise ValueError(f"missing available artifact: {artifact['locator']}")
                verify_bound_artifact(root, artifact["locator"], artifact["sha256"])
    if outputs & inputs:
        raise ValueError("canonical trace output aliases an input")

    translated = []
    for arm, paths in zip(payload["arms"], resolved):
        trajectory = validate_trajectory(_read(paths["trajectory"]))
        terminal = _read(paths["terminal"])
        if trajectory["record_mode"] != "prospective" or trajectory["owner"] != "server":
            raise ValueError("server closure requires a server-owned prospective trajectory")
        if terminal.get("format") != "molgap-rml-terminal-package-v1":
            raise ValueError("unsupported existing RML terminal package")
        if trajectory["trajectory_id"] != arm["trajectory_id"] or terminal.get("trajectory_id") != arm["trajectory_id"]:
            raise ValueError("trajectory/terminal identity mismatch")
        if (v2 and trajectory["state_at_start"]["source_config_identity"]
                != canonical_fingerprint(expected[arm["arm_id"]])):
            raise ValueError("prospective trajectory arm identity mismatch")
        paired = pair_binding(trajectory)
        if paired is not None:
            same_run = declaration["prospective"].get("same_run_replay") if v2 else None
            if (same_run is None or paired["reference_arm_id"] != same_run["reference_arm_id"]
                    or (arm["arm_id"] != same_run["reference_arm_id"]
                        and arm["arm_id"] not in same_run["candidate_arm_ids"])):
                raise ValueError("same-run replay was not declared by the frozen ExperimentSpec")
            prospective = {entry["arm_id"]: entry for entry in declaration["prospective"]["arms"]}
            reference = prospective[paired["reference_arm_id"]]
            if (paired["spec_identity"], paired["logical_run_id"], paired["arm_id"],
                    paired["comparison_role"], paired["reference_trajectory_id"], paired["reference_trajectory_ref"]) != (
                    spec.identity, declaration["logical_run_id"], arm["arm_id"],
                    expected[arm["arm_id"]]["scientific_role"], reference["trajectory_id"],
                    reference["output"] + "/trajectory.json"):
                raise ValueError("same-run replay binding differs from frozen ExperimentSpec")
            manifest = terminal.get("trace_manifest")
            if isinstance(manifest, dict) and manifest.get("backtest_eligibility", {}).get("eligible") is True:
                observed = arm["observed"]["identity"]
                observation = validate_pair_observation(terminal.get("same_run_observation"))
                expected_observation = {
                    "schema": observation["schema"], "spec_identity": spec.identity,
                    "logical_run_id": declaration["logical_run_id"],
                    "platform_name": observed["platform"]["name"],
                    "platform_run_reference": observed["platform"]["run_reference"]["value"],
                    "attempt_id": observed["attempt_id"]["value"],
                    "source_commit": observed["source_commit"]["value"],
                    "source_package_sha256": observed["source_package_sha256"]["value"],
                }
                if observation != expected_observation:
                    raise ValueError("same-run terminal observation differs from descriptor evidence")
                paired_observations.append(observation)
        if terminal.get("run_id") != arm["run_id"]:
            raise ValueError("terminal run identity mismatch")
        if not any(action["action_id"] == terminal.get("action_id") and arm["run_id"] in action["run_ids"]
                   for action in trajectory["actions"]):
            raise ValueError("terminal run/action was not frozen in trajectory")
        action = next(action for action in trajectory["actions"]
                      if action["action_id"] == terminal["action_id"] and arm["run_id"] in action["run_ids"])
        identity = arm["observed"]["identity"]
        attempt = identity["attempt_id"]["value"]
        if attempt is not None and attempt not in action["attempt_ids"]:
            raise ValueError("observed attempt identity mismatch with frozen action")
        commit = identity["source_commit"]["value"]
        if commit is not None and commit != action["source_commit"]:
            raise ValueError("observed source commit identity mismatch with frozen action")
        if "trace" in paths:
            trace = validate_canonical_trace(_read(paths["trace"]))
            if any(trace[field] != arm[field] for field in ("trajectory_id", "run_id")):
                raise ValueError("trace identity mismatch")
        item = {"arm_identifier": arm["arm_id"], "trajectory": arm["trajectory"], "terminal": arm["terminal"]}
        item.update({field: arm[field] for field in sorted(_OPTIONAL & arm.keys())})
        if "recovery_spec" in paths:
            item["recovery_spec"] = _recovery(_read(paths["recovery_spec"]), arm)
        translated.append(item)
    if paired_observations and any(row != paired_observations[0] for row in paired_observations[1:]):
        raise ValueError("same-run arms have inconsistent observed platform jobs")
    return translated


def execute_terminal_descriptor(
    repo_root: str | Path, spec: ExperimentSpec, descriptor: TerminalDescriptor,
) -> list[dict[str, Any]]:
    """Explicitly run the existing fail-closed closure, including its side effects."""
    from .research_memory.terminal_wiring import close_terminal_multi_arm

    translated = translate_terminal_descriptor(repo_root, spec, descriptor)
    from .research_memory.paired import pair_binding

    if not any(pair_binding(validate_trajectory(_read(repo_local_path(Path(repo_root).resolve(), arm["trajectory"]))))
               is not None for arm in translated):
        return close_terminal_multi_arm(repo_root, translated)
    roles = {arm["arm_id"]: arm["scientific_role"] for arm in spec.to_dict()["arms"]}
    ordered = sorted(translated, key=lambda item: 0 if roles[item["arm_identifier"]] == "reference" else 1)
    results = close_terminal_multi_arm(repo_root, ordered)
    by_arm = {arm["arm_identifier"]: result for arm, result in zip(ordered, results)}
    return [by_arm[arm["arm_identifier"]] for arm in translated]
