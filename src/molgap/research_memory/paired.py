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


def _continuation_attempt_id(
    origin_attempt_id: str, platform_version: str | None, *, origin: bool
) -> str:
    """Map the legacy continuation ABI without manufacturing the origin ID."""
    if origin:
        return origin_attempt_id
    if not isinstance(platform_version, str) or not platform_version:
        raise ValueError("continued receipt lacks an observed physical platform version")
    return f"{origin_attempt_id}-v{platform_version}"


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


def validate_terminal_continuation(
    root: Path,
    terminal: Mapping[str, Any],
    *,
    require_strict: bool = False,
    _native_verification=None,
) -> dict[str, str]:
    """Bind a physical successor to retained state; V5 owns eligibility."""
    from .paths import artifact_locator, bound_artifact_path, verify_bound_artifact
    from .trace import load_canonical_trace

    pointer = terminal.get("continuation_ref")
    bindings = terminal.get("artifact_hashes", {})
    if not isinstance(pointer, str) or pointer not in bindings:
        raise ValueError("different platform jobs require a bound continuation record")
    verify_bound_artifact(root, pointer, bindings[pointer])
    path = bound_artifact_path(root, pointer)
    record = load_json_object(path)
    common = {
        "format", "origin", "continued", "source_trace", "continued_trace",
        "source_context", "continued_manifest", "checkpoint", "authority",
    }
    strict_fields = {
        "source_package", "origin_receipt", "continued_receipt",
        "native_certificate", "actual_resume_certificate",
    }
    optional = strict_fields | {"recovery_repair"}
    resume_key = "resume_bundle" if "resume_bundle" in record else "resume_binding"
    if (
        not common | {resume_key} <= record.keys()
        or record.keys() - common - {resume_key} - optional
        or record["format"] != "molgap-terminal-continuation-v1"
    ):
        raise ValueError("invalid terminal continuation record")
    origin = validate_pair_observation(record["origin"])
    continued = validate_pair_observation(record["continued"])
    if continued != validate_pair_observation(terminal.get("same_run_observation")):
        raise ValueError("continuation terminal job mismatch")
    for key in (
        "spec_identity", "logical_run_id", "source_commit",
        "source_package_sha256", "platform_name",
    ):
        if origin[key] != continued[key]:
            raise ValueError("continuation changed frozen training identity")
    retained = {}
    for key in (
        "source_trace", "continued_trace", "source_context", resume_key,
        "continued_manifest", "checkpoint", "authority",
        *sorted(optional & record.keys()),
    ):
        item = record[key]
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256"}
            or bindings.get(item["path"]) != item["sha256"]
        ):
            raise ValueError("continuation artifact is not bound")
        verify_bound_artifact(root, item["path"], item["sha256"])
        retained[key] = bound_artifact_path(root, item["path"])
    source = load_canonical_trace(retained["source_trace"])
    final = load_canonical_trace(retained["continued_trace"])
    for key in ("trajectory_id", "run_id", "metric_semantics"):
        if source[key] != final[key]:
            raise ValueError("continuation trace identity changed")
    before, after = source["observations"], final["observations"]
    if not before or len(after) <= len(before) or after[: len(before)] != before:
        raise ValueError("continuation must preserve an exact retained trace prefix")
    if final["run_id"] != terminal["run_id"] or final["trajectory_id"] != terminal["trajectory_id"]:
        raise ValueError("continuation trace is not the terminal run")
    manifest = load_json_object(retained["continued_manifest"])
    if (
        manifest.get("format") != "molgap-family-output-v1"
        or manifest["artifacts"]["trace"]["sha256"] != record["continued_trace"]["sha256"]
    ):
        raise ValueError("continuation output/trace binding mismatch")
    context = load_json_object(retained["source_context"])["context"]
    if (
        {k: v for k, v in context.items() if k != "platform_version"}
        != {k: v for k, v in manifest["context"].items() if k != "platform_version"}
    ):
        raise ValueError("continuation output context changed")
    for field, observed_field in (
        ("spec_identity", "spec_identity"),
        ("logical_run_id", "logical_run_id"),
        ("source_commit", "source_commit"),
        ("source_archive_sha256", "source_package_sha256"),
        ("platform", "platform_name"),
    ):
        if context[field] != origin[observed_field]:
            raise ValueError("continuation observation/output identity mismatch")
    for observation in (origin, continued):
        if observation["platform_run_reference"].partition("@")[0] != context["run_reference"]:
            raise ValueError("continuation observation/output run mismatch")
    for observed, identity in ((context, origin), (manifest["context"], continued)):
        version = observed.get("platform_version")
        if version is not None and not identity["attempt_id"].endswith("-v" + str(version)):
            raise ValueError("continuation producer version differs from its receipt observation")
    if resume_key == "resume_bundle":
        from molgap.experiment_resume import validate_resume_bundle

        checked = validate_resume_bundle(retained[resume_key], context=context)
        bundle = checked["manifest"]
        if (
            bundle["artifacts"]["checkpoint"]["sha256"] != record["checkpoint"]["sha256"]
            or bundle["artifacts"]["trace"]["sha256"] != record["source_trace"]["sha256"]
            or bundle["trajectory"]["id"] != terminal["trajectory_id"]
            or bundle["run"]["run_id"] != terminal["run_id"]
            or bundle["run"]["attempt_id"] != origin["attempt_id"]
        ):
            raise ValueError("continuation bundle is not the retained source endpoint")
    else:
        resume = load_json_object(retained[resume_key])
        if (
            resume["spec_identity"], resume["package_identity"], resume["arm_id"],
            resume["start_epoch"], resume["end_epoch"],
        ) != (
            context["spec_identity"], context["package_identity"], context["arm_id"],
            before[-1]["epoch_or_pass"], after[-1]["epoch_or_pass"],
        ):
            raise ValueError("continuation resume identity/progress mismatch")
        if (
            resume["files"].get(retained["checkpoint"].name) != record["checkpoint"]["sha256"]
            or resume["files"].get(retained["source_trace"].name) != record["source_trace"]["sha256"]
        ):
            raise ValueError("continuation checkpoint is not the retained source endpoint")
    if not any(
        artifact.get("locator") == record["continued_trace"]["path"]
        and artifact.get("sha256") == record["continued_trace"]["sha256"]
        for artifact in terminal["evidence"]["artifacts"]
    ):
        raise ValueError("continuation trace is absent from accepted evidence")
    if terminal["trace_manifest"]["trace_artifact_ref"] != record["continued_trace"]["path"]:
        raise ValueError("continuation manifest selects a different trace")
    if require_strict:
        _validate_strict_continuation(
            root, terminal, record, retained, origin, continued, manifest,
            bundle if resume_key == "resume_bundle" else None,
            strict_fields, bindings, _native_verification,
        )
    return origin


def validate_ineligible_continuation(
    root: Path, terminal: Mapping[str, Any]
) -> dict[str, str]:
    """Compatibility route for retained continuations without replay admission."""
    return validate_terminal_continuation(root, terminal)


def _validate_strict_continuation(
    root: Path,
    terminal: Mapping[str, Any],
    record: Mapping[str, Any],
    retained: Mapping[str, Path],
    origin: Mapping[str, str],
    continued: Mapping[str, str],
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any] | None,
    strict_fields: set[str],
    bindings: Mapping[str, str],
    native_verification,
) -> None:
    """Run the expensive continuation qualification only on explicit request."""
    if bundle is None or not strict_fields <= retained.keys():
        raise ValueError(
            "strict continuation requires a verified bundle, receipts, package and native qualification"
        )
    from molgap.experiment_spec import ExperimentSpec
    from molgap.experiment_family_workflow import RunContext, _check_context
    from molgap.experiment_package import read_packaged_text, verify_experiment_source_package
    from molgap.experiment_resume import validate_resume_bundle
    from molgap.family_training_runner import inspect_retained_native_qualification
    from .paths import artifact_locator, verify_bound_artifact

    package_dir = retained["source_package"].parent
    if retained["source_package"].name != "package_manifest.json":
        raise ValueError("strict continuation source package must bind its manifest")
    package = verify_experiment_source_package(package_dir)
    spec = ExperimentSpec.from_json((package_dir / "experiment_spec.json").read_text(encoding="utf-8"))
    context = load_json_object(retained["source_context"])["context"]
    receipt_contexts = []
    for key, observation in (("origin_receipt", origin), ("continued_receipt", continued)):
        observed_context = RunContext.from_launch(
            spec, retained[key], package_dir,
            expected_package_identity=package["package_identity"], arm_id=context["arm_id"],
        )
        _check_context(context if key == "origin_receipt" else manifest["context"], observed_context)
        if (
            observed_context.run_reference != observation["platform_run_reference"]
        ):
            raise ValueError("strict continuation receipt/attempt identity mismatch")
        receipt_contexts.append(observed_context)
    if origin["attempt_id"] != bundle["run"]["attempt_id"]:
        raise ValueError("strict continuation origin attempt differs from frozen resume bundle")
    if continued["attempt_id"] != _continuation_attempt_id(
        origin["attempt_id"], receipt_contexts[1].platform_version, origin=False
    ):
        raise ValueError("strict continuation successor attempt differs from physical receipt version")
    versions = [item.platform_version for item in receipt_contexts]
    if (
        not all(value is not None and value.isdecimal() for value in versions)
        or int(versions[1]) <= int(versions[0])
    ):
        raise ValueError("strict continuation successor must follow its origin")
    prospective = next(
        item for item in spec.to_dict()["prospective"]["arms"] if item["arm_id"] == context["arm_id"]
    )
    validate_resume_bundle(
        retained["resume_bundle"], spec, arm_id=context["arm_id"], context=context,
        trajectory=root / prospective["output"] / "trajectory.json",
    )
    recipe = manifest["artifacts"]["contract"]
    recipe_path = retained["continued_manifest"].parent / recipe["path"]
    verify_bound_artifact(root, artifact_locator(root, recipe_path), recipe["sha256"])
    native_root = retained["native_certificate"].parent
    if (
        retained["native_certificate"].name != "runtime_certificate.json"
        or retained["actual_resume_certificate"].name != "actual_resume_certificate.json"
    ):
        raise ValueError("strict continuation must bind the native certificate files")
    from molgap.screen_policy import canonical_fingerprint

    if manifest.get("runtime", {}).get("runtime_certificate_id") != canonical_fingerprint(
        load_json_object(retained["native_certificate"])
    ):
        raise ValueError("strict continuation output/native certificate mismatch")
    original_provenance = [
        path for path in validate_resume_bundle(retained["resume_bundle"], spec, context=context)["sidecars"]["provenance"]
        if path.name == "runtime_provenance.json"
    ]
    provenance = load_json_object(native_root / "runtime_provenance.json")
    if len(original_provenance) != 1 or load_json_object(original_provenance[0]) != provenance:
        raise ValueError("strict continuation changed the original runtime/data/initialization identity")
    checkpoint_sha = bundle["artifacts"]["checkpoint"]["sha256"]
    if native_verification is None:
        native = inspect_retained_native_qualification(
            native_root, recipe_path=recipe_path, provenance=provenance,
            context=RunContext(**context), resume_checkpoint_sha256=checkpoint_sha,
            resume_directory=retained["actual_resume_certificate"].parent,
        )
    else:
        from .trace import file_digest

        native, verified_checkpoint = native_verification
        if verified_checkpoint != checkpoint_sha:
            raise ValueError("Cached continuation proof belongs to another checkpoint")
        dependencies = {Path(item["local_path"]).resolve(): item for item in native["dependencies"]}
        required = {retained["native_certificate"].resolve(), retained["actual_resume_certificate"].resolve()}
        if not required <= dependencies.keys():
            raise ValueError("Cached continuation proof omits native/resume certificates")
        for path, item in dependencies.items():
            if file_digest(path) != item.get("verified_sha256"):
                raise ValueError("Continuation proof dependency changed after verification")
    if native["status"] != "VERIFIED":
        raise ValueError("strict continuation native qualification is incomplete: " + str(native))
    if "recovery_repair" in retained:
        import hashlib

        repair = load_json_object(retained["recovery_repair"])
        if (
            repair.get("format") != "molgap-recovery-repair-v1"
            or repair.get("spec_identity") != spec.identity
            or repair.get("package_identity") != package["package_identity"]
            or repair.get("archive_sha256") != context["source_archive_sha256"]
            or not _COMMIT.fullmatch(repair.get("commit", ""))
        ):
            raise ValueError("strict continuation repair identity mismatch")
        reviewed = {"src/molgap/common/io.py", "src/molgap/experiment_resume.py", "src/molgap/family_training_runner.py"}
        if not repair.get("files") or repair["files"].keys() - reviewed:
            raise ValueError("strict continuation repair exceeds the reviewed infrastructure scope")
        for name, pins in repair["files"].items():
            original = read_packaged_text(package_dir, name).encode("utf-8")
            if hashlib.sha256(original).hexdigest() != pins["original_sha256"]:
                raise ValueError("strict continuation repair original source mismatch")
            matches = [pointer for pointer, digest in bindings.items()
                       if pointer.endswith("/" + name) and digest == pins["repaired_sha256"]]
            if len(matches) != 1:
                raise ValueError("strict continuation repaired source is not independently bound")
            verify_bound_artifact(root, matches[0], pins["repaired_sha256"])


def build_terminal_continuation(
    root: Path,
    *,
    package_dir: Path,
    origin_receipt: Path,
    continued_receipt: Path,
    resume_bundle: Path,
    continued_output: Path,
    authority: Path,
    output: Path,
    native_root: Path,
    actual_resume_root: Path,
    recovery_repair: Path | None = None,
    repair_source_root: Path | None = None,
) -> dict[str, Any]:
    """Stage retained continuation evidence without changing its frozen plan."""
    from molgap.experiment_family_workflow import RunContext
    from molgap.experiment_launch import publish_immutable_bytes
    from molgap.experiment_package import verify_experiment_source_package
    from molgap.experiment_resume import validate_resume_bundle
    from molgap.experiment_spec import ExperimentSpec
    from .paths import artifact_locator, repo_local_path
    from .trace import file_digest, json_bytes

    package = verify_experiment_source_package(package_dir)
    spec = ExperimentSpec.from_json(
        (package_dir / "experiment_spec.json").read_text(encoding="utf-8")
    )
    checked = validate_resume_bundle(resume_bundle, spec)
    bundle, arm_id = checked["manifest"], checked["manifest"]["arm"]["id"]
    manifest = load_json_object(continued_output / "output_manifest.json")
    hashes: dict[str, str] = {}

    def pin(path: Path) -> dict[str, str]:
        pointer = artifact_locator(root, path)
        digest = file_digest(path)
        hashes[pointer] = digest
        return {"path": pointer, "sha256": digest}

    origin_attempt_id = bundle["run"]["attempt_id"]

    def observation(receipt: Path, *, origin: bool) -> dict[str, str]:
        context = RunContext.from_launch(
            spec,
            receipt,
            package_dir,
            expected_package_identity=package["package_identity"],
            arm_id=arm_id,
        )
        return {
            "schema": PAIR_OBSERVATION_SCHEMA,
            "spec_identity": spec.identity,
            "logical_run_id": context.logical_run_id,
            "platform_name": context.platform,
            "platform_run_reference": context.run_reference,
            "attempt_id": _continuation_attempt_id(
                origin_attempt_id, context.platform_version, origin=origin
            ),
            "source_commit": context.source_commit,
            "source_package_sha256": context.source_archive_sha256,
        }

    context_entries = [
        path for path in checked["sidecars"]["context"] if "context" in load_json_object(path)
    ]
    if len(context_entries) != 1:
        raise ValueError("Continuation requires a unique retained source context")
    record: dict[str, Any] = {
        "format": "molgap-terminal-continuation-v1",
        "origin": observation(origin_receipt, origin=True),
        "continued": observation(continued_receipt, origin=False),
        "source_trace": pin(checked["artifacts"]["trace"]),
        "continued_trace": pin(
            repo_local_path(continued_output, manifest["artifacts"]["trace"]["path"])
        ),
        "source_context": pin(context_entries[0]),
        "resume_bundle": pin(checked["manifest_path"]),
        "continued_manifest": pin(continued_output / "output_manifest.json"),
        "checkpoint": pin(checked["artifacts"]["checkpoint"]),
        "authority": pin(authority),
        "source_package": pin(package_dir / "package_manifest.json"),
        "origin_receipt": pin(origin_receipt),
        "continued_receipt": pin(continued_receipt),
        "native_certificate": pin(native_root / "runtime_certificate.json"),
        "actual_resume_certificate": pin(
            actual_resume_root / "actual_resume_certificate.json"
        ),
    }
    if recovery_repair is not None:
        if repair_source_root is None:
            raise ValueError("Repair evidence requires its retained source root")
        record["recovery_repair"] = pin(recovery_repair)
        for name in load_json_object(recovery_repair)["files"]:
            pin(repair_source_root / name)
    publish_immutable_bytes(output.absolute(), json_bytes(record))
    record_pin = pin(output)
    return {
        "continuation_ref": record_pin["path"],
        "artifact_hashes": hashes,
        "same_run_observation": record["continued"],
        "origin": record["origin"],
    }


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
    """Bind a completed same-run control, including an ineligible trace."""
    binding = pair_binding(trajectory)
    if binding is None:
        raise ValueError("trajectory has no same-run replay binding")
    reference = reference_trajectory(root, trajectory)
    reference_path = resolve_repo_pointer(root, binding["reference_trajectory_ref"])
    assert reference_path is not None
    from .finalize import current_finalized_directory, load_terminal_input, prospective_bytes, verified_receipt

    destination = current_finalized_directory(root, reference_path)

    if not (destination / "finalization.json").is_file():
        raise ValueError("same-run reference is not yet terminally accepted")
    verified_receipt(destination)
    if prospective_bytes(destination) != reference_path.read_bytes():
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
    manifest_path = destination / "trace_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("same-run reference has no canonical trace manifest")
    manifest = load_json_object(manifest_path)
    if manifest["comparison_role"] != "reference" or manifest["reference_id"] != evidence_id:
        raise ValueError("same-run reference trace differs from its accepted evidence")
    observation = validate_pair_observation(load_terminal_input(destination).get("same_run_observation"))
    if (observation["spec_identity"], observation["logical_run_id"], observation["source_commit"]) != (
        binding["spec_identity"], binding["logical_run_id"], reference["state_at_start"]["source_commit"]
    ):
        raise ValueError("same-run reference observation differs from frozen plan")
    return evidence_id, finalized


def accepted_reference_evidence(root: Path, trajectory: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a replay-eligible control after immutable terminal acceptance."""
    evidence_id, finalized = terminal_reference_evidence(root, trajectory)
    binding = pair_binding(trajectory)
    assert binding is not None
    reference_path = resolve_repo_pointer(root, binding["reference_trajectory_ref"])
    assert reference_path is not None
    from .finalize import current_finalized_directory

    destination = current_finalized_directory(root, reference_path)
    manifest_path = destination / "trace_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("same-run reference has no canonical trace manifest")
    manifest = load_json_object(manifest_path)
    if (
        manifest["comparison_role"] != "reference"
        or manifest["reference_id"] != evidence_id
        or not manifest["backtest_eligibility"]["eligible"]
        or manifest["backtest_eligibility"]["exclusion_reasons"]
    ):
        raise ValueError("same-run reference trace is not eligible")
    return evidence_id, finalized
