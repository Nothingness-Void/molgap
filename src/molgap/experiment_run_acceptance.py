"""Resolve acceptance inputs from one frozen run without moving its artifacts."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .common.identity import validate_execution_identity
from .common.paths import MachinePaths, validate_relative_path
from .experiment_family_artifacts import artifact_adapter
from .experiment_family_workflow import (EXPECTED, EXPECTED_FAMILY, RunContext,
                                         TargetIdentityBinding, _check_context, _json)
from .experiment_launch import _safe_local, canonical_json, publish_immutable_bytes
from .experiment_package import read_packaged_text
from .experiment_spec import _canonical, _digest, _unique_object
from .research_memory.paths import artifact_locator, bound_artifact_path, repo_local_path
from .research_memory.trace import file_digest
from .screen_policy import canonical_fingerprint


def _packaged_json(package, name, digest=None, *, _verification=None):
    validate_relative_path(name)
    raw = read_packaged_text(package, name, _verification=_verification)
    if digest is not None and hashlib.sha256(raw.encode()).hexdigest() != digest:
        raise ValueError("Packaged acceptance input differs from its frozen pin")
    value = json.loads(raw, object_pairs_hook=_unique_object)
    _canonical(value)
    if type(value) is not dict:
        raise ValueError("Expected a packaged acceptance object")
    return value


def _output_path(value, *, paths, repo_root, run, external_only=False):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("Expected an explicit retained output directory")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    _safe_local(path)
    path = path.absolute()
    checkout = Path(repo_root).absolute()
    if path.is_relative_to(run) or run.is_relative_to(path):
        raise ValueError("Retained outputs must not overlap frozen run records")
    if path == checkout or path == paths.artifact_root:
        raise ValueError("A storage root is not an arm output directory")
    if not ((not external_only and path.is_relative_to(checkout))
            or path.is_relative_to(paths.artifact_root)):
        raise ValueError("Outputs must be retained in the checkout or configured artifact_root")
    return path


def _mapping_path(run, arm_id, platform_version=None):
    """Return the immutable retrieval map for an original or successor run."""
    directory = run / "operations" / "retrieved_outputs"
    if platform_version is not None:
        validate_relative_path(platform_version)
        directory /= platform_version
    return directory / (arm_id + ".json")


def _continuation_options(spec, options, receipt, repo_root, verification=None):
    """Bind a physical successor while retaining the original scientific identity."""
    if receipt is None:
        return dict(options)
    from .experiment_launch import read_launch_receipt

    original_receipt, package = options["receipt_path"], options["package_dir"]
    successor = Path(receipt)
    successor = successor if successor.is_absolute() else Path(repo_root) / successor
    _safe_local(successor.absolute())
    read_launch_receipt(successor.absolute(), spec, package,
        expected_package_identity=options["expected_package_identity"], _verification=verification)
    for arm in spec.to_dict()["arms"]:
        before = RunContext.from_launch(spec, original_receipt, package,
            expected_package_identity=options["expected_package_identity"], arm_id=arm["arm_id"],
            _verification=verification)
        after = RunContext.from_launch(spec, successor, package,
            expected_package_identity=options["expected_package_identity"], arm_id=arm["arm_id"],
            _verification=verification)
        if ({k: v for k, v in before.to_dict().items() if k != "platform_version"}
                != {k: v for k, v in after.to_dict().items() if k != "platform_version"}
                or not isinstance(after.platform_version, str) or not after.platform_version.isdecimal()
                or not isinstance(before.platform_version, str) or not before.platform_version.isdecimal()
                or int(after.platform_version) <= int(before.platform_version)):
            raise ValueError("Continuation receipt must bind a later attempt of the original frozen run")
    return {**options, "receipt_path": successor.absolute()}


def _accepted_continuation_receipt(spec, options, repo_root):
    """Discover one common, hash-bound successor already accepted for every arm."""
    from .research_memory.finalize import current_finalized_directory, load_terminal_input
    from .research_memory.paths import verify_bound_artifact

    root = Path(repo_root).resolve()
    successors, canonical = [], {}
    declaration = spec.to_dict()
    bindings = declaration["prospective"]["arms"]
    for binding in bindings:
        current = current_finalized_directory(root, binding["output"] + "/trajectory.json")
        if not current.is_dir():
            continue
        terminal = load_terminal_input(current)
        canonical[binding["arm_id"]] = (current, terminal)
        pointer = terminal.get("continuation_ref")
        if pointer is None:
            continue
        if (terminal.get("trajectory_id") != binding["trajectory_id"]
                or terminal.get("run_id") != declaration["logical_run_id"] + ":" + binding["arm_id"] + ":downstream"):
            raise ValueError("Accepted continuation differs from the original prospective identity")
        hashes = terminal.get("artifact_hashes", {})
        if not isinstance(pointer, str) or pointer not in hashes:
            raise ValueError("Accepted continuation record is not bound")
        verify_bound_artifact(root, pointer, hashes[pointer])
        record = _json(bound_artifact_path(root, pointer))
        if (record.get("format") != "molgap-terminal-continuation-v1"
                or record.get("continued") != terminal.get("same_run_observation")):
            raise ValueError("Accepted continuation differs from the terminal observation")
        retained = {}
        for key in ("origin_receipt", "continued_receipt"):
            item = record.get(key)
            if (type(item) is not dict or set(item) != {"path", "sha256"}
                    or hashes.get(item["path"]) != item["sha256"]):
                raise ValueError("Accepted continuation receipt is not bound")
            verify_bound_artifact(root, item["path"], item["sha256"])
            retained[key] = bound_artifact_path(root, item["path"])
        if file_digest(retained["origin_receipt"]) != file_digest(options["receipt_path"]):
            raise ValueError("Accepted continuation does not start at the original receipt")
        successors.append((retained["continued_receipt"].resolve(), record["continued_receipt"]["sha256"]))
    if not successors:
        return None, canonical
    if len(successors) != len(bindings) or len({digest for _, digest in successors}) != 1:
        raise ValueError("Accepted arms require one common continuation receipt")
    return successors[0][0], canonical


def record_run_retrieval(profile, repo_root, run_id, attempt_id, *, context,
                         destination, manifest_sha256, continuation_receipt=None,
                         _verification=None):
    """Record only successfully retained manifest-bound bytes; never copy outputs."""
    from .experiment_run import _original_run
    spec, options, prepared = _original_run(profile, repo_root, run_id, attempt_id,
                                           _verification=_verification)
    options = _continuation_options(spec, options, continuation_receipt, repo_root, _verification)
    expected = RunContext.from_launch(spec, options["receipt_path"], options["package_dir"],
        expected_package_identity=options["expected_package_identity"], arm_id=context.arm_id,
        _verification=_verification)
    if context != expected:
        raise ValueError("Retrieval mapping differs from the original launch context")
    paths = MachinePaths.from_file(profile).outside_checkout(repo_root)
    run = prepared.parent
    destination = _output_path(destination, paths=paths, repo_root=repo_root, run=run,
                               external_only=True)
    _digest(manifest_sha256, "retrieved manifest digest")
    manifest_path = destination / "output_manifest.json"
    _safe_local(manifest_path)
    if file_digest(manifest_path) != manifest_sha256:
        raise ValueError("Retrieved output manifest differs from its pinned bytes")
    manifest = _json(manifest_path)
    _check_context(manifest["context"], context)
    # Recheck all downloaded artifact bytes before advertising the destination.
    for item in manifest["artifacts"].values():
        path = repo_local_path(destination, item["path"])
        _safe_local(path)
        if file_digest(path) != item["sha256"]:
            raise ValueError("Retrieved output artifact differs from its pinned bytes")
    value = {"format": "molgap-run-retrieved-output-v1", "context": context.to_dict(),
             "output_dir": str(destination), "manifest_sha256": manifest_sha256}
    target = _mapping_path(run, context.arm_id,
        context.platform_version if continuation_receipt is not None else None)
    publish_immutable_bytes(target, canonical_json(value).encode())
    return target


def resolve_run_acceptance_inputs(profile, repo_root, run_id, attempt_id, *,
                                  retrieved_root=None, supplied=None, locations=None,
                                  continuation_receipt=None):
    """Resolve all arms from frozen plans, explicit paths or verified retrieval maps."""
    from .experiment_run import _original_run
    from .experiment_package import _PackageVerification
    verification = _PackageVerification()
    spec, options, prepared = _original_run(profile, repo_root, run_id, attempt_id,
                                           _verification=verification)
    run, package = prepared.parent, options["package_dir"]
    original_receipt = options["receipt_path"]
    canonical_finalizations = {}
    if continuation_receipt is None:
        continuation_receipt, canonical_finalizations = _accepted_continuation_receipt(
            spec, options, repo_root)
    options = _continuation_options(spec, options, continuation_receipt, repo_root, verification)
    paths = MachinePaths.from_file(profile).outside_checkout(repo_root)
    workflow = _json(run / "inputs/workflow_plan.json")
    acceptance = _packaged_json(package, workflow["acceptance_plan"], _verification=verification)
    if (acceptance.get("format") not in {"molgap-family-acceptance-plan-v1",
                                         "molgap-family-same-run-acceptance-plan-v1",
                                         "molgap-modern2d-formal-reference-plan-v1",
                                         "molgap-modern2d-configured-plan-v1"}
            or acceptance.get("spec_identity") != spec.identity):
        raise ValueError("Frozen acceptance plan differs from the original Spec")
    declaration = spec.to_dict()
    arms = {arm["arm_id"]: arm for arm in declaration["arms"]}
    entries = acceptance.get("arms")
    if type(entries) is not list or [entry.get("arm_id") for entry in entries] != list(arms):
        raise ValueError("Frozen acceptance requires exactly every original arm")
    bindings = {entry["arm_id"]: entry for entry in declaration["prospective"]["arms"]}
    if set(bindings) != set(arms):
        raise ValueError("Missing original prospective arm bindings")
    for value, label in ((supplied, "outputs"), (locations, "locations")):
        if value is not None and (type(value) is not dict or set(value) != set(arms)):
            raise ValueError("Explicit " + label + " must name every original arm")
    root = (_output_path(retrieved_root, paths=paths, repo_root=repo_root, run=run)
            if retrieved_root is not None else None)
    outputs, contexts, resolved_locations, raw_traces = {}, {}, {}, {}
    for entry in entries:
        arm_id, expected = entry["arm_id"], entry["expected"]
        arm, binding = arms[arm_id], bindings[arm_id]
        artifact_adapter(entry["adapter"], (arm["family"]["name"], arm["family"]["version"]))
        if type(expected) is not dict or set(expected) not in (EXPECTED, EXPECTED_FAMILY):
            raise ValueError("Incomplete frozen acceptance expectations")
        contract = entry["contract"]
        if set(contract) != {"path", "sha256"} or contract["sha256"] != arm["training"]["recipe"]["sha256"]:
            raise ValueError("Frozen acceptance recipe differs from the Spec")
        recipe = _packaged_json(package, contract["path"], contract["sha256"], _verification=verification)
        if recipe.get("acceptance_requirements") != expected:
            raise ValueError("Frozen acceptance expectations differ from the recipe")
        context = RunContext.from_launch(spec, options["receipt_path"], package,
            expected_package_identity=options["expected_package_identity"], arm_id=arm_id,
            _verification=verification)
        contexts[arm_id] = context
        proposal = _packaged_json(package, binding["plan_spec_ref"], binding["plan_spec_sha256"],
                                  _verification=verification)
        trajectory = proposal["trajectory"]
        if (trajectory.get("trajectory_id") != binding["trajectory_id"]
                or trajectory.get("state_at_start", {}).get("source_config_identity") != canonical_fingerprint(arm)):
            raise ValueError("Frozen prospective trajectory differs from its arm")
        validate_execution_identity(spec, arm_id, trajectory, context.source_commit)
        run_key = declaration["logical_run_id"] + ":" + arm_id + ":downstream"
        canonical = {"trajectory_id": binding["trajectory_id"], "run_id": run_key,
                     "trajectory": binding["output"] + "/trajectory.json",
                     "terminal": binding["output"] + "/terminal.json",
                     "trace": binding["output"] + "/trace_manifest.json"}
        where = deepcopy(locations[arm_id]) if locations is not None else canonical
        if (type(where) is not dict or set(where) != set(canonical)
                or any(where[k] != canonical[k] for k in ("trajectory_id", "run_id", "trajectory"))):
            raise ValueError("Explicit locations differ from the original prospective identity")
        for key in ("trajectory", "terminal", "trace"):
            path = (bound_artifact_path if key == "trace" else repo_local_path)(repo_root, where[key])
            _safe_local(path)
            where[key] = (artifact_locator(repo_root, path) if key == "trace" else
                          path.relative_to(Path(repo_root).resolve()).as_posix())
        if len({where[k] for k in ("trajectory", "terminal", "trace")}) != 3:
            raise ValueError("Canonical closure locations must be distinct")
        resolved_locations[arm_id] = where
        item = supplied[arm_id] if supplied is not None else None
        if item is not None:
            if type(item) is not dict or set(item) - {"output_dir", "expected"} or "output_dir" not in item:
                raise ValueError("Explicit outputs require output_dir and optional frozen expected")
            if "expected" in item and item["expected"] != expected:
                raise ValueError("Output overrides cannot change frozen expectations")
            destination = _output_path(item["output_dir"], paths=paths, repo_root=repo_root, run=run)
        elif root is not None:
            destination = root / arm_id
            _safe_local(destination)
        else:
            mapping_path = _mapping_path(run, arm_id,
                context.platform_version if continuation_receipt is not None else None)
            _safe_local(mapping_path)
            mapping = _json(mapping_path)
            if set(mapping) != {"format", "context", "output_dir", "manifest_sha256"} or mapping["format"] != "molgap-run-retrieved-output-v1":
                raise ValueError("Invalid retained retrieval mapping")
            if mapping["context"] != context.to_dict():
                raise ValueError("Retained retrieval mapping differs from the original context")
            destination = _output_path(mapping["output_dir"], paths=paths, repo_root=repo_root,
                                       run=run, external_only=True)
            _digest(mapping["manifest_sha256"], "retrieved manifest digest")
            manifest_path = destination / "output_manifest.json"
            _safe_local(manifest_path)
            if file_digest(manifest_path) != mapping["manifest_sha256"]:
                raise ValueError("Retained retrieval manifest changed")
        outputs[arm_id] = {"output_dir": str(destination), "expected": deepcopy(expected)}
        manifest_path = destination / "output_manifest.json"
        _safe_local(manifest_path)
        if manifest_path.is_file():
            manifest = _json(manifest_path)
            _check_context(manifest["context"], context)
            trace = repo_local_path(destination, manifest["artifacts"]["trace"]["path"])
            _safe_local(trace)
            raw_traces[arm_id] = str(trace)
    target_identity = TargetIdentityBinding.from_frozen_package(
        spec, Path(repo_root), package, workflow["acceptance_plan"], _verification=verification)
    return {"spec": spec, "options": options, "prepared": prepared,
            "_canonical_finalizations": canonical_finalizations,
            "original_receipt": original_receipt,
            "outputs": outputs, "locations": resolved_locations, "contexts": contexts,
            "acceptance_plan": acceptance, "run_directory": run, "raw_traces": raw_traces,
            "target_identities": {arm_id: target_identity for arm_id in arms}}
