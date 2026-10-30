"""Stage existing terminal-package metadata from independently inspected outputs.

Only compact metadata is published in the checkout. Retained tensors and traces
keep their original custody and exact digests. Staging never finalizes RML.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from .common.io import publish_immutable_bytes
from .experiment_family_workflow import _artifact_path, _check_context, _json, _runtime, inspect_terminal_output, terminal_attempt_id
from .research_memory.paths import artifact_locator, repo_local_path, verify_bound_artifact
from .research_memory.roles import validate_observed_role_truth
from .research_memory.schemas import TRAJECTORY_OUTCOMES, validate_cost_event, validate_role_event, validate_trajectory
from .research_memory.trace import file_digest, json_bytes, load_canonical_trace
from .screen_policy import canonical_fingerprint


PROTECTED_ROLES = ("official_validation", "test_dev", "test_challenge")
DATA_OPEN_BOUNDARY = (
    "This record covers only Python-level file opens in the training main process. It does not "
    "cover native code or child processes, nor network access (the Kaggle kernel has internet), "
    "nor opens of files outside the directory that holds the frozen dataset manifest. "
    "'untouched' means that process did not open protected-role files through Python; it does not "
    "prove the data was unreachable.")


STRUCTURAL_BOUNDARY = "'untouched' means the protected roles were not among this run's declared mounted inputs."
# Normalized sha256 of src/molgap/family_training_data.py (git blob 6c6bc503) as frozen into legacy packages.
LEGACY_TRAINING_DATA_SHA256 = "8f7539309cb86083104ae3ded83e6606a3a2bbd31b4c0cc6127806f798582f31"
# Normalized sha256 of the two reviewed pre-recorder family_training_runner.py blobs (717b0548, 4d4feff8):
# each calls load_training_data once and opens no other dataset file.
LEGACY_RUNNER_SHA256 = frozenset({"1711486e74211f7a430ef9814000f1bdbf27208ee9e00a4c54499db3107175fb",
                                  "fe382d03a98067f5b3a6a184ff511f21fb5799c30bab568e36b46ff4d8b4065e"})


def _per_arm(value, arm_id):
    return value.get(arm_id) if isinstance(value, dict) else value


def _cost_events(native, context, where, action_id, source_ref, *, category="training", suffix=""):
    """Translate only measured native seconds; retain everything else as unknown."""
    events = []
    for index, cost in enumerate(native):
        if (cost["status"] != "measured" or cost["unit"] != "seconds"
                or cost["hardware"].lower() in {"unknown", "unavailable", "unspecified"}):
            continue
        field = {("wall_seconds", "process_wall"): "wall_hours",
                 ("device_seconds", "allocated_device"): "device_hours"}.get(
                     (cost["metric"], cost["semantics"]))
        # Busy time is not allocated device hours, and CPU/wall are not interchangeable.
        if field is None:
            continue
        measurement = {name: {"value": None, "status": "measurement_missing"}
                       for name in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")}
        measurement[field] = {"value": cost["value"] / 3600, "status": "measured"}
        event = {"schema": "molgap-cost-event-v1",
            "cost_event_id": f"cost-terminal-{where['trajectory_id']}-{suffix}{index}",
            "trajectory_id": where["trajectory_id"], "action_id": action_id, "run_id": where["run_id"],
            "attempt_id": context["attempt_id"] or "unknown", "platform": context["platform"],
            "hardware": cost["hardware"], "category": category, "evidence_ref": source_ref,
            "measurement": measurement}
        events.append(validate_cost_event(event))
    return events


def _allocation_observation(root, retention, arm_id):
    """Bind the observed window; assigned device records need no cost apportionment."""
    from .experiment_retention import _manifest_bytes
    manifest = retention["manifest"]
    manifest_path = Path(retention["manifest_path"])
    directory = manifest_path.parent
    refs = {"execution_retention": {
        "path": manifest_path.name,
        "sha256": hashlib.sha256(_manifest_bytes(manifest)).hexdigest()},
        "allocation_ledger": manifest["allocation_ledger"]["root"],
        "arm_allocation_ledger": manifest["allocation_ledger"]["arms"][arm_id]}
    bindings, artifacts = {}, []
    for name, reference in refs.items():
        locator = artifact_locator(root, directory / reference["path"])
        verify_bound_artifact(root, locator, reference["sha256"])
        bindings[locator] = reference["sha256"]
        artifacts.append({"name": name, "locator": locator, "sha256": reference["sha256"],
            "availability": "machine_local_verified" if locator.startswith("external://") else "durable_local_verified"})
    ledger = retention["ledgers"]["root"]
    devices = [device for device in ledger["devices"] if device["arm_id"] == arm_id]
    observation = {"scope": ledger["scope"], "started_at": ledger["started_at"],
        "observed_at": ledger["observed_at"], "status": ledger["status"], "devices": devices,
        "prior_segments_recorded": len(ledger["prior_segments"]),
        "prior_unobserved_intervals": ledger["prior_unobserved_intervals"],
        "cumulative_costs_evaluated": False, "artifact_refs": sorted(bindings)}
    native = [{"metric": "device_seconds", "unit": "seconds", "value": device["allocated_seconds"],
        "status": "measured", "semantics": "allocated_device", "hardware": device["hardware"]}
        for device in devices]
    return observation, native, bindings, artifacts


def _roles(root, audit_ref, context, where, action_id):
    if audit_ref is None:
        return {role: "unknown" for role in PROTECTED_ROLES}, [], ["observed_protected_role_audit_missing"], None
    audit_path = repo_local_path(root, audit_ref)
    audit_ref = audit_path.relative_to(root).as_posix()
    audit = _json(audit_path)
    if isinstance(audit, dict) and "context" not in audit:
        audit = audit.get(context.arm_id)
    if not isinstance(audit, dict):
        raise ValueError("Observed role audit requires an audit object for the frozen arm")
    if audit.get("context") != context.to_dict():
        raise ValueError("Observed role audit differs from the frozen run/source/attempt context")
    role_use, events = audit.get("role_use"), audit.get("roles")
    if not isinstance(role_use, dict) or not isinstance(events, list):
        raise ValueError("Role audit requires observed role_use and existing-schema roles")
    for role in PROTECTED_ROLES:
        if role_use.get(role) not in {"untouched", "consumed", "not_applicable", "unknown"}:
            raise ValueError("Role audit lacks explicit protected-role truth: " + role)
    for event in events:
        validate_role_event(event)
        if (event["trajectory_id"], event["run_id"], event["action_id"], event["evidence_ref"]) != (
                where["trajectory_id"], where["run_id"], action_id, audit_ref):
            raise ValueError("Observed role event differs from its bound run/action/source audit")
    validate_observed_role_truth([role_use], events)
    blockers = (["observed_protected_role_audit_incomplete"]
                if any(role_use[role] == "unknown" for role in PROTECTED_ROLES) else [])
    return role_use, events, blockers, audit_ref


def _protected_roles_from_data_open(output_dir, artifacts, context, dataset, selected_roles):
    """Derive protected-role truth from the hash-bound record of files the worker opened.

    Returns None when the arm has no record. Otherwise an evidence dict whose ``status``
    is ``untouched``; any other status (unverified record, failed positive control,
    protected file opened) leaves the states ``unknown``. Consumption is never filled in
    silently: it needs a role event and an explicit ``--protected-role-use``.
    """
    from .experiment_family_workflow import DATA_OPEN_FORMAT
    if "data_open" not in artifacts:
        return None
    entry = artifacts["data_open"]
    try:
        path = _artifact_path(output_dir, entry["path"])
        if file_digest(path) != entry["sha256"]:
            raise ValueError("Retained data-open record differs from its output manifest digest")
        record = _json(path)
        _check_context(record.get("context"), context)
        if record.get("format") != DATA_OPEN_FORMAT or record.get("dataset_manifest_sha256") != dataset:
            raise ValueError("Data-open record differs from the frozen dataset identity")
        shards = {item["file"]: item["role"] for item in record["shards"]}
        invocations = record["invocations"]
        flags = record["manifest_protected_role_flags"]
        manifest_name = record["manifest"]
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return {"status": "record_unverified", "record": entry, "reason": str(exc)}
    selected = {name for name, role in shards.items() if role in selected_roles}
    opened = set()
    for invocation in invocations:
        opened.update(invocation["opened"])
    consumed = sorted({shards[name] for name in opened if shards.get(name) in PROTECTED_ROLES})
    if consumed:
        return {"status": "protected_role_file_opened", "record": entry, "roles": consumed}
    if (set(flags) != {role + "_role_read" for role in PROTECTED_ROLES}
            or any(value is not False for value in flags.values())
            or any(role not in {*selected_roles, *PROTECTED_ROLES} for role in shards.values())):
        return {"status": "unattributable_dataset_manifest", "record": entry}
    stray = opened - set(shards) - {manifest_name}
    if invocations and invocations[0]["had_checkpoint"]:
        # Work before this record's first invocation was not observed.
        return {"status": "positive_control_failed", "record": entry, "unobserved_earlier_invocation": True}
    # Each invocation loads the data again, so each must show every selected shard.
    missing = sorted(selected - set.intersection(*(set(item["opened"]) for item in invocations))) if invocations else sorted(selected)
    if stray or missing or not invocations:
        return {"status": "positive_control_failed", "record": entry,
                "missing_selected_shards": missing, "unattributed_opens": sorted(stray)}
    text = ("Hash-bound record " + entry["path"] + " (sha256 " + entry["sha256"] + ", listed in output_manifest.json) "
            "shows the frozen training process opened " + str(len(opened & set(shards))) + " dataset shard file(s): every selected "
            + "/".join(sorted(selected_roles)) + " shard in each of " + str(len(invocations)) + " invocation(s), and no "
            "official_validation, test_dev or test_challenge file. " + DATA_OPEN_BOUNDARY)
    return {"status": "untouched", "record": entry, "text": text}


def _protected_roles_from_frozen_inputs(package_dir, contract, dataset, data_root=None):
    """Structural (not observed) protected-role evidence for a legacy package.

    Applies only to packages frozen before the data-open recorder existed, whose
    data loader, pinned dataset manifest and kernel mounts show that protected roles
    were not among the run's inputs. A package that has the recorder never uses
    this path: its missing record keeps the states ``unknown``.
    """
    from .experiment_package import read_packaged_text
    from .family_training_data import DatasetSelection, resolve_pinned_manifest
    from .kaggle_fixed_mirror import FIXED_MIRRORS, sha256_file
    if package_dir is None:
        return None
    package = Path(package_dir)

    def unverified(reason):
        return {"status": "structural_unverified", "reason": reason}

    def normalized(text):
        text = text.replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    try:
        runner = read_packaged_text(package, "src/molgap/family_training_runner.py")
        if "_begin_data_open_record" in runner:
            return {"status": "recorder_expected_record_missing"}
        if normalized(runner) not in LEGACY_RUNNER_SHA256:
            return unverified("packaged family_training_runner.py is not a reviewed legacy runner")
        loader_sha = normalized(read_packaged_text(package, "src/molgap/family_training_data.py"))
        if loader_sha != LEGACY_TRAINING_DATA_SHA256:
            return unverified("packaged family_training_data.py is not the reviewed legacy loader")
        if data_root is None:
            return unverified("no data root to read the pinned dataset manifest")
        manifest_path = resolve_pinned_manifest(Path(data_root), DatasetSelection(
            contract["dataset"]["manifest"], dataset, contract["dataset"]["development"]))
        manifest_sha = sha256_file(manifest_path)
        manifest = _json(manifest_path)
        if (manifest_sha != dataset or set(manifest["roles"]) != {"train", "development"}
                or {item["role"] for item in manifest["geometry_shards"]} - {"train", "development"}
                or any(manifest.get(role + "_role_read") is not False for role in PROTECTED_ROLES)):
            return unverified("pinned dataset manifest does not declare only train/development with protected reads False")
        mirror = next((item for item in FIXED_MIRRORS.values() if item["manifest_sha256"] == dataset), None)
        metadata_path = package.parent / "kernel" / "kernel-metadata.json"
        run = package.parent.parent
        plan = _json(run / "inputs" / "workflow_plan.json")
        source_id = (plan.get("platform") or plan["kaggle"])["source_dataset"]
        journal = run / "operations" / "submission_state.json"
        if journal.is_file() and _json(journal)["binding"]["dataset"] != source_id:
            return unverified("the submission journal binds a different source dataset than the frozen plan")
        if mirror is None or not metadata_path.is_file():
            return unverified("no fixed mirror for the pinned manifest, or kernel metadata missing")
        metadata = _json(metadata_path)
        sources = metadata["dataset_sources"]
        mounted = next((item for item in sources if item.endswith("/" + mirror["slug"])), None)
        if (metadata["competition_sources"] != [] or metadata.get("kernel_sources", []) != []
                or type(sources) is not list or len(sources) != 2 or mounted is None
                or sorted(sources) != sorted([mounted, source_id])):
            return unverified("kernel mounts are not exactly the source dataset and the fixed mirror")
        metadata_sha = sha256_file(metadata_path)
        # A staged release records the kernel metadata hash; legacy Kaggle-route runs have no such record.
        staging = package.parent / "release" / "staging.json"
        if staging.is_file() and _json(staging).get("kernel_metadata_sha256") != metadata_sha:
            return unverified("kernel-metadata.json differs from the hash recorded at release staging")
        commit = (package / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        return unverified("structural inputs unreadable: " + str(exc))
    text = ("Structural, not observed. Frozen source " + commit + " loads data only through load_training_data "
            "(blob 6c6bc503), which selects only train and development. Pinned manifest " + manifest_sha
            + " declares only those roles, with all protected-read flags False. The kernel mounted only "
            + source_id + ", " + mounted + " (kernel-metadata sha " + metadata_sha + "). " + STRUCTURAL_BOUNDARY)
    return {"status": "structural_untouched", "text": text, "bindings": {
        "source_commit": commit, "family_training_data_sha256": loader_sha, "dataset_manifest_sha256": manifest_sha,
        "kernel_metadata_sha256": metadata_sha, "dataset_sources": sources}}


def draft_role_audit(spec_arm, context, where, action_id, output_dir, evidence_ref, protected_role_use=None,
                     frozen_package=None, data_root=None):
    """Draft one arm's role audit for a family-session screen from retained artifacts.

    Events are drafted from the frozen Spec data roles and the arm's retained,
    digest-checked training contract; terminal inspection verifies the run itself.
    Protected-role states stay ``unknown`` (which blocks closure) unless the human
    passes them, or the arm retained a hash-bound record of the dataset files its
    worker opened that shows only train/development files (``untouched``), or a legacy
    package without that recorder passes the structural check (``structural_untouched``:
    evidence about the run's inputs, not an observation of file opens).
    """
    states = {role: "unknown" for role in PROTECTED_ROLES}
    for role, state in (protected_role_use or {}).items():
        if role not in states or state not in {"untouched", "consumed", "not_applicable", "unknown"}:
            raise ValueError("Invalid protected-role use: " + str(role) + "=" + str(state))
        states[role] = state
    output_dir = Path(output_dir)
    manifest = _json(output_dir / "output_manifest.json")
    artifacts = manifest.get("artifacts", {})
    for name in ("contract", "predictions", "trace"):
        if name not in artifacts:
            raise ValueError("Output manifest lacks the " + name + " artifact needed for a role audit")
        if file_digest(_artifact_path(output_dir, artifacts[name]["path"])) != artifacts[name]["sha256"]:
            raise ValueError("Retained " + name + " differs from its output manifest digest")
    if artifacts["contract"]["sha256"] != context.training_recipe_sha256:
        raise ValueError("Retained training contract differs from the frozen recipe identity")
    contract = _json(_artifact_path(output_dir, artifacts["contract"]["path"]))
    roles = {item["role"]: item for item in spec_arm["data"]["roles"]}
    if set(roles) != {"train", "development"} or not contract["dataset"]["development"]:
        raise ValueError("Role audit drafting covers only train and development session screens")
    if not contract["selection"].startswith("best-development"):
        raise ValueError("Role audit drafting needs development-based selection in the training contract")
    train, development = roles["train"]["membership_sha256"], roles["development"]["membership_sha256"]
    dataset = spec_arm["data"]["dataset"]["sha256"]
    derived = _protected_roles_from_data_open(output_dir, artifacts, context, dataset, set(roles))
    if derived is None:
        derived = _protected_roles_from_frozen_inputs(frozen_package, contract, dataset, data_root)
    if derived is not None and derived["status"] in {"untouched", "structural_untouched"}:
        for role in PROTECTED_ROLES:
            if role not in (protected_role_use or {}):
                states[role] = "untouched"
    elif derived is not None and derived["status"] == "protected_role_file_opened":
        print("WARNING: data-open record shows protected-role file(s) opened for "
              + ", ".join(derived["roles"]) + "; states stay unknown", file=sys.stderr)
        if any((protected_role_use or {}).get(role) in {"untouched", "not_applicable"} for role in PROTECTED_ROLES):
            raise ValueError("Data-open record shows a protected-role file was opened; "
                             "it contradicts an explicit untouched/not_applicable state")
    required = contract["acceptance_requirements"]
    if (contract["dataset"]["manifest_sha256"] != dataset
            or required.get("source_idx_sha256") != development or "target_sha256" not in required):
        raise ValueError("Training contract differs from the frozen Spec data roles")
    prefix = "role-" + context.logical_run_id + "-" + context.arm_id + "-"
    events = [{"schema": "molgap-role-event-v1", "role_event_id": prefix + role + "-" + kind,
        "trajectory_id": where["trajectory_id"], "action_id": action_id, "run_id": where["run_id"],
        "dataset_identity": dataset, "row_manifest_hash": rows, "role_name": role,
        "access_kind": kind, "selection_used": role == "development", "evidence_ref": evidence_ref}
        for role, rows, kinds in (("train", train, ("training_membership",)),
            ("development", development, ("prediction_input", "labels_read", "metric_computed", "selection_used")))
        for kind in kinds]
    audit = {"context": context.to_dict(), "role_use": states, "roles": events}
    if derived is not None:
        audit["protected_role_evidence"] = derived
    return audit


def load_closure_input(path):
    """Read the one human closure file; return ({arm: {outcome, decision_ref}}, role_audit or None)."""
    from .experiment_launch import _unique
    closure = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique)
    arms = closure.get("arms") if type(closure) is dict else None
    if (type(closure) is not dict or set(closure) - {"arms", "role_audit"} or type(arms) is not dict or not arms
            or any(type(item) is not dict or set(item) != {"outcome", "decision_ref"}
                   or not all(type(value) is str and value for value in item.values())
                   for item in arms.values())
            or not all(type(arm) is str and arm and "=" not in arm for arm in arms)
            or ("role_audit" in closure and (type(closure["role_audit"]) is not str or not closure["role_audit"]))):
        raise ValueError("Closure input must be {arms: {arm: {outcome, decision_ref}}, role_audit?}")
    return arms, closure.get("role_audit")


def draft_run_role_audits(profile, repo_root, run_id, attempt_id, *, retrieved_root, output_dir,
                          protected_role_use=None, continuation_receipt=None):
    """Write ``roles_<arm>.json`` for every arm of a run; return {arm: repo-relative path}."""
    from .experiment_run import resolve_run_acceptance_inputs
    resolved = resolve_run_acceptance_inputs(profile, repo_root, run_id, attempt_id,
        retrieved_root=retrieved_root, continuation_receipt=continuation_receipt)
    from .common.paths import MachinePaths
    drafted = write_role_audits(repo_root, resolved, output_dir, protected_role_use,
                                MachinePaths.from_file(profile).data_root)
    root = Path(repo_root).resolve()
    publish_immutable_bytes(repo_local_path(root, Path(output_dir) / "audit_map.json"), json_bytes(drafted))
    return drafted


def write_role_audits(repo_root, resolved, output_dir, protected_role_use=None, data_root=None):
    """Draft and publish one audit per resolved arm (``spec``, ``contexts``, ``locations``, ``outputs``)."""
    root, drafted = Path(repo_root).resolve(), {}
    arms = {arm["arm_id"]: arm for arm in resolved["spec"].to_dict()["arms"]}
    for arm_id, arm in arms.items():
        where = resolved["locations"][arm_id]
        trajectory = _json(repo_local_path(root, where["trajectory"]))
        actions = [item for item in trajectory["actions"] if where["run_id"] in item["run_ids"]]
        if len(actions) != 1:
            raise ValueError("Run must bind exactly one frozen prospective action: " + arm_id)
        path = repo_local_path(root, Path(output_dir) / ("roles_" + arm_id + ".json"))
        ref = path.relative_to(root).as_posix()
        drafted[arm_id] = (path, ref, draft_role_audit(arm, resolved["contexts"][arm_id], where,
            actions[0]["action_id"], resolved["outputs"][arm_id]["output_dir"], ref, protected_role_use,
            resolved.get("options", {}).get("package_dir"), data_root))
    # Every arm validated first, so one bad arm never leaves a partial set behind.
    for path, _, audit in drafted.values():
        publish_immutable_bytes(path, json_bytes(audit))
    return {arm_id: ref for arm_id, (_, ref, _) in drafted.items()}


def _diagnostic_binding(root, resolved, report):
    if report is None:
        return None
    summary = report.get("written_files", {}).get("analysis_summary")
    if summary is None:
        # Standalone API callers may supply an in-memory report. Publish it once
        # beside the original run, then share that immutable pointer across arms.
        run = Path(resolved["run_directory"]).absolute()
        digest = canonical_fingerprint(report)
        summary = run / "reports" / ("terminal_diagnostic_" + digest[:16] + ".json")
        publish_immutable_bytes(summary, json_bytes(report))
    path = Path(summary)
    path = path if path.is_absolute() else root / path
    if canonical_fingerprint(_json(path)) != canonical_fingerprint(report):
        raise ValueError("Retained diagnostic summary differs from the supplied report")
    # An explicit working report may be replaced by a later analysis. Bind one
    # immutable shared copy for terminal evidence, rather than that mutable name.
    identity = canonical_fingerprint({key: value for key, value in report.items() if key != "written_files"})
    own_names = {"diagnostic_" + identity + ".json",
                 "terminal_diagnostic_" + canonical_fingerprint(report)[:16] + ".json"}
    if path.name not in own_names:
        path = Path(resolved["run_directory"]) / "reports" / ("terminal_diagnostic_" + canonical_fingerprint(report) + ".json")
        publish_immutable_bytes(path, json_bytes(report))
    locator = artifact_locator(root, path)
    digest = file_digest(path)
    verify_bound_artifact(root, locator, digest)
    return {"locator": locator, "sha256": digest, "spec_identity": resolved["spec"].identity,
            "logical_run_id": resolved["spec"].to_dict()["logical_run_id"]}


def stage_verified_terminal_packages(profile, repo_root, run_id, attempt_id, *,
        retrieved_root=None, supplied=None, locations=None, outcome=None, decision_ref=None,
        diagnostic_report=None, role_audit=None, resolved=None, reports=None,
        execution_root=None, _retention=None, _strict_transaction=False):
    """Publish drafts, or explicit decision packages, without canonical RML mutation.

    ``resolved`` and ``reports`` share observations inside one local acceptance
    transaction. Independent callers resolve and inspect again. An optional
    repository-local role audit binds its exact ``context``, observed ``role_use``
    and existing-schema ``roles``; recipe authorization is never an audit.
    Positive decisions use the same explicit outcome/decision inputs inside the
    run owner's strict transaction; that owner qualifies the pair before closure.
    """
    if resolved is None:
        from .experiment_run import resolve_run_acceptance_inputs
        resolved = resolve_run_acceptance_inputs(profile, repo_root, run_id, attempt_id,
            retrieved_root=retrieved_root, supplied=supplied, locations=locations)
    root = Path(repo_root).resolve()
    spec = resolved["spec"]
    frozen_options = resolved.get("options", {})
    frozen_package = frozen_options.get("package_dir")
    verification = frozen_options.get("_verification")
    retention = _retention
    if execution_root is not None and retention is None:
        from .experiment_retention import validate_execution_retention
        context = next(iter(resolved["contexts"].values()))
        retention = validate_execution_retention(Path(execution_root), spec,
            package_identity=frozen_options["expected_package_identity"], source_commit=context.source_commit,
            source_archive_sha256=context.source_archive_sha256)
    outputs = {arm_id: {**item,
        **({"context": resolved["contexts"][arm_id]} if "context" not in item else {}),
        **({"target_identity": resolved["target_identities"][arm_id]}
           if arm_id in resolved.get("target_identities", {}) else {})}
        for arm_id, item in resolved["outputs"].items()}
    locations = locations if locations is not None else resolved["locations"]
    arm_ids = {arm["arm_id"] for arm in spec.to_dict()["arms"]}
    if set(outputs) != arm_ids or set(locations) != arm_ids:
        raise ValueError("Terminal staging requires every independent frozen arm")
    if reports is None:
        reports = {arm_id: inspect_terminal_output(item) for arm_id, item in outputs.items()}
    if set(reports) != arm_ids:
        raise ValueError("Terminal staging requires every independent mechanical report")
    if any(report["status"] != "MECHANICALLY_VERIFIED" for report in reports.values()):
        return {"status": "BLOCKED", "arms": reports, "paths": {}, "locations": {},
                "blockers": ["mechanical_output_verification_failed"], "executed": False}
    diagnostic_binding = _diagnostic_binding(root, resolved, diagnostic_report)
    staged, paths, all_blockers = {}, {}, []
    for arm_id in sorted(arm_ids):
        item, report, where = outputs[arm_id], reports[arm_id], dict(locations[arm_id])
        context = item["context"]
        arm = next(arm for arm in spec.to_dict()["arms"] if arm["arm_id"] == arm_id)
        if context.spec_identity != spec.identity or context.arm_identity != canonical_fingerprint(arm):
            raise ValueError("Terminal context differs from the frozen Spec arm")
        if report.get("context") != context.to_dict():
            raise ValueError("Mechanical report context differs from the frozen launch")
        from .research_memory.plan import resolve_decision_state
        trajectory_path = repo_local_path(root, where["trajectory"])
        trajectory = validate_trajectory(resolve_decision_state(root, _json(trajectory_path)))
        # Restore only exact metadata pins from the original source package.
        from .research_memory.terminal_wiring import _restore_frozen_generated_metadata
        restored = _restore_frozen_generated_metadata(
            root, trajectory, frozen_package, _verification=verification)
        if trajectory["trajectory_id"] != where["trajectory_id"] or trajectory["record_mode"] != "prospective":
            raise ValueError("Terminal staging requires the owning prospective trajectory")
        actions = [action for action in trajectory["actions"] if where["run_id"] in action["run_ids"]]
        if len(actions) != 1:
            raise ValueError("Terminal run must bind exactly one frozen prospective action")
        action = actions[0]
        observed_attempt = terminal_attempt_id(spec, arm_id, context.platform_version)
        continuation = None
        if arm_id in resolved.get("continuations", {}):
            from .research_memory.paired import build_terminal_continuation
            continuation = build_terminal_continuation(root,
                output=trajectory_path.parent / "terminal_staging" / "continuations" /
                    str(context.platform_version) / "continuation.json",
                **resolved["continuations"][arm_id])
            if (continuation["origin"]["attempt_id"] not in action["attempt_ids"]
                    or continuation["same_run_observation"]["attempt_id"] != observed_attempt):
                raise ValueError("Continuation does not connect the frozen action to this observed attempt")
        if (action["source_commit"] != context.source_commit or
                (observed_attempt is not None and observed_attempt not in action["attempt_ids"] and continuation is None)):
            raise ValueError("Terminal source/attempt differs from the frozen prospective action; use prepared resume evidence and recovery authority")
        observed = report["observed"]
        output_root = Path(item["output_dir"]).absolute()
        manifest_path = _artifact_path(output_root, "output_manifest.json")
        manifest = _json(manifest_path)
        _check_context(manifest.get("context"), context)
        _runtime(manifest.get("runtime"), context, item["expected"].get("precision", "fp32"))
        if (manifest.get("artifacts") != observed["artifacts"] or manifest.get("costs") != observed["costs"]
                or manifest.get("progress") != observed["progress"]):
            raise ValueError("Inspected output manifest changed before terminal staging")
        artifacts, hashes = [], dict(restored)
        if continuation is not None:
            hashes.update(continuation["artifact_hashes"])
        allocation, allocation_costs = None, []
        if retention is not None:
            allocation, allocation_costs, bindings, retained_artifacts = _allocation_observation(root, retention, arm_id)
            hashes.update(bindings)
            artifacts.extend(retained_artifacts)
        for name, binding in observed["artifacts"].items():
            path = _artifact_path(output_root, binding["path"])
            locator = artifact_locator(root, path)
            verify_bound_artifact(root, locator, binding["sha256"])
            hashes[locator] = binding["sha256"]
            artifacts.append({"name": "training_trace" if name == "trace" else name,
                "locator": locator, "sha256": binding["sha256"],
                "availability": "machine_local_verified" if locator.startswith("external://") else "durable_local_verified"})
            if name == "trace":
                trace_path, trace_locator = path, locator
        trace = load_canonical_trace(trace_path)
        if (trace["trajectory_id"], trace["run_id"]) != (where["trajectory_id"], where["run_id"]):
            raise ValueError("Actual canonical trace differs from the prospective owner/run")
        explicit_outcome, explicit_decision = _per_arm(outcome, arm_id), _per_arm(decision_ref, arm_id)
        blockers = []
        if explicit_outcome is None or explicit_decision is None:
            blockers.append("explicit_outcome_and_decision_ref_required")
        if explicit_outcome is not None and explicit_outcome not in TRAJECTORY_OUTCOMES - {"ACTIVE"}:
            raise ValueError("Expected an explicit existing terminal trajectory outcome")
        if (explicit_outcome == "READY_FOR_DESKTOP" or
                explicit_outcome in {"POSITIVE_UNDER_CONTRACT", "POSITIVE_BELOW_GATE"} and not _strict_transaction):
            blockers.append("positive_or_ready_requires_owning_scientific_acceptance")
        if explicit_decision is not None:
            explicit_decision = repo_local_path(root, explicit_decision).relative_to(root).as_posix()
            hashes[explicit_decision] = file_digest(root / explicit_decision)
        role_use, roles, role_blockers, audit_ref = _roles(root, _per_arm(role_audit, arm_id), context, where, action["action_id"])
        blockers.extend(role_blockers)
        if audit_ref is not None:
            hashes[audit_ref] = file_digest(root / audit_ref)
        manifest_locator = artifact_locator(root, manifest_path)
        hashes[manifest_locator] = file_digest(manifest_path)
        if diagnostic_binding is not None:
            hashes[diagnostic_binding["locator"]] = diagnostic_binding["sha256"]
        # Input-addressed metadata avoids overwriting a draft when a later audit
        # or explicit decision completes it. Retried identical staging is stable.
        identity = canonical_fingerprint({"report": report, "outcome": explicit_outcome,
            "decision_ref": explicit_decision, "bindings": hashes, "role_use": role_use,
            "roles": roles, "diagnostic": diagnostic_binding,
            **({"strict_reviewed_outcome": explicit_outcome}
               if _strict_transaction and explicit_outcome in {"POSITIVE_UNDER_CONTRACT", "POSITIVE_BELOW_GATE"} else {})})
        directory = repo_local_path(root, trajectory_path.parent / "terminal_staging" / attempt_id / identity[:16])
        pointer = lambda name: (directory / name).relative_to(root).as_posix()
        objective_ref, acceptance_ref = pointer("observed.json"), pointer("acceptance.json")
        terminal_ref = pointer("terminal.json")
        costs = _cost_events(observed["costs"], {"attempt_id": observed_attempt, "platform": context.platform},
            where, action["action_id"], objective_ref)
        if allocation is not None:
            existing_allocation = any(event["measurement"]["device_hours"]["status"] == "measured"
                                      for event in costs)
            # Retained scopes may overlap; retain both sources without adding
            # a second measurement to RML's additive device-hour ledger.
            allocation["event_status"] = ("not_aggregated_existing_device_measurement" if existing_allocation
                                           else "recorded_current_window")
            if not existing_allocation:
                # The bootstrap/runtime window includes preflight and idle
                # allocation; it does not establish training-only device time.
                costs.extend(_cost_events(allocation_costs,
                    {"attempt_id": observed_attempt, "platform": context.platform}, where, action["action_id"],
                    objective_ref, category="other", suffix="allocation-"))
        objective = {"context": context.to_dict(), "mechanical_verification": report,
            "native_costs": observed["costs"], "costs": costs, "roles": [],
            "missing_evidence": ["scientific_comparison_not_evaluated", "runtime_qualification_not_evaluated",
                "replay_readiness_not_evaluated", *role_blockers], "diagnostic_report": diagnostic_binding}
        if allocation is not None:
            objective["allocation_observation"] = allocation
        publish_immutable_bytes(directory / "observed.json", json_bytes(objective))
        hashes[objective_ref] = file_digest(directory / "observed.json")
        closure_ready = not blockers
        final_decision = {"outcome": explicit_outcome if closure_ready else "ACTIVE", "final": closure_ready,
            "decision_ref": explicit_decision, "next_allowed_actions": [], "reopen_conditions": []}
        dimensions = {"execution_status": "complete", "artifact_status": "mechanically_verified",
            "comparison_status": "NOT_EVALUATED", "scientific_status": explicit_outcome if closure_ready else "NOT_EVALUATED",
            "transfer_status": "NOT_EVALUATED", "budget_decision": "NOT_EVALUATED", "full_handoff_status": "NOT_EVALUATED"}
        evidence_id = "ev-terminal-" + where["trajectory_id"]
        acceptance = {"run_id": where["run_id"], "evidence_id": evidence_id, "outcome": dimensions,
            "trajectory_decision": final_decision, "role_use": role_use,
            "mechanical_status": "MECHANICALLY_VERIFIED", "closure_blockers": blockers}
        publish_immutable_bytes(directory / "acceptance.json", json_bytes(acceptance))
        hashes[acceptance_ref] = file_digest(directory / "acceptance.json")
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
            "evidence_id": evidence_id, "track": trajectory["track"], "scope": "terminal_output_inspection",
            "legacy_contract": trajectory["state_at_start"]["contract_refs"][0],
            "outcome": dimensions, "authority": {"pointers": [explicit_decision or objective_ref]},
            "artifacts": artifacts, "role_use": role_use,
            "metrics": {"source_ref": objective_ref, "unit": "eV",
                "development_mae_eV": observed.get("development_mae_eV"),
                "selected_weights": observed.get("selected_weights"),
                "selected_optimizer_step": observed.get("selected_optimizer_step"),
                "missing_reason": None if observed.get("development_mae_eV") is not None else "Development metric not observed"},
            "migration": {"migrated_at": "terminal_staging", "verification_scope": "retained output bytes and explicit decision",
                "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False}}
        from .research_memory.terminal_wiring import build_default_trace_manifest
        trace_manifest = build_default_trace_manifest(root, trajectory, {"run_id": where["run_id"]}, trace)
        trace_manifest.update(trace_artifact_ref=trace_locator, trace_artifact_sha256=hashes[trace_locator],
            terminal_evidence_ref=terminal_ref)
        for field in trace_manifest["trace_fields"]:
            trace_manifest["trace_fields"][field] = any(row.get(field) is not None for row in trace["observations"])
        package = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": where["trajectory_id"],
            "run_id": where["run_id"], "action_id": action["action_id"], "acceptance_ref": acceptance_ref,
            "decision": final_decision, "evidence": evidence, "costs": costs, "roles": roles,
            "role_use": role_use, "artifact_hashes": hashes, "trace_manifest": trace_manifest,
            "closure_blockers": blockers, "requested_outcome": explicit_outcome}
        from .research_memory.paired import pair_binding
        paired = pair_binding(trajectory)
        if paired is not None:
            trace_manifest["comparison_role"] = paired["comparison_role"]
            trace_manifest["reference_id"] = (evidence_id if paired["comparison_role"] == "reference"
                else "ev-terminal-" + paired["reference_trajectory_id"])
            package["same_run_observation"] = {"schema": "molgap-same-run-observation-v1",
                "spec_identity": spec.identity, "logical_run_id": context.logical_run_id,
                "platform_name": context.platform, "platform_run_reference": context.run_reference,
                "attempt_id": observed_attempt, "source_commit": context.source_commit,
                "source_package_sha256": context.source_archive_sha256}
        if continuation is not None:
            from .research_memory.paired import validate_terminal_continuation
            package.update(continuation_ref=continuation["continuation_ref"],
                same_run_observation=continuation["same_run_observation"])
            validate_terminal_continuation(root, package)
        existing = directory / "terminal.json"
        package["finalized_at"] = (_json(existing)["finalized_at"] if existing.is_file()
            else datetime.now(timezone.utc).isoformat())
        publish_immutable_bytes(existing, json_bytes(package))
        staged[arm_id] = {**where, "terminal": terminal_ref, "trace": trace_locator}
        paths[arm_id] = {"terminal": str(existing), "observed": str(directory / "observed.json"),
            "acceptance": str(directory / "acceptance.json"), "closure_ready": not blockers, "blockers": blockers}
        all_blockers.extend(f"{arm_id}: {blocker}" for blocker in blockers)
    return {"status": "STAGED_DRAFT" if all_blockers else "STAGED_CLOSURE_READY",
        "arms": reports, "paths": paths, "locations": staged, "blockers": all_blockers, "executed": False}
