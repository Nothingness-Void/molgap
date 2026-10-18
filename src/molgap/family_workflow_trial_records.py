"""Host-only terminal translation for the frozen mature-family workflow trial.

This never finalizes, rebuilds RML, executes a model or contacts a platform.
Use ``research_memory.finalize.finalize`` explicitly on each returned arm.
Large output/proof artifacts stay in the external inspection directory.

``inspection_root/acceptance.json`` is an independently retained host report,
whose exact SHA256 must be supplied separately. Required bindings are ``format``
(``molgap-family-workflow-trial-acceptance-v1``), ``verified=True``, source_commit,
spec_identity, package_identity, source_archive_sha256 and ``arms``. Each arm
contains ``inspection_snapshot_sha256``, ``role_use`` and ``native_qualification``.
The latter contains status=VERIFIED, certificate_sha256, provenance_sha256 and
proof_artifacts: [{path: <inspection-root relative>, sha256: ...}]. This report
attests the host's native proof verification, not merely producer certificate
acceptance. The adapter additionally verifies all those bytes and reconstructs
the output inspection itself (or validates an owner-created snapshot).

Metadata migration performs no training; its V5 migration flags remain false.
Actual earlier training is explicitly true in execution_observation/acceptance.
These are different operations, not a NO_TRAIN classification.
"""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from .common.io import _safe_local, publish_immutable_bytes
from .common.wire import canonical_fingerprint
from .experiment_family_workflow import (
    RunContext, _check_context, _json, capture_output_inspection,
)
from .experiment_inspection import validate_snapshot
from .experiment_launch import _unique
from .experiment_retention import (
    validate_allocation_ledger, validate_execution_retention_manifest,
)
from .experiment_spec import ExperimentSpec
from .family_workflow_trial import check_plan
from .research_memory.paths import repo_local_path
from .research_memory.roles import validate_observed_role_truth
from .research_memory.schemas import validate_cost_event, validate_role_event, validate_trajectory
from .research_memory.trace import file_digest, json_bytes, validate_canonical_trace
from .v5_common import validate_v5_evidence_envelope

BASE = "experiments/family_workflow_trial_100k"
ARMS = ("k1", "gptrans")
SOURCE = "33276db9600d3c90a9039201fde90da4e8a64b9a"
SPEC = "3d2fe2faa7a7b0dd7136c319d19f19b5a70542c5356a11be26a23e69c3360b82"
PACKAGE = "a10bf26135763db93b927043e528ec71d49213055bb9f63ca3029cc29fda5f3f"
ARCHIVE = "13e670f2c53d6a83d2e259b8f40d014ff4e5e4fbde033b7f95913987080e4102"
REPORT_FORMAT = "molgap-family-workflow-trial-acceptance-v1"
ROLE_USE = {"train_100k": "consumed", **{r: "untouched" for r in (
    "development", "common_dev", "audit", "official_validation", "test_dev", "test_challenge")}}


def _match(value, expected, label):
    if type(value) is not dict or any(
            type(value.get(k)) is not type(v) or value.get(k) != v for k, v in expected.items()):
        raise ValueError(label + ": identity/scope mismatch")


def _checked_file(root, relative, digest=None):
    path = repo_local_path(Path(root).absolute(), relative)
    _safe_local(path)
    if not path.is_file() or (digest is not None and file_digest(path) != digest):
        raise ValueError("Missing or changed artifact SHA256: " + str(relative))
    return path


def prepare_trial_terminal(repo_root, output_relative, *, run_directory, inspection_root,
                           finalized_at, acceptance_sha256, inspection_snapshots=None):
    """Prepare small immutable terminal packages, without publishing RML.

    Layout: inspection_root/{k1,gptrans}/<family outputs + native proofs>, plus
    bootstrap_allocation.json and experiment/{execution_report,allocation_ledger,
    execution_retention}.json. ``acceptance.json`` follows the module docstring.
    Pass owner-created InspectionSnapshots keyed by both arm IDs, or omit them
    to perform the existing CPU retained-output inspector once per arm. Arbitrary
    dictionaries/pass booleans cannot replace an output inspection snapshot.

    Full provisioning/billing remains missing. The common bootstrap/ledger
    windows are retained once as non-additive observations, never summed across
    arms or emitted twice as measured full-allocation cost events. Canonical
    traces are retained as byte-bound evidence, not admitted to Replay.
    """
    if (not isinstance(finalized_at, str) or datetime.fromisoformat(
            finalized_at.replace("Z", "+00:00")).utcoffset() != timedelta(0)):
        raise ValueError("Explicit UTC finalized_at required")
    if (type(acceptance_sha256) is not str or len(acceptance_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in acceptance_sha256)):
        raise ValueError("Explicit independently retained acceptance SHA256 required")
    root, run, inspection = map(lambda p: Path(p).resolve(), (repo_root, run_directory, inspection_root))
    destination = repo_local_path(root, output_relative)
    _safe_local(destination)
    results = root / BASE / "results"
    destination.relative_to(results)
    if destination == results or {"rml", "rml_finalized"} & set(destination.parts):
        raise ValueError("Use a dedicated results preparation directory")
    prefix, pending, sources = destination.relative_to(root).as_posix(), {}, {}

    def read(folder, ref, *, digest=None, retain=None):
        path = _checked_file(folder, ref, digest)
        if path.suffix != ".json" or path.stat().st_size > 1024**2:
            raise ValueError("Only bounded JSON metadata may be retained")
        payload = path.read_bytes()
        if digest is not None and hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Metadata changed during read: " + ref)
        value = json.loads(payload, object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite metadata JSON")))
        if type(value) is not dict:
            raise ValueError("Metadata must be a JSON object")
        sources[path] = hashlib.sha256(payload).hexdigest()
        pending[retain or f"{prefix}/metadata/{ref}"] = payload
        return value

    declaration = read(run, "inputs/experiment_spec.json")
    spec = ExperimentSpec(declaration)
    if spec.identity != SPEC or tuple(a["arm_id"] for a in declaration["arms"]) != ARMS:
        raise ValueError("Frozen Spec identity differs")
    report = read(inspection, "acceptance.json", digest=acceptance_sha256)
    _match(report, dict(format=REPORT_FORMAT, verified=True, source_commit=SOURCE,
        spec_identity=SPEC, package_identity=PACKAGE, source_archive_sha256=ARCHIVE), "host acceptance")
    if type(report.get("arms")) is not dict or set(report["arms"]) != set(ARMS):
        raise ValueError("Host acceptance requires exactly both arms")
    plan = read(root, f"{BASE}/acceptance_plan.json", retain=f"{BASE}/acceptance_plan.json")
    check_plan(spec, root, plan)
    read(root, plan["authority"]["path"], digest=plan["authority"]["sha256"], retain=plan["authority"]["path"])
    read(run, "operations/launch_receipt.json")
    retention = read(inspection, "experiment/execution_retention.json")
    validate_execution_retention_manifest(retention, spec, package_identity=PACKAGE,
        source_commit=SOURCE, source_archive_sha256=ARCHIVE)
    execution = read(inspection, "experiment/" + retention["execution_report"]["path"],
        digest=retention["execution_report"]["sha256"])
    ledger_binding = retention["allocation_ledger"]["root"]
    ledger = read(inspection, "experiment/" + ledger_binding["path"], digest=ledger_binding["sha256"])
    validate_allocation_ledger(ledger, spec)
    for name in ARMS:
        arm_ledger = retention["allocation_ledger"]["arms"][name]
        observed_ledger = read(inspection, "experiment/" + arm_ledger["path"], digest=arm_ledger["sha256"])
        if observed_ledger != ledger:
            raise ValueError("Overlapping arm allocation ledger differs from root")
    _match(ledger, dict(status="complete", prior_segments=[], prior_unobserved_intervals=False), "allocation ledger")
    _match(execution, dict(spec_identity=SPEC, status="complete"), "execution")
    if [p.get("phase") for p in execution.get("phases", [])] != ["preflight", "train"]:
        raise ValueError("Exactly preflight and train phases required")
    for phase in execution["phases"]:
        _match(phase, dict(status="complete"), "execution phase")
        workers = phase.get("workers", [])
        if len(workers) != 2:
            raise ValueError("Exactly two workers required")
        for device, arm in enumerate(ARMS):
            _match(workers[device], dict(arm_id=arm, device=device, returncode=0), "worker")
    bootstrap = read(inspection, "bootstrap_allocation.json")
    _match(bootstrap, dict(status="complete", allocation_device_count=2, hardware=["Tesla T4"] * 2), "bootstrap")
    for window in (bootstrap, ledger):
        wall, seconds = window["wall_seconds"], window["allocated_device_seconds"]
        if (type(wall) not in (int, float) or not 0 < wall <= 1800 or
                type(seconds) not in (int, float) or seconds != wall * 2):
            raise ValueError("Observed two-T4 window exceeds budget or is inconsistent")
    if inspection_snapshots is not None and set(inspection_snapshots) != set(ARMS):
        raise ValueError("Provide owner snapshots for exactly both arms")
    checked = {}
    for arm, entry in zip(declaration["arms"], plan["arms"]):
        name = arm["arm_id"]
        ctx = RunContext.from_launch(spec, run / "operations/launch_receipt.json", run / "prepared/package",
            expected_package_identity=PACKAGE, arm_id=name)
        _match(ctx.to_dict(), dict(source_commit=SOURCE, source_archive_sha256=ARCHIVE,
            package_identity=PACKAGE, account="kaseichou", platform="kaggle",
            run_reference="kaseichou/molgap-family-workflow-trial-100k-v1", platform_version="1"), "launch context")
        _match(execution["arms"][name], dict(device=ARMS.index(name), training_started=True,
            resumed_from_checkpoint=False, terminal_status="complete", exit_reason="train_worker_exit_0"), "arm execution")
        contract_ref = entry["contract"]["path"]
        read(root, contract_ref, digest=entry["contract"]["sha256"], retain=contract_ref)
        output = _checked_file(inspection, name + "/output_manifest.json").parent
        snapshot = (capture_output_inspection(output, context=ctx, expected=entry["expected"])
            if inspection_snapshots is None else inspection_snapshots[name])
        inspected = validate_snapshot(snapshot, output_dir=output, context=ctx.to_dict(),
            expected=entry["expected"], file_digest=file_digest, load_json=_json, safe_local=_safe_local)
        for artifact in snapshot.artifact_bindings().values():
            sources[_checked_file(output, artifact["path"], artifact["sha256"])] = artifact["sha256"]
        arm_report = report["arms"][name]
        _match(arm_report, dict(inspection_snapshot_sha256=snapshot.digest, role_use=ROLE_USE), "arm host acceptance")
        _match(inspected["observed"]["progress"], dict(optimizer_steps=32, sample_presentations=4096,
            completed_passes=0, next_batch=32, status="complete"), "acknowledged endpoint")
        _match(inspected["observed"], dict(selected_optimizer_step=32,
            selected_weights="live" if name == "k1" else "ema"), "selected endpoint")
        manifest = read(inspection, name + "/output_manifest.json", digest=snapshot.manifest_sha256)
        if "predictions" in manifest["artifacts"]:
            raise ValueError("Train-only trial may not produce development predictions")
        trace_binding = snapshot.artifact_bindings()["trace"]
        trace = read(output, trace_binding["path"], digest=trace_binding["sha256"],
            retain=f"{prefix}/metadata/{name}/canonical_trace.json")
        validate_canonical_trace(trace)
        if any(r.get(k) is not None for r in trace["observations"] for k in ("live_dev_metric", "ema_dev_metric")):
            raise ValueError("Development metrics contradict train-only role truth")
        companion = snapshot.artifact_bindings().get("trace_context")
        if companion is None:
            raise ValueError("Hash-bound trace context required")
        read(output, companion["path"], digest=companion["sha256"], retain=f"{prefix}/metadata/{name}/trace_context.json")
        native = arm_report["native_qualification"]
        _match(native, dict(status="VERIFIED"), "host native verification")
        certificate = read(inspection, name + "/runtime_certificate.json", digest=native["certificate_sha256"])
        provenance = read(inspection, name + "/runtime_provenance.json", digest=native["provenance_sha256"])
        _check_context(provenance["context"], ctx)
        _match(certificate, dict(format="molgap-runtime-certificate-v1", status="accepted",
            provenance_sha256=canonical_fingerprint(provenance), formal_training_exposure=0), "native certificate")
        if manifest["runtime"].get("runtime_certificate_id") != canonical_fingerprint(certificate):
            raise ValueError("Output runtime certificate differs")
        proof_map = {}
        for proof in native["proof_artifacts"]:
            path = _checked_file(inspection, proof["path"], proof["sha256"])
            path.relative_to(output)
            if proof["path"] in proof_map:
                raise ValueError("Duplicate native proof binding")
            sources[path] = proof["sha256"]
            proof_map[proof["path"]] = proof["sha256"]
        for proof in [*certificate["proofs"], certificate["resume_origin"]]:
            if proof_map.get(name + "/" + proof["path"]) != proof["sha256"]:
                raise ValueError("Host native verification does not cover certificate proof")
        frozen_ref = f"{BASE}/{name}/rml/trajectory.json"
        frozen = validate_trajectory(read(root, frozen_ref, retain=frozen_ref))
        _match(frozen, dict(owner="server", record_mode="prospective"), "prospective")
        _match(frozen["state_at_start"], dict(source_commit=SOURCE,
            source_config_identity=canonical_fingerprint(arm), reference_ids=[]), "prospective source")
        _match(frozen["decision"], dict(outcome="ACTIVE"), "prospective decision")
        run_id = ctx.logical_run_id + ":" + name + ":downstream"
        if len(frozen["actions"]) != 1:
            raise ValueError("One frozen execution action required")
        action = frozen["actions"][0]
        _match(action, dict(type="execution_diagnostic", source_commit=SOURCE, run_ids=[run_id]), "prospective action")
        if len(action["attempt_ids"]) != 1:
            raise ValueError("One independently bound attempt required")
        checked[name] = (arm, ctx, snapshot, frozen, frozen_ref, run_id, action)

    bound = {ref: hashlib.sha256(data).hexdigest() for ref, data in pending.items()}
    result = {"status": "PREPARED", "arms": {}}
    allocation = dict(allocation_identity=canonical_fingerprint({"run": declaration["logical_run_id"],
        "spec": SPEC, "ledger": ledger_binding["sha256"]}), allocation_device_count=2,
        bootstrap_wall_seconds=bootstrap["wall_seconds"], bootstrap_device_seconds=bootstrap["allocated_device_seconds"],
        ledger_wall_seconds=ledger["wall_seconds"], ledger_device_seconds=ledger["allocated_device_seconds"],
        additive_with_other_arm=False, bootstrap_additive_with_ledger=False,
        full_billing_status="measurement_missing", provisioning_before_python_seconds=None, queue_seconds=None)
    for name, (arm, ctx, snapshot, frozen, frozen_ref, run_id, action) in checked.items():
        tid, aid, attempt = frozen["trajectory_id"], action["action_id"], action["attempt_ids"][0]
        acceptance_ref, decision_ref = f"{prefix}/{name}/acceptance.json", f"{prefix}/{name}/decision.json"
        roles = [dict(schema="molgap-role-event-v1", role_event_id=f"role-{tid}-{kind}", trajectory_id=tid,
            action_id=aid, run_id=run_id, dataset_identity=arm["data"]["dataset"]["sha256"],
            row_manifest_hash=arm["data"]["roles"][0]["membership_sha256"], role_name="train_100k",
            access_kind=kind, selection_used=False, evidence_ref=acceptance_ref)
            for kind in ("training_membership", "prediction_input", "labels_read")]
        costs = [dict(schema="molgap-cost-event-v1", cost_event_id=f"cost-{tid}-full-missing", trajectory_id=tid,
            action_id=aid, run_id=run_id, attempt_id=attempt, platform="kaggle2", hardware="Tesla_T4_16GB",
            category="other", measurement={unit: dict(status="measurement_missing", value=None)
                for unit in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")}, evidence_ref=acceptance_ref)]
        for event in roles:
            validate_role_event(event)
        validate_cost_event(costs[0])
        validate_observed_role_truth([ROLE_USE], roles)
        outcome = dict(execution_status="complete", artifact_status="accepted", comparison_status="context_only",
            scientific_status="INCONCLUSIVE", transfer_status="not_evaluated", budget_decision="measurement_missing",
            full_handoff_status="not_authorized")
        decision = dict(decision_ref=decision_ref, outcome="INCONCLUSIVE", final=True,
            next_allowed_actions=[], reopen_conditions=["separate human authority"])
        facts = dict(context=ctx.to_dict(), training_executed=True, optimizer_updates=32,
            sample_presentations=4096, development_rows=0, noReplay=True, promotion_authorized=False,
            automatic_successor_authorized=False, inspection_snapshot_sha256=snapshot.digest,
            shared_allocation_observation=allocation, large_artifacts_retained_at=str(snapshot.output_dir))
        pending[decision_ref] = json_bytes({**decision, **facts})
        evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL",
            legacy_contract="none-prospective-v5", evidence_id=f"family-workflow-trial-{name}-100k-s42-v1",
            track=frozen["track"], scope="WORKFLOW_EXECUTION_TRIAL", outcome=outcome, role_use=ROLE_USE,
            execution_observation=facts, authority=dict(pointers=[frozen_ref, plan["authority"]["path"], decision_ref]),
            artifacts=[dict(name=ref, locator=ref, sha256=sha, availability="repository_retained")
                for ref, sha in sorted(bound.items())], migration=dict(migrated_at=finalized_at,
                training_executed=False, inference_executed=False, scientific_reinterpretation=False,
                verification_scope="Host-only metadata translation of observed32-update train-only execution; no model run during migration."))
        validate_v5_evidence_envelope(evidence)
        pending[acceptance_ref] = json_bytes(dict(format="molgap-rml-terminal-acceptance-adapter-v1",
            evidence_id=evidence["evidence_id"], run_id=run_id, outcome=outcome, trajectory_decision=decision,
            role_use=ROLE_USE, roles=roles, costs=costs, **facts))
        hashes = {**bound, **{ref: hashlib.sha256(pending[ref]).hexdigest() for ref in (acceptance_ref, decision_ref)}}
        terminal_ref = f"{prefix}/{name}/terminal.json"
        pending[terminal_ref] = json_bytes(dict(format="molgap-rml-terminal-package-v1", trajectory_id=tid,
            run_id=run_id, action_id=aid, finalized_at=finalized_at, acceptance_ref=acceptance_ref,
            artifact_hashes=hashes, evidence=evidence, decision=decision, costs=costs, roles=roles, role_use=ROLE_USE))
        result["arms"][name] = dict(trajectory=frozen_ref, terminal=terminal_ref)

    files = {ref: data for ref, data in pending.items() if ref.startswith(prefix + "/")}
    marker = f"{prefix}/preparation.json"
    files[marker] = json_bytes({**result, "finalized_at": finalized_at,
        "files": {ref: hashlib.sha256(data).hexdigest() for ref, data in files.items()}})
    existing = {p.relative_to(root).as_posix() for p in destination.rglob("*") if p.is_file()} if destination.exists() else set()
    if not existing <= set(files) or any(repo_local_path(root, ref).read_bytes() != files[ref] for ref in existing):
        raise ValueError("PREPARATION_CONFLICT: immutable output differs")
    # Revalidate external byte bindings before the first publication; failed
    # preconditions never leave an apparently accepted terminal package.
    if any(file_digest(path) != sha for path, sha in sources.items()):
        raise ValueError("Input artifacts changed during terminal preparation")
    for ref, data in files.items():
        publish_immutable_bytes(repo_local_path(root, ref), data)
    return {**result, "status": "ALREADY_PREPARED" if existing == set(files) else "PREPARED"}
