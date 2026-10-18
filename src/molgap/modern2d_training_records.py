"""Metadata-only preparation for formal V1's preflight failure; never finalize.

Each returned arm can independently enter research_memory.finalize.finalize
using its original trajectory and prepared terminal, without a training trace.
Receipt/package verification consumes pinned saved observations, not archives.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from .common.io import _safe_local, publish_immutable_bytes
from .common.wire import canonical_fingerprint
from .experiment_family_workflow import RunContext, incomplete_observations_from_execution
from .experiment_launch import _parse, _response, canonical_json
from .experiment_retention import validate_allocation_ledger, validate_execution_retention_manifest
from .experiment_spec import ExperimentSpec
from .research_memory.paths import repo_local_path
from .research_memory.roles import validate_observed_role_truth
from .research_memory.schemas import validate_cost_event, validate_role_event, validate_trajectory
from .research_memory.trace import json_bytes
from .v5_common import validate_v5_evidence_envelope

BASE = "experiments/modern2d_100k_formal_screen"
ARMS = ("ppgt_author_v2", "esa_native_v3")
RUN = "modern2d-formal-100k-s42-v1"
KERNEL = "kaseichou/molgap-modern2d-formal-100k-v1"
SOURCE = "d67e30894089ffaa40569b93ca9c5130cfbb87ba"
SPEC = "5436d6b9a9eb05a8a76a96d21911b1a4e82b03f2c9aa8647aab27940a98f941b"
PACKAGE = "1a64c0dadc0e610565ea692a38df64f1bd1c95570ddfe14492cc485517f60a85"
ARCHIVE = "6870a62e99d159d0239a7c350e006b805f815c0fc81c4a6f2928be6f714e5c90"
_INPUT_PINS = {
    "bootstrap_allocation.json": "781078816e00c2e6a4be04f425c1b3765263f8f787cb60c9eac989543a41a00d",
    "experiment/execution_report.json": "3995072c40b02ff1b471cd58c32d0860894cfc65b1e3b627c58e152524bc0e58",
    "experiment/allocation_ledger.json": "092f41a91073e56aa3b0f190334695e315c9962d635747b4ae017151c59cc44c",
    "experiment/execution_retention.json": "0cb6fe761bb364ba47bfaa9c0f091650c04399d5797b32c4a0381e67a70c2d94",
    **{f"worker_logs/experiment/{arm}/preflight.log": "b50f81b3a96c11ba2f811ff41cf8e92eaade4b5c7244c1ad49d62caf23d9328b" for arm in ARMS},
}
_OPERATION_PINS = {
    "launch_receipt.json": "5b3519dfe97051fec49733c23f3539acb521e86ce11101c0bc7c21fee92b2916",
    "source_verification.json": "ba1d4cb89d765d52856a41876e636e9a6b76ea6c5b93ff6b5021fdadc6cc7835",
}
_REPO_PINS = {
    f"{BASE}/ppgt_author_v2/rml/trajectory.json": "bfe6e371dbfe095306ee5a40f29c9331260b2bd67e5698e7249849eb200fe498",
    f"{BASE}/esa_native_v3/rml/trajectory.json": "1eebd17568c9104a1ce5cf71de693e4fe5fa77d279f8d87ad724ce32887d822b",
    f"{BASE}/roles/train_100k.json": "4652a55e21c1622af57d61aef6318a8c23b1f0379a122cb4f093467b412989fc",
    f"{BASE}/authority.json": "8c60c607abe743a4772657201ff81e3777a5859ec37cefa1ce475e9391d8a315",
}


def _match(record, expected, label):
    if any(type(record.get(k)) is not type(v) or record.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{label}: binding or scope mismatch")


def prepare_preflight_failure_terminal(repo_root, output, inspection_root, operations_root, finalized_at):
    """Prepare immutable root-relative small evidence; explicit UTC time required.

    Full billing is unknown. Overlapping Python windows are lower bounds only.
    The retention manifest binds identical arm/root ledgers; only the retained
    root copy is inspected. No checkpoint, model, prediction or label is opened.
    """
    if not isinstance(finalized_at, str) or datetime.fromisoformat(finalized_at.replace("Z", "+00:00")).utcoffset() != timedelta(0):
        raise ValueError("Explicit UTC finalized_at required")
    root = Path(repo_root).resolve()
    destination = repo_local_path(root, output)
    _safe_local(destination)
    destination.relative_to(root / BASE / "results")
    if destination == root / BASE / "results" or {"rml", "rml_finalized"} & set(destination.parts):
        raise ValueError("Use a dedicated results preparation directory")
    prefix, pending = destination.relative_to(root).as_posix(), {}

    def read(folder, ref, digest=None, retain=None):
        path = repo_local_path(folder, ref)
        _safe_local(path)
        if path.suffix not in {".json", ".log"} or path.stat().st_size > 1024**2:
            raise ValueError("Only bounded JSON/log metadata may be read")
        payload = path.read_bytes()
        if digest is not None and hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"SHA256 mismatch: {ref}")
        pending[retain or f"{prefix}/metadata/{ref}"] = payload
        return json.loads(payload) if path.suffix == ".json" else payload.decode("utf-8")

    saved = {ref: read(Path(inspection_root), ref, sha) for ref, sha in _INPUT_PINS.items()}
    operations = {ref: read(Path(operations_root), ref, sha, f"{prefix}/operations/{ref}") for ref, sha in _OPERATION_PINS.items()}
    repo = {ref: read(root, ref, sha, ref) for ref, sha in _REPO_PINS.items()}
    spec = ExperimentSpec(read(root, f"{BASE}/experiment_spec.json", retain=f"{BASE}/experiment_spec.json"))
    declaration = spec.to_dict()
    _match(declaration, {"logical_run_id": RUN}, "Spec run")
    if spec.identity != SPEC or tuple(a["arm_id"] for a in declaration["arms"]) != ARMS:
        raise ValueError("Spec/arm identity mismatch")
    receipt = _parse(pending[f"{prefix}/operations/launch_receipt.json"].decode())
    response = _response(canonical_json(receipt["response"]))
    _match(receipt, {"format": "molgap-experiment-launch-receipt", "version": 1,
        "submission_state": "ACCEPTED", "reconciliation_result": "ACCEPTED",
        "receipt_identity": canonical_fingerprint({k: v for k, v in receipt.items() if k != "receipt_identity"})}, "receipt")
    _match(response, {"mode": "observed", "outcome": "accepted", "binding": receipt["binding"]}, "response")
    for key in ("canonical_platform_reference", "platform_version", "physical_runs", "timestamp", "monitor_paths"):
        _match(receipt, {key: response[key]}, "receipt observation")
    binding = receipt["binding"]
    expected_binding = {"spec_identity": SPEC, "spec_sha256": SPEC, "source_commit": SOURCE,
        "package_identity": PACKAGE, "source_archive_sha256": ARCHIVE, "logical_run_id": RUN,
        "experiment_id": declaration["experiment_id"], "requested_logical_platform": "kaggle",
        "arms": [{"arm_id": a["arm_id"], "arm_identity": canonical_fingerprint(a)} for a in declaration["arms"]]}
    _match(binding, expected_binding, "receipt binding")
    _match(receipt, {"launch_identity": canonical_fingerprint({k: binding[k] for k in
        ("experiment_id", "logical_run_id", "requested_logical_platform")})}, "launch identity")
    _match(receipt, {"canonical_platform_reference": {"value": KERNEL, "missing_reason": None},
        "platform_version": {"value": "1", "missing_reason": None}, "physical_runs": {"missing_reason": None,
        "value": [{"arm_ids": list(ARMS), "canonical_reference": KERNEL, "run_identity": "kernel139364627/version1",
                   "platform_version": {"value": "1", "missing_reason": None}}]}}, "physical version")
    _match(operations["source_verification.json"], {"kernel": KERNEL, "kernel_id": 139364627, "version": 1,
        "entry_bytes_equal": True, "private": True, "machine_shape": "NvidiaTeslaT4",
        "entry_sha256": "8cc7f173e150915de3902f7eacb956a91303d2846442991a21da71fe2e01a3b8",
        "dataset_sources": ["kaseichou/molgap-modern2d-100k-commondev-v1", "kaseichou/molgap-modern2d-formal-source-v1"]}, "source verification")
    retention = saved["experiment/execution_retention.json"]
    _parse(pending[f"{prefix}/metadata/experiment/execution_retention.json"].decode())
    validate_execution_retention_manifest(retention, spec, package_identity=PACKAGE, source_commit=SOURCE, source_archive_sha256=ARCHIVE)
    for ref in [retention["execution_report"], retention["allocation_ledger"]["root"], *retention["allocation_ledger"]["arms"].values()]:
        name = Path(ref["path"]).name
        if hashlib.sha256(pending[f"{prefix}/metadata/experiment/{name}"]).hexdigest() != ref["sha256"]:
            raise ValueError("Retention report/ledger SHA256 mismatch")
    ledger = validate_allocation_ledger(saved["experiment/allocation_ledger.json"], spec)
    _match(ledger, {"status": "failed", "prior_segments": [], "prior_unobserved_intervals": False}, "ledger")
    bootstrap, execution = saved["bootstrap_allocation.json"], saved["experiment/execution_report.json"]
    _match(bootstrap, {"status": "failed", "hardware": ["Tesla T4"] * 2, "allocation_device_count": 2,
        "wall_seconds": 71.42726459599999, "allocated_device_seconds": 142.85452919199997,
        "provisioning_before_python_seconds": None, "queue_seconds": None}, "bootstrap window")
    observations = incomplete_observations_from_execution(spec, execution)
    if len(execution["phases"]) != 1:
        raise ValueError("Only failed preflight may precede terminal preparation")
    phase = execution["phases"][0]
    _match(phase, {"phase": "preflight", "status": "failed"}, "phase")
    if len(phase["workers"]) != 2:
        raise ValueError("Exactly two preflight workers required")
    for device, arm in enumerate(ARMS):
        _match(phase["workers"][device], {"arm_id": arm, "device": device, "returncode": 1}, "preflight worker")
        _match(execution["arms"][arm], {"device": device, "training_started": False, "resumed_from_checkpoint": False,
            "terminal_status": "failed", "exit_reason": "preflight_worker_exit_1"}, "training not started")
        log = saved[f"worker_logs/experiment/{arm}/preflight.log"]
        if not all(text in log for text in ("_bootstrap(training_only=True", "_assert_target_statistics",
                "observed=(5.338300266634226, 1.2750902063230574), expected=(5.338300266634226, 1.275090206323082)")):
            raise ValueError("Unexpected preflight failure log")
        if log != saved[f"worker_logs/experiment/{ARMS[0]}/preflight.log"] or log.count("Traceback (most recent call last):") != 1:
            raise ValueError("Preflight failure logs differ")
    result, bound = {"status": "PREPARED", "arms": {}}, {ref: hashlib.sha256(data).hexdigest() for ref, data in pending.items()}
    train = repo[f"{BASE}/roles/train_100k.json"]
    _match(train, {"role": "train_100k", "rows": 100000, "protected": False, "status": "complete"}, "train role")
    for a in declaration["arms"]:
        arm, trajectory_ref = a["arm_id"], f"{BASE}/{a['arm_id']}/rml/trajectory.json"
        frozen = validate_trajectory(repo[trajectory_ref])
        _match(frozen, {"owner": "server", "record_mode": "prospective",
            "trajectory_id": f"TC-modern2d-formal-100k-{arm}-s42-v1"}, "prospective")
        _match(frozen["state_at_start"], {"source_commit": SOURCE, "source_config_identity": canonical_fingerprint(a)}, "prospective source")
        _match(frozen["decision"], {"outcome": "ACTIVE"}, "prospective decision")
        run_id, tid = f"{RUN}:{arm}:downstream", frozen["trajectory_id"]
        if len(frozen["actions"]) != 1:
            raise ValueError("One frozen action required")
        _match(frozen["actions"][0], {"action_id": "A001", "run_ids": [run_id], "attempt_ids": [f"{arm}-v1"], "source_commit": SOURCE, "type": "training"}, "action")
        context = RunContext(declaration["experiment_id"], RUN, arm, canonical_fingerprint(a), SPEC,
            a["family"]["name"], a["family"]["version"], SOURCE, ARCHIVE, PACKAGE,
            a["training"]["recipe"]["sha256"], "kaggle", "kaseichou", KERNEL, "1")
        train_role = next(r for r in a["data"]["roles"] if r["role"] == "train")
        _match(train_role, {"membership_sha256": train["source_index_sha256"], "row_order_sha256": train["source_index_sha256"]}, "train membership")
        acceptance_ref, decision_ref = f"{prefix}/{arm}/acceptance.json", f"{prefix}/{arm}/decision.json"
        roles = [{"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-{kind}", "trajectory_id": tid,
            "action_id": "A001", "run_id": run_id, "dataset_identity": train["version_identity"],
            "row_manifest_hash": train["source_index_sha256"], "role_name": "train_100k", "access_kind": kind,
            "selection_used": False, "evidence_ref": acceptance_ref} for kind in ("labels_read", "prediction_input")]
        role_use = {r: "untouched" for r in ("common_dev", "audit", "official_validation", "test_dev", "test_challenge")}
        role_use["train_100k"] = "consumed"
        costs = [{"schema": "molgap-cost-event-v1", "cost_event_id": f"cost-{tid}-full-missing", "trajectory_id": tid,
            "action_id": "A001", "run_id": run_id, "attempt_id": f"{arm}-v1", "platform": "kaggle2", "hardware": "Tesla_T4_16GB",
            "category": "infrastructure_failure", "measurement": {unit: {"status": "measurement_missing", "value": None}
            for unit in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")}, "evidence_ref": acceptance_ref}]
        for event in roles:
            validate_role_event(event)
        validate_cost_event(costs[0])
        validate_observed_role_truth([role_use], roles)
        outcome = dict(execution_status="failed", artifact_status="metadata_verified", comparison_status="not_evaluated",
            scientific_status="NOT_EVALUATED", transfer_status="not_evaluated", budget_decision="measurement_missing", full_handoff_status="not_authorized")
        decision = dict(decision_ref=decision_ref, outcome="INFRASTRUCTURE_ONLY", final=True, next_allowed_actions=[], reopen_conditions=["separate authority"])
        facts = dict(context=context.to_dict(), training_started=False, optimizer_updates=0, model_updates=0, noReplay=True,
            scientific_acceptance="NOT_EVALUATED", platform_status="ERROR", progress=observations[arm]["progress"],
            exit_reason=observations[arm]["exit_reason"], full_cost_status="measurement_missing",
            observed_python_window_lower_bound={"wall_seconds": bootstrap["wall_seconds"], "device_count": 2,
                "device_seconds": bootstrap["allocated_device_seconds"], "additive_with_ledger": False,
                "scope": "Shared allocation lower bound only; excludes pre-Python provisioning and queue."})
        pending[decision_ref] = json_bytes({**decision, **facts})
        evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL", legacy_contract="none-prospective-v5",
            evidence_id=f"modern2d-formal-100k-{arm}-s42-v1-preflight-failure", track=frozen["track"], scope="PREFLIGHT_INFRASTRUCTURE_FAILURE",
            outcome=outcome, role_use=role_use, authority={"pointers": [trajectory_ref, f"{BASE}/authority.json", decision_ref]},
            artifacts=[dict(name=ref, locator=ref, availability="repository_retained", sha256=sha) for ref, sha in sorted(bound.items())],
            migration=dict(migrated_at=finalized_at, training_executed=False, inference_executed=False, scientific_reinterpretation=False,
                verification_scope="Pinned saved metadata/logs only; train-only normalization failed before model/optimizer updates."))
        validate_v5_evidence_envelope(evidence)
        pending[acceptance_ref] = json_bytes(dict(format="molgap-rml-terminal-acceptance-adapter-v1", evidence_id=evidence["evidence_id"],
            run_id=run_id, outcome=outcome, trajectory_decision=decision, role_use=role_use, costs=costs, roles=roles, **facts))
        hashes = {**bound, **{ref: hashlib.sha256(pending[ref]).hexdigest() for ref in (acceptance_ref, decision_ref)}}
        terminal_ref = f"{prefix}/{arm}/terminal.json"
        pending[terminal_ref] = json_bytes(dict(format="molgap-rml-terminal-package-v1", trajectory_id=tid, action_id="A001", run_id=run_id,
            finalized_at=finalized_at, acceptance_ref=acceptance_ref, artifact_hashes=hashes, evidence=evidence, decision=decision, costs=costs, roles=roles, role_use=role_use))
        result["arms"][arm] = dict(trajectory=trajectory_ref, terminal=terminal_ref)
    files = {ref: data for ref, data in pending.items() if ref.startswith(prefix + "/")}
    if destination.exists():
        inventory = {p.relative_to(root).as_posix() for p in destination.rglob("*") if p.is_file()}
        if inventory != set(files) or any(repo_local_path(root, ref).read_bytes() != data for ref, data in files.items()):
            raise ValueError("PREPARATION_CONFLICT: existing output differs")
        return {**result, "status": "ALREADY_PREPARED"}
    destination.mkdir(parents=True, exist_ok=False)
    for ref, data in files.items():
        publish_immutable_bytes(repo_local_path(root, ref), data)
    return result
