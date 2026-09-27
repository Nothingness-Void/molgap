"""Close the v1 K1 joint-objective preflight failure without scientific claims.

This adapter only translates retained native Kaggle metadata into the shared
V5/RML terminal package.  It does not inspect checkpoints, reconstruct traces,
or infer role events from the runner's stack trace.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import REPO_ROOT
from .research_memory.finalize import finalize
from .research_memory.trace import atomic_write, file_digest, json_bytes


REL = "experiments/pcqm_k1_joint_atom_reconstruction_100k"
ROOT = REPO_ROOT / REL
FAILURE_REL = f"{REL}/failure_v1"
FAILURE_ROOT = REPO_ROOT / FAILURE_REL
NATIVE_REL = (
    "platforms/_records/kaggle/training/k1_joint_atom_s42_v1/"
    "pcqm_k1_joint_atom_reconstruction"
)
NATIVE_ROOT = REPO_ROOT / NATIVE_REL

RUN_ID = "kaseichou/molgap-k1-joint-atom-s42:v1"
SOURCE_COMMIT = "6576b4797b79308d1772a5fd6001ba86f0d74f43"
RECIPES = ("k1_corrupt_gap", "k1_corrupt_gap_atom_aux")
TRAJECTORIES = {
    "k1_corrupt_gap": "TC-k1-corrupt-gap-100k-s42",
    "k1_corrupt_gap_atom_aux": "TC-k1-corrupt-gap-atom-aux-100k-s42",
}
AUDIT_TRAJECTORY = "TC-k1-joint-atom-post100k-audit-s42"
ALL_TRAJECTORIES = (*TRAJECTORIES.values(), AUDIT_TRAJECTORY)
TERMINAL_OBSERVATION = REPO_ROOT / "platforms/_records/kaggle/training/k1_joint_atom_s42_v1/terminal_observation.json"

FAILURE_FORMAT = "molgap-k1-joint-study-failure-v1"
TERMINAL_FORMAT = "molgap-rml-terminal-package-v1"
ACCEPTANCE_FORMAT = "molgap-rml-terminal-acceptance-adapter-v1"

COMMON_AUTHORITY = (
    f"{REL}/protocol.md",
    f"{REL}/training_contract.json",
    f"{REL}/source_config.json",
    f"{REL}/role_plan.json",
    f"{REL}/audit_role_plan.json",
    f"{REL}/budget_snapshot.json",
    f"{FAILURE_REL}/decision.md",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _retained_finalized_at() -> str:
    """Bind closure time to the retained platform terminal observation."""
    value = _load(TERMINAL_OBSERVATION).get("at")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("retained terminal observation has no closure timestamp")
    return value


def _write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite retained failure record: {path}")
    atomic_write(path, json_bytes(value))


def _sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return file_digest(path)


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _native(recipe: str, name: str) -> Path:
    return NATIVE_ROOT / recipe / name


def _validate_native() -> dict[str, Any]:
    summary = _load(NATIVE_ROOT / "execution_summary.json")
    launch = _load(NATIVE_ROOT / "launch_identity.json")
    if summary.get("run_id") != RUN_ID or launch.get("run_id") != RUN_ID:
        raise ValueError("native failure records do not bind the frozen v1 run")
    if summary.get("complete") is not False or summary.get("used_device_count") != 2:
        raise ValueError("native summary is not the recorded failed two-device job")
    if summary.get("allocated_device_names") != ["Tesla T4", "Tesla T4"]:
        raise ValueError("native device allocation differs from the retained job")
    if summary.get("automatic_successor_submitted") is not False:
        raise ValueError("native summary cannot authorize an automatic successor")
    if summary.get("total_job_wall_seconds") != 231.290926357:
        raise ValueError("native total wall time differs from the retained job")
    if summary.get("total_allocated_device_seconds") != 462.581852714:
        raise ValueError("native allocated device time differs from the retained job")
    if launch.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("native source commit differs from the supplied v1 identity")
    if launch.get("recipes") != list(RECIPES):
        raise ValueError("native recipe list differs from the prospective plans")

    workers = {row.get("mode"): row for row in summary.get("workers", [])}
    if set(workers) != set(RECIPES):
        raise ValueError("native summary does not contain exactly both arms")
    checked: dict[str, Any] = {"execution_summary": summary, "launch_identity": launch}
    for recipe in RECIPES:
        row = workers[recipe]
        failure = _load(_native(recipe, "failure.json"))
        native_cost = _load(_native(recipe, "native_cost.json"))
        if row.get("complete") is not False or row.get("exit_code") != 1:
            raise ValueError(f"native worker status is not a failed attempt: {recipe}")
        if failure.get("run_id") != RUN_ID or failure.get("recipe") != recipe:
            raise ValueError(f"failure identity mismatch: {recipe}")
        if failure.get("stage") != "preflight_training":
            raise ValueError(f"failure stage is not preflight_training: {recipe}")
        traceback = failure.get("traceback", "")
        if "pcqm_k1_variants_runner.py\", line 1675" not in traceback:
            raise ValueError(f"failure does not bind runner line 1675: {recipe}")
        if "Computed calibration target statistics differ from immutable asset" not in traceback:
            raise ValueError(f"failure does not bind target-stat identity error: {recipe}")
        if native_cost.get("run_id") != RUN_ID or native_cost.get("trajectory_id") != TRAJECTORIES[recipe]:
            raise ValueError(f"native cost identity mismatch: {recipe}")
        if native_cost.get("complete") is not False or native_cost.get("training_completed") is not False:
            raise ValueError(f"native cost claims completion: {recipe}")
        checked[recipe] = {
            "worker": row,
            "failure": failure,
            "native_cost": native_cost,
        }
    return checked


def _cost(recipe: str, native: dict[str, Any], evidence_ref: str) -> dict[str, Any]:
    return {
        "schema": "molgap-cost-event-v1",
        "cost_event_id": f"cost-TC-{recipe.replace('_', '-')}-100k-s42-training",
        "trajectory_id": TRAJECTORIES[recipe],
        "action_id": "A001",
        "run_id": RUN_ID,
        "attempt_id": "v1-preflight-target-stat-failure",
        "category": "infrastructure_failure",
        "platform": "kaggle2",
        "hardware": "Tesla_T4_16GB",
        "measurement": {
            "device_hours": {
                "value": native["allocated_device_seconds"] / 3600.0,
                "status": "measured",
            },
            "cpu_hours": {
                "value": native["cpu_process_seconds"] / 3600.0,
                "status": "measured",
            },
            "wall_hours": {
                "value": native["wall_seconds"] / 3600.0,
                "status": "measured",
            },
            "queue_hours": {"value": None, "status": "measurement_missing"},
        },
        "evidence_ref": evidence_ref,
        "native_cost_ref": _rel(NATIVE_ROOT / native["recipe"] / "native_cost.json"),
        "native_cpu_scope": native["cpu_scope"],
        "native_cost_scope": "one arm worker interval; parent job allocation includes bootstrap and both T4 devices",
    }


def _audit_cost(evidence_ref: str) -> dict[str, Any]:
    missing = {"value": None, "status": "measurement_missing"}
    return {
        "schema": "molgap-cost-event-v1",
        "cost_event_id": f"cost-{AUDIT_TRAJECTORY}",
        "trajectory_id": AUDIT_TRAJECTORY,
        "action_id": "A001",
        "run_id": RUN_ID,
        "attempt_id": "v1-training-failure-audit-not-entered",
        "category": "infrastructure_failure",
        "platform": "kaggle2",
        "hardware": "Tesla_T4_16GB",
        "measurement": {
            "device_hours": dict(missing),
            "cpu_hours": dict(missing),
            "wall_hours": dict(missing),
            "queue_hours": dict(missing),
        },
        "evidence_ref": evidence_ref,
        "cost_caveat": "The separate NO_TRAIN audit was never entered; the job summary is retained but not allocated to this action.",
    }


def _decision() -> dict[str, Any]:
    return {
        "decision_ref": f"{FAILURE_REL}/decision.md",
        "outcome": "INFRASTRUCTURE_ONLY",
        "next_allowed_actions": [],
        "reopen_conditions": [
            "new prospective v2 authority after immutable target-stat identity repair",
        ],
        "final": True,
    }


def _role_use() -> dict[str, str]:
    return {
        "internal_development_100000_150000": "unknown",
        "official_validation": "untouched",
        "test_dev": "untouched",
        "test_challenge": "untouched",
    }


def _outcome() -> dict[str, str]:
    return {
        "execution_status": "failed_before_model_preflight",
        "artifact_status": "failure_logs_preserved",
        "comparison_status": "not_evaluated",
        "scientific_status": "not_evaluated",
        "transfer_status": "unavailable",
        "budget_decision": "infrastructure_failure_recorded",
        "full_handoff_status": "not_authorized",
    }


def _authority(recipe: str | None) -> list[str]:
    trajectory = (
        f"{REL}/audit/rml_plan/trajectory.json"
        if recipe is None
        else f"{REL}/arms/{recipe}/rml_plan/trajectory.json"
    )
    prelaunch = [] if recipe is None else [f"{REL}/arms/{recipe}/comparison_readiness_prelaunch.json"]
    return [
        *COMMON_AUTHORITY,
        _rel(TERMINAL_OBSERVATION),
        trajectory,
        *prelaunch,
        f"{FAILURE_REL}/failure_summary.json",
        f"{FAILURE_REL}/target_reduction_diagnostic.json",
        f"{NATIVE_REL}/execution_summary.json",
        f"{NATIVE_REL}/launch_identity.json",
        *(
            [
                f"{NATIVE_REL}/{recipe}/failure.json",
                f"{NATIVE_REL}/{recipe}/native_cost.json",
                f"{NATIVE_REL}/{recipe}.log",
            ]
            if recipe is not None
            else []
        ),
    ]


def _artifact_paths(recipe: str | None) -> list[Path]:
    paths = [
        REPO_ROOT / f"{FAILURE_REL}/failure_summary.json",
        REPO_ROOT / f"{FAILURE_REL}/target_reduction_diagnostic.json",
        REPO_ROOT / f"{NATIVE_REL}/execution_summary.json",
        REPO_ROOT / f"{NATIVE_REL}/launch_identity.json",
    ]
    if recipe is not None:
        paths.extend(
            [
                _native(recipe, "failure.json"),
                _native(recipe, "native_cost.json"),
                NATIVE_ROOT / f"{recipe}.log",
            ]
        )
    return paths


def _evidence(
    *, recipe: str | None, evidence_id: str, costs: list[dict[str, Any]], artifact_paths: list[Path], finalized_at: str
) -> dict[str, Any]:
    authority = _authority(recipe)
    artifact_rows = [
        {
            "name": path.stem,
            "locator": _rel(path),
            "availability": "repository_retained_native_record",
            "sha256": _sha(path),
        }
        for path in artifact_paths
    ]
    return {
        "format": "molgap-v5-evidence-envelope-v1",
        "contract": "MOLGAP-COMMON-V5-FINAL",
        "legacy_contract": "none-v5-prospective-infrastructure-failure",
        "evidence_id": evidence_id,
        "track": "C",
        "scope": "prospective_fixed_100k_joint_objective_execution_infrastructure_failure",
        "outcome": _outcome(),
        "role_use": _role_use(),
        "authority": {"pointers": authority},
        "artifacts": artifact_rows,
        "migration": {
            "migrated_at": finalized_at,
            "verification_scope": (
                "native preflight failure and target-stat diagnostic only; internal-development "
                "role access is unknown because no native role events were emitted; no model "
                "metrics, checkpoints, traces, 500K audit, or protected-role access"
            ),
            "training_executed": False,
            "inference_executed": False,
            "scientific_reinterpretation": False,
        },
        "cost_summary": {
            "job_total_wall_seconds": 231.290926357,
            "job_total_allocated_device_seconds": 462.581852714,
            "device_count": 2,
            "audit_entered": False,
            "parent_job_scope": "includes bootstrap and both Tesla T4 devices; not allocated to either arm cost event",
            "arm_cost_scope": "native per-worker intervals only",
        },
        "costs_bound": [cost["cost_event_id"] for cost in costs],
    }


def _bindings(paths: list[str], extra: list[Path] | None = None) -> dict[str, str]:
    resolved = [REPO_ROOT / path for path in paths]
    if extra:
        resolved.extend(extra)
    bindings: dict[str, str] = {}
    for path in resolved:
        bindings[_rel(path)] = _sha(path)
    return bindings


def _write_terminal(
    *, recipe: str | None, evidence_id: str, costs: list[dict[str, Any]], acceptance_ref: str,
    terminal_path: Path, finalized_at: str
) -> None:
    trajectory_ref = (
        f"{REL}/audit/rml_plan/trajectory.json"
        if recipe is None
        else f"{REL}/arms/{recipe}/rml_plan/trajectory.json"
    )
    authority = _authority(recipe)
    artifacts = _artifact_paths(recipe)
    evidence = _evidence(
        recipe=recipe,
        evidence_id=evidence_id,
        costs=costs,
        artifact_paths=artifacts,
        finalized_at=finalized_at,
    )
    acceptance = {
        "format": ACCEPTANCE_FORMAT,
        "evidence_id": evidence_id,
        "run_id": RUN_ID,
        "finalized_at": finalized_at,
        "outcome": evidence["outcome"],
        "trajectory_decision": _decision(),
        "role_use": _role_use(),
        "costs": costs,
        "roles": [],
        "cost_caveat": (
            "Measured native worker wall/device/CPU-process intervals are retained for arms; "
            "the parent device total includes bootstrap and both T4 devices; the separate "
            "NO_TRAIN audit was not entered and remains measurement_missing."
        ),
    }
    _write(REPO_ROOT / acceptance_ref, acceptance)
    bound = set(authority)
    bound.update(row["locator"] for row in evidence["artifacts"])
    bound.add(acceptance_ref)
    bound.add(trajectory_ref)
    artifact_hashes = _bindings(sorted(bound))
    terminal = {
        "format": TERMINAL_FORMAT,
        "trajectory_id": TRAJECTORIES[recipe] if recipe is not None else AUDIT_TRAJECTORY,
        "run_id": RUN_ID,
        "action_id": "A001",
        "finalized_at": finalized_at,
        "acceptance_ref": acceptance_ref,
        "artifact_hashes": artifact_hashes,
        "evidence": evidence,
        "decision": _decision(),
        "costs": costs,
        "roles": [],
        "role_use": _role_use(),
    }
    _write(terminal_path, terminal)


def _summary(native: dict[str, Any], finalized_at: str) -> dict[str, Any]:
    return {
        "format": FAILURE_FORMAT,
        "run_id": RUN_ID,
        "source_commit": SOURCE_COMMIT,
        "finalized_at": finalized_at,
        "failure_stage": "preflight_training",
        "failure_location": "src/molgap/pcqm_k1_variants_runner.py:1675",
        "failure_reason": "Computed calibration target statistics differ from immutable asset",
        "model_constructed": False,
        "training_executed": False,
        "model_metrics_emitted": False,
        "checkpoints_emitted": False,
        "traces_emitted": False,
        "fixed_500k_audit_entered": False,
        "protected_roles_read": False,
        "role_events_emitted": False,
        "role_access_statement": (
            "Only training/internal-100K data loading needed for role validation and target-stat "
            "calculation is evidenced; no native role events were emitted, so internal role use is unknown."
        ),
        "job_summary": {
            "total_job_wall_seconds": native["execution_summary"]["total_job_wall_seconds"],
            "total_allocated_device_seconds": native["execution_summary"]["total_allocated_device_seconds"],
            "allocated_devices": native["execution_summary"]["allocated_device_names"],
            "used_device_count": native["execution_summary"]["used_device_count"],
            "scope": "parent execution total includes bootstrap and both devices; arm costs are worker-scoped",
        },
        "arms": {
            recipe: {
                "trajectory_id": TRAJECTORIES[recipe],
                "worker_wall_seconds": native[recipe]["worker"]["worker_wall_seconds"],
                "native_cost_ref": _rel(_native(recipe, "native_cost.json")),
                "failure_ref": _rel(_native(recipe, "failure.json")),
                "failure_sha256": _sha(_native(recipe, "failure.json")),
                "native_cost_sha256": _sha(_native(recipe, "native_cost.json")),
            }
            for recipe in RECIPES
        },
        "diagnostic_ref": f"{FAILURE_REL}/target_reduction_diagnostic.json",
        "diagnostic_sha256": _sha(FAILURE_ROOT / "target_reduction_diagnostic.json"),
    }


def close_failure_v1(
    repo_root: str | Path = REPO_ROOT, *, finalized_at: str | None = None
) -> dict[str, Any]:
    """Create immutable failure packages and finalize all three v1 plans."""
    root = Path(repo_root).resolve()
    if root != REPO_ROOT.resolve():
        raise ValueError("the v1 failure adapter is bound to this repository")
    retained_at = _retained_finalized_at()
    if finalized_at is not None and finalized_at != retained_at:
        raise ValueError("closure timestamp must equal retained terminal observation")
    native = _validate_native()
    if not (FAILURE_ROOT / "target_reduction_diagnostic.json").is_file():
        raise FileNotFoundError(FAILURE_ROOT / "target_reduction_diagnostic.json")
    summary_path = FAILURE_ROOT / "failure_summary.json"
    if summary_path.exists():
        raise FileExistsError("v1 failure closure already prepared")
    FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
    _write(summary_path, _summary(native, retained_at))

    jobs: list[dict[str, Any]] = []
    for recipe in RECIPES:
        acceptance_ref = f"{FAILURE_REL}/terminal_acceptance_{recipe}.json"
        terminal = FAILURE_ROOT / f"terminal_{recipe}.json"
        native_cost = native[recipe]["native_cost"]
        cost = _cost(recipe, native_cost, acceptance_ref)
        _write_terminal(
            recipe=recipe,
            evidence_id=f"pcqm-{recipe}-100k-s42-infrastructure-failure-v1",
            costs=[cost],
            acceptance_ref=acceptance_ref,
            terminal_path=terminal,
            finalized_at=retained_at,
        )
        result = finalize(
            root,
            f"{REL}/arms/{recipe}/rml_plan/trajectory.json",
            _rel(terminal),
        )
        jobs.append({"recipe": recipe, "trajectory_id": TRAJECTORIES[recipe], "finalize": result})

    audit_acceptance_ref = f"{FAILURE_REL}/terminal_acceptance_audit.json"
    audit_terminal = FAILURE_ROOT / "terminal_audit.json"
    audit_cost = _audit_cost(audit_acceptance_ref)
    _write_terminal(
        recipe=None,
        evidence_id="pcqm-k1-joint-atom-post100k-audit-s42-infrastructure-failure-v1",
        costs=[audit_cost],
        acceptance_ref=audit_acceptance_ref,
        terminal_path=audit_terminal,
        finalized_at=retained_at,
    )
    audit_result = finalize(root, f"{REL}/audit/rml_plan/trajectory.json", _rel(audit_terminal))
    jobs.append({"trajectory_id": AUDIT_TRAJECTORY, "finalize": audit_result})
    return {
        "format": FAILURE_FORMAT,
        "status": "FINALIZED",
        "run_id": RUN_ID,
        "source_commit": SOURCE_COMMIT,
        "failure_summary": _rel(summary_path),
        "trajectories": jobs,
        "scientific_verdict": None,
        "retry_authority": "new prospective v2 only",
    }


def close_cancelled_attempt(attempt: int, *, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Translate an empty, identity-bound cancellation into existing RML closure.

    No log does not establish zero cost, zero training or untouched roles.
    This adapter records the scheduler terminal without inventing worker facts.
    """
    root = Path(repo_root).resolve()
    if type(attempt) is not int or attempt < 2:
        raise ValueError("expected a separately planned retry attempt")
    release = f"{REL}/attempts/v{attempt}"
    cancelled = f"{REL}/cancellation_v{attempt}"
    observation_ref = f"{cancelled}/platform_observation.json"
    observation = _load(root / observation_ref)
    receipt_ref = f"{REL}/submission_receipt_v{attempt}.json"
    receipt = _load(root / receipt_ref)
    job = receipt["job"]
    config_ref = f"{release}/source_config.json"
    config = _load(root / config_ref)
    run_id = f"{job['kernel']}:v{attempt}"
    if (observation["raw_status"] != "CANCEL_ACKNOWLEDGED"
            or observation["files"] or observation["log"] or observation["next_page_token"]
            or (observation["kernel"], observation["kernel_id"], observation["version"])
            != (job["kernel"], job["kernel_id"], attempt)
            or job["version"] != attempt or job["run_id"] != run_id
            or config["run_id"] != run_id or config["source_commit"] != receipt["source_commit"]):
        raise ValueError("cancellation evidence does not bind an empty terminal attempt")
    observed_at = observation["observed_at"]
    decision = {"decision_ref": f"{cancelled}/decision.md", "outcome": "INFRASTRUCTURE_ONLY",
        "next_allowed_actions": [], "reopen_conditions": ["explicitly authorized new physical attempt"], "final": True}
    outcome = {"execution_status": "cancelled", "artifact_status": "no_native_outputs_available",
        "comparison_status": "not_evaluated", "scientific_status": "not_evaluated",
        "transfer_status": "unavailable", "budget_decision": "cancellation_recorded_cost_unknown",
        "full_handoff_status": "not_authorized"}
    roles = {name: "unknown" for name in ("train", "internal_development_100000_150000",
        "internal_development_500000_550000")}
    # This is scheduler-metadata evidence, not an attestation of remote role use.
    roles.update({name: "not_applicable" for name in ("official_validation", "test_dev", "test_challenge")})
    jobs = []
    for recipe in (*RECIPES, "audit"):
        sub = "audit" if recipe == "audit" else f"arms/{recipe}"
        trajectory_ref = f"{release}/{sub}/rml_plan/trajectory.json"
        trajectory = _load(root / trajectory_ref)
        if trajectory["actions"][0]["run_ids"] != [run_id]:
            raise ValueError("prospective trajectory does not bind cancelled run")
        tid = trajectory["trajectory_id"]
        evidence_id = f"pcqm-k1-joint-{recipe.replace('_', '-')}-cancelled-v{attempt}"
        authority = [f"{REL}/protocol.md", f"{REL}/training_contract.json", config_ref,
            receipt_ref, observation_ref, trajectory_ref, decision["decision_ref"]]
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
            "legacy_contract": "none-v5-prospective-cancellation", "evidence_id": evidence_id,
            "track": "C", "scope": "scheduler_cancellation_without_native_worker_evidence",
            "outcome": outcome, "role_use": roles, "authority": {"pointers": authority},
            "artifacts": [{"name": "platform_observation", "locator": observation_ref,
                "availability": "repository_retained_platform_record", "sha256": file_digest(root / observation_ref)}],
            "migration": {"migrated_at": observed_at,
                "verification_scope": "Scheduler cancellation only; remote training, inference, cost and role access remain unknown. Closure executes no models.",
                "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False}}
        acceptance_ref = f"{cancelled}/terminal_acceptance_{recipe}.json"
        terminal_ref = f"{cancelled}/terminal_{recipe}.json"
        acceptance = {"format": ACCEPTANCE_FORMAT, "evidence_id": evidence_id, "run_id": run_id,
            "finalized_at": observed_at, "outcome": outcome, "trajectory_decision": decision,
            "role_use": roles, "costs": [], "roles": [], "native_measurements_available": False}
        from .v5_common import validate_v5_evidence_envelope
        validate_v5_evidence_envelope(evidence, repo_root=root)
        _write(root / acceptance_ref, acceptance)
        terminal = {"format": TERMINAL_FORMAT, "trajectory_id": tid, "run_id": run_id,
            "action_id": "A001", "finalized_at": observed_at, "acceptance_ref": acceptance_ref,
            "artifact_hashes": {ref: file_digest(root / ref) for ref in [*authority, acceptance_ref]},
            "evidence": evidence, "decision": decision, "costs": [], "roles": [], "role_use": roles}
        _write(root / terminal_ref, terminal)
        jobs.append(finalize(root, trajectory_ref, terminal_ref))
    return {"status": "FINALIZED", "run_id": run_id, "results": jobs, "scientific_claim": None}


__all__ = [
    "AUDIT_TRAJECTORY",
    "RECIPES",
    "RUN_ID",
    "SOURCE_COMMIT",
    "TRAJECTORIES",
    "close_failure_v1",
    "close_cancelled_attempt",
]
