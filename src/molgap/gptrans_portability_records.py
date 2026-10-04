"""Translate retained pre-worker audit failure into the existing RML finalizer."""
from __future__ import annotations

import json
from pathlib import Path

from .gptrans_portability import verify_file
from .training_reproducibility import atomic_json, sha256_file
from .research_memory.finalize import finalize


def close_preworker_failure(root: Path, output: Path, *, release_path: Path,
                           plan: Path, result: Path, finalized_at: str):
    root = root.resolve()
    for path in (output, release_path, plan, result):
        path.resolve().relative_to(root)
    if (plan.parent/"rml_finalized").exists():
        return finalize(root, plan, result/"terminal.json")
    manifest = json.loads((output/"output_manifest.json").read_text())
    release = json.loads(release_path.read_text())["release"]
    if manifest["status"] != "ERROR" or manifest["identity"] != release:
        raise ValueError("Failure is not the frozen release")
    if set(manifest["files"]) != {"allocation.json", "cost.json", "failure.json"}:
        raise ValueError("Only failures before any worker/role execution qualify")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"output_manifest.json"}:
        raise ValueError("Unexpected execution artifacts")
    for name, digest in manifest["files"].items():
        verify_file(output/name, digest)
    if any(manifest.get(k) is not False for k in (
        "training_executed", "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")):
        raise ValueError("Failure role/execution attestations differ")
    native = json.loads((output/"cost.json").read_text())
    allocation = json.loads((output/"allocation.json").read_text())
    if native["allocated_devices"] != 2 or allocation["allocated_devices"] != 2:
        raise ValueError("Unexpected allocation")
    wall = native["allocation_wall_seconds"]
    if not 0 < wall <= 5400 or abs(native["allocated_device_hours"]-2*wall/3600) > 1e-9:
        raise ValueError("Invalid measured failure cost")
    frozen = json.loads(plan.read_text())
    tid = frozen["trajectory_id"]
    run = frozen["actions"][0]["run_ids"][0]
    def rel(p):
        return p.relative_to(root).as_posix()
    decision = dict(final=True, outcome="INFRASTRUCTURE_ONLY", decision_ref=rel(result/"decision.md"),
        next_allowed_actions=[], reopen_conditions=["New prospective attempt after CPU-only environment qualification"])
    outcome = dict(execution_status="failed_before_workers", artifact_status="failure_metadata_verified",
        comparison_status="not_evaluated", scientific_status="not_evaluated", transfer_status="not_evaluated",
        budget_decision="native_allocation_cost_retained", full_handoff_status="not_authorized")
    role_use = {k:"untouched" for k in ("internal_development_100000_150000", "internal_development_500000_550000",
        "official_validation", "test_dev", "test_challenge")}
    role_use["train"] = "not_applicable"
    acceptance = result/"acceptance.json"
    missing = dict(status="measurement_missing", value=None)
    cost = dict(schema="molgap-cost-event-v1", cost_event_id=frozen["actions"][0]["cost_event_ids"][0],
        trajectory_id=tid, action_id="A001", run_id=run, attempt_id="v1", category="infrastructure_failure",
        platform="kaggle2", hardware="Tesla_T4x2", evidence_ref=rel(acceptance), measurement=dict(
            device_hours=dict(status="measured", value=native["allocated_device_hours"]),
            wall_hours=dict(status="measured", value=wall/3600), cpu_hours=missing, queue_hours=missing))
    atomic_json(acceptance, dict(evidence_id="pcqm-gptrans-ema-portability-infrastructure-v1", run_id=run,
        outcome=outcome, trajectory_decision=decision, role_use=role_use, costs=[cost], roles=[],
        remote_identity=dict(kernel="kaseichou/molgap-gptrans-ema-portability-audit", kernel_id=136990464, version=1),
        model_inference_executed=False, cost_scope=native["scope"]))
    pointers = [plan, release_path, result/"decision.md", acceptance, output/"output_manifest.json",
                *(output/name for name in manifest["files"])]
    hashes = {rel(p):sha256_file(p) for p in pointers}
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL",
        evidence_id="pcqm-gptrans-ema-portability-infrastructure-v1", track="C", scope="preworker_runtime_failure",
        legacy_contract="none-v5-prospective", outcome=outcome, role_use=role_use,
        authority=dict(pointers=list(hashes)), artifacts=[dict(name=p.name, locator=rel(p), sha256=hashes[rel(p)],
            availability="retained_metadata") for p in pointers[4:]], migration=dict(migrated_at=finalized_at,
            training_executed=False, inference_executed=False, scientific_reinterpretation=False,
            verification_scope="Hash-bound pre-worker failure only; no model deserialization, graph or role execution"))
    terminal = result/"terminal.json"
    atomic_json(terminal, dict(format="molgap-rml-terminal-package-v1", trajectory_id=tid, run_id=run,
        action_id="A001", finalized_at=finalized_at, acceptance_ref=rel(acceptance), artifact_hashes=hashes,
        evidence=evidence, decision=decision, costs=[cost], roles=[], role_use=role_use))
    return finalize(root, plan, terminal)


def accept_environment(output: Path, binding: dict) -> dict:
    """CPU qualification is import proof only, never model/data acceptance."""
    from .kaggle_python_environment import REQUIREMENTS, UV_VERSION, PYTHON_VERSION
    qualification = json.loads((output/"environment_qualification.json").read_text())
    native = json.loads((output/"environment_cost.json").read_text())
    if qualification.get("source_identity") != binding["source_release"]["identity"]:
        raise ValueError("CPU result is not the intended audit source")
    if (qualification.get("status") != "IMPORTS_QUALIFIED_NOT_GPU_CALIBRATED"
        or qualification.get("requirements") != list(REQUIREMENTS)
        or qualification.get("bootstrap_uv") != UV_VERSION
        or qualification.get("requested_python") != PYTHON_VERSION
        or qualification.get("cuda_build") != "12.1"
        or not qualification.get("python", "").startswith("3.12.")):
        raise ValueError("Pinned environment qualification differs")
    expected = {r.split("==")[0]:r.split("==")[1] for r in REQUIREMENTS}
    if qualification.get("packages") != expected:
        raise ValueError("Imported workload packages differ")
    for record in (qualification, native):
        if any(record.get(k) is not False for k in (
            "training_executed", "model_inference_executed", "graph_role_read")):
            raise ValueError("Qualification exceeded import-only scope")
    if (qualification.get("model_constructed") is not False or native.get("status") != "COMPLETE"
        or native.get("allocated_devices") != 0 or native.get("device_hours") != 0
        or not 0 < native.get("wall_seconds", 0) <= 1850):
        raise ValueError("CPU qualification execution/cost differs")
    return dict(accepted=True, qualification=qualification, cost=native,
        model_inference_executed=False, gpu_calibrated=False,
        scientific_comparison_status="NOT_EVALUATED", next_decision="SOL_REQUIRED")
