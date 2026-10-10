"""Translate verified scratch artifacts using the existing V5/RML finalizer."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
EID = "pcqm-k1-execution-efficiency-20261010"


def main():
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    inputs, trajectory = read(HERE / "inputs.json"), read(HERE / "rml/trajectory.json")
    result_dir = HERE / "results/cpu-attempt-001"
    runtime, result = read(result_dir / "runtime.json"), read(result_dir / "result.json")
    failure = read(HERE / "infrastructure_failure.json")
    if failure["phase"] != "checkpoint_publication_metadata" or (result_dir / "completion.json").exists():
        raise ValueError("Unexpected partial diagnostic disposition")
    for name, digest in inputs["files"].items():
        if sha256_file(HERE / "executed_source" / name) != digest:
            raise ValueError("Frozen execution source differs")
    if (runtime["input_sha256"] != sha256_file(HERE / "inputs.json")
            or runtime["prospective_sha256"] != sha256_file(HERE / "rml/trajectory.json")
            or runtime["source_commit"] != inputs["source_commit"]
            or runtime["execution_device"] != "cpu" or runtime["development_rows"]
            or runtime["observed_train_source_idx"] != inputs["source_idx"]
            or result["status"] != "complete" or result["contract_promotion"]):
        raise ValueError("Source/input/role/completion identity differs")
    if not result["cases"]["baseline"]["equivalence"]["exact"]:
        raise ValueError("Baseline scratch replay failed")
    wall = result["wall_seconds"]
    if not math.isfinite(wall) or wall < 0 or wall > inputs["worker_wall_limit_seconds"]:
        raise ValueError("Worker bound/cost invalid")
    for name in ("terminal_decision.md", "attribution.md", "analysis_zh.md"):
        if not (HERE / name).exists():
            raise ValueError("Parent interpretation required")
    tid, action = trajectory["trajectory_id"], trajectory["actions"][0]
    run = action["run_ids"][0]
    outcome = dict(execution_status="scratch_probe_complete_checkpoint_metadata_failed", artifact_status="local_hash_verified",
        comparison_status="matched_cpu_scratch_execution_only", scientific_status="NO_TRAIN",
        transfer_status="not_evaluated", budget_decision="scratch_probe_complete_worker_accounting_partial",
        full_handoff_status="not_applicable")
    decision = dict(outcome="NO_TRAIN", decision_ref=f"{REL}/terminal_decision.md", next_allowed_actions=[],
        reopen_conditions=["Fresh prospective and idle local GPU qualification; separate native T4 runtime contract"])
    cost = dict(schema="molgap-cost-event-v1", cost_event_id="cost-k1-execution-efficiency-cpu-observed",
        trajectory_id=tid, action_id=action["action_id"], run_id=run, attempt_id="cpu-attempt-001",
        platform="local-windows", hardware="CPU four intra-op threads; no accelerator work", category="other",
        evidence_ref=f"{REL}/acceptance.json", measurement={
            "wall_hours": {"value": wall / 3600, "status": "measured"},
            "cpu_hours": {"value": None, "status": "measurement_missing"},
            "device_hours": {"value": None, "status": "not_applicable"},
            "queue_hours": {"value": None, "status": "not_applicable"}})
    roles = []
    for role, rows, kinds in (
        ("train_probe_decoded", inputs["source_idx"], ("labels_read",)),
        ("train_probe_executed", inputs["source_idx"][:128 * (inputs["timing"]["warmup"] + inputs["timing"]["steps"])],
         ("prediction_input", "metric_computed"))):
        row_hash = hashlib.sha256(b"".join(struct.pack("<q", row) for row in rows)).hexdigest()
        for kind in kinds:
            roles.append(dict(schema="molgap-role-event-v1", role_event_id=f"role-k1-execution-efficiency-{role}-{kind}",
                trajectory_id=tid, action_id=action["action_id"], run_id=run,
                dataset_identity="pcqm4mv2-ogb-fixed-500k-scnet-v1", row_manifest_hash=row_hash,
                role_name=role, access_kind=kind, selection_used=False, evidence_ref=f"{REL}/acceptance.json"))
    role_use = dict(train_probe_decoded="labels_read", train_probe_executed="prediction_input",
        development="untouched", official_validation="untouched", test_dev="untouched", test_challenge="untouched")
    acceptance = dict(evidence_id=EID, run_id=run, outcome=outcome, trajectory_decision=decision,
        roles=roles, costs=[cost], role_use=role_use,
        checks=dict(prospective=True, source_snapshot=True, artifact_hashes=True, baseline_replay=True,
                    train_only=True, bounded_scratch_probe=True),
        case_qualification={k: {"exact": row.get("equivalence", {}).get("exact"),
            "execution_only_admissible": row["execution_only_admissible"], "status": row["status"]}
            for k, row in result["cases"].items()}, checkpoint_resume_exact=None,
        execution_scope="Discarded scratch updates on training-only members; no scientific fitting/selection",
        gpu_qualification="not_run_background_load", training_replay_ready=False,
        cost_scope="Measured scratch-probe wall only; total worker wall, process CPU, preparation and test costs unavailable",
        infrastructure_failure_ref=f"{REL}/infrastructure_failure.json")
    atomic_json(HERE / "acceptance.json", acceptance)
    paths = [p for p in HERE.rglob("*") if p.is_file() and not any(
        part in p.parts for part in ("rml", "__pycache__")) and p.name not in ("terminal.json", "closure_receipt.json")]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL",
        evidence_id=EID, track="B", scope="desktop_k1_execution_efficiency", legacy_contract="pcqm-k1-execution-efficiency-v1",
        outcome=outcome, role_use=role_use,
        authority={"pointers": [f"{REL}/{n}" for n in ("protocol.md", "terminal_decision.md", "attribution.md", "acceptance.json")]},
        migration=dict(migrated_at=datetime.now(timezone.utc).date().isoformat(), training_executed=False,
            inference_executed=False, scientific_reinterpretation=False, verification_scope="Metadata translation of separately executed CPU scratch diagnostic"),
        observed_execution=dict(training_executed=False, inference_executed=True, training_replay_ready=False,
                                execution_ref=f"{REL}/results/cpu-attempt-001/result.json"),
        artifacts=[dict(name=p.removeprefix(REL + "/"), locator=p, sha256=h,
                        availability="locally_retained_hash_verified") for p, h in hashes.items()])
    terminal = dict(format="molgap-rml-terminal-package-v1", trajectory_id=tid, run_id=run,
        action_id=action["action_id"], finalized_at=datetime.now(timezone.utc).isoformat(),
        acceptance_ref=f"{REL}/acceptance.json", artifact_hashes=hashes, evidence=evidence,
        decision=decision, costs=[cost], roles=roles)
    atomic_json(HERE / "terminal.json", terminal)
    atomic_json(HERE / "closure_receipt.json", finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json"))


if __name__ == "__main__":
    main()
