"""Translate the accepted K1 timing diagnostic into existing excluded RML closure."""
from pathlib import Path
import json

from .research_memory.trace import atomic_write, file_digest, json_bytes, load_canonical_trace
from .research_memory.terminal_wiring import build_default_trace_manifest, close_terminal_arm


def close_execution(*, repo_root, output_root, analysis_path, summary_path, decision_path, finalized_at):
    """No training/inference; each frozen prospective record keeps its own trace."""
    from .k1_infrastructure_comparison import BASE
    root, outputs = Path(repo_root).resolve(), Path(output_root).resolve()
    analysis_path, summary_path, decision_path = map(Path, (analysis_path, summary_path, decision_path))
    analysis = json.loads(analysis_path.read_bytes())
    if analysis["status"] != "EXECUTION_COMPARISON_VERIFIED":
        raise ValueError("Verified saved-output comparison is required")
    rel = lambda p: Path(p).resolve().relative_to(root).as_posix()
    save = lambda p, d: atomic_write(p, json_bytes(d))
    bootstrap = outputs.parent / "bootstrap_allocation.json"
    allocation = json.loads(bootstrap.read_bytes())
    if (allocation["status"] != "complete" or allocation["allocation_device_count"] != 2
            or allocation["allocated_device_seconds"] != 2 * allocation["wall_seconds"]):
        raise ValueError("Complete two-device allocation observation is required")
    results = {}
    for arm in ("original", "optimized"):
        plan = root / BASE / arm / "rml/trajectory.json"
        prospective = json.loads(plan.read_bytes())
        tid = prospective["trajectory_id"]
        run = prospective["actions"][0]["run_ids"][0]
        arm_root = outputs / arm
        target = root / BASE / "results" / arm
        target.mkdir(parents=True, exist_ok=True)
        metadata = target / "observed_metadata.json"
        evidence_id = "pcqm-k1-infrastructure-" + arm + "-100k-s42"
        role_use = {"internal_train": "consumed", "internal_development": "consumed",
            "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
        outcome = {"execution_status": "complete", "artifact_status": "accepted",
            "comparison_status": "context_only", "scientific_status": "INCONCLUSIVE",
            "transfer_status": "not_evaluated", "budget_decision": "stop_under_contract",
            "full_handoff_status": "not_authorized"}
        decision = {"outcome": "INCONCLUSIVE", "final": True, "decision_ref": rel(decision_path),
            "next_allowed_actions": ["CLOSE"], "reopen_conditions": ["explicit new compute decision"]}
        roles = []
        for role, kinds in (("internal_train", ("training_membership", "labels_read", "metric_computed")),
                ("internal_development", ("prediction_input", "labels_read", "metric_computed", "selection_used"))):
            for kind in kinds:
                roles.append({"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-{role}-{kind}",
                    "trajectory_id": tid, "action_id": "A001", "run_id": run,
                    "dataset_identity": "pcqm4mv2-ogb-fixed-100k-v1",
                    "row_manifest_hash": "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d",
                    "role_name": role, "access_kind": kind, "selection_used": kind == "selection_used",
                    "evidence_ref": rel(metadata)})
        cost = {"schema": "molgap-cost-event-v1", "cost_event_id": f"cost-{tid}-observed-v1",
            "trajectory_id": tid, "action_id": "A001", "run_id": run, "attempt_id": arm + "-v1",
            "category": "training", "platform": "kaggle", "hardware": "Tesla_T4",
            "evidence_ref": rel(metadata), "measurement": {
                "device_hours": {"status": "measured", "value": allocation["wall_seconds"] / 3600},
                "wall_hours": {"status": "measured", "value": allocation["wall_seconds"] / 3600},
                "cpu_hours": {"status": "measurement_missing", "value": None},
                "queue_hours": {"status": "measurement_missing", "value": None}}}
        save(metadata, {"evidence_id": evidence_id, "trajectory_id": tid, "run_id": run,
            "outcome": outcome, "trajectory_decision": decision, "role_use": role_use,
            "costs": [cost], "roles": roles, "source_commit": analysis["source_commit"],
            "analysis_ref": rel(analysis_path), "summary_ref": rel(summary_path),
            "observed_role_sources": [rel(arm_root / "output_manifest.json"), rel(arm_root / "canonical_trace.json"),
                rel(arm_root / "development_predictions.pt")],
            "allocation_ref": rel(bootstrap), "cost_attribution": "one full allocated device per arm, including setup and idle tail; excludes unknown queue/pre-Python provisioning",
            "local_training_executed": False, "model_inference_executed": False})
        trace_path = arm_root / "canonical_trace.json"
        trace = load_canonical_trace(trace_path)
        terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid, "run_id": run,
            "action_id": "A001", "finalized_at": finalized_at, "acceptance_ref": rel(metadata),
            "decision": decision, "roles": roles, "costs": [cost], "role_use": role_use}
        authorities = [root / BASE / name for name in ("protocol.md", "authority.json", "role_plan.json")]
        authorities += [plan, decision_path, root / BASE / arm / "contract.json", root / BASE / "submission/launch_receipt.json"]
        artifacts = [p for p in arm_root.rglob("*") if p.is_file()]
        artifacts += [metadata, summary_path, analysis_path, bootstrap,
            outputs / "execution_report.json", outputs / "execution_retention.json", outputs / "allocation_ledger.json"]
        hashes = {rel(p): file_digest(p) for p in artifacts + authorities}
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
            "evidence_id": evidence_id, "track": "C", "scope": "execution_diagnostic_100k",
            "legacy_contract": "none-prospective-v5", "outcome": outcome, "role_use": role_use,
            "authority": {"pointers": [rel(p) for p in authorities]},
            "artifacts": [{"name": p.name, "locator": rel(p), "sha256": hashes[rel(p)],
                "availability": "local_verified", "purpose": "training_trace" if p == trace_path else
                ("execution_profile" if "profiles" in p.parts or "profile_qualification" in p.parts else "analysis_diagnostics")}
                for p in artifacts],
            "migration": {"migrated_at": finalized_at, "verification_scope": "native proofs, sealed outputs and saved predictions; execution diagnostic only",
                "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False}}
        terminal["evidence"] = evidence
        manifest = build_default_trace_manifest(root, prospective, terminal, trace)
        manifest.update(reference_id=evidence_id, comparison_role="reference",
            model_identity="neural-atom-k1-reference-192", evaluation_role_identity="pcqm4mv2-ogb-fixed-100k-v1:development-100000-150000",
            selection_semantics="best-development-live", metric_semantics="Gap development MAE eV; training unit declared by canonical trace",
            backtest_eligibility={"eligible": False, "exclusion_reasons": ["standalone execution diagnostic; self ID anchors storage only", "no frozen scientific reference or causal/replay qualification"]})
        for field, observed in {"learning_rate": "learning_rate", "checkpoint_identity": "checkpoint_identity"}.items():
            manifest["trace_fields"][field] = any(row[observed] is not None for row in trace["observations"])
        terminal.update(artifact_hashes=hashes, trace_manifest=manifest)
        terminal_path = target / "terminal.json"
        if terminal_path.exists() and terminal_path.read_bytes() != json_bytes(terminal):
            raise ValueError("Preserve the existing terminal declaration")
        save(terminal_path, terminal)
        closure = close_terminal_arm(root, plan, terminal_path, trace=trace_path, arm_identifier=arm)
        save(target / "closure.json", {key: closure[key] for key in (
            "trajectory_id", "pipeline_status", "finalization_status", "finalization_id",
            "validation_status", "outcome", "next_decision", "replay_pool_delta")})
        results[arm] = closure
    return results
