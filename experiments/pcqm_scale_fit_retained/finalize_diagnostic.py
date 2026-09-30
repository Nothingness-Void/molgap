"""Accept retained predictions and delegate canonical publication to RML."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone

import torch

from molgap.constants import REPO_ROOT as ROOT
from molgap.research_memory.finalize import finalize, verified_receipt
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v4_runtime import state_dict_sha256, torch_load_compat


HERE = ROOT / "experiments/pcqm_scale_fit_retained"
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-pcqm-scale-fit-retained-20261001"
RUN = "local-scale-fit-retained-20261001"
EID = "pcqm-scale-fit-retained-diagnostic-20261001"


def main():
    destination = HERE / "rml/rml_finalized"
    if destination.exists():
        print(json.dumps(verified_receipt(destination)))
        return
    spec = importlib.util.spec_from_file_location("accepted_scale_fit", HERE / "run_diagnostic.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    plan = adapter.check()
    report = adapter.read(HERE / "results/analysis.json")
    if (report["status"] != "EXECUTED_PENDING_ACCEPTANCE"
            or report["plan_sha256"] != sha256_file(HERE / "frozen_plan.json")
            or report["molecule_forwards"] != 60000
            or report["settings"]["precision"] != "fp32"
            or report["settings"]["tf32_enabled"]):
        raise ValueError("Execution or frozen plan mismatch")
    expected = {f"{arm}.{state}.{role}" for arm, states in adapter.STATES.items()
                for state in states for role in adapter.COHORTS}
    if set(report["state_reports"]) != expected:
        raise ValueError("Incomplete retained-state coverage")
    records, prediction_hashes = {}, {}
    for key, observed in report["state_reports"].items():
        path = HERE / "results" / f"{key}.pt"
        adapter.require_hash(path, observed["artifact_sha256"])
        record = torch_load_compat(path, map_location="cpu", weights_only=False)
        role = key.rsplit(".", 1)[1]
        adapter.validate_prediction(record, role)
        for tensor, field in (("source_idx", "source_idx_sha256"), ("target_eV", "target_sha256")):
            if state_dict_sha256({tensor: record[tensor]}) != observed[field]:
                raise ValueError("Observed row/target hash mismatch")
        mae = float((record["prediction_eV"] - record["target_eV"]).abs().double().mean())
        if abs(mae - observed["mae_eV"]) > 1e-12:
            raise ValueError("Retained MAE mismatch")
        records[key] = record
        prediction_hashes[path.relative_to(ROOT).as_posix()] = observed["artifact_sha256"]
    if adapter.paired_analysis(records) != report["paired_metrics"]:
        raise ValueError("Paired diagnostic does not reproduce from retained predictions")

    outcome = dict(execution_status="complete_no_training", artifact_status="local_hash_verified",
                   comparison_status="fixed_prefix_consumed_role_diagnostic",
                   scientific_status="NO_TRAIN", transfer_status="context_only_not_full_qualification",
                   budget_decision="no_training_release", full_handoff_status="not_applicable")
    decision = dict(decision_ref=f"{REL}/terminal_decision.md", outcome="NO_TRAIN",
                    next_allowed_actions=[], reopen_conditions=["new_decision_relevant_matched_factorial_contract"])
    role_use = dict(train_prefix="used", internal_development="selection_used",
                    official_validation="untouched", test_dev="untouched", test_challenge="untouched")
    cost = dict(schema="molgap-cost-event-v1", cost_event_id="cost-scale-fit-retained-observed",
                trajectory_id=TID, action_id="A001", run_id=RUN, attempt_id="attempt-001",
                category="inference", platform="local-windows", hardware=report["gpu_name"],
                evidence_ref=f"{REL}/acceptance.json", measurement=dict(
                    device_hours=dict(value=None, status="measurement_missing"),
                    cpu_hours=dict(value=None, status="measurement_missing"),
                    wall_hours=dict(value=report["total_script_wall_seconds"] / 3600, status="measured"),
                    queue_hours=dict(value=None, status="not_applicable")))
    roles = []
    for role in adapter.COHORTS:
        observed = report["state_reports"][f"reference_100k.live.{role}"]
        kinds = ["prediction_input", "labels_read", "metric_computed"]
        if role == "development":
            kinds.append("selection_used")
        for kind in kinds:
            roles.append(dict(schema="molgap-role-event-v1",
                role_event_id=f"role-scale-fit-retained-{role}-{kind}", trajectory_id=TID,
                action_id="A001", run_id=RUN, dataset_identity="pcqm-fixed500k-dev50k-matched60-v4",
                row_manifest_hash=observed["source_idx_sha256"],
                role_name="internal_development" if role == "development" else "train_prefix",
                access_kind=kind, selection_used=role == "development", evidence_ref=f"{REL}/acceptance.json"))
    acceptance = dict(format="molgap-scale-fit-retained-acceptance-v1", evidence_id=EID,
                      run_id=RUN, outcome=outcome, trajectory_decision=decision, role_use=role_use,
                      checks=dict(prospective_plan=True, frozen_source_equivalence=True,
                                  checkpoint_hashes=True, exact_subset_order=True,
                                  aligned_targets=True, finite_predictions=True, reconstructed_metrics=True,
                                  no_optimizer_updates=True, protected_roles_untouched=True),
                      comparison_class="CONTEXT_ONLY", missing_states=report["missing_states"],
                      limitations=report["limitations"], costs=[cost], roles=roles,
                      telemetry_scope=dict(gpu_resident_seconds=report["synchronized_gpu_resident_seconds"],
                          cuda_event_forward_seconds=report["cuda_event_forward_seconds"],
                          script_wall_seconds=report["total_script_wall_seconds"],
                          note="Distinct measured intervals; full allocation and CPU hours unknown"),
                      executed_source_hashes=plan["source_hashes"],
                      target_transform_limit=report["joint_100k_transform_authority"])
    atomic_json(HERE / "acceptance.json", acceptance)
    names = ["protocol.md", "terminal_decision.md", "inputs.json", "frozen_plan.json",
             "run_review.json", "run_diagnostic.py", "finalize_diagnostic.py", "acceptance.json",
             "results/analysis.json", "results/execution.json"]
    hashes = {f"{REL}/{name}": sha256_file(HERE / name) for name in names}
    hashes.update(prediction_hashes)
    authority = [f"{REL}/{name}" for name in ("protocol.md", "terminal_decision.md", "acceptance.json", "results/analysis.json")]
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL",
                    evidence_id=EID, track="B", scope="desktop_retained_same_cohort_fit_diagnostic",
                    legacy_contract="pcqm-scale-fit-retained-diagnostic-v1", outcome=outcome,
                    authority=dict(pointers=authority), role_use=role_use,
                    migration=dict(migrated_at="2026-10-01", training_executed=False,
                        inference_executed=False, scientific_reinterpretation=False,
                        verification_scope="Metadata-only export of a separately executed prospective inference diagnostic; export runs no inference and does not reinterpret historical decisions"),
                    observed_execution=dict(training_executed=False, inference_executed=True,
                        execution_ref=f"{REL}/results/execution.json"),
                    artifacts=[dict(name=name, locator=f"{REL}/{name}", sha256=hashes[f"{REL}/{name}"],
                                    availability="locally_retained_hash_verified")
                               for name in ("results/analysis.json", "acceptance.json", "terminal_decision.md")])
    package = dict(format="molgap-rml-terminal-package-v1", trajectory_id=TID,
                   run_id=RUN, action_id="A001", finalized_at=datetime.now(timezone.utc).isoformat(),
                   acceptance_ref=f"{REL}/acceptance.json", artifact_hashes=hashes,
                   evidence=evidence, decision=decision, costs=[cost], roles=roles)
    atomic_json(HERE / "terminal.json", package)
    result = finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json")
    print(json.dumps({"status": result["status"], "finalization_id": result["finalization_id"]}))


if __name__ == "__main__":
    main()
