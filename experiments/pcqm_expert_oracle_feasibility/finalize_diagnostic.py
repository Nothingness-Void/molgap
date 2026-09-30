"""Bind retained diagnostic evidence and call the existing RML finalizer."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from molgap.constants import REPO_ROOT
from molgap.research_memory.finalize import finalize, verified_receipt
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = REPO_ROOT
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-pcqm-expert-oracle-feasibility-20260930"
RUN = "local-expert-oracle-feasibility-20260930"
EID = "pcqm-expert-oracle-feasibility-no-train-20260930"


def main():
    finalized = HERE/"rml/rml_finalized"
    if finalized.exists():
        print(json.dumps({"status":"ALREADY_FINALIZED", "finalization_id":verified_receipt(finalized)["finalization_id"]}))
        return
    report = json.loads((HERE/"analysis.json").read_text())
    prospective = json.loads((HERE/"rml/trajectory.json").read_text())
    for item in report["identity"]["inputs"].values():
        if sha256_file(ROOT/item["retained_path"]) != item["sha256"]:
            raise ValueError("Retained prediction changed")
    oof = report["retained_oof"]
    if sha256_file(ROOT/oof["path"]) != oof["sha256"]:
        raise ValueError("OOF bytes changed")
    with np.load(ROOT/oof["path"], allow_pickle=False) as data:
        if not np.array_equal(data["source_idx"], np.arange(500000,550000)):
            raise ValueError("OOF source identity changed")
        for name in ("global_crossfit", "hard_gate", "soft_gate"):
            if not np.isfinite(data[name]).all():
                raise ValueError("Non-finite OOF output")
            actual = float(np.abs(data[name]-data["target"]).mean())
            expected = report["crossfit"]["global_mae_eV"] if name == "global_crossfit" else report["crossfit"]["gates"][name]["mae_eV"]
            if not np.isclose(actual, expected, rtol=0, atol=1e-12):
                raise ValueError("OOF endpoint mismatch")
    if any(x["nomination_rule_pass"] for x in report["crossfit"]["gates"].values()):
        raise ValueError("NO_TRAIN disposition conflicts with gate")
    source_commit = subprocess.check_output(["git","rev-parse","71976042"],cwd=ROOT,text=True).strip()
    source = subprocess.check_output(["git","show",f"{source_commit}:{REL}/analyze.py"],cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != report["provenance"]["analyzer_sha256"]:
        raise ValueError("Executed source is not the frozen Git blob")
    for relative in ("src/molgap/hierarchical_oracle.py", "src/molgap/multi2d.py"):
        blob = subprocess.check_output(["git","show",f"{source_commit}:{relative}"],cwd=ROOT)
        # Shared Python sources may be CRLF on disk; record both bytes, do not rewrite them.
        if blob.replace(b"\r\n",b"\n") != (ROOT/relative).read_bytes().replace(b"\r\n",b"\n"):
            raise ValueError("Oracle helper differs from frozen source")
    outcome = dict(execution_status="complete_no_training", artifact_status="hash_verified_saved_predictions", comparison_status="exploratory_gate_crossfit_consumed_development", scientific_status="NO_TRAIN", transfer_status="not_v5_qualified", budget_decision="no_accelerator_release", full_handoff_status="not_applicable")
    decision = dict(decision_ref=f"{REL}/decision.md", outcome="NO_TRAIN", next_allowed_actions=["await_separate_pretraining_acceptance", "review_conditional_experiment_plan"], reopen_conditions=["new_accepted_pretraining_artifacts_and_independently_qualified_role"])
    role_use = dict(fixed50k_development="selection_used", official_validation="untouched", test_dev="untouched", test_challenge="untouched")
    acceptance = dict(format="molgap-expert-oracle-acceptance-v1", evidence_id=EID, run_id=RUN, outcome=outcome, trajectory_decision=decision, role_use=role_use, checks=dict(input_hashes=True, ordered_50k_rows=True, target_alignment=True, oof_mae_recomputed=True, frozen_execution_source_verified=True, nomination_rule_failed=True, protected_roles_untouched=True), observed_execution=dict(source_commit=source_commit, analyzer_sha256=report["provenance"]["analyzer_sha256"], shared_oracle_sha256={r:sha256_file(ROOT/r) for r in ("src/molgap/hierarchical_oracle.py","src/molgap/multi2d.py")}), provenance_discrepancy=dict(field="prospective.actions.A001.source_commit", declared=prospective["actions"][0]["source_commit"], planned_desktop_base=prospective["state_at_start"]["source_commit"], observed_execution_source=source_commit, reason="Example action scalar was not rebound; preserved original snapshot, actual source independently verified; no training replay claim"))
    atomic_json(HERE/"acceptance.json", acceptance)
    bound = ["analysis.json","acceptance.json","protocol.md","decision.md","paper_review.md","experiment_plan.md","analyze.py"]
    hashes = {f"{REL}/{name}":sha256_file(HERE/name) for name in bound}
    for item in report["identity"]["inputs"].values():
        hashes[item["retained_path"]] = item["sha256"]
    hashes[oof["path"]] = oof["sha256"]
    authority = [f"{REL}/{name}" for name in ("protocol.md","decision.md","acceptance.json","analysis.json","experiment_plan.md")]
    evidence = dict(format="molgap-v5-evidence-envelope-v1", evidence_id=EID, track="B", contract="MOLGAP-COMMON-V5-FINAL", legacy_contract="pcqm-expert-oracle-feasibility-v1", scope="desktop_saved_prediction_oracle_and_gate_feasibility", authority=dict(pointers=authority), artifacts=[dict(name=n, locator=f"{REL}/{n}", sha256=hashes[f"{REL}/{n}"], availability="committed_metadata_verified") for n in ("analysis.json","acceptance.json","decision.md","paper_review.md","experiment_plan.md")], outcome=outcome, role_use=role_use, migration=dict(migrated_at="2026-09-30", training_executed=False, inference_executed=False, scientific_reinterpretation=False, verification_scope="CPU saved-prediction diagnostic; source transcription discrepancy retained in acceptance", observed_source_commit=source_commit))
    cost = dict(schema="molgap-cost-event-v1", cost_event_id="cost-TB-pcqm-expert-oracle-feasibility-measured", trajectory_id=TID, action_id="A001", run_id=RUN, attempt_id="attempt-001", category="audit", platform="local-windows", hardware="LOCAL_CPU_UNSPECIFIED", evidence_ref=f"{REL}/analysis.json", measurement=dict(device_hours=dict(value=None,status="not_applicable"), queue_hours=dict(value=None,status="not_applicable"), cpu_hours=dict(value=report["local_cost"]["process_cpu_seconds"]/3600,status="measured"), wall_hours=dict(value=report["local_cost"]["wall_seconds"]/3600,status="measured")))
    old_role = next(r for p in (ROOT/"experiments/pcqm_k1_residual_reconciliation/roles").glob("*.json") if (r := json.loads(p.read_text()))["access_kind"] == "labels_read")
    roles = [dict(schema="molgap-role-event-v1", role_event_id=f"role-{TID}-{kind}", trajectory_id=TID, action_id="A001", run_id=RUN, dataset_identity=old_role["dataset_identity"], row_manifest_hash=old_role["row_manifest_hash"], role_name="fixed50k_development", access_kind=kind, selection_used=True, evidence_ref=f"{REL}/acceptance.json") for kind in ("prediction_input","labels_read","metric_computed","selection_used")]
    cost["evidence_ref"] = f"{REL}/acceptance.json"
    acceptance.update(costs=[cost], roles=roles)
    atomic_json(HERE/"acceptance.json", acceptance)
    hashes[f"{REL}/acceptance.json"] = sha256_file(HERE/"acceptance.json")
    for artifact in evidence["artifacts"]:
        artifact["sha256"] = hashes[artifact["locator"]]
    terminal = dict(format="molgap-rml-terminal-package-v1", trajectory_id=TID, run_id=RUN, action_id="A001", finalized_at=datetime.now(timezone.utc).isoformat(), acceptance_ref=f"{REL}/acceptance.json", artifact_hashes=hashes, evidence=evidence, decision=decision, costs=[cost], roles=roles)
    atomic_json(HERE/"terminal.json", terminal)
    print(json.dumps(finalize(ROOT,f"{REL}/rml",f"{REL}/terminal.json")))


if __name__ == "__main__":
    main()
