"""Publish retained train-only engineering checks through the shared finalizer."""
from datetime import datetime, timezone
import json
from pathlib import Path

from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v4_runtime import normalized_source_sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-flag-cpu-qualification-20261002"
RUN = "local-k1-flag-qualification-20261002"
EID = "pcqm-k1-flag-cpu-qualification-20261002"


def main():
    frozen = json.loads((HERE / "cpu_frozen.json").read_text(encoding="utf-8"))
    result = json.loads((HERE / "qualification/qualification_result.json").read_text(encoding="utf-8"))
    if ((HERE / "qualification/cpu_failure.json").exists() or result["accepted"] is not True or
            result["freeze_sha256"] != sha256_file(HERE / "cpu_frozen.json") or
            result["parameters"] != 3658817 or result["rows"] != 256 or
            result["development_role_read"] or result["protected_roles_read"] or
            result["wall_seconds"] > 300 or
            sha256_file(HERE / "cpu_prospective/trajectory.json") != frozen["prospective_sha256"]):
        raise ValueError("Retained CPU result violates the published contract")
    for name, digest in frozen["source_hashes"].items():
        if normalized_source_sha256(ROOT / name) != digest:
            raise ValueError("Frozen executed source changed: " + name)
    outcome = dict(execution_status="complete_no_training", artifact_status="local_hash_verified",
        comparison_status="engineering_checks_only", scientific_status="NO_TRAIN",
        transfer_status="not_evaluated", budget_decision="cpu_closure_no_training_release",
        full_handoff_status="not_applicable")
    decision = dict(outcome="NO_TRAIN", decision_ref=f"{REL}/cpu_decision.md",
        next_allowed_actions=[], reopen_conditions=["Separate authorized training prospective and actual T4 qualification."])
    acceptance_ref = f"{REL}/cpu_acceptance.json"
    cost = dict(schema="molgap-cost-event-v1", cost_event_id="cost-k1-flag-cpu-observed-20261002",
        trajectory_id=TID, action_id="A001", run_id=RUN, attempt_id="cpu-001",
        platform="local-windows", hardware="CPU4threads; no accelerator", category="preflight",
        evidence_ref=acceptance_ref, measurement={
            "wall_hours": dict(status="measured", value=result["wall_seconds"] / 3600),
            "cpu_hours": dict(status="measured", value=result["process_cpu_seconds"] / 3600),
            "device_hours": dict(status="not_applicable", value=None),
            "queue_hours": dict(status="not_applicable", value=None)})
    role_use = dict(train_prefix="diagnostic_labels_used_no_selection", internal_development="untouched",
        official_validation="untouched", test_dev="untouched", test_challenge="untouched")
    roles = [dict(schema="molgap-role-event-v1", role_event_id=f"role-k1-flag-cpu-{kind}-20261002",
        trajectory_id=TID, action_id="A001", run_id=RUN, dataset_identity="pcqm4mv2-ogb-fixed-100k-v1",
        row_manifest_hash=result["source_idx_sha256"], role_name="train_prefix_0_256", access_kind=kind,
        selection_used=False, evidence_ref=acceptance_ref) for kind in ("prediction_input", "labels_read", "metric_computed")]
    acceptance = dict(format="molgap-k1-flag-cpu-acceptance-v1", evidence_id=EID, run_id=RUN,
        outcome=outcome, trajectory_decision=decision, role_use=role_use, costs=[cost], roles=roles,
        qualification_result_ref=f"{REL}/qualification/qualification_result.json",
        scientific_nomination=False, role_scope="This CPU action only; retained reference role history remains consumed.", cost_scope="Worker timer only; launcher/planning/publication overhead unmeasured.")
    atomic_json(HERE / "cpu_acceptance.json", acceptance)
    names = ["cpu_frozen.json", "cpu_prospective/trajectory.json", "qualification/qualification_result.json",
        "qualification/sensitivity.json", "qualification/initial_state.pt", "qualification/cpu_resume.pt",
        "cpu_decision.md", "cpu_attribution.md", "cpu_acceptance.json", "protocol.md", "evidence_review.md",
        "rml_review.md", "rml_inventory.json", "role_plan.json", "finalize_cpu.py"]
    hashes = {f"{REL}/{name}": sha256_file(HERE / name) for name in names}
    authority = [f"{REL}/{name}" for name in ("protocol.md", "cpu_decision.md", "cpu_attribution.md",
        "cpu_acceptance.json", "qualification/qualification_result.json")]
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL",
        evidence_id=EID, track="B", scope="desktop_k1_flag_cpu_qualification", legacy_contract="k1-flag-cpu-qualification-v1",
        outcome=outcome, authority=dict(pointers=authority), role_use=role_use,
        migration=dict(migrated_at="2026-10-02", training_executed=False, inference_executed=False,
            scientific_reinterpretation=False, verification_scope="Publication of independently executed engineering diagnostic."),
        observed_execution=dict(training_executed=False, diagnostic_optimizer_updates_executed=True,
            inference_executed=True, execution_ref=f"{REL}/qualification/qualification_result.json"),
        artifacts=[dict(name=name, locator=f"{REL}/{name}", sha256=hashes[f"{REL}/{name}"],
            availability="locally_retained_hash_verified") for name in
            ("qualification/qualification_result.json", "qualification/sensitivity.json", "qualification/initial_state.pt",
             "cpu_acceptance.json", "cpu_decision.md")])
    terminal = dict(format="molgap-rml-terminal-package-v1", trajectory_id=TID, run_id=RUN, action_id="A001",
        finalized_at=datetime.now(timezone.utc).isoformat(), acceptance_ref=acceptance_ref,
        artifact_hashes=hashes, evidence=evidence, decision=decision, costs=[cost], roles=roles)
    atomic_json(HERE / "cpu_terminal.json", terminal)
    print(json.dumps(finalize(ROOT, f"{REL}/cpu_prospective", f"{REL}/cpu_terminal.json")))


if __name__ == "__main__":
    main()
