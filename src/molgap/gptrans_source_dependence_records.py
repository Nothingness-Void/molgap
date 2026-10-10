"""Observation terminal adapter over the existing two-role NO_TRAIN RML owner."""
from datetime import datetime, timezone
import json

from .gptrans_source_dependence import BASE, MODELS, ROLES, accept
from .training_reproducibility import atomic_json, sha256_file


def close(root,output,inputs):
    from .k1_relation_audit_records import prepare_no_train_terminal
    from .research_memory.finalize import finalize
    if (root/BASE/"rml_plan/rml_finalized").exists():
        return finalize(root,root/BASE/"rml_plan/trajectory.json",root/BASE/"results/terminal.json")
    if not (root/BASE/"decision_terminal.md").is_file() or "decision_terminal.md" not in (root/BASE/"decision.md").read_text(encoding="utf-8"):
        raise ValueError("Controller interpretation required before terminal closure")
    accepted=accept(output,inputs)
    frozen=json.loads((root/BASE/"rml_plan/trajectory.json").read_text())
    results=root/BASE/"results"
    ledger=accepted["native_cost"]
    atomic_json(results/"acceptance_summary.json",accepted)
    atomic_json(results/"execution.json",dict(allocation=ledger,training_executed=False,local_model_inference_executed=False))
    contract=json.loads((inputs/"contract.json").read_text())
    atomic_json(results/"role_row_manifests.json",{role:dict(rows=512,role_range=contract["roles"][role],
        panel_sha256=contract["reference_payloads"]["panels.json"]["sha256"],reused_development=True,
        checkpoint_sha256={arm:contract["model_assets"][arm+"_best.pt"]["sha256"] for arm in MODELS}) for role in ROLES})
    manifest=json.loads((output/"output_manifest.json").read_text())
    refs=[(output/name).relative_to(root).as_posix() for name in ("output_manifest.json",*manifest["files"]) if name.endswith(".json")]
    # Bind all observation chunks in evidence, not only a summary or worker flags.
    refs += [(output/name).relative_to(root).as_posix() for name in manifest["files"] if name.endswith(".pt")]
    refs += [BASE+"/results/"+n+".json" for n in ("acceptance_summary","execution","role_row_manifests","role_history","cost_records")]
    missing=dict(status="measurement_missing",value=None)
    paths=prepare_no_train_terminal(prefix=BASE,frozen=frozen,run_id=frozen["actions"][0]["run_ids"][0],
        evidence_id="pcqm-gptrans-source-dependence-s42-v1",outcome=dict(execution_status="complete",artifact_status="accepted",
            comparison_status="context_only",scientific_status="no_train_source_observations",transfer_status="not_tested",
            budget_decision="within_frozen_cap",full_handoff_status="not_authorized"),
        scope="frozen_source_dependence_NO_TRAIN_no_causal_training_claim",finalized_at=datetime.now(timezone.utc).isoformat(),
        artifact_refs=refs,authority=[BASE+"/"+n for n in ("protocol.md","contract.json","role_plan.json","budget.json",
            "release_binding.json","submission_receipt.json","decision.md","decision_terminal.md")],acceptance_name="acceptance_summary",repo_root=root,
        cost_measurement=dict(device_hours=dict(status="measured",value=ledger["allocated_device_seconds"]/3600),
            wall_hours=dict(status="measured",value=ledger["wall_seconds"]/3600),cpu_hours=missing,queue_hours=missing),cost_semantics=ledger["scope"])
    return finalize(root,root/paths["trajectory"],root/paths["terminal"])
