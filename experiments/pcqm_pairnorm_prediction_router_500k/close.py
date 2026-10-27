"""Finalize the gate-only diagnostic through the existing RML API."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct

from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
read = lambda p: json.loads((HERE / p).read_text(encoding="utf-8"))
result, trajectory = read("analysis.json"), read("rml/trajectory.json")
if result["rows"] != 50000 or result["training_executed"] or result["backbone_inference_executed"] or not result["gate_fit"]:
    raise ValueError("Unexpected execution scope")
for item in result["inputs"].values():
    if sha256_file(Path(item["path"])) != item["sha256"]:
        raise ValueError("Input changed")
if sha256_file(Path(result["retained_oof"]["path"])) != result["retained_oof"]["sha256"]:
    raise ValueError("Cross-fitted outputs changed")
if result["source_sha256"] != sha256_file(HERE / "analyze.py"):
    raise ValueError("Analyzer changed")
tid = trajectory["trajectory_id"]
run = trajectory["actions"][0]["run_ids"][0]
eid = "pcqm-pairnorm-prediction-router-500k-20261027"
outcome = dict(execution_status="complete_gate_fit_no_backbone_training",artifact_status="local_hash_verified",
    comparison_status="consumed_role_crossfit_diagnostic",scientific_status="NO_TRAIN",transfer_status="not_evaluated",
    budget_decision="cpu_diagnostic_complete",full_handoff_status="not_applicable")
decision = dict(outcome="NO_TRAIN",decision_ref=f"{REL}/decision.md",next_allowed_actions=[],
    reopen_conditions=["New input features or independent role with separately prospective authorization"])
cost = dict(schema="molgap-cost-event-v1",cost_event_id="cost-pcqm-pairnorm-prediction-router-observed",
    trajectory_id=tid,action_id="A001",run_id=run,attempt_id="attempt-001",platform="local-windows",hardware="CPU four threads; no accelerator",
    category="audit",evidence_ref=f"{REL}/acceptance.json",measurement={
        "wall_hours":{"value":result["wall_seconds"]/3600,"status":"measured"},
        "cpu_hours":{"value":result["cpu_seconds"]/3600,"status":"measured"},
        "device_hours":{"value":None,"status":"not_applicable"},"queue_hours":{"value":None,"status":"not_applicable"}})
row_hash = hashlib.sha256(b"".join(struct.pack("<q",i) for i in range(500000,550000))).hexdigest()
roles = [dict(schema="molgap-role-event-v1",role_event_id=f"role-pairnorm-prediction-router-{kind}",trajectory_id=tid,
    action_id="A001",run_id=run,dataset_identity="pcqm4mv2-ogb-fixed-500k-scnet-v1",row_manifest_hash=row_hash,
    role_name="internal_development_retained_predictions",access_kind=kind,selection_used=True,evidence_ref=f"{REL}/acceptance.json")
    for kind in ("labels_read","metric_computed","selection_used")]
role_use = dict(internal_development_retained_predictions="consumed",official_validation="untouched",test_dev="untouched",test_challenge="untouched",common="untouched",ood="untouched")
acceptance = dict(format="molgap-local-saved-prediction-acceptance-v1",evidence_id=eid,run_id=run,outcome=outcome,
    trajectory_decision=decision,role_use=role_use,costs=[cost],roles=roles,comparison_class="CONTEXT_ONLY",
    analysis_ref=f"{REL}/analysis.json",execution_scope="Prediction-only gate fit and cross-fit; no backbone execution")
atomic_json(HERE / "acceptance.json",acceptance)
names = ["protocol.md","decision.md","attribution.md","analysis.json","acceptance.json","analyze.py"]
hashes = {f"{REL}/{n}":sha256_file(HERE/n) for n in names}
evidence = dict(format="molgap-v5-evidence-envelope-v1",contract="MOLGAP-COMMON-V5-FINAL",evidence_id=eid,track="B",
    scope="pcqm-pairnorm-prediction-router-500k",legacy_contract="pcqm-pairnorm-prediction-router-500k-v1",outcome=outcome,
    authority={"pointers":[f"{REL}/{n}" for n in ("protocol.md","decision.md","attribution.md","acceptance.json","analysis.json")]},role_use=role_use,
    migration=dict(migrated_at=datetime.now(timezone.utc).date().isoformat(),training_executed=False,inference_executed=False,
        scientific_reinterpretation=False,verification_scope="CPU gate fit and prediction-only cross-fit"),
    observed_execution=dict(training_executed=False,inference_executed=False,training_replay_ready=False,gate_fit=True,execution_ref=f"{REL}/analysis.json"),
    artifacts=[dict(name=n,locator=f"{REL}/{n}",sha256=hashes[f"{REL}/{n}"],availability="locally_retained_hash_verified") for n in names])
terminal = dict(format="molgap-rml-terminal-package-v1",trajectory_id=tid,run_id=run,action_id="A001",
    finalized_at=datetime.now(timezone.utc).isoformat(),acceptance_ref=f"{REL}/acceptance.json",artifact_hashes=hashes,
    evidence=evidence,decision=decision,costs=[cost],roles=roles)
atomic_json(HERE/"terminal.json",terminal)
print(finalize(ROOT,f"{REL}/rml",f"{REL}/terminal.json"))
