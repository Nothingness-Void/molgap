"""Freeze this saved-prediction routing question using the RML planner."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-pcqm-pairnorm-prediction-router-500k-20261027"
RUN = "local-pcqm-pairnorm-prediction-router-20261027"
POLICY = "pcqm-pairnorm-prediction-router-500k"
cost_id = "cost-pcqm-pairnorm-prediction-router-expected"
spec = json.loads((ROOT / "experiments/pcqm_k1_gptrans_pairnorm_fusion_500k/plan_input.json").read_text())
t = spec["trajectory"]
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
refs = ["pcqm-k1-gptrans-pairnorm-fusion-500k-20261027", "pcqm-expert-oracle-feasibility-no-train-20260930"]
t.update(trajectory_id=TID, family_id=POLICY,
    question="Can prediction-only routing beat cross-fitted global fusion for the new PairNorm checkpoints?",
    comparison_class="CONTEXT_ONLY", comparison_readiness_ref=f"{REL}/protocol.md",
    comparison_blockers=["Model-selection-used development role", "Cross-fit gate is not independent model validation"])
t["hypothesis"].update(hypothesis_id="H-pcqm-pairnorm-prediction-router-500k-20261027",
    observed_deficiency="New K1/PairNorm fixed fusion has a large label-informed oracle gap; old prediction-only gate failed its 1 meV nomination.",
    supporting_evidence_ids=refs, alternative_explanations=["New predictions retain the old gate's limited learnability", "Cross-fitted gain does not transfer beyond repeatedly consumed development"],
    changed_mechanism="Existing hard and convex soft cross-fitted gates on retained model predictions",
    cheapest_falsifier="Five-fold OOF fixed/global/gate comparison on the pinned 50K role",
    decision_changed_if_positive="Consider separately authorized independent-role evaluation only",
    decision_changed_if_negative="Close prediction-only routing for this pair under this diagnostic",
    expected_native_cost_ref=cost_id, related_closed_family_ids=["pcqm-prediction-disagreement-expert-routing"])
t["state_at_start"].update(source_commit=commit, source_config_identity=sha256_file(HERE / "protocol.md"),
    contract_refs=[f"{REL}/protocol.md"], reference_ids=[], parent_trajectory_ids=[], prior_evidence_ids=refs,
    role_snapshot_refs=[f"{REL}/roles.json"], budget_snapshot_ref=f"{REL}/protocol.md")
t["actions"] = [{"action_id":"A001", "type":"local_crossfit_prediction_router", "run_ids":[RUN],
    "attempt_ids":["attempt-001"], "source_commit":commit, "evidence_refs":[f"{REL}/protocol.md"], "cost_event_ids":[cost_id]}]
t["decision"].update(outcome="ACTIVE", decision_ref=f"{REL}/decision.md", next_allowed_actions=["CPU prediction-only cross-fit"], reopen_conditions=[])
c = spec["costs"][0]
c.update(cost_event_id=cost_id, trajectory_id=TID, run_id=RUN, platform="local-windows", hardware="CPU; no accelerator", category="audit", evidence_ref=f"{REL}/protocol.md")
c["measurement"] = {"wall_hours":{"value":2.0,"status":"estimated"}, "cpu_hours":{"value":None,"status":"measurement_missing"},
    "device_hours":{"value":None,"status":"not_applicable"}, "queue_hours":{"value":None,"status":"not_applicable"}}
spec["decision_state"].update(policy_id=POLICY, source_commit=commit, state_timestamp=datetime.now(timezone.utc).isoformat(),
    budget_snapshot_ref=f"{REL}/protocol.md", role_snapshot_refs=[f"{REL}/roles.json"])
policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-bn-mechanism-a100.1.json").read_text())
policy.update(policy_id=POLICY, comparability_selector={"scientific_contract":POLICY+"-v1"}, created_from_source_digest=sha256_file(HERE / "protocol.md"))
atomic_json(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
atomic_json(HERE / "roles.json", {"internal_development_retained_predictions":"fit gate and compute cross-fitted metrics; model-selection-used", "official_validation":"untouched", "test_dev":"untouched", "test_challenge":"untouched"})
atomic_json(HERE / "plan_input.json", spec)
atomic_json(HERE / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))
print("prospective published; labels and predictions not read")
