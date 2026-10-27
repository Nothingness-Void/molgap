"""Publish a prospective, no-training fixed-prediction diagnostic."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-gptrans-pairnorm-fusion-500k-20261027"
RUN = "local-k1-gptrans-pairnorm-fusion-20261027"
POLICY = "pcqm-k1-gptrans-pairnorm-fusion-500k"
cost_id = "cost-k1-gptrans-pairnorm-fusion-expected"

prior = json.loads((ROOT / "experiments/pcqm_k1_complete_local_audit/plan_input.json").read_text())
spec = prior
t = spec["trajectory"]
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
refs = ["pcqm-k1-complete-module-audit-20261008"]
t.update(trajectory_id=TID, family_id=POLICY,
         question="Does retained K1 improve fixed fusion with the new GPTrans PairNorm runs?",
         comparison_class="CONTEXT_ONLY", comparison_readiness_ref=f"{REL}/protocol.md",
         comparison_blockers=["Cross-run exploratory blend on selection-used development", "No independent official evaluation"])
t["hypothesis"].update(hypothesis_id="H-k1-gptrans-pairnorm-fusion-20261027",
    observed_deficiency="Single-model 500K development MAE exceeds previous heterogeneous blend.",
    supporting_evidence_ids=refs, alternative_explanations=["New GPTrans predictions make K1 redundant", "The observed advantage does not generalize"],
    changed_mechanism="Fixed averaging of saved model predictions; no fitting or model execution",
    cheapest_falsifier="Compare fixed equal blends on exactly aligned saved development predictions",
    decision_changed_if_positive="Consider further independent evaluation, without model promotion",
    decision_changed_if_negative="Do not prioritize this blend on this evidence",
    expected_native_cost_ref=cost_id, related_closed_family_ids=["k1-complete-module-audit"])
t["state_at_start"].update(source_commit=commit, source_config_identity=sha256_file(HERE / "protocol.md"),
    contract_refs=[f"{REL}/protocol.md"], reference_ids=[], parent_trajectory_ids=[], prior_evidence_ids=refs,
    role_snapshot_refs=[f"{REL}/roles.json"], budget_snapshot_ref=f"{REL}/protocol.md")
t["actions"] = [{"action_id":"A001", "type":"local_saved_prediction_diagnostic", "run_ids":[RUN],
    "attempt_ids":["attempt-001"], "source_commit":commit, "evidence_refs":[f"{REL}/protocol.md"], "cost_event_ids":[cost_id]}]
t["decision"].update(outcome="ACTIVE", decision_ref=f"{REL}/decision.md", next_allowed_actions=["CPU saved-prediction diagnostic"], reopen_conditions=[])
c = spec["costs"][0]
c.update(cost_event_id=cost_id, trajectory_id=TID, run_id=RUN, platform="local-windows", hardware="CPU; no accelerator",
         category="audit", evidence_ref=f"{REL}/protocol.md")
c["measurement"] = {"wall_hours":{"value":None,"status":"measurement_missing"},
    "cpu_hours":{"value":None,"status":"measurement_missing"}, "device_hours":{"value":None,"status":"not_applicable"},
    "queue_hours":{"value":None,"status":"not_applicable"}}
ds = spec["decision_state"]
ds.update(policy_id=POLICY, source_commit=commit, state_timestamp=datetime.now(timezone.utc).isoformat(),
    budget_snapshot_ref=f"{REL}/protocol.md", role_snapshot_refs=[f"{REL}/roles.json"])
policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-bn-mechanism-a100.1.json").read_text())
policy.update(policy_id=POLICY, comparability_selector={"scientific_contract":POLICY+"-v1"}, created_from_source_digest=sha256_file(HERE / "protocol.md"))
atomic_json(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
atomic_json(HERE / "roles.json", {"internal_development_retained_predictions":"read labels and pre-existing predictions; historically selection-used", "official_validation":"untouched", "test_dev":"untouched", "test_challenge":"untouched"})
atomic_json(HERE / "plan_input.json", spec)
atomic_json(HERE / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))
print("prospective published; no prediction data consumed")
