"""Publish the new saved-prediction question with the existing RML planner."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-pcqm-expert-oracle-feasibility-20260930"
RUN = "local-expert-oracle-feasibility-20260930"

def main():
    protocol = f"{REL}/protocol.md"
    policy = json.loads((ROOT / "research_memory/policies/pcqm-geometry-reliability-screen.1.json").read_text())
    policy.update(policy_id="pcqm-expert-oracle-feasibility", comparability_selector={"scientific_contract": "pcqm-expert-oracle-feasibility-v1"}, required_observable_fields=["user_requested_local_oracle_review"], action_rule={"field": "user_requested_local_oracle_review", "operator": "eq", "threshold": 1, "action": "RUN_DIAGNOSTIC"}, created_from_source_digest=sha256_file(HERE / "protocol.md"))
    atomic_json(ROOT / "research_memory/policies/pcqm-expert-oracle-feasibility.1.json", policy)
    spec = json.loads((ROOT / "experiments/pcqm_geometry_reliability_gate/plan_input.json").read_text())
    t = spec["trajectory"]
    costid = "cost-TB-pcqm-expert-oracle-feasibility-estimate"
    t.update(trajectory_id=TID, family_id="pcqm-prediction-disagreement-expert-routing", question="Does label-free prediction disagreement realize Oracle complementarity beyond the existing global blend?")
    t["hypothesis"] = dict(hypothesis_id="H-pcqm-expert-oracle-feasibility-20260930", observed_deficiency="Coarse molecular summaries fail to predict expert winners although accepted errors differ", supporting_evidence_ids=["pcqm-k1-residual-reconciliation"], alternative_explanations=["Fusion succeeds by cancellation rather than winner selection", "Winner identity depends on latent features absent from retained predictions", "Consumed development selection overstates learnability"], changed_mechanism="Saved-prediction-only postdispatch hard/soft gate diagnostic; encoders unchanged", cheapest_falsifier="Hash-gated Oracle bounds and fixed five-fold prediction-space gate comparison", related_closed_family_ids=["k1-existing-prediction-attribution", "k1-gptrans-fusion"], expected_native_cost_ref=costid, decision_changed_if_positive="Propose one independent-role specialist diagnostic after pretraining acceptance", decision_changed_if_negative="Keep fixed fusion comparator and do not train a prediction-only router")
    t["state_at_start"].update(contract_refs=[protocol, "experiments/pcqm_k1_residual_reconciliation/protocol.md"], reference_ids=["pcqm-k1-residual-reconciliation"], parent_trajectory_ids=["TB-k1-residual-reconciliation"], prior_evidence_ids=["pcqm-k1-residual-reconciliation"], role_snapshot_refs=[protocol], budget_snapshot_ref=protocol, source_config_identity=sha256_file(HERE / "protocol.md"))
    t["actions"] = [dict(action_id="A001", type="local_saved_prediction_oracle_and_crossfit_diagnostic", source_commit=t["state_at_start"]["source_commit"], run_ids=[RUN], attempt_ids=["attempt-001"], evidence_refs=[], cost_event_ids=[costid])]
    t["decision"].update(decision_ref=f"{REL}/decision_plan.md", next_allowed_actions=["one_local_saved_prediction_diagnostic"])
    spec["decision_state"].update(policy_id=policy["policy_id"], budget_snapshot_ref=protocol, state_timestamp=datetime.now(timezone.utc).isoformat())
    cost = copy.deepcopy(spec["costs"][0])
    cost.update(cost_event_id=costid, trajectory_id=TID, run_id=RUN, evidence_ref=protocol)
    cost["measurement"]["wall_hours"] = dict(value=0.25, status="estimated")
    spec["costs"] = [cost]
    atomic_json(HERE / "plan_input.json", spec)
    print(json.dumps(plan(ROOT, spec, f"{REL}/rml")))

if __name__ == "__main__":
    main()
