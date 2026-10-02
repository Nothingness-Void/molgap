"""Freeze a train-only FLAG qualification with the existing RML planner."""
from pathlib import Path
import json
import subprocess

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v4_runtime import normalized_source_sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-flag-cpu-qualification-20261002"
RUN = "local-k1-flag-qualification-20261002"


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    refs = ["pcqm-k1-slot-width96-kaggle3-100k-s42-v1-terminal",
            "pcqm-k1-slot-readout-diagnostic-20261002",
            "pcqm-k1-v4-192-kaggle3-reference-custody-20261002"]
    protocol, review, roles = [f"{REL}/{name}" for name in
                              ("protocol.md", "evidence_review.md", "role_plan.json")]
    policy_id = "k1-flag-cpu-qualification"
    cost_id = "cost-k1-flag-cpu-expected"
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id,
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "cpu-001",
        "platform": "local-windows", "hardware": "CPU4threads; no accelerator", "category": "preflight",
        "evidence_ref": protocol, "measurement": {
            "wall_hours": {"status": "estimated", "value": 300 / 3600},
            "cpu_hours": {"status": "estimated", "value": 1200 / 3600},
            "device_hours": {"status": "not_applicable", "value": None},
            "queue_hours": {"status": "not_applicable", "value": None}}}
    state = {"source_commit": commit, "source_config_identity": "K1 original192/64/64 + train-only FLAG M3 alpha0.001",
        "contract_refs": [protocol, review, f"{REL}/rml_review.md", f"{REL}/rml_inventory.json"],
        "reference_ids": [refs[-1]], "prior_evidence_ids": refs,
        "parent_trajectory_ids": ["TB-k1-slot-width96-kaggle3-100k-s42-v1"],
        "role_snapshot_refs": [roles], "budget_snapshot_ref": protocol}
    hypothesis = {"hypothesis_id": "H-k1-flag-cpu-20261002",
        "observed_deficiency": "Simple atom/slot widening misses material gains; useful preserved paths motivate an untested supervised embedding-robustness objective, not a diagnosed fitting failure.",
        "supporting_evidence_ids": refs,
        "alternative_explanations": ["Perturbations may harm clean Gap regression.",
            "Embedding sensitivity need not predict generalization.", "Repeated evaluations and dropout can account for effects."],
        "changed_mechanism": "Three gradient-directed embedding perturbations averaged before one optimizer step; clean K1 state and architecture unchanged.",
        "cheapest_falsifier": "First256 training rows only: exact clean identity, finite gradients, independent gradient accumulation, resume, storage and descriptive frozen-model sensitivity; CPU300s maximum.",
        "related_closed_family_ids": ["k1-slot-width96", "k1-node-width", "k1-slot-readout-diagnostic", "gptrans-noisy-nodes", "k1-pretraining"],
        "expected_native_cost_ref": cost_id,
        "decision_changed_if_positive": "Review one authorized fixed100K candidate after accepted CPU checks and remote T4 qualification; no automatic scale-up.",
        "decision_changed_if_negative": "NO_TRAIN on implementation/numeric/resource failure; preserve failure attribution.",
        "historical_unknowns": ["Clean Gap transfer benefit of FLAG.", "Single-seed stochasticity."]}
    trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": TID,
        "record_mode": "prospective", "track": "B", "owner": "desktop", "family_id": "k1-flag",
        "question": "Does the frozen train-only FLAG algorithm qualify without changing clean K1 identity?",
        "hypothesis": hypothesis, "state_at_start": state,
        "actions": [{"action_id": "A001", "type": "cpu_train_only_algorithm_and_sensitivity_qualification",
            "source_commit": commit, "run_ids": [RUN], "attempt_ids": ["cpu-001"],
            "evidence_refs": [protocol], "cost_event_ids": [cost_id]}],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {"outcome": "ACTIVE", "decision_ref": protocol,
            "next_allowed_actions": ["A001"], "reopen_conditions": []}}
    policy = {"schema": "molgap-policy-v1", "policy_id": policy_id, "version": "1",
        "policy_type": "research_action", "status": "candidate",
        "comparability_selector": {"scientific_contract": "k1-flag-cpu-qualification-v1"},
        "required_observable_fields": ["evidence_review_complete"],
        "action_rule": {"field": "evidence_review_complete", "operator": "eq", "threshold": 1, "action": "QUALIFY_CPU"},
        "borderline_action": "NO_TRAIN", "observation_point": None, "promotion_rule": None, "early_stop_rule": None,
        "cost_model": {"kind": "measured_only", "assumptions": []},
        "approval": {"approved_by": None, "approved_at": None, "authority_ref": None},
        "created_from_source_digest": sha256_file(HERE / "protocol.md")}
    atomic_json(ROOT / "research_memory/policies/k1-flag-cpu-qualification.1.json", policy)
    atomic_json(HERE / "cpu_action_inputs.json", {"trajectory_id": TID, "state_timestamp": "2026-10-02",
        "evidence_ids": refs, "evidence_review_complete": 1,
        "authority": "User authorized detailed RML review, one justified new experiment and Kaggle3 submission"})
    spec = {"trajectory": trajectory, "costs": [cost], "action_inputs_ref": f"{REL}/cpu_action_inputs.json",
        "decision_state": {"known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
            "available_actions": ["QUALIFY_CPU", "NO_TRAIN"], "chosen_action": "QUALIFY_CPU",
            "policy_id": policy_id, "policy_version": "1", "role_snapshot_refs": [roles],
            "budget_snapshot_ref": protocol, "state_timestamp": "2026-10-02", "source_commit": commit}}
    atomic_json(HERE / "qualification_plan.json", spec)
    result = plan(ROOT, spec, HERE / "cpu_prospective")
    names = [f"{REL}/qualify.py", f"{REL}/protocol.md", f"{REL}/role_plan.json",
             f"{REL}/evidence_review.md", "src/molgap/k1_flag.py", "src/molgap/k1_screen_training.py",
             "src/molgap/qm9_neural_atom.py", "src/molgap/training_reproducibility.py", "src/molgap/v4_runtime.py"]
    checkpoint = ROOT / "platforms/_records/kaggle/training/pcqm_k1_width256_reference_kaggle3_v1/k1_pair/reference/selected_model.pt"
    frozen = {"source_hashes": {name: normalized_source_sha256(ROOT / name) for name in names},
        "prospective_path": f"{REL}/cpu_prospective/trajectory.json",
        "prospective_sha256": sha256_file(HERE / "cpu_prospective/trajectory.json"),
        "graph_shard_path": "D:/文档/molgap/data/pcqm_fixed_100k_v1/train/train_shard_0000.pt",
        "graph_shard_sha256": "8cc4c6373c22c8f72ac302f514bb657c5b40a9a1c620c6228b760380ebec705e",
        "selected_checkpoint_path": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": "4a7dab43f50f5f3216b224374c213a01cc82cfb4160dcd6d54aab1ef614fce17",
        "selected_state_sha256": "828fa756b0fb757d3ff1a125fc7ee5b6c2ddab6fd36c072f35b26b72b2f77822",
        "initial_state_sha256": "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd",
        "target_mean": 5.3383002281188965, "target_std": 1.275090217590332,
        "row_start": 0, "row_stop": 256, "cpu_threads": 4, "max_seconds": 300,
        "source_commit": commit, "trajectory_id": TID, "run_id": RUN}
    atomic_json(HERE / "cpu_frozen.json", frozen)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
