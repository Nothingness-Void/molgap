"""Verify retained inputs and publish the bounded BN diagnostic prospective."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from molgap.constants import REPO_ROOT as ROOT
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = ROOT / "experiments/pcqm_k1_500k_bn_calibration"
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-500k-bn-calibration-20261007"
RUN = "local-k1-500k-bn-calibration-20261007"
POLICY = "pcqm-k1-500k-bn-calibration"


def main():
    if (HERE / "rml").exists():
        raise FileExistsError("Prospective already published")
    prior_path = ROOT / "experiments/pcqm_k1_500k_bottleneck_diagnostic/inputs.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    keys = ["owner_commit", "archive", "frozen_source_root", "frozen_source_files", "cache_root", "manifest_sha256"]
    inputs = {key: prior[key] for key in keys}
    inputs["prior_inputs_sha256"] = sha256_file(prior_path)
    inputs["checkpoints"] = {name: prior["checkpoints"][name] for name in ["best_model.pt", "best_predictions.pt"]}
    for name, item in {**inputs["checkpoints"], "source archive": inputs["archive"]}.items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Accepted input differs: {name}")
    frozen = Path(inputs["frozen_source_root"]).parent
    for name, digest in inputs["frozen_source_files"].items():
        if sha256_file(frozen / name) != digest:
            raise ValueError(f"Accepted executable source differs: {name}")
    cache = Path(inputs["cache_root"])
    if sha256_file(cache / "manifest.json") != inputs["manifest_sha256"]:
        raise ValueError("Accepted cache manifest differs")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    for shard in manifest["geometry_shards"]:
        if sha256_file(cache / shard["file"]) != shard["sha256"]:
            raise ValueError(f"Accepted graph shard differs: {shard['file']}")
    authority = HERE / "input_authority"
    authority.mkdir(exist_ok=True)
    owner = Path("D:/w/k1-gptrans-500k-package/experiments/pcqm_k1_gptrans_package_transfer_500k/terminal_acceptance")
    for origin, name in [
        (owner / "k1_pretrained_consistency/acceptance.json", "acceptance.snapshot.json"),
        (owner / "cost_accuracy_attribution.md", "cost_accuracy_attribution.md"),
    ]:
        shutil.copyfile(origin, authority / name)
    inputs.update(sample_seed=20261008, calibration_rows=16384, train_range=[0, 500000],
        development_range=[500000, 550000], development_rows=50000, batch_size=128,
        cpu_threads=4, worker_wall_ceiling_seconds=600, prediction_tolerance_eV=1e-4,
        calibration={"sampling": "uniform_without_replacement_shuffled", "batch_norm_momentum": None,
            "reset_running_statistics": True, "dropout_enabled": False, "parameter_updates": False,
            "gradients": False, "label_objective": False},
        material_gain_eV=0.001, paired_row_confidence=0.95,
        authorization="User-authorized bounded local BN-only training-feature state adaptation and previously consumed internal development evaluation; no training or protected roles.",
        executed_source_files={name: sha256_file(ROOT / name) for name in [
            f"{REL}/run.py", f"{REL}/prepare.py", "src/molgap/k1_frozen_inference.py",
            "src/molgap/k1_bn_calibration.py"]})
    atomic_json(HERE / "inputs.json", inputs)
    atomic_json(HERE / "roles.json", {
        "train_prefix": {"range": [0, 500000], "decoded_labels_read": 500000,
            "prediction_input_rows": 16384, "labels_used_for_calibration": False},
        "internal_development": {"range": [500000, 550000], "decoded_labels_read": 50000,
            "prediction_input_rows": 50000, "previously_selection_used": True},
        "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"})
    timestamp = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    refs = ["pcqm-k1-500k-bottleneck-diagnostic-20261007", "pcqm-matched-500k-v4-three-arm"]
    cost_id = "cost-k1-500k-bn-calibration-expected"
    state = {"source_commit": commit, "source_config_identity": sha256_file(HERE / "inputs.json"),
        "contract_refs": [f"{REL}/protocol.md", f"{REL}/inputs.json"], "reference_ids": [],
        "parent_trajectory_ids": [], "prior_evidence_ids": refs,
        "role_snapshot_refs": [f"{REL}/roles.json"], "budget_snapshot_ref": f"{REL}/protocol.md"}
    trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": TID,
        "record_mode": "prospective", "owner": "desktop", "track": "B",
        "family_id": "k1-500k-bn-calibration",
        "question": "Does training-member BN-only recalibration improve the selected epoch49 K1 predictor on its consumed full50K development cohort?",
        "hypothesis": {"hypothesis_id": "H-k1-500k-bn-calibration-20261007",
            "observed_deficiency": "Frozen bottleneck evidence leaves clean inference normalization mismatch unresolved.",
            "supporting_evidence_ids": refs,
            "alternative_explanations": ["Running statistics already adequate", "Generalization or optimization limits", "Selected-development diagnostic gain fails independent-role transfer"],
            "changed_mechanism": "Reset and cumulatively recompute BN running buffers on16384 uniformly sampled train members with dropout disabled; all learned parameters fixed.",
            "cheapest_falsifier": "One seeded training-feature calibration and paired baseline/calibrated full50K development inference under600s worker ceiling.",
            "decision_changed_if_positive": "A gain>=0.001eV with positive paired-row95% lower bound nominates normalization mismatch for separate validation; close NO_TRAIN.",
            "decision_changed_if_negative": "Deprioritize this BN-buffer explanation under the exact diagnostic; close NO_TRAIN.",
            "expected_native_cost_ref": cost_id,
            "related_closed_family_ids": ["k1-500k-frozen-bottleneck-diagnostic"],
            "historical_unknowns": ["Training seed variance", "Independent-role transfer", "Training-time BN intervention effect"]},
        "state_at_start": state,
        "actions": [{"action_id": "A001", "type": "local_bn_buffer_calibration",
            "run_ids": [RUN], "attempt_ids": ["attempt-001"], "source_commit": commit,
            "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": [cost_id]}],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md",
            "next_allowed_actions": ["Bounded local BN-only calibration diagnostic"], "reopen_conditions": []},
        "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{REL}/protocol.md",
        "comparison_blockers": ["Selected, previously consumed development cohort and single trained seed", "Parameter-free BN state adaptation is not training replay or adoption"],
        "reference_bundle_id": None}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": TID,
        "action_id": "A001", "run_id": RUN, "attempt_id": "attempt-001", "platform": "local-windows",
        "hardware": "CPU; four intra-op threads", "category": "inference", "evidence_ref": f"{REL}/protocol.md",
        "measurement": {"wall_hours": {"value": 600 / 3600, "status": "estimated"},
            "cpu_hours": {"value": None, "status": "measurement_missing"},
            "device_hours": {"value": None, "status": "not_applicable"},
            "queue_hours": {"value": None, "status": "not_applicable"}}}
    spec = {"trajectory": trajectory, "costs": [cost], "decision_state": {
        "known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
        "available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC",
        "policy_id": POLICY, "policy_version": "1", "budget_snapshot_ref": f"{REL}/protocol.md",
        "role_snapshot_refs": [f"{REL}/roles.json"], "source_commit": commit, "state_timestamp": timestamp}}
    atomic_json(HERE / "plan_input.json", spec)
    receipt = plan(ROOT, spec, f"{REL}/rml")
    atomic_json(HERE / "plan_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
