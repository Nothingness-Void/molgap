"""Hash retained inputs and publish the new prospective using the RML owner."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import os
import sys
import numpy as np
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-clean-fit-generalization-500k-20261009"
RUN = "local-k1-clean-fit-generalization-500k-20261009"
POLICY = "pcqm-k1-clean-fit-generalization-500k"


def main():
    if (HERE / "rml").exists():
        raise FileExistsError("Prospective already exists; never overwrite")
    owner = Path("D:/w/k1-consistency-500k")
    previous = owner / "experiments/pcqm_k1_consistency_ablation_500k/clean_bn/results_attempt3_20261009"
    report = json.loads((previous / "pair_report.json").read_text())
    package = Path(report["inputs"]["source_archive"]["path"]).parent
    # The accepted owner supports this newer family contract; desktop's older
    # Spec validator must not be relaxed merely to inspect retained source.
    verification = subprocess.check_output([sys.executable, "-c",
        "import json,sys; from pathlib import Path; from molgap.experiment_package import verify_experiment_source_package; print(json.dumps(verify_experiment_source_package(Path(sys.argv[1]))))",
        str(package)], cwd=owner, env=dict(os.environ, PYTHONPATH=str(owner / "src")), text=True)
    atomic_json(HERE / "package_verification_receipt.json", json.loads(verification))
    if report["status"] != "complete" or report["inputs"]["source_archive"]["sha256"] != "b2b7539d7fbd0fbff6e674ed632ab90be00f86e8484eb6f2e8cb3ddafe06a3fd":
        raise ValueError("Wrong accepted pair/package")
    bind = lambda path: {"path": str(path.resolve()), "sha256": sha256_file(path)}
    arms = {}
    for arm, observation in report["arms"].items():
        if observation["status"] != "complete" or observation["worker_exitcode"] != 0 or not observation["restored_prediction_exact"]:
            raise ValueError("Selected diagnostic not accepted")
        directory = Path(report["inputs"]["arms"][arm]["checkpoint"]["path"]).parent
        artifacts = {"selected": bind(directory / "best_model.pt"), "final": bind(directory / "last_checkpoint.pt")}
        for key, filename in (("original", "original.pt"), ("calibrated", "calibrated.pt"), ("buffers", "calibrated_buffers.pt")):
            artifacts[key] = bind(previous / arm / filename)
            if artifacts[key]["sha256"] != observation["artifacts"][filename]:
                raise ValueError("Accepted selected artifacts differ")
        artifacts["prior_report"] = observation
        arms[arm] = artifacts
    files = [f"{REL}/{name}" for name in ("prepare.py", "run.py", "protocol.md", "evidence_review.md", "reuse.md")]
    files += ["src/molgap/k1_clean_fit_diagnostic.py", "src/molgap/k1_bn_diagnostic.py",
              "src/molgap/k1_frozen_inference.py", "src/molgap/k1_bn_calibration.py", "src/molgap/frozen_diagnostic_closure.py"]
    train = np.random.default_rng(20261008).choice(500000, 16384, replace=False).tolist()
    example = next(iter(report["arms"].values()))["runtime"]
    inputs = {key: report["inputs"][key] for key in ("source_archive", "source_inventory", "frozen_source_root", "cache_root", "manifest_sha256")}
    inputs.update(arms=arms, prior_pair_report=bind(previous / "pair_report.json"),
        cpu_threads=4, ceiling_seconds=1200, train_source_idx=train, development_source_idx=list(range(500000, 550000)),
        train_decoded_source_bounds=[0, 500000], prior_runtime_torch=example["torch"],
        prior_software_sha256=example["installed_distributions_sha256"],
        executed_source_files={name: sha256_file(ROOT / name) for name in files})
    atomic_json(HERE / "inputs.json", inputs)
    atomic_json(HERE / "roles.json", {"train_labels_decoded": [0, 500000], "train_clean_metric_sample": train,
        "calibration_same_sample": True, "development_prediction_and_metric": [500000, 550000],
        "development_selection_previously_used": True, "protected_roles": "untouched"})
    policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-late-weight-average.1.json").read_text())
    policy.update(policy_id=POLICY, comparability_selector={"scientific_contract": POLICY + "-v1"}, created_from_source_digest=sha256_file(HERE / "protocol.md"))
    atomic_json(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    refs = ["pcqm-k1-late-weight-average-20261008", "k1-saved-bn-rows-20261008-evidence"]
    cost_id = "cost-k1-clean-fit-generalization-expected"
    trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": TID, "record_mode": "prospective", "owner": "desktop", "track": "B",
        "family_id": "k1-clean-fit-generalization", "question": "How do matched clean train-sample and consumed-development errors change across frozen arms/epochs/BN states?",
        "hypothesis": {"hypothesis_id": "H-k1-clean-fit-generalization-20261009", "observed_deficiency": "Optimizer-scaled train trace is not clean fit; late clean dev loss and BN sensitivity leave clean train gap unmeasured.",
            "supporting_evidence_ids": refs, "alternative_explanations": ["Clean train and development move together", "Development degrades despite improved sampled train fit", "Calibration changes in-sample fit and dev differently"],
            "changed_mechanism": "Frozen clean evaluation only; no model or training intervention", "cheapest_falsifier": "Bounded CPU fixed16384 clean train predictions and retained/full50K dev endpoints",
            "decision_changed_if_positive": "Describe observed clean-fit discrepancy; no causal proof or training release", "decision_changed_if_negative": "Rule out this exact descriptive gap pattern; no blind retry", "expected_native_cost_ref": cost_id,
            "related_closed_family_ids": ["k1-late-weight-average", "k1-saved-bn-rows"]},
        "state_at_start": {"source_commit": commit, "source_config_identity": sha256_file(HERE / "inputs.json"), "contract_refs": [f"{REL}/protocol.md", f"{REL}/inputs.json", f"{REL}/evidence_review.md"],
            "reference_ids": [], "parent_trajectory_ids": [], "prior_evidence_ids": refs, "role_snapshot_refs": [f"{REL}/roles.json"], "budget_snapshot_ref": f"{REL}/protocol.md"},
        "actions": [{"action_id": "A001", "type": "authorized_local_clean_fit_diagnostic", "run_ids": [RUN], "attempt_ids": ["attempt-001"], "source_commit": commit,
                     "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": [cost_id]}], "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md", "next_allowed_actions": ["One bounded CPU clean-fit diagnostic"], "reopen_conditions": []},
        "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{REL}/protocol.md", "comparison_blockers": ["Consumed selected development", "Train calibration in-sample", "No independent-role or strict training replay qualification"], "reference_bundle_id": None}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "attempt-001",
        "platform": "local-windows", "hardware": "CPU four threads; no accelerator", "category": "inference", "evidence_ref": f"{REL}/protocol.md",
        "measurement": {"wall_hours": {"value": 1200 / 3600, "status": "estimated"}, "cpu_hours": {"value": None, "status": "measurement_missing"},
                        "device_hours": {"value": None, "status": "not_applicable"}, "queue_hours": {"value": None, "status": "not_applicable"}}}
    spec = {"trajectory": trajectory, "costs": [cost], "decision_state": {"available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC", "policy_id": POLICY,
        "policy_version": "1", "state_timestamp": datetime.now(timezone.utc).isoformat(), "budget_snapshot_ref": f"{REL}/protocol.md", "role_snapshot_refs": [f"{REL}/roles.json"], "source_commit": commit}}
    atomic_json(HERE / "plan_input.json", spec)
    atomic_json(HERE / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))
    print("PROSPECTIVE_PUBLISHED_NO_INFERENCE")


if __name__ == "__main__":
    main()
