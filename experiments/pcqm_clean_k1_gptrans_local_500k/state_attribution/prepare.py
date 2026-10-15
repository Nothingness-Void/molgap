"""Bind existing artifacts and publish prospective metadata before inference."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.experiment_package import verify_experiment_source_package

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-clean-ema-500k-state-attribution-20261015"
POLICY = "pcqm-k1-session-state-attribution"
RUN = "local-k1-session-state-attribution-20261015"


def main():
    if (HERE / "rml").exists():
        raise FileExistsError("Prospective already exists; never overwrite")
    source = ROOT / "platforms/_records/kaggle/staging/clean_k1_gptrans_local_500k/release001/package"
    package = verify_experiment_source_package(source)
    accepted = HERE.parent / "terminal_acceptance"
    output = accepted / "outputs/experiment/k1_clean_ema"
    manifest = json.loads((output / "output_manifest.json").read_text(encoding="utf-8"))
    context = manifest["context"]
    if context["source_commit"] != "4097e696493a6e87132314c46e332f087b41a196":
        raise ValueError("Wrong producer source")
    bind = lambda p: {"path": str(p.resolve()), "sha256": sha256_file(p)}
    artifacts = {}
    for key, record in manifest["artifacts"].items():
        item = bind(output / record["path"])
        if item["sha256"] != record["sha256"]:
            raise ValueError("Accepted bytes changed: " + key)
        artifacts[key] = item
    prior = json.loads((ROOT / "experiments/pcqm_k1_500k_bn_calibration/inputs.json").read_text(encoding="utf-8"))
    cache = Path(prior["cache_root"])
    recipe = json.loads((output / "training_contract.json").read_text(encoding="utf-8"))
    if sha256_file(cache / "manifest.json") != recipe["dataset"]["manifest_sha256"]:
        raise ValueError("Wrong fixed500K cache")
    prefix = list(range(16384))
    uniform = np.random.default_rng(20261015).choice(500000, 16384, replace=False).tolist()
    available = np.setdiff1d(np.arange(500000), np.union1d(prefix, uniform))
    train = np.random.default_rng(20261016).choice(available, 4096, replace=False).tolist()
    files = [f"{REL}/{name}" for name in ("prepare.py", "run.py", "close.py", "protocol.md")]
    files += ["src/molgap/k1_session_state_diagnostic.py", "src/molgap/k1_frozen_inference.py",
              "src/molgap/k1_bn_calibration.py", "src/molgap/family_training_state.py",
              "src/molgap/family_training_data.py", "src/molgap/frozen_diagnostic_closure.py"]
    inputs = {"context": context, "source_archive": bind(source / "source.tar.gz"),
        "source_inventory": bind(source / "SOURCE_FILES.json"), "package_verification": package,
        "recipe": artifacts["contract"], "retained_predictions": artifacts["predictions"],
        "checkpoints": {"selected": {"artifact": artifacts["selected_model"], "step": 124992},
                        "final": {"artifact": artifacts["resume"], "step": 234360}},
        "cache_root": str(cache.resolve()), "dataset": dict(recipe["dataset"], manifest="manifest.json"),
        "train_decoded_source_bounds": [0, 500000], "train_source_idx": train,
        "development_source_idx": list(range(500000, 550000)),
        "calibration_source_idx": {"prefix": prefix, "uniform": uniform},
        "additional_calibrations": [["live", "live", "prefix"], ["live", "live", "uniform"], ["ema", "raw-ema", "uniform"]],
        "model_source_files": [f"src/molgap/{n}.py" for n in ("gps", "pcqm_gap_architecture", "qm9_neural_atom", "k1_screen_training")],
        "executed_source_files": {n: sha256_file(ROOT / n) for n in files},
        "cpu_threads": 4, "ceiling_seconds": 1200, "prediction_tolerance_eV": 1e-4}
    atomic_json(HERE / "inputs.json", inputs)
    roles = {"train_decoded_labels": [0, 500000], "train_descriptive_rows": train,
             "calibration_features_only": {"prefix_rows": 16384, "uniform_rows": 16384},
             "development": [500000, 550000], "previously_selection_used": True,
             "new_selection": False, "protected_roles": "untouched"}
    atomic_json(HERE / "roles.json", roles)
    policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-clean-fit-generalization-500k.1.json").read_text())
    policy.update(policy_id=POLICY, comparability_selector={"scientific_contract": POLICY + "-v1"},
                  created_from_source_digest=sha256_file(HERE / "protocol.md"))
    atomic_json(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    evidence = json.loads((HERE.parent / "rml/k1_clean_ema/rml_finalized/v5_evidence.json").read_text())
    refs = [evidence["evidence_id"]]
    cost_id = "cost-k1-session-state-attribution-expected"
    trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": TID, "record_mode": "prospective", "owner": "desktop", "track": "B",
        "family_id": "k1-clean-ema-state-attribution", "question": "Which retained weight/buffer states explain this K1 endpoint and tail behavior?",
        "hypothesis": {"hypothesis_id": "H-k1-session-state-20261015", "observed_deficiency": "Complete500K Clean EMA result is weak; same-step Live/Raw dev outputs were not retained.",
            "supporting_evidence_ids": refs, "alternative_explanations": ["Weight averaging", "BN buffer mismatch or prefix sensitivity", "Learned recipe/generalization gap not rescued by state changes"],
            "changed_mechanism": "Frozen weight-view and BN-buffer interventions only", "cheapest_falsifier": "Bounded CPU same-checkpoint state predictions and training-only recalibration",
            "decision_changed_if_positive": "Localize output-state effects before another module; no promotion", "decision_changed_if_negative": "Rule out these state interventions; retain missing training controls",
            "expected_native_cost_ref": cost_id, "related_closed_family_ids": ["k1-clean-fit-generalization", "k1-saved-bn-rows"]},
        "state_at_start": {"source_commit": commit, "source_config_identity": sha256_file(HERE / "inputs.json"),
            "contract_refs": [f"{REL}/protocol.md", f"{REL}/inputs.json", "experiments/pcqm_clean_k1_gptrans_local_500k/terminal_acceptance/attribution.md"],
            "reference_ids": [], "parent_trajectory_ids": ["TB-clean-k1-gptrans-local-500k-k1_clean_ema-s42"], "prior_evidence_ids": refs,
            "role_snapshot_refs": [f"{REL}/roles.json"], "budget_snapshot_ref": f"{REL}/protocol.md"},
        "actions": [{"action_id": "A001", "type": "authorized_local_frozen_state_attribution", "run_ids": [RUN], "attempt_ids": ["attempt-001"],
            "source_commit": commit, "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": [cost_id]}],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md", "next_allowed_actions": ["One bounded local CPU state diagnostic"], "reopen_conditions": []},
        "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{REL}/protocol.md",
        "comparison_blockers": ["Consumed selection role", "Single realization; no matched historical training reference"], "reference_bundle_id": None}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "attempt-001",
        "platform": "local-windows", "hardware": "CPU four threads; no accelerator", "category": "inference", "evidence_ref": f"{REL}/protocol.md",
        "measurement": {"wall_hours": {"value": 1 / 3, "status": "estimated"}, "cpu_hours": {"value": None, "status": "measurement_missing"},
                        "device_hours": {"value": None, "status": "not_applicable"}, "queue_hours": {"value": None, "status": "not_applicable"}}}
    spec = {"trajectory": trajectory, "costs": [cost], "decision_state": {"available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC",
        "policy_id": POLICY, "policy_version": "1", "state_timestamp": datetime.now(timezone.utc).isoformat(),
        "budget_snapshot_ref": f"{REL}/protocol.md", "role_snapshot_refs": [f"{REL}/roles.json"], "source_commit": commit}}
    atomic_json(HERE / "plan_input.json", spec)
    atomic_json(HERE / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))
    print("PROSPECTIVE_PUBLISHED_NO_INFERENCE")


if __name__ == "__main__":
    main()
