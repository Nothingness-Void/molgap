"""Bind retained inputs and use the existing RML prospective planner."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from molgap.experiment_preflight import _unpack
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-500k-bottleneck-diagnostic-20261007"
RUN = "local-k1-500k-bottleneck-20261007"
OWNER = Path("D:/w/k1-gptrans-500k-package")
CACHE = Path("D:/文档/molgap-exp/molgap-500k-v4-evidence/data/cache/pcqm4mv2_500k_v4")


def main():
    if (HERE / "rml").exists():
        raise FileExistsError("Prospective already published")
    retained = OWNER / "platforms/_records/kaggle/training/pcqm_k1_gptrans_package_transfer_500k_s42_final/k1_pretrained_consistency"
    archive = OWNER / "platforms/_records/kaggle/staging/pcqm_k1_gptrans_package_transfer_500k/kaggle-v5-k1-resume-v4/package/source.tar.gz"
    expected_archive = "c294a785f9d0b702f63152e4448511875f7124a830f1ad088c9acdd64203814a"
    if sha256_file(archive) != expected_archive:
        raise ValueError("Accepted executable archive differs")
    frozen = ROOT / "platforms/_records/local/diagnostics/k1_500k_bottleneck/source"
    frozen.mkdir(parents=True, exist_ok=False)
    _unpack(archive.parent, frozen)
    pins = {
        "best_model.pt": "2e9cee13ebbf67031470e89acaf46d551238f937255fed601eb0922b93fdad92",
        "last_checkpoint.pt": "29afd0d3fadc77b939bda61f60c21512eebef9606893b5c6c92b0dea982f15c2",
        "best_predictions.pt": "57463e4e578db6310eadfd2b46d9c57f50d0ef7e2aa12c3f883ace72e7265f55",
    }
    for name, digest in pins.items():
        if sha256_file(retained / name) != digest:
            raise ValueError(f"Accepted input differs: {name}")
    manifest_path = CACHE / "manifest.json"
    manifest_sha = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
    if sha256_file(manifest_path) != manifest_sha:
        raise ValueError("Accepted cache manifest differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for shard in manifest["geometry_shards"]:
        if sha256_file(CACHE / shard["file"]) != shard["sha256"]:
            raise ValueError("Accepted graph shard differs")
    authority = HERE / "input_authority"
    authority.mkdir()
    owner_exp = OWNER / "experiments/pcqm_k1_gptrans_package_transfer_500k"
    for origin, name in [
        (owner_exp / "terminal_acceptance/k1_pretrained_consistency/acceptance.json", "acceptance.snapshot.json"),
        (owner_exp / "terminal_acceptance/k1_pretrained_consistency/artifact_inventory.json", "artifact_inventory.snapshot.json"),
        (owner_exp / "terminal_acceptance/cost_accuracy_attribution.md", "cost_accuracy_attribution.md"),
    ]:
        shutil.copyfile(origin, authority / name)
    inputs = {
        "owner_commit": "031a890b4d604fc4614cde2da10e6e76cf2f836e",
        "checkpoints": {name: {"path": (retained / name).as_posix(), "sha256": digest} for name, digest in pins.items()},
        "archive": {"path": archive.as_posix(), "sha256": expected_archive},
        "frozen_source_root": (frozen / "src").as_posix(),
        "frozen_source_files": {p.relative_to(frozen).as_posix(): sha256_file(p) for p in frozen.rglob("*.py")},
        "cache_root": CACHE.as_posix(), "manifest_sha256": manifest_sha,
        "sample_seed": 20261007, "train_prefix_rows": 1024, "train_extension_rows": 1024,
        "development_rows": 2048, "batch_size": 64, "cpu_threads": 4,
        "worker_wall_ceiling_seconds": 600, "prediction_tolerance_eV": 1e-4,
        "interventions": {"last_slot_off": [9], "all_slots_off": [3, 6, 9]},
        "authorization": "User requests verification of K1 dimension/exposure hypotheses; bounded local clean inference and slot interventions only.",
        "executed_source_files": {name: sha256_file(ROOT / name) for name in [
            f"{REL}/prepare.py", f"{REL}/run.py", "src/molgap/k1_frozen_inference.py"]},
    }
    atomic_json(HERE / "inputs.json", inputs)
    roles = {"train_prefix": {"range": [0, 500000], "sample_rows": 2048},
             "internal_development": {"range": [500000, 550000], "sample_rows": 2048,
                 "previously_selection_used": True},
             "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    atomic_json(HERE / "roles.json", roles)
    timestamp = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    references = ["pcqm-k1-slot-readout-diagnostic-20261002", "pcqm-matched-500k-v4-three-arm"]
    state = {"source_commit": commit, "source_config_identity": sha256_file(HERE / "inputs.json"),
             "contract_refs": [f"{REL}/protocol.md", f"{REL}/inputs.json"],
             "reference_ids": [], "parent_trajectory_ids": [], "prior_evidence_ids": references,
             "role_snapshot_refs": [f"{REL}/roles.json"], "budget_snapshot_ref": f"{REL}/protocol.md"}
    trajectory = {
        "schema": "molgap-trajectory-v1", "trajectory_id": TID, "record_mode": "prospective",
        "owner": "desktop", "track": "B", "family_id": "k1-500k-frozen-bottleneck-diagnostic",
        "question": "Does the accepted 500K K1 tail improve clean training fit, and does its selected predictor depend on global slot returns?",
        "hypothesis": {"hypothesis_id": "H-k1-500k-bottleneck-20261007",
            "observed_deficiency": "The composed 500K K1 endpoint is close to older plain K1 while mandatory double passes increase cost.",
            "supporting_evidence_ids": references,
            "alternative_explanations": ["Useful global signal with adequate capacity", "Finite-budget or learning-rate effects", "Regularization or optimization advantage attenuates at scale"],
            "changed_mechanism": "Frozen inference only: zero layer9 or layers3/6/9 mixer returns; compare retained best/final states in identical eval mode.",
            "cheapest_falsifier": "4096 fixed clean rows per retained state plus two2048-row selected-development slot interventions; fail on reconstruction or600s ceiling.",
            "decision_changed_if_positive": "Identify frozen slot utility or a late fit/generalization tradeoff as a bounded observation; separately design any training discriminator.",
            "decision_changed_if_negative": "Do not prioritize an inactive-slot explanation or extending the current tail without further evidence.",
            "expected_native_cost_ref": "cost-k1-500k-bottleneck-expected",
            "related_closed_family_ids": ["k1-slot-readout-diagnostic", "k1-slot-width96"],
            "historical_unknowns": ["Training seed variance", "Capacity saturation", "Effect of expanded pretraining budget"]},
        "state_at_start": state,
        "actions": [{"action_id": "A001", "type": "local_frozen_checkpoint_intervention",
            "run_ids": [RUN], "attempt_ids": ["attempt-001"], "source_commit": commit,
            "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": ["cost-k1-500k-bottleneck-expected"]}],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md",
            "next_allowed_actions": ["Bounded local diagnostic only"], "reopen_conditions": []},
        "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{REL}/protocol.md",
        "comparison_blockers": ["Selected, previously consumed internal development; single trained seed", "Inference deletion does not test added training capacity"],
        "reference_bundle_id": None,
    }
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-k1-500k-bottleneck-expected",
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "attempt-001",
        "platform": "local-windows", "hardware": "CPU; four intra-op threads", "category": "inference",
        "evidence_ref": f"{REL}/protocol.md", "measurement": {
            "wall_hours": {"value": 600 / 3600, "status": "estimated"},
            "cpu_hours": {"value": None, "status": "measurement_missing"},
            "device_hours": {"value": None, "status": "not_applicable"},
            "queue_hours": {"value": None, "status": "not_applicable"}}}
    spec = {"trajectory": trajectory, "costs": [cost], "decision_state": {
        "known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
        "available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC",
        "policy_id": "pcqm-k1-500k-bottleneck-diagnostic", "policy_version": "1",
        "budget_snapshot_ref": f"{REL}/protocol.md", "role_snapshot_refs": [f"{REL}/roles.json"],
        "source_commit": commit, "state_timestamp": timestamp}}
    atomic_json(HERE / "plan_input.json", spec)
    receipt = plan(ROOT, spec, f"{REL}/rml")
    atomic_json(HERE / "plan_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
