"""Bind completed CPU qualification observations through the shared RML finalizer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from molgap.constants import REPO_ROOT
from molgap.research_memory.finalize import finalize, verified_receipt
from molgap.training_reproducibility import atomic_json, sha256_file

REL = "experiments/pcqm_k1_slot_width96"
TID = "TB-k1-slot96-cpu-qualification-20261002"
RUN = "local-k1-slot96-qualification-20261002"
EID = "pcqm-k1-slot96-cpu-qualification-20261002"
GATE_EV = 0.001
ROW_HASH = "3270b30071692d712c468fa52fa247d37f0a8e4b5444dc7c913aadb8ca0832c4"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def bound(root, item):
    path = Path(item["path"])
    path = path if path.is_absolute() else root / path
    require(sha256_file(path) == item["sha256"], f"Frozen input changed: {path}")
    return path


def close(actual, expected, name):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and abs(actual - expected) <= 1e-12,
            f"Retained aggregate differs: {name}")


def validate_inputs(root, here):
    frozen = read(here / "cpu_frozen.json")
    trajectory = read(here / "qualification/trajectory.json")
    require(trajectory["trajectory_id"] == TID and trajectory["record_mode"] == "prospective",
            "Wrong CPU prospective trajectory")
    require(sha256_file(here / "qualification/trajectory.json") == frozen["trajectory_sha256"],
            "Prospective trajectory changed after freeze")
    require(trajectory["state_at_start"]["source_commit"] == frozen["source_commit"],
            "Prospective source differs from CPU freeze")
    require(frozen["max_wall_seconds"] == 300, "CPU ceiling changed")
    require(any(a["action_id"] == "A001" and RUN in a["run_ids"]
                for a in trajectory["actions"]), "Run was not prospectively frozen")
    required = {f"{REL}/qualify.py", f"{REL}/protocol.md", f"{REL}/role_plan.json",
                "src/molgap/k1_slot_width96.py", "src/molgap/qm9_neural_atom.py",
                "src/molgap/k1_screen_training.py", "src/molgap/training_reproducibility.py",
                "src/molgap/v4_runtime.py"}
    require(required <= frozen["source_sha256"].keys(), "Required frozen sources absent")
    for name, digest in frozen["source_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and sha256_file(path) == digest,
                f"Executed source/protocol changed: {name}")
    for name in ("development_shard", "reference_checkpoint", "reference_predictions", "target_transform"):
        bound(root, frozen[name])
    q, a = read(here / "qualification_result.json"), read(here / "ablation_result.json")
    require(not (here / "cpu_failure.json").exists(), "CPU failure record requires reconciliation")
    require(q["schema"] == "molgap-k1-slot96-cpu-initialization-v1"
            and q["status"] == "CPU_INITIALIZATION_QUALIFIED", "Incomplete initialization")
    require(a["schema"] == "molgap-k1-slot96-zero-layer9-ablation-v1"
            and a["status"] == "CPU_ABLATION_COMPLETE", "Incomplete ablation")
    for report in (q, a):
        require(report["trajectory_id"] == TID and report["action_id"] == "A001",
                "CPU report identity changed")
    declarations = frozen["model_declarations"]
    require(q["candidate_parameters"] == declarations["expected_parameter_count"] == 3853793
            and q["reference_parameters"] == declarations["reference_parameter_count"] == 3658817,
            "Parameter counts differ from frozen declaration")
    require(q["reference_default_state_sha256"] ==
            "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd",
            "Accepted default initialization changed")
    for field in ("reference_default_matches_explicit_latent64", "finite", "saved_state_roundtrip",
                  "real_pure2d_forward_finite"):
        require(q[field] is True, f"Initialization check failed: {field}")
    require(q["data_access"] is True and q["label_access"] is False and q["fixture_rows"] == 64,
            "Candidate forward input/label scope differs")
    require(q["fixture_role"] == "development-input-only-no-label-metric"
            and q["gpu_qualification"] == "pending" and q["training_submission"] == "not_performed",
            "CPU initialization cannot imply GPU qualification or submission")
    require(q["runtime"]["seed"] == 42 and q["runtime"]["precision"] == "fp32"
            and q["runtime"]["tf32_enabled"] is False
            and q["runtime"]["deterministic_algorithms"] is True, "Frozen CPU arithmetic differs")
    require(sha256_file(here / "qualification_initial_state.pt") == q["initial_state_file_sha256"],
            "Retained random initialization file changed")
    for field in ("target_source_exact", "all_weights_preserved", "layer3_layer6_preserved", "finite"):
        require(a[field] is True, f"Ablation check failed: {field}")
    for field in ("training_role_read", "official_validation_used", "test_dev_used", "test_challenge_used"):
        require(a[field] is False, f"Unexpected role use: {field}")
    require(a["rows"] == 2048 and a["subset_seed"] == 20261002 and a["batch_size"] == 64
            and a["loader_workers"] == 0, "Frozen CPU cohort/loader changed")
    reference = read(here / "reference_binding/reference_reuse_decision.json")
    require(a["reference_context"] == reference["reference_source"]
            and a["selected_epoch"] == 40 and a["selected_optimizer_step"] == 31240
            and a["selected_weights"] == "live", "Selected reference identity differs")
    require(sha256_file(here / "ablation_rows.npz") == a["rows_sha256"], "Ablation rows changed")
    with np.load(here / "ablation_rows.npz", allow_pickle=False) as rows:
        required_rows = {"offsets", "source_idx", "target_eV", "original_prediction_eV",
                         "zero9_prediction_eV", "per_row_error_delta_eV", "per_row_prediction_delta_eV"}
        require(set(rows.files) == required_rows, "Unexpected ablation row fields")
        data = {name: rows[name] for name in rows.files}
    require(all(x.shape == (2048,) and np.isfinite(x).all() for x in data.values()),
            "Ablation rows have invalid shape or nonfinite values")
    offsets = np.sort(np.random.default_rng(20261002).choice(50000, 2048, replace=False))
    require(data["offsets"].dtype.kind in "iu" and data["source_idx"].dtype.kind in "iu"
            and np.array_equal(data["offsets"], offsets)
            and np.array_equal(data["source_idx"], offsets + 100000), "Subset membership/order differs")
    target = data["target_eV"]
    original, zero = data["original_prediction_eV"].astype(np.float64), data["zero9_prediction_eV"].astype(np.float64)
    delta = np.abs(zero - target) - np.abs(original - target)
    require(np.array_equal(delta, data["per_row_error_delta_eV"])
            and np.array_equal(zero - original, data["per_row_prediction_delta_eV"]),
            "Retained row deltas do not reproduce")
    close(a["original_mae_eV"], float(np.abs(original - target).mean()), "original MAE")
    close(a["zero9_mae_eV"], float(np.abs(zero - target).mean()), "zero-layer9 MAE")
    close(a["zero9_minus_original_mae_eV"], float(delta.mean()), "utility delta")
    require(a["gate_eV"] == GATE_EV and type(a["proceed_to_slot_capacity_screen"]) is bool
            and a["proceed_to_slot_capacity_screen"] == (float(delta.mean()) >= GATE_EV),
            "Utility boolean disagrees with the frozen 1 meV operational gate")
    require(0 <= a["original_replicate_max_delta_eV"] <= 1e-4,
            "Original saved prediction reconstruction failed")
    ci = a["paired_row_bootstrap"]
    require(ci["replicates"] == 1000 and ci["seed"] == 42
            and len(ci["mean_delta_ci95_eV"]) == 2
            and all(math.isfinite(x) for x in ci["mean_delta_ci95_eV"])
            and ci["mean_delta_ci95_eV"][0] <= ci["mean_delta_ci95_eV"][1], "Invalid row uncertainty")
    rng = np.random.default_rng(42)
    bootstrap = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(1000)])
    require(np.allclose(ci["mean_delta_ci95_eV"], np.quantile(bootstrap, [0.025, 0.975]),
                        rtol=0, atol=1e-12), "Retained row uncertainty does not reproduce")
    for report in (q["cost"], a["inference_cost"], a["total_cost"]):
        for field in ("wall_seconds", "process_cpu_seconds"):
            value = report[field]
            require(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0, "Invalid measured CPU cost")
        require(report["cpu_threads"] == 4
                and report["device_seconds"] == {"status": "not_applicable", "value": None}
                and report["queue_seconds"] == {"status": "not_applicable", "value": None},
                "CPU native cost semantics changed")
    require(a["total_cost"]["wall_seconds"] <= frozen["max_wall_seconds"], "CPU budget exceeded")
    for field in ("wall_seconds", "process_cpu_seconds"):
        require(q["cost"][field] + a["inference_cost"][field] <= a["total_cost"][field] + 1e-6,
                "Initialization and inference are not subsets of total measured cost")
    return frozen, q, a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    root = parser.parse_args().repo_root.resolve()
    here = root / REL
    frozen, qualification, ablation = validate_inputs(root, here)
    destination = here / "qualification/rml_finalized"
    if destination.exists():
        print(json.dumps(verified_receipt(destination)))
        return
    eligible = ablation["proceed_to_slot_capacity_screen"]
    # Dated interpretation is reviewed separately; finalization binds its bytes.
    require((here / "cpu_decision.md").is_file() and (here / "cpu_attribution.md").is_file(),
            "Reviewed CPU decision/attribution required before publication")
    outcome = dict(execution_status="complete_no_training", artifact_status="local_hash_verified",
                   comparison_status="consumed_role_frozen_final_slot_intervention", scientific_status="NO_TRAIN",
                   transfer_status="not_evaluated", budget_decision="cpu_closure_no_training_release",
                   full_handoff_status="not_applicable")
    decision = dict(outcome="NO_TRAIN", decision_ref=f"{REL}/cpu_decision.md", next_allowed_actions=[],
                    reopen_conditions=["Separate candidate training prospective and applicable release authority; CPU gate alone is not qualification or promotion."])
    role_use = dict(internal_development="selection_used", train_prefix="untouched",
                    official_validation="untouched", test_dev="untouched", test_challenge="untouched")
    acceptance_ref = f"{REL}/cpu_acceptance.json"
    cost = dict(schema="molgap-cost-event-v1", cost_event_id="cost-k1-slot96-cpu-observed-20261002",
                trajectory_id=TID, action_id="A001", run_id=RUN, attempt_id="cpu-001",
                platform="local-windows", hardware="CPU4threads; no accelerator", category="inference",
                evidence_ref=acceptance_ref, measurement={
                    "device_hours": dict(value=None, status="not_applicable"),
                    "queue_hours": dict(value=None, status="not_applicable"),
                    "wall_hours": dict(value=ablation["total_cost"]["wall_seconds"] / 3600, status="measured"),
                    "cpu_hours": dict(value=ablation["total_cost"]["process_cpu_seconds"] / 3600, status="measured")})
    roles = [dict(schema="molgap-role-event-v1", role_event_id=f"role-k1-slot96-cpu-{kind}-20261002",
                  trajectory_id=TID, action_id="A001", run_id=RUN,
                  dataset_identity="pcqm4mv2-ogb-fixed-100k-v1", row_manifest_hash=ROW_HASH,
                  role_name="internal_development", access_kind=kind, selection_used=True,
                  evidence_ref=acceptance_ref) for kind in
             ("prediction_input", "labels_read", "metric_computed", "selection_used")]
    acceptance = dict(format="molgap-k1-slot96-cpu-acceptance-v1", evidence_id=EID, run_id=RUN,
                      outcome=outcome, trajectory_decision=decision, role_use=role_use, costs=[cost], roles=roles,
                      utility=dict(delta_mae_eV=ablation["zero9_minus_original_mae_eV"], gate_eV=GATE_EV,
                                   eligible_for_separate_capacity_screen=eligible, proves_latent64_bottleneck=False),
                      checks=dict(frozen_sources_and_inputs=True, prospective_identity=True, finite_rows=True,
                                  deterministic_subset_identity=True, reconstructed_row_deltas=True,
                                  utility_boolean_correct=True, candidate_pure2d_forward=True,
                                  no_training_role_read=True, protected_roles_untouched=True),
                      role_history="Internal development was already selection-used by the retained reference and prior diagnostic.",
                      cost_scope="Single total ablation worker timer includes initialization and inference; subset timers are not additional costs. Launcher/planning/publication overhead unmeasured.",
                      qualification_result_ref=f"{REL}/qualification_result.json",
                      ablation_result_ref=f"{REL}/ablation_result.json")
    atomic_json(here / "cpu_acceptance.json", acceptance)
    names = ["cpu_frozen.json", "qualification/trajectory.json", "qualification_result.json",
             "ablation_result.json", "ablation_rows.npz", "qualification_initial_state.pt",
             "protocol.md", "role_plan.json", "evidence_review.md", "cpu_decision.md",
             "cpu_attribution.md", "cpu_acceptance.json", "finalize_cpu.py"]
    hashes = {f"{REL}/{name}": sha256_file(here / name) for name in names}
    hashes.update(frozen["source_sha256"])
    authority = [f"{REL}/{name}" for name in
                 ("protocol.md", "cpu_decision.md", "cpu_attribution.md", "cpu_acceptance.json",
                  "qualification_result.json", "ablation_result.json")]
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL",
                    evidence_id=EID, track="B", scope="desktop_k1_slot96_cpu_qualification",
                    legacy_contract="k1-slot96-cpu-qualification-v1", outcome=outcome,
                    authority=dict(pointers=authority), role_use=role_use,
                    migration=dict(migrated_at="2026-10-02", training_executed=False,
                                   inference_executed=False, scientific_reinterpretation=False,
                                   verification_scope="Metadata publication of separately executed prospective CPU qualification; no model execution in finalizer."),
                    observed_execution=dict(training_executed=False, inference_executed=True,
                                            execution_ref=f"{REL}/ablation_result.json"),
                    artifacts=[dict(name=name, locator=f"{REL}/{name}", sha256=hashes[f"{REL}/{name}"],
                                    availability="locally_retained_hash_verified") for name in
                               ("qualification_result.json", "ablation_result.json", "ablation_rows.npz",
                                "qualification_initial_state.pt", "cpu_acceptance.json", "cpu_decision.md")])
    terminal = dict(format="molgap-rml-terminal-package-v1", trajectory_id=TID, run_id=RUN,
                    action_id="A001", finalized_at=datetime.now(timezone.utc).isoformat(),
                    acceptance_ref=acceptance_ref, artifact_hashes=hashes, evidence=evidence,
                    decision=decision, costs=[cost], roles=roles)
    atomic_json(here / "cpu_terminal.json", terminal)
    print(json.dumps(finalize(root, f"{REL}/qualification", f"{REL}/cpu_terminal.json")))


if __name__ == "__main__":
    main()
