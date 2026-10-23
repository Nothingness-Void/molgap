"""Bind strictly accepted selected states and publish the existing RML plan."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess

from molgap.constants import REPO_ROOT
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-verification", required=True)
    parser.add_argument("--output-dir", choices=("calibration", "calibration_local_runtime"), default="calibration")
    parser.add_argument("--local-inference-runtime", action="store_true")
    args = parser.parse_args()
    root = Path(REPO_ROOT).resolve()
    question = root / "experiments/pcqm_k1_batch_renorm_100k"
    here = question / args.output_dir
    if args.local_inference_runtime != (args.output_dir == "calibration_local_runtime"):
        raise ValueError("Local runtime qualification requires its separate prospective directory")
    rel = here.relative_to(root).as_posix()
    if (here / "rml").exists() or (here / "inputs.json").exists():
        raise FileExistsError("Prospective inputs already exist; reconcile rather than overwrite")
    verification_path = Path(args.strict_verification).resolve()
    verified = json.loads(verification_path.read_text(encoding="utf-8"))
    if (verified["status"] != "COMPLETE" or verified["executed"] is not True or
            verified["strict_validation"]["status"] != "VERIFIED"):
        raise ValueError("Training strict acceptance is not VERIFIED")

    def binding(path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": sha256_file(path)}

    import torch
    retained_root = Path("D:/molgap-artifacts/reconciliation/k1-batch-renorm-100k/outputs")
    arms, evidence_ids, parent_ids = {}, [], []
    related_families = []
    for arm in ("brn_scale", "brn_affine"):
        strict_arm = verified["strict_validation"]["arms"][arm]
        if strict_arm["native_qualification"]["status"] != "VERIFIED" or strict_arm["replay"]["status"] != "VERIFIED":
            raise ValueError("Training arm native qualification/replay not verified")
        output = retained_root / arm
        selected = torch.load(output / "selected_model.pt", map_location="cpu", weights_only=True)
        canonical = question / f"runs/pcqm-k1-batch-renorm-100k-s42/prospective/{arm}/rml_finalized"
        evidence = json.loads((canonical / "v5_evidence.json").read_text(encoding="utf-8"))
        terminal = json.loads((canonical / "trajectory.json").read_text(encoding="utf-8"))
        if terminal["decision"]["outcome"] == "ACTIVE" or selected["weights"] != "live":
            raise ValueError("Training arm has no terminal selected LIVE custody")
        evidence_ids.append(evidence["evidence_id"])
        parent_ids.append(terminal["trajectory_id"])
        related_families.append(terminal["family_id"])
        arms[arm] = {"recipe": binding(output / "training_contract.json"),
            "selected": binding(output / "selected_model.pt"),
            "predictions": binding(output / "development_predictions.pt"),
            "context": selected["context"],
            "accepted_evidence": binding(canonical / "v5_evidence.json"),
            "accepted_trajectory": binding(canonical / "trajectory.json")}
    # Pin the existing tracked runtime source inventory; no graph/data traversal.
    tracked = subprocess.check_output(["git", "ls-files", "src/molgap"], cwd=root, text=True).splitlines()
    source_names = [name for name in tracked if name.endswith(".py")]
    source_names += [(question / name).relative_to(root).as_posix()
                     for name in ("diagnose_clean_bn.py", "prepare_clean_bn.py")]
    sources = {name: sha256_file(root / name) for name in sorted(set(source_names))}
    inputs = {"format": "molgap-brn-clean-bn-input-v1", "ceiling_seconds": 1800,
        "prediction_tolerance_eV": 1e-4, "device": "cuda:0", "cpu_threads": 4,
        "trajectory_ref": f"{rel}/rml/trajectory.json", "source_sha256": sources,
        "role_authority": json.loads((here / "roles.json").read_text(encoding="utf-8")),
        "arms": arms, "strict_verification": binding(verification_path),
        "initialization": binding("D:/molgap-artifacts/initializations/structural-bias-norm-100k/k1.pt")}
    if args.local_inference_runtime:
        prior_root = question / "calibration/rml/rml_finalized"
        prior = json.loads((prior_root / "trajectory.json").read_text(encoding="utf-8"))
        prior_evidence = json.loads((prior_root / "v5_evidence.json").read_text(encoding="utf-8"))
        prior_result_path = question / "calibration/results/result.json"
        prior_result = json.loads(prior_result_path.read_text(encoding="utf-8"))
        if (prior["decision"]["outcome"] != "NO_TRAIN" or prior_result["status"] != "failed" or
                prior_result["error"] != "Local Torch differs from frozen recipe pin" or prior_result["arms"]):
            raise ValueError("Attempt001 runtime failure must be finalized before revised qualification")
        runtime = prior_result["runtime"]
        if runtime != {"device": "cuda:0", "torch": "2.7.1+cu128", "cpu_threads": 4,
                       "hardware": "NVIDIA GeForce RTX 5060"} or str(torch.__version__) != runtime["torch"]:
            raise ValueError("Explicitly scoped observed local runtime differs")
        inputs.update(ceiling_seconds=1780, local_inference_runtime=runtime,
            prior_attempt={"trajectory": binding(prior_root / "trajectory.json"),
                           "evidence": binding(prior_root / "v5_evidence.json"), "result": binding(prior_result_path)})
        evidence_ids.append(prior_evidence["evidence_id"])
        parent_ids.append(prior["trajectory_id"])
        related_families.append(prior["family_id"])
    atomic_json(here / "inputs.json", inputs)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tid, hid = "TB-k1-brn-clean-bn-100k-20261024", "H-k1-brn-clean-bn-100k-20261024"
    run, cost_id, policy = "local-k1-brn-clean-bn-100k-20261024", "cost-k1-brn-clean-bn-100k-ceiling", "pcqm-k1-brn-clean-bn-100k"
    attempt = "attempt-001"
    if args.local_inference_runtime:
        tid += "-local-runtime-attempt002"
        hid += "-local-runtime-attempt002"
        run += "-local-runtime-attempt002"
        cost_id += "-attempt002"
        attempt = "attempt-002"
    state = {"source_commit": commit, "source_config_identity": sha256_file(here / "inputs.json"),
        "contract_refs": [f"{rel}/protocol.md", f"{rel}/inputs.json", "experiments/pcqm_k1_batch_renorm_100k/protocol.md"],
        "reference_ids": [], "parent_trajectory_ids": parent_ids, "prior_evidence_ids": evidence_ids,
        "role_snapshot_refs": [f"{rel}/roles.json"], "budget_snapshot_ref": f"{rel}/protocol.md"}
    trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": tid, "record_mode": "prospective",
        "owner": "desktop", "track": "B", "family_id": "k1-brn-clean-bn-100k",
        "question": "Does a common cumulative Clean-BN pass alter the accepted selected-LIVE affine-minus-scale contrast?",
        "hypothesis": {"hypothesis_id": hid,
            "observed_deficiency": "The accepted endpoint contrast does not distinguish BN buffer mediation from learned-function differences.",
            "supporting_evidence_ids": evidence_ids,
            "alternative_explanations": ["BN running-buffer mediation", "Learned-function differences", "Selected-development and one-seed uncertainty"],
            "changed_mechanism": "Reset and cumulatively recompute only BN running buffers on the same16384 ordered training members, dropout off, selected LIVE weights fixed.",
            "cheapest_falsifier": f"Original predictions must reproduce within1e-4eV under the bound runtime; then128 calibration batches/arm and50K development inference within{inputs['ceiling_seconds']}s total.",
            "decision_changed_if_positive": "A disappearing contrast supports buffer mediation as a compatible explanation; close NO_TRAIN without promotion.",
            "decision_changed_if_negative": "A persistent contrast supports learned-function differences as compatible, without identifying a unique cause; close NO_TRAIN.",
            "expected_native_cost_ref": cost_id, "related_closed_family_ids": sorted(set(related_families)),
            "historical_unknowns": ["Training stochasticity", "Independent-role transfer", "Unique causal attribution"]},
        "state_at_start": state,
        "actions": [{"action_id": "A001", "type": "local_bn_buffer_calibration", "run_ids": [run],
            "attempt_ids": [attempt], "source_commit": commit, "evidence_refs": [f"{rel}/inputs.json"], "cost_event_ids": [cost_id]}],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {"outcome": "ACTIVE", "decision_ref": f"{rel}/decision.md",
            "next_allowed_actions": ["Bounded local selected-LIVE Clean-BN diagnostic"], "reopen_conditions": []},
        "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{rel}/protocol.md",
        "comparison_blockers": ["Previously selection-used development cohort", "Single training seed; no independent transfer"],
        "reference_bundle_id": None}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": tid,
        "action_id": "A001", "run_id": run, "attempt_id": attempt, "platform": "local-windows",
        "hardware": "Existing desktop GPU cuda:0; CPU4 threads", "category": "inference", "evidence_ref": f"{rel}/protocol.md",
        "measurement": {"wall_hours": {"value": inputs["ceiling_seconds"] / 3600, "status": "estimated"},
            "device_hours": {"value": inputs["ceiling_seconds"] / 3600, "status": "estimated"}, "cpu_hours": {"value": None, "status": "measurement_missing"},
            "queue_hours": {"value": None, "status": "not_applicable"}}}
    spec = {"trajectory": trajectory, "costs": [cost], "decision_state": {
        "known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
        "available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC",
        "policy_id": policy, "policy_version": "1", "budget_snapshot_ref": f"{rel}/protocol.md",
        "role_snapshot_refs": [f"{rel}/roles.json"], "source_commit": commit,
        "state_timestamp": datetime.now(timezone.utc).isoformat()}}
    atomic_json(here / "plan_input.json", spec)
    receipt = plan(root, spec, f"{rel}/rml")
    atomic_json(here / "plan_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
