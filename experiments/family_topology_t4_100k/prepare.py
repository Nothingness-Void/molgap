"""Fresh question metadata; reuse the existing profile/family staging adapter."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

from molgap.experiment_launch import publish_immutable_bytes
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_runner import family_arm_bindings, read_recipe
from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file
from molgap.v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
BASE = ROOT / "experiments/family_100k_operator_profile"
STORAGE = ROOT / "platforms/_records/kaggle/staging/family_topology_t4_100k"
CONFIG = json.loads((HERE / "config.json").read_bytes())


def draft(initial_root):
    if (HERE / "rml").exists():
        raise ValueError("Prospective already published; reconcile rather than redraft")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    spec = json.loads((BASE / "experiment_spec.json").read_bytes())
    spec["experiment_id"] = spec["logical_run_id"] = CONFIG["run_id"]
    preparation, baseline = [], {}
    for device, arm in enumerate(spec["arms"]):
        name = arm["arm_id"]
        recipe_ref = REL + "/training_recipe_" + name + ".json"
        _, recipe = read_recipe(ROOT / recipe_ref)
        bindings = family_arm_bindings(recipe)
        for key in ("base",):
            arm[key] = bindings[key]
        for key in ("objective", "sampler", "transform"):
            arm["training"][key] = bindings[key]
        arm["training"]["recipe"]["sha256"] = normalized_source_sha256(ROOT / recipe_ref)
        module = "k1_screen_training.py" if name == "k1_mean2" else "gptrans_bond_local.py"
        arm["addons"][0]["source_sha256"] = normalized_source_sha256(ROOT / "src/molgap" / module)
        evidence = json.loads((BASE / "rml" / name / "rml_finalized/v5_evidence.json").read_bytes())
        parent = json.loads((BASE / "rml" / name / "rml_finalized/trajectory.json").read_bytes())["trajectory_id"]
        prior = evidence["evidence_id"]
        tid = "TB-family-topology-t4-100k-" + name + "-s42"
        cost_id = "cost-" + tid + "-planned"
        plan = json.loads((BASE / ("plan_" + name + ".json")).read_bytes())
        trajectory = plan["trajectory"]
        trajectory.update(trajectory_id=tid, family_id="family-topology-t4-execution",
            question="Does shared topology reuse reduce T4 time while retaining complete recipe endpoints?")
        hypothesis = trajectory["hypothesis"]
        hypothesis.update(hypothesis_id="H-" + tid, supporting_evidence_ids=[prior],
            observed_deficiency="Accepted profiler shows repeated topology operations, but no measured speedup.",
            changed_mechanism="Shared mask/index reuse and exact simple-path expansion bound; mathematical recipe unchanged",
            cheapest_falsifier="Native repeat/resume barrier followed by the same bounded operator window",
            decision_changed_if_positive="Retain matched-recipe delivery gain with cross-job uncertainty; no scientific promotion",
            decision_changed_if_negative="Record no gain or regression; do not resubmit or change thresholds",
            expected_native_cost_ref=cost_id,
            related_closed_family_ids=["family-100k-operator-profile"])
        trajectory["state_at_start"].update(source_commit=commit, source_config_identity=canonical_fingerprint(arm),
            contract_refs=[REL + "/protocol.md", REL + "/config.json", REL + "/evidence_review.md"],
            parent_trajectory_ids=[parent], prior_evidence_ids=[prior], reference_ids=[],
            role_snapshot_refs=[REL + "/role_plan.json"], budget_snapshot_ref=REL + "/config.json")
        action = trajectory["actions"][0]
        action.update(source_commit=commit, run_ids=[CONFIG["run_id"] + ":" + name + ":downstream"],
            attempt_ids=[name + "-v1"], evidence_refs=[REL + "/config.json"], cost_event_ids=[cost_id])
        trajectory["decision"]["decision_ref"] = REL + "/decision.md"
        cost = plan["costs"][0]
        cost.update(cost_event_id=cost_id, trajectory_id=tid, run_id=action["run_ids"][0],
            attempt_id=action["attempt_ids"][0], evidence_ref=REL + "/protocol.md")
        plan["decision_state"].update(known_trajectory_ids=[parent], known_evidence_ids=[prior],
            policy_id=CONFIG["run_id"], source_commit=commit, state_timestamp=datetime.now(timezone.utc).isoformat(),
            role_snapshot_refs=[REL + "/role_plan.json"], budget_snapshot_ref=REL + "/config.json")
        plan_ref = REL + "/plan_" + name + ".json"
        atomic_json(ROOT / plan_ref, plan)
        spec["prospective"]["arms"][device] = {"arm_id": name, "trajectory_id": tid,
            "plan_spec_ref": plan_ref, "plan_spec_sha256": sha256_file(ROOT / plan_ref), "output": REL + "/rml/" + name}
        initial = initial_root / (name + ".pt")
        inspect_frozen_state_artifact(initial, expected_state_sha256=recipe.initialization_sha256)
        preparation.append({"arm_id": name, "device": device, "recipe": recipe_ref, "initial_state": str(initial)})
        manifest = BASE / "terminal_acceptance/outputs/experiment" / name / "output_manifest.json"
        baseline[name] = {"evidence_id": prior, "trajectory_id": parent,
            "initial_file_sha256": sha256_file(initial), "recipe_sha256": sha256_file(ROOT / recipe_ref),
            "output_manifest_ref": manifest.relative_to(ROOT).as_posix(),
            "output_manifest_sha256": sha256_file(manifest),
            "source_commit": json.loads(manifest.read_bytes())["context"]["source_commit"]}
    validated = ExperimentSpec(spec)
    from molgap.experiment_prospective import validate_execution_identity
    for binding in validated.to_dict()["prospective"]["arms"]:
        plan = json.loads((ROOT / binding["plan_spec_ref"]).read_bytes())
        validate_execution_identity(validated, binding["arm_id"], plan["trajectory"], commit)
    atomic_json(HERE / "experiment_spec.json", validated.to_dict())
    STORAGE.mkdir(parents=True, exist_ok=True)
    atomic_json(STORAGE / "preparation_plan.json", {"format": "molgap-family-training-preparation-v1",
        "spec_identity": validated.identity, "arms": preparation,
        "source_files": [REL + "/config.json", REL + "/protocol.md", REL + "/evidence_review.md"]})
    atomic_json(HERE / "baseline_binding.json", {"kernel": CONFIG["baseline_kernel"],
        "kernel_version": CONFIG["baseline_version"], "arms": baseline, "strict_reference": False})
    print(json.dumps({"status": "DRAFT_ONLY", "spec_identity": validated.identity, "source_commit": commit}))


def stage(output, data_root):
    # Reuse the accepted staging composition; all trainer/IO/release owners are shared.
    module_spec = importlib.util.spec_from_file_location("profile_preparation", BASE / "prepare.py")
    adapter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(adapter)
    adapter.ROOT, adapter.HERE, adapter.REL = ROOT, HERE, REL
    adapter.STORAGE, adapter.CONFIG = STORAGE, CONFIG
    adapter.stage(output, data_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("draft", "stage"))
    parser.add_argument("--initial-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("D:/文档/molgap/data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "draft":
        if args.initial_root is None:
            parser.error("draft requires the frozen baseline initialization directory")
        draft(args.initial_root)
    elif args.output is None:
        parser.error("stage requires a fresh output directory")
    else:
        stage(args.output, args.data_root)
