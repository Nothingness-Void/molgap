"""Build local single-arm inputs with existing owners; never execute training."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from molgap.common.declarations import ExecutionProfile, ExecutionQuestion, build_execution_proposal
from molgap.common.paths import MachinePaths
from molgap.experiment_execution import build_addon_declaration
from molgap.experiment_source_inventory import registered_source_files, runtime_source_files
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_runner import build_screen_recipe, family_arm_bindings, read_recipe
from molgap.research_memory.trace import atomic_write, json_bytes
from molgap.v4_runtime import inspect_frozen_state_artifact


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    atomic_write(path, json_bytes(value))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", choices=("bn_reference", "ffn_rms", "local_residual_gain"))
    args = parser.parse_args()
    config = read(HERE / "input.json")
    paths = MachinePaths.from_file(config["machine_profile"]).outside_checkout(ROOT)
    if args.prepare:
        from molgap.experiment_workflow import prepare_family_training
        directory = HERE / args.prepare
        spec = ExperimentSpec(read(directory / "experiment_spec.json"))
        plan = read(directory / "workflow_plan.json")
        output = paths.artifact_root / "runs" / spec.to_dict()["logical_run_id"] / "v1" / "prepared"
        report = prepare_family_training(spec, ROOT, plan, output)
        print(json.dumps({"prepared": str(output), "report": report}))
        return
    record, baseline = read_recipe(ROOT / config["baseline_recipe_ref"])
    original = read(ROOT / config["baseline_spec_ref"])
    source = next(a for a in original["arms"] if a["arm_id"] == config["baseline_arm_id"])
    assert source["addons"] == [] and baseline.mode == "reference"
    assert (baseline.train_rows, baseline.development_rows, baseline.optimizer_steps,
            baseline.batch_size, baseline.seed, baseline.schedule.total_units) == (100000, 50000, 31240, 128, 42, 40)
    manifest_path = paths.data_root / baseline.dataset.manifest
    assert digest(manifest_path) == baseline.dataset.manifest_sha256
    manifest = read(manifest_path)
    assert manifest["roles"]["train"]["source_idx_stop"] == 100000
    assert manifest["roles"]["development"]["source_idx_stop"] == 150000
    inspect_frozen_state_artifact(Path(config["initial_state_ref"]), expected_state_sha256=baseline.initialization_sha256)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    timestamp = datetime.now(timezone.utc).isoformat()
    prefix = HERE.relative_to(ROOT).as_posix()
    policy = read(HERE / "policy_registration.json")
    policy["created_from_source_digest"] = digest(HERE / "protocol.md")
    from molgap.research_memory.policy import validate_policy
    validate_policy(policy)
    atomic_json(HERE / "policy_registration.json", policy)
    roles = {"allowed_roles": ["train", "development"], "role_bindings": source["data"]["roles"],
             "official_validation": False, "test_dev": False, "test_challenge": False,
             "authority_ref": prefix + "/authority.json", "observed_consumption": "not yet observed"}
    atomic_json(HERE / "role_scope.json", roles)
    context = {"status": "PENDING", "reference_arm_id": "bn_reference",
               "candidate_arm_ids": ["ffn_rms", "local_residual_gain"],
               "strict_reference_bundle": None, "runtime_changed_from_historical": "2.11.0 -> 2.7.1",
               "local_runtime": {"torch": "2.7.1+cu128", "pyg": "2.7.0"},
               "manifest_sha256": baseline.dataset.manifest_sha256,
               "initialization_state_sha256": baseline.initialization_sha256,
               "historical_recipe_ref": config["baseline_recipe_ref"], "arms": []}
    for item in config["arms"]:
        arm_id = item["arm_id"]
        addons = (() if item["addon"] is None else (build_addon_declaration(
            ("neural_atom_k1", "3"), item["addon"], repo_root=ROOT, config={}),))
        recipe = replace(baseline, mode="composed" if addons else "reference",
                         addons=addons, torch_version=config["torch_version"])
        built = build_screen_recipe(recipe.mode, configuration=recipe, family=("neural_atom_k1", "3"),
            source_idx_sha256=record["acceptance_requirements"]["source_idx_sha256"],
            target_sha256=record["acceptance_requirements"]["target_sha256"],
            train_role_identity=record["train_role_identity"], development_role_identity=record["development_role_identity"],
            logging=record["logging"])
        directory = HERE / arm_id
        base = prefix + "/" + arm_id
        atomic_json(directory / "recipe.json", built)
        arm = copy.deepcopy(source)
        bindings = family_arm_bindings(recipe)
        arm.update(arm_id=arm_id, addons=list(addons), addon_semantics="ordered" if addons else "baseline",
                   scientific_role="candidate" if addons else "reference", base=bindings["base"])
        arm["training"].update({k: bindings[k] for k in ("sampler", "transform", "objective")})
        arm["training"]["recipe"]["sha256"] = digest(directory / "recipe.json")
        run = "pcqm-k1-selective-100k-" + arm_id.replace("_", "-") + "-s42"
        question = ExecutionQuestion(policy["policy_id"], policy["action_rule"]["action"],
            "Does " + arm_id + " improve the matched local original-BN reference at100K40passes?",
            "Whole-block norm replacements did not establish selective FFN or local residual effects.",
            item["addon"] or "Matched original BN, no MsgNorm; new local reference.",
            ("ev-terminal-T-pcqm-k1-rms-position-160ep-100k-s42-20261022-post_node_rms",
             "ev-terminal-T-pcqm-k1-rms-position-160ep-100k-s42-20261022-pre_node_rms"),
            ("No generalization gain", "Changed optimization rather than improved capacity"),
            ("Pre-Norm AdaRMS100K negative", "Whole-block Post-LN100K negative"),
            "Review matched local evidence; no automatic adoption or scale-up.", "Reject this exact mechanism under this contract.")
        profile = ExecutionProfile("desktop", "B", "local", "NVIDIA GeForce RTX 5060 8GiB",
            "Three sequential single-arm100K40pass runs; each candidate compared with the new local BN arm.", None, None)
        tid = "TB-" + run
        proposal = build_execution_proposal(question=question, arm=arm, trajectory_id=tid,
            source_commit=commit, run=run, version="v1", contract_ref=base + "/recipe.json",
            contract_refs=[base + "/recipe.json", prefix + "/protocol.md", prefix + "/input.json"],
            authority_ref=prefix + "/authority.json", role_ref=prefix + "/role_scope.json",
            decision_ref=prefix + "/protocol.md", timestamp=timestamp, profile=profile, action_type="training")
        proposal["decision_state"]["policy_version"] = policy["version"]
        atomic_json(directory / "plan_input.json", proposal)
        spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2",
            "experiment_id": config["experiment_id"], "logical_run_id": run, "arms": [arm],
            "platform": {"name": "local", "accelerator": "NVIDIA GeForce RTX 5060", "device_count": 1,
                         "cpu_cores": 4, "memory_gib": 32, "atomic_checkpoints": True, "retrievable_chunks": True},
            "prospective": {"arms": [{"arm_id": arm_id, "trajectory_id": tid,
                "plan_spec_ref": base + "/plan_input.json", "plan_spec_sha256": digest(directory / "plan_input.json"),
                "output": base + "/prospective"}]},
            "evidence": original["evidence"], "terminal_protocol": original["terminal_protocol"]})
        atomic_json(directory / "experiment_spec.json", spec.to_dict())
        records = [prefix + "/input.json", prefix + "/authority.json", prefix + "/protocol.md",
                   prefix + "/role_scope.json", prefix + "/policy_registration.json", base + "/recipe.json"]
        plan = {"format": "molgap-family-training-preparation-v1", "spec_identity": spec.identity,
                "arms": [{"arm_id": arm_id, "device": 0, "recipe": base + "/recipe.json",
                          "initial_state": config["initial_state_ref"]}],
                "source_files": registered_source_files(spec, sorted(set(runtime_source_files(ROOT)) | set(records)))}
        atomic_json(directory / "workflow_plan.json", plan)
        context["arms"].append({"arm_id": arm_id, "logical_run_id": run, "spec_identity": spec.identity,
                                 "spec_ref": base + "/experiment_spec.json"})
        print(json.dumps({"arm": arm_id, "spec_identity": spec.identity, "status": "INPUT_DECLARED_NOT_PREPARED"}))
    atomic_json(HERE / "comparison_context.json", context)


if __name__ == "__main__":
    main()
