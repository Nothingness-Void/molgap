"""Question configuration over existing family preparation and Kaggle staging."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from molgap.experiment_execution import build_family_recipe
from molgap.experiment_family_workflow import tensor_digest
from molgap.experiment_launch import canonical_json, publish_immutable_bytes
from molgap.experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL
from molgap.family_training_data import DatasetSelection, load_training_data
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_runner import INTERFACES, family_arm_bindings
from molgap.training_reproducibility import atomic_json, atomic_torch_save, canonical_fingerprint, sha256_file
from molgap.v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256, state_dict_sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
CONFIG = json.loads((HERE / "config.json").read_text())
STORAGE = ROOT / "platforms/_records/kaggle/staging/family_100k_operator_profile"


def inputs(data_root, k1_initial):
    """CPU admission of authorized fixed views and frozen random initial tensors."""
    import torch
    from molgap.gptrans_bond_local import make_bond_local_model
    torch.set_num_threads(1)
    STORAGE.mkdir(parents=True, exist_ok=True)
    data = load_training_data(data_root, DatasetSelection("pcqm_fixed_100k_v1/manifest.json",
        CONFIG["dataset_manifest_sha256"], True))
    payloads = {}
    for arm_id, arm in CONFIG["arms"].items():
        destination = STORAGE / (arm_id + ".pt")
        if arm_id == "k1_mean2":
            inspect_frozen_state_artifact(k1_initial, expected_state_sha256=arm["initialization_sha256"])
            if destination.exists() and sha256_file(destination) != sha256_file(k1_initial):
                raise ValueError("Staged K1 initialization changed")
            if not destination.exists():
                shutil.copyfile(k1_initial, destination)
        else:
            model = make_bond_local_model()
            if state_dict_sha256(model.state_dict()) != arm["initialization_sha256"]:
                raise ValueError("Retained local-bond initialization differs")
            if not destination.exists():
                atomic_torch_save(destination, model.state_dict())
            del model
        inspect_frozen_state_artifact(destination, expected_state_sha256=arm["initialization_sha256"])
        payloads[arm_id] = {"file": destination.relative_to(ROOT).as_posix(), "sha256": sha256_file(destination),
            "state_sha256": arm["initialization_sha256"]}
    target = torch.cat([part._data.y.view(-1).float() for part in data.development.datasets])
    source = torch.arange(CONFIG["train_rows"], CONFIG["train_rows"] + CONFIG["development_rows"])
    identity = {"manifest_sha256": data.manifest_sha256, "train_rows": data.train_rows,
        "development_rows": data.development_rows, "source_idx_sha256": tensor_digest(source, role="source_idx"),
        "target_sha256": tensor_digest(target, role="target"), "initializations": payloads,
        "train_mean_double_observation": data.target_mean_eV, "train_std_double_observation": data.target_std_eV,
        "normalization_authority": "config retains accepted K1 asset and retained GPTrans double-reduction asset"}
    atomic_json(HERE / "input_identity.json", identity)
    roles = []
    for kind, indices in (("train", torch.arange(data.train_rows)), ("development", source)):
        digest = tensor_digest(indices, role="source_idx")
        roles.append({"role":kind, "membership_sha256":digest, "row_order_sha256":digest,
            "usage_sha256":canonical_fingerprint({"role":kind, "membership":digest, "contract":CONFIG["run_id"]})})
    for arm_id, arm in CONFIG["arms"].items():
        recipe = FamilyRecipe(arm["family"], arm["mode"], CONFIG["seed"], arm["initialization_sha256"],
            DatasetSelection("pcqm4mv2-ogb-fixed-100k-v1/manifest.json", data.manifest_sha256, True),
            data.train_rows, data.development_rows, data.train_rows // 128 * arm["epochs"],
            arm["learning_rate"], arm["weight_decay"], ScheduleSpec(arm["schedule"], arm["epochs"],
                warmup_units=arm["warmup_units"]), arm["selection"], arm["sampler"], 128,
            arm["target_mean_eV"], arm["target_std_eV"], ema_decay=arm["ema_decay"], torch_version=CONFIG["torch_version"])
        record = build_family_recipe(INTERFACES[recipe.family], addon=arm["addon"], configuration=recipe,
            source_idx_sha256=identity["source_idx_sha256"], target_sha256=identity["target_sha256"],
            train_role_identity=roles[0]["usage_sha256"], development_role_identity=roles[1]["usage_sha256"])
        record["profiling"] = CONFIG["profiling"] | {"module_paths":
            ["local_blocks.0", "local_blocks.8"] if recipe.family == "neural_atom_k1" else ["blocks.0", "blocks.11"]}
        atomic_json(HERE / ("training_recipe_" + arm_id + ".json"), record)
    atomic_json(HERE / "role_plan.json", {"roles":roles, "official_validation": "forbidden", "test_dev":"forbidden", "test_challenge":"forbidden"})
    policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-colab-execution-profile.1.json").read_text())
    policy.update(policy_id=CONFIG["run_id"], created_from_source_digest=sha256_file(HERE / "protocol.md"),
        comparability_selector={"scientific_contract":CONFIG["run_id"]})
    policy["action_rule"]["action"] = "RUN_PROFILE_TRAINING"
    atomic_json(ROOT / "research_memory/policies" / (CONFIG["run_id"] + ".1.json"), policy)
    print(json.dumps({"status":"INPUTS_PREPARED", "rows":data.train_rows, "initializations":payloads}))


def draft():
    from molgap.family_training_runner import read_recipe
    if (HERE / "rml").exists():
        raise ValueError("Prospective already published; reconcile the existing stage")
    identity = json.loads((HERE / "input_identity.json").read_text())
    roles = json.loads((HERE / "role_plan.json").read_text())["roles"]
    commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    ref = lambda name, digest: {"name":name, "version":"1", "sha256":digest}
    arms, bindings, preparation = [], [], []
    prior = {"k1_mean2":"pcqm-k1-clean-second-100k-s42-v1-mean2-terminal", "gptrans_local":"pcqm-gptrans-t-100k-v4-reference"}
    for device, (arm_id, facts) in enumerate(CONFIG["arms"].items()):
        recipe_ref = REL + "/training_recipe_" + arm_id + ".json"
        _, recipe = read_recipe(ROOT / recipe_ref)
        b = family_arm_bindings(recipe)
        module = "k1_screen_training.py" if device == 0 else "gptrans_bond_local.py"
        arm = {"arm_id":arm_id, "scientific_role":"candidate", "family":{"name":recipe.family,"version":INTERFACES[recipe.family][1]},
            "base":b["base"], "initialization":{"kind":"frozen_state","seed":recipe.seed,"state_sha256":recipe.initialization_sha256},
            "data":{"dataset":ref("pcqm4mv2",identity["manifest_sha256"]), "split":ref("fixed100k-next50k",canonical_fingerprint(roles)),
                "roles":deepcopy(roles), "feature_schema":"ogb-atom9-bond3-rwse16-v1", "feature_sha256":identity["manifest_sha256"],
                "target":"pcqm4mv2-gap-eV-direct"},
            "training":{"recipe":ref("shared-family-training",normalized_source_sha256(ROOT / recipe_ref)),
                "overrides":{}, **{k:b[k] for k in ("objective","sampler","transform")}},
            "addons":[{"name":facts["addon"],"version":"1","config":{},
                "source_sha256":normalized_source_sha256(ROOT / "src/molgap" / module)}], "addon_semantics":"ordered"}
        arms.append(arm)
        tid = "TB-family-100k-operator-profile-" + arm_id + "-s42"
        cost_id = "cost-" + tid + "-planned"
        action = {"action_id":"A001", "type":"paired_100k_training", "source_commit":commit,
            "run_ids":[CONFIG["run_id"] + ":" + arm_id + ":downstream"], "attempt_ids":[CONFIG["run_id"] + "-v1"],
            "evidence_refs":[REL + "/config.json"], "cost_event_ids":[cost_id]}
        trajectory = {"schema":"molgap-trajectory-v1", "trajectory_id":tid,"record_mode":"prospective","track":"B","owner":"desktop",
            "family_id":"family-100k-operator-profile", "question":"Where does the selected shared family execution spend native T4 time?",
            "hypothesis":{"hypothesis_id":"H-"+tid, "supporting_evidence_ids":[prior[arm_id]],
                "observed_deficiency":"Synthetic short-pass timings do not identify molecular operator bottlenecks.",
                "alternative_explanations":["Concurrent CPU contention", "Instrumentation overhead", "Checkpoint and asynchronous dispatch waits"],
                "changed_mechanism":"Observation only: shared phase/operator profiling with existing selected family recipe",
                "cheapest_falsifier":"A short native capture during the user-authorized complete100K run",
                "decision_changed_if_positive":"Target the measured largest operator in a separately controlled equivalence/speed test",
                "decision_changed_if_negative":"Preserve missing/partial attribution; no speculative optimization or retry",
                "expected_native_cost_ref":cost_id,"related_closed_family_ids":["family-session-speed"]},
            "state_at_start":{"source_commit":commit,"source_config_identity":canonical_fingerprint(arm),
                "contract_refs":[REL+"/protocol.md",REL+"/config.json",REL+"/evidence_review.md"],
                "reference_ids":[],"parent_trajectory_ids":[],"prior_evidence_ids":[prior[arm_id]],
                "role_snapshot_refs":[REL+"/role_plan.json"],"budget_snapshot_ref":REL+"/config.json"},
            "actions":[action],"result":{"evidence_ids":[],"evidence_refs":[]},
            "decision":{"outcome":"ACTIVE","decision_ref":REL+"/decision.md","next_allowed_actions":["A001"],"reopen_conditions":[]}}
        cost = {"schema":"molgap-cost-event-v1","cost_event_id":cost_id,"trajectory_id":tid,"action_id":"A001",
            "run_id":action["run_ids"][0],"attempt_id":action["attempt_ids"][0],"platform":"kaggle","hardware":"Tesla T4",
            "category":"training","evidence_ref":REL+"/protocol.md",
            "measurement":{k:{"value":7.0 if k in ("wall_hours","device_hours") else None,
                "status":"estimated" if k in ("wall_hours","device_hours") else "measurement_missing"}
                for k in ("wall_hours","device_hours","cpu_hours","queue_hours")}}
        plan = {"trajectory":trajectory,"costs":[cost],"decision_state":{"known_trajectory_ids":[],"known_evidence_ids":[prior[arm_id]],
            "active_reference_ids":[],"available_actions":["RUN_PROFILE_TRAINING","NO_TRAIN"],"chosen_action":"RUN_PROFILE_TRAINING",
            "policy_id":CONFIG["run_id"],"policy_version":"1","budget_snapshot_ref":REL+"/config.json",
            "role_snapshot_refs":[REL+"/role_plan.json"],"source_commit":commit,"state_timestamp":datetime.now(timezone.utc).isoformat()}}
        plan_ref = REL + "/plan_" + arm_id + ".json"
        atomic_json(ROOT / plan_ref, plan)
        bindings.append({"arm_id":arm_id,"trajectory_id":tid,"plan_spec_ref":plan_ref,"plan_spec_sha256":sha256_file(ROOT/plan_ref),
            "output":REL+"/rml/"+arm_id})
        preparation.append({"arm_id":arm_id,"device":device,"recipe":recipe_ref,"initial_state":str(ROOT/identity["initializations"][arm_id]["file"])})
    spec = ExperimentSpec({"schema_version":SCHEMA_VERSION_V2,"experiment_id":CONFIG["run_id"],"logical_run_id":CONFIG["run_id"],"arms":arms,
        "platform":{k:CONFIG[k] for k in ("accelerator","device_count","cpu_cores","memory_gib")} | {"name":"kaggle","atomic_checkpoints":True,"retrievable_chunks":True},
        "prospective":{"arms":bindings},"evidence":{"policy":ref("molgap-v5",sha256_file(ROOT/"docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md")),
            "required_artifacts":["v5_evidence","costs","roles","trace_manifest","terminal_artifact"]},"terminal_protocol":TERMINAL_PROTOCOL})
    plan = {"format":"molgap-family-training-preparation-v1","spec_identity":spec.identity,"arms":preparation,
        "source_files":[REL+"/config.json",REL+"/protocol.md",REL+"/evidence_review.md"]}
    atomic_json(HERE/"experiment_spec.json",spec.to_dict())
    atomic_json(STORAGE/"preparation_plan.json",plan)
    print(json.dumps({"status":"DRAFT_ONLY","spec_identity":spec.identity,"source_commit":commit}))


def stage(output, data_root):
    from molgap.experiment_workflow import prepare_family_training
    from molgap.kaggle_workflow import _metadata, freeze_inputs
    from molgap.experiment_preflight import check_release_inputs
    spec = ExperimentSpec(json.loads((HERE/"experiment_spec.json").read_text()))
    plan = json.loads((STORAGE/"preparation_plan.json").read_text())
    result = prepare_family_training(spec,ROOT,plan,output)
    if result["status"] != "SOURCE_INPUTS_PREPARED":
        print(json.dumps(result)); raise SystemExit(1)
    source, kernel = output/"inputs", output/"kernel"
    kernel.mkdir()
    metadata = _metadata({"account":CONFIG["account"],"kernel":CONFIG["account"]+"/"+CONFIG["title"].lower().replace(" ","-"),
        "title":CONFIG["title"],"datasets":[CONFIG["source_dataset"],CONFIG["dataset"]],"source_dataset":CONFIG["source_dataset"],"accelerator":CONFIG["accelerator"]})
    publish_immutable_bytes(source/"dataset-metadata.json",canonical_json({"id":CONFIG["source_dataset"],"title":"K1 GPTrans 100K Profiler Source","licenses":[{"name":"CC0-1.0"}]}).encode())
    publish_immutable_bytes(kernel/"kernel-metadata.json",canonical_json(metadata).encode())
    manifest = json.loads((output/"package/package_manifest.json").read_text())
    launch = {"format":"molgap-execution-launch-v1","spec_identity":spec.identity,"expected_package_identity":manifest["package_identity"],
        "expected_source_archive_sha256":manifest["archive_sha256"],"account":CONFIG["account"],"run_reference":metadata["id"],
        "dataset_sources":metadata["dataset_sources"],"accelerator":CONFIG["accelerator"],"device_count":2,"jobs":result["jobs"],
        "allocation_wall_limit_seconds":CONFIG["allocation_wall_limit_seconds"],"cleanup_reserve_seconds":CONFIG["cleanup_reserve_seconds"]}
    frozen = {"input_root":source,"kernel_dir":kernel,"launch_path":source/"experiment_launch.json","entry_path":kernel/"run.py",
        "launch":launch,"entry_text":(ROOT/"platforms/kaggle/run_experiment.py").read_text()}
    freeze_inputs(frozen,{b["arm_id"]:ROOT/b["output"] / "trajectory.json" for b in spec.to_dict()["prospective"]["arms"]})
    release = check_release_inputs(spec,output/"package",expected_package_identity=manifest["package_identity"],
        recipe_files={a["arm_id"]:a["recipe"] for a in plan["arms"]},initial_states={a["arm_id"]:source/"initial_states"/(a["arm_id"]+".pt") for a in plan["arms"]},
        required_modules=["molgap.family_training_runner","molgap.family_execution_profile","molgap.gptrans_bond_local","molgap.pcqm_wedge"],
        pickle_inputs=[data_root/"pcqm_fixed_100k_v1/train/train_shard_0000.pt"],input_root=source,entry_script=kernel/"run.py",
        launch_config=frozen["launch_path"],kernel_metadata=kernel/"kernel-metadata.json",
        dataset_manifests={a["arm_id"]:data_root/"pcqm_fixed_100k_v1/manifest.json" for a in plan["arms"]})
    atomic_json(output/"release_report.json",release)
    print(json.dumps({"status":release["status"],"errors":release["errors"]}))
    if release["errors"]: raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("inputs","draft","stage"))
    parser.add_argument("--data-root",type=Path,default=Path("D:/文档/molgap/data"))
    parser.add_argument("--k1-initial",type=Path,default=Path("D:/文档/molgap/experiments/pcqm_k1_dropout_consistency/initial_state.pt"))
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.phase == "inputs": inputs(args.data_root,args.k1_initial)
    elif args.phase == "draft": draft()
    elif args.output is None: parser.error("stage requires --output")
    else: stage(args.output,args.data_root)
