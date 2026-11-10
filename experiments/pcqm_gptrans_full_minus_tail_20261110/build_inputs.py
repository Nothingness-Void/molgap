"""Freeze the GPT-T full-minus-tail metadata through the registered shared-family owners."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from molgap.common.io import publish_immutable_bytes
from molgap.experiment_execution import build_addon_declaration, build_family_recipe
from molgap.experiment_family_workflow import tensor_digest
from molgap.experiment_spec import ExperimentSpec, FAMILIES, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL
from molgap.experiment_source_inventory import runtime_source_files
from molgap.family_training_data import DatasetSelection
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_runner import INTERFACES, family_arm_bindings, read_recipe
from molgap.research_memory.trace import atomic_write, json_bytes
from molgap.screen_policy import canonical_fingerprint
from molgap.v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
QUESTION = "pcqm-gptrans-full-minus-tail-20261110"
PARENT_SHA = "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"
TRAIN_ROWS, TAIL_END, SHARD_ROWS, PASSES = 3_250_000, 3_378_606, 50_000, 60
STEPS = PASSES * (TRAIN_ROWS // 128)
MANIFEST_FILE = HERE / "ogb-train-full-minus-tail.manifest.json"
STATS_FILE = HERE / "train_statistics.json"
ARM = "gptrans_pairnorm"
ITEM = {"family": "gptrans_t", "mode": "degree_bond_local_pair_update_norm_ema999",
        "addon": "gptrans_degree_bond_local_pair_update_norm",
        "state": "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b",
        "initial": "D:/molgap-artifacts/initializations/structural-bias-norm-100k/gptrans.pt"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def manifest():
    parent = json.loads((ROOT / "platforms/_records/ims/pcqm_fixed_datasets_v1/ogb-train-full.manifest.json").read_text())
    assert hashlib.sha256(canonical(parent)).hexdigest() == PARENT_SHA
    keep = TRAIN_ROWS // SHARD_ROWS
    derived = json.loads(json.dumps(parent))
    for kind in ("topology", "geometry"):
        derived["assets"][kind] = parent["assets"][kind][:keep]
    assert derived["assets"]["topology"][-1]["source_idx_max"] == TRAIN_ROWS - 1
    derived["roles"]["train"] = {"rows": TRAIN_ROWS, "source_idx_start": 0, "source_idx_stop": TRAIN_ROWS}
    derived["identity"].update(name="ogb-train-full-minus-tail-v1", train_rows=TRAIN_ROWS)
    derived["aggregates"] = {"derived": "parent aggregates describe all 68 shards and no longer apply"}
    derived["derived_from"] = {"manifest_sha256": PARENT_SHA, "excluded_source_idx": [TRAIN_ROWS, TAIL_END],
                               "excluded_role": "tail_holdout_v1"}
    raw = canonical(derived)
    publish_immutable_bytes(MANIFEST_FILE, raw)
    print(json.dumps({"derived_manifest_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}))


def manifest_sha():
    return hashlib.sha256(MANIFEST_FILE.read_bytes()).hexdigest()


def statistics(root):
    from molgap.family_training_data import load_training_data
    data = load_training_data(Path(root), DatasetSelection("ogb-train-full/manifest.json", manifest_sha(), False))
    assert data.train_rows == TRAIN_ROWS and data.development_rows == 0
    atomic_write(STATS_FILE, json_bytes({"train_rows": data.train_rows, "target_mean_eV": data.target_mean_eV,
                                         "target_std_eV": data.target_std_eV, "manifest_sha256": manifest_sha()}))
    print(json.dumps({"mean": data.target_mean_eV, "std": data.target_std_eV}))


def recipes():
    import torch
    sha = manifest_sha()
    stats = json.loads(STATS_FILE.read_text())
    assert stats["manifest_sha256"] == sha
    membership = tensor_digest(torch.arange(TRAIN_ROWS, dtype=torch.int64), role="source_idx")
    role = {"role": "train", "membership_sha256": membership, "row_order_sha256": membership,
            "usage_sha256": canonical_fingerprint({"role": "train", "membership": membership, "contract": QUESTION})}
    atomic_write(HERE / "role_plan.json", json_bytes({"roles": [role], "development": "absent",
        "held_out": "tail_holdout_v1 rows [3250000, 3378606) offline only",
        "official_validation": "forbidden", "test_dev": "forbidden", "test_challenge": "forbidden"}))
    observation = inspect_frozen_state_artifact(Path(ITEM["initial"]), expected_state_sha256=ITEM["state"])
    recipe = FamilyRecipe(family=ITEM["family"], mode=ITEM["mode"], seed=42, initialization_sha256=ITEM["state"],
        dataset=DatasetSelection("ogb-train-full/manifest.json", sha, False), train_rows=TRAIN_ROWS, development_rows=0,
        optimizer_steps=STEPS, learning_rate=1e-3, weight_decay=.05,
        schedule=ScheduleSpec("warmup_cosine_epoch", PASSES, minimum_lr=1e-6, warmup_units=4),
        selection="final-ema", sampler="torch-pass-randperm", checkpoint_every=128,
        target_mean_eV=stats["target_mean_eV"], target_std_eV=stats["target_std_eV"],
        ema_decay=.999, resume_chunks=True, optimizer="adamw")
    record = build_family_recipe(INTERFACES[recipe.family], addon=ITEM["addon"], configuration=recipe,
        train_role_identity=role["usage_sha256"], development_role_identity=None)
    assert record["acceptance_requirements"]["sample_presentations"] == STEPS * 128
    atomic_write(HERE / ("training_recipe_" + ARM + ".json"), json_bytes(record))
    atomic_write(HERE / "initialization_inspection.json", json_bytes({ARM: observation}))
    policy = {"schema": "molgap-policy-v1", "policy_id": QUESTION, "version": "1",
        "policy_type": "research_action", "status": "candidate",
        "required_observable_fields": ["user_requested_scope_reviewed"], "observation_point": None,
        "comparability_selector": {"scientific_contract": QUESTION},
        "action_rule": {"action": "PREPARE_FULL_COLAB", "field": "user_requested_scope_reviewed", "operator": "eq", "threshold": 1},
        "borderline_action": "NO_TRAIN", "early_stop_rule": None, "promotion_rule": None,
        "cost_model": {"kind": "measured_only", "assumptions": []},
        "approval": {"approved_by": None, "approved_at": None, "authority_ref": None},
        "created_from_source_digest": normalized_source_sha256(HERE / "protocol.md")}
    atomic_write(ROOT / "research_memory/policies" / (QUESTION + ".1.json"), json_bytes(policy))


def plans():
    sha = manifest_sha()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    role_plan = json.loads((HERE / "role_plan.json").read_text())
    evidence = json.loads((ROOT / "experiments/pcqm_structural_bias_norm_100k/k1/experiment_spec.json").read_text())["evidence"]
    recipe_ref = REL + "/training_recipe_" + ARM + ".json"
    _, recipe = read_recipe(ROOT / recipe_ref)
    bindings = family_arm_bindings(recipe)
    addons = [build_addon_declaration(INTERFACES[recipe.family], ITEM["addon"], repo_root=ROOT)]
    arm = {"arm_id": ARM, "scientific_role": "candidate", "family": {"name": recipe.family, "version": INTERFACES[recipe.family][1]},
        "base": bindings["base"], "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": ITEM["state"]},
        "data": {"dataset": {"name": "pcqm4mv2", "version": "1", "sha256": sha},
            "split": {"name": "official-train-minus-tail", "version": "1", "sha256": canonical_fingerprint({"train": [0, TRAIN_ROWS], "development": None})},
            "roles": role_plan["roles"], "feature_schema": FAMILIES[INTERFACES[recipe.family]].feature_schema, "feature_sha256": sha,
            "target": "pcqm4mv2-gap-eV-direct"},
        "training": {"recipe": {"name": "shared-family-training", "version": "1", "sha256": normalized_source_sha256(ROOT / recipe_ref)},
            "overrides": {}, **{key: bindings[key] for key in ("sampler", "transform", "objective")}},
        "addons": addons, "addon_semantics": "ordered"}
    run = QUESTION + "-" + ARM
    tid, cid = "TB-" + run, "cost-" + run + "-planned"
    contracts = [REL + "/protocol.md", REL + "/evidence_review.md", recipe_ref]
    trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": tid, "record_mode": "prospective", "owner": "desktop", "track": "B",
        "family_id": QUESTION, "question": "Does a 60-pass GPT-T full-train endpoint without the tail holdout support a later fixed fusion, and where does tail generalization stop improving?",
        "hypothesis": {"hypothesis_id": "H-" + run, "observed_deficiency": "The earlier full package trained on every row, so its endpoint could not be diagnosed on a held-out role.",
            "supporting_evidence_ids": ["pcqm-k1-gptrans-full-fusion-r3"], "alternative_explanations": ["Bounded-scale improvements may not transfer to full training."],
            "changed_mechanism": ITEM["mode"], "cheapest_falsifier": "Verify frozen metadata and native repeat/resume qualification before full training in the uploaded Notebook.",
            "expected_native_cost_ref": cid, "related_closed_family_ids": ["pcqm-k1-gptrans-full-fusion"],
            "decision_changed_if_positive": "Retain the final endpoint for separately authorized fixed-fusion evaluation.",
            "decision_changed_if_negative": "Do not adopt the endpoint; retain attribution and exact artifacts."},
        "state_at_start": {"source_commit": commit, "source_config_identity": canonical_fingerprint(arm), "contract_refs": contracts,
            "reference_ids": [], "parent_trajectory_ids": [], "prior_evidence_ids": ["pcqm-k1-gptrans-full-fusion-r3"],
            "role_snapshot_refs": [REL + "/role_plan.json"], "budget_snapshot_ref": REL + "/authority.json"},
        "actions": [{"action_id": "A001", "type": "training", "source_commit": commit, "run_ids": [run + ":" + ARM + ":downstream"],
            "attempt_ids": [ARM + "-v1"], "evidence_refs": contracts, "cost_event_ids": [cid]}],
        "result": {"evidence_ids": [], "evidence_refs": []}, "decision": {"outcome": "ACTIVE", "decision_ref": REL + "/decision.md",
            "next_allowed_actions": ["Prepare upload-ready full Colab package; no cloud execution by the preparing agent."], "reopen_conditions": []}}
    plan = {"trajectory": trajectory, "decision_state": {"available_actions": ["PREPARE_FULL_COLAB", "NO_TRAIN"], "chosen_action": "PREPARE_FULL_COLAB",
        "policy_id": QUESTION, "policy_version": "1", "state_timestamp": datetime.now(timezone.utc).isoformat(), "source_commit": commit,
        "budget_snapshot_ref": REL + "/authority.json", "role_snapshot_refs": [REL + "/role_plan.json"]},
        "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": cid, "trajectory_id": tid, "action_id": "A001",
            "run_id": run + ":" + ARM + ":downstream", "attempt_id": ARM + "-v1", "platform": "colab", "hardware": "NVIDIA A100/L4",
            "category": "training", "evidence_ref": REL + "/authority.json", "measurement": {key: {"value": None, "status": "measurement_missing"}
                for key in ("wall_hours", "cpu_hours", "device_hours", "queue_hours")}}]}
    plan_ref = REL + "/plan_input_" + ARM + ".json"
    atomic_write(ROOT / plan_ref, json_bytes(plan))
    spec = ExperimentSpec({"schema_version": SCHEMA_VERSION_V2, "experiment_id": run, "logical_run_id": run, "arms": [arm],
        "platform": {"name": "colab", "accelerator": "A100/L4", "device_count": 1, "cpu_cores": 4, "memory_gib": 50,
            "atomic_checkpoints": True, "retrievable_chunks": True}, "prospective": {"arms": [{"arm_id": ARM, "trajectory_id": tid,
            "plan_spec_ref": plan_ref, "plan_spec_sha256": hashlib.sha256((ROOT / plan_ref).read_bytes()).hexdigest(), "output": REL + "/rml/" + ARM}]},
        "evidence": evidence, "terminal_protocol": TERMINAL_PROTOCOL})
    atomic_write(HERE / ("experiment_spec_" + ARM + ".json"), spec.to_json().encode())
    names = ("build_inputs.py", "protocol.md", "authority.json", "evidence_review.md", "decision.md", "role_plan.json",
             "ogb-train-full-minus-tail.manifest.json", "train_statistics.json")
    preparation = {"format": "molgap-family-training-preparation-v1", "spec_identity": spec.identity,
        "source_files": runtime_source_files(ROOT, [REL + "/" + name for name in names]),
        "arms": [{"arm_id": ARM, "device": 0, "recipe": recipe_ref, "initial_state": ITEM["initial"]}]}
    atomic_write(HERE / ("prepare_plan_" + ARM + ".json"), json_bytes(preparation))
    print(json.dumps({"arm_id": ARM, "spec_identity": spec.identity, "source_commit": commit}))


PHASES = {"manifest": manifest, "recipes": recipes, "plans": plans}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("manifest", "stats", "recipes", "plans"), required=True)
    parser.add_argument("--data-root", help="stats only: folder containing ogb-train-full/manifest.json and store/")
    args = parser.parse_args()
    statistics(args.data_root) if args.phase == "stats" else PHASES[args.phase]()
