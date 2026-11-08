"""Freeze full-train metadata through the registered shared-family owners."""
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
QUESTION = "pcqm-k1-muon-gptrans-full-colab-20261108"
MANIFEST_SHA = "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"
ARMS = {
    "k1_muon": {"family": "neural_atom_k1", "mode": "reference", "addon": None,
        "state": "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd",
        "initial": "D:/molgap-artifacts/initializations/structural-bias-norm-100k/k1.pt",
        "lr": 4e-4, "wd": 1e-5, "schedule": "cosine_epoch", "warmup": 0,
        "selection": "final-live", "sampler": "torch-pass-randperm",
        "optimizer": "k1-muon-batched-adamw-fp32-v1"},
    "gptrans_pairnorm": {"family": "gptrans_t", "mode": "degree_bond_local_pair_update_norm_ema999",
        "addon": "gptrans_degree_bond_local_pair_update_norm",
        "state": "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b",
        "initial": "D:/molgap-artifacts/initializations/structural-bias-norm-100k/gptrans.pt",
        "lr": 1e-3, "wd": .05, "schedule": "warmup_cosine_epoch", "warmup": 4,
        "selection": "final-ema", "sampler": "torch-pass-randperm", "optimizer": "adamw"}}


def recipes():
    import torch
    manifest = json.loads((ROOT / "platforms/_records/ims/pcqm_fixed_datasets_v1/ogb-train-full.manifest.json").read_text())
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA
    publish_immutable_bytes(HERE / "ogb-train-full.manifest.json", raw)
    membership = tensor_digest(torch.arange(3378606, dtype=torch.int64), role="source_idx")
    role = {"role": "train", "membership_sha256": membership, "row_order_sha256": membership,
        "usage_sha256": canonical_fingerprint({"role": "train", "membership": membership, "contract": QUESTION})}
    atomic_write(HERE / "role_plan.json", json_bytes({"roles": [role], "development": "absent",
        "official_validation": "forbidden", "test_dev": "forbidden", "test_challenge": "forbidden"}))
    observations = {}
    for arm_id, item in ARMS.items():
        observations[arm_id] = inspect_frozen_state_artifact(Path(item["initial"]), expected_state_sha256=item["state"])
        recipe = FamilyRecipe(family=item["family"], mode=item["mode"], seed=42,
            initialization_sha256=item["state"], dataset=DatasetSelection("ogb-train-full/manifest.json", MANIFEST_SHA, False),
            train_rows=3378606, development_rows=0, optimizer_steps=1583700,
            learning_rate=item["lr"], weight_decay=item["wd"],
            schedule=ScheduleSpec(item["schedule"], 60, minimum_lr=1e-6, warmup_units=item["warmup"]),
            selection=item["selection"], sampler=item["sampler"], checkpoint_every=128,
            target_mean_eV=5.689459009120812, target_std_eV=1.1621397733722936,
            ema_decay=.999, resume_chunks=True, optimizer=item["optimizer"])
        record = build_family_recipe(INTERFACES[recipe.family], addon=item["addon"], configuration=recipe,
            train_role_identity=role["usage_sha256"], development_role_identity=None)
        assert record["acceptance_requirements"]["sample_presentations"] == 202713600
        atomic_write(HERE / ("training_recipe_" + arm_id + ".json"), json_bytes(record))
    atomic_write(HERE / "initialization_inspection.json", json_bytes(observations))
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
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    role_plan = json.loads((HERE / "role_plan.json").read_text())
    evidence = json.loads((ROOT / "experiments/pcqm_structural_bias_norm_100k/k1/experiment_spec.json").read_text())["evidence"]
    for arm_id, item in ARMS.items():
        recipe_ref = REL + "/training_recipe_" + arm_id + ".json"
        _, recipe = read_recipe(ROOT / recipe_ref)
        bindings = family_arm_bindings(recipe)
        addons = [] if item["addon"] is None else [build_addon_declaration(INTERFACES[recipe.family], item["addon"], repo_root=ROOT)]
        arm = {"arm_id": arm_id, "scientific_role": "candidate", "family": {"name": recipe.family, "version": INTERFACES[recipe.family][1]},
            "base": bindings["base"], "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": item["state"]},
            "data": {"dataset": {"name": "pcqm4mv2", "version": "1", "sha256": MANIFEST_SHA},
                "split": {"name": "official-train-full", "version": "1", "sha256": canonical_fingerprint({"train": [0, 3378606], "development": None})},
                "roles": role_plan["roles"], "feature_schema": FAMILIES[INTERFACES[recipe.family]].feature_schema, "feature_sha256": MANIFEST_SHA,
                "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": {"name": "shared-family-training", "version": "1", "sha256": normalized_source_sha256(ROOT / recipe_ref)},
                "overrides": {}, **{key: bindings[key] for key in ("sampler", "transform", "objective")}},
            "addons": addons, "addon_semantics": "ordered" if addons else "baseline"}
        run = QUESTION + "-" + arm_id
        tid = "TB-" + run
        cid = "cost-" + run + "-planned"
        contracts = [REL + "/protocol.md", REL + "/evidence_review.md", recipe_ref]
        trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": tid, "record_mode": "prospective", "owner": "desktop", "track": "B",
            "family_id": QUESTION, "question": "Can full official-train endpoints for K1 Muon and GPTrans PairNorm support a later fixed fusion?",
            "hypothesis": {"hypothesis_id": "H-" + run, "observed_deficiency": "Retained 500K evidence does not establish full-scale endpoint quality.",
                "supporting_evidence_ids": ["pcqm-k1-gptrans-full-fusion-r3"], "alternative_explanations": ["Bounded-scale improvements may not transfer to full training."],
                "changed_mechanism": item["optimizer"] if arm_id == "k1_muon" else item["mode"],
                "cheapest_falsifier": "Verify frozen metadata and native repeat/resume qualification before full training in the uploaded Notebook.",
                "expected_native_cost_ref": cid, "related_closed_family_ids": ["pcqm-k1-gptrans-full-fusion"],
                "decision_changed_if_positive": "Retain the final endpoint for separately authorized fixed-fusion evaluation.",
                "decision_changed_if_negative": "Do not adopt the endpoint; retain attribution and exact artifacts."},
            "state_at_start": {"source_commit": commit, "source_config_identity": canonical_fingerprint(arm), "contract_refs": contracts,
                "reference_ids": [], "parent_trajectory_ids": [], "prior_evidence_ids": ["pcqm-k1-gptrans-full-fusion-r3"],
                "role_snapshot_refs": [REL + "/role_plan.json"], "budget_snapshot_ref": REL + "/authority.json"},
            "actions": [{"action_id": "A001", "type": "training", "source_commit": commit, "run_ids": [run + ":" + arm_id + ":downstream"],
                "attempt_ids": [arm_id + "-v1"], "evidence_refs": contracts, "cost_event_ids": [cid]}],
            "result": {"evidence_ids": [], "evidence_refs": []}, "decision": {"outcome": "ACTIVE", "decision_ref": REL + "/decision.md",
                "next_allowed_actions": ["Prepare upload-ready full Colab package; no cloud execution by the preparing agent."], "reopen_conditions": []}}
        plan = {"trajectory": trajectory, "decision_state": {"available_actions": ["PREPARE_FULL_COLAB", "NO_TRAIN"], "chosen_action": "PREPARE_FULL_COLAB",
            "policy_id": QUESTION, "policy_version": "1", "state_timestamp": datetime.now(timezone.utc).isoformat(), "source_commit": commit,
            "budget_snapshot_ref": REL + "/authority.json", "role_snapshot_refs": [REL + "/role_plan.json"]},
            "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": cid, "trajectory_id": tid, "action_id": "A001",
                "run_id": run + ":" + arm_id + ":downstream", "attempt_id": arm_id + "-v1", "platform": "colab", "hardware": "NVIDIA A100",
                "category": "training", "evidence_ref": REL + "/authority.json", "measurement": {key: {"value": None, "status": "measurement_missing"}
                    for key in ("wall_hours", "cpu_hours", "device_hours", "queue_hours")}}]}
        plan_ref = REL + "/plan_input_" + arm_id + ".json"
        atomic_write(ROOT / plan_ref, json_bytes(plan))
        spec = ExperimentSpec({"schema_version": SCHEMA_VERSION_V2, "experiment_id": run, "logical_run_id": run, "arms": [arm],
            "platform": {"name": "colab", "accelerator": "A100", "device_count": 1, "cpu_cores": 4, "memory_gib": 80,
                "atomic_checkpoints": True, "retrievable_chunks": True}, "prospective": {"arms": [{"arm_id": arm_id, "trajectory_id": tid,
                "plan_spec_ref": plan_ref, "plan_spec_sha256": hashlib.sha256((ROOT / plan_ref).read_bytes()).hexdigest(), "output": REL + "/rml/" + arm_id}]},
            "evidence": evidence, "terminal_protocol": TERMINAL_PROTOCOL})
        atomic_write(HERE / ("experiment_spec_" + arm_id + ".json"), spec.to_json().encode())
        preparation = {"format": "molgap-family-training-preparation-v1", "spec_identity": spec.identity,
            "source_files": runtime_source_files(ROOT, [REL + "/" + name for name in ("build_inputs.py", "protocol.md", "authority.json", "evidence_review.md", "decision.md", "role_plan.json", "ogb-train-full.manifest.json")]),
            "arms": [{"arm_id": arm_id, "device": 0, "recipe": recipe_ref, "initial_state": item["initial"]}]}
        atomic_write(HERE / ("prepare_plan_" + arm_id + ".json"), json_bytes(preparation))
        print(json.dumps({"arm_id": arm_id, "spec_identity": spec.identity, "source_commit": commit}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("recipes", "plans"), required=True)
    args = parser.parse_args()
    recipes() if args.phase == "recipes" else plans()
