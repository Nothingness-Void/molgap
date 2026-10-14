"""Question declarations over shared family/identity/platform preparation owners."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess

from molgap.experiment_execution import build_family_recipe
from molgap.experiment_family_workflow import tensor_digest
from molgap.experiment_launch import canonical_json, publish_immutable_bytes
from molgap.experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL
from molgap.family_training_data import DatasetSelection
from molgap.family_training_recipe import ScheduleSpec
from molgap.family_training_runner import INTERFACES, family_arm_bindings, read_recipe
from molgap.family_training_state import CalibrationPolicy, source_order_sha256
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import sha256_file
from molgap.research_memory.trace import atomic_write, json_bytes
from molgap.v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
RUN = "molgap-k1-clean-ema-gptrans-local-500k"
MANIFEST_SHA = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
DATASET = "nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1"
TITLE = "MolGap K1 Clean EMA GPTrans Local 500K"
SOURCE_DATASET = "nothingnessvoid/" + RUN + "-src"


def write(path, record):
    atomic_write(path, json_bytes(record))


def recipes():
    import torch
    role_records = []
    for role, start, rows in (("train", 0, 500000), ("development", 500000, 50000)):
        digest = tensor_digest(torch.arange(start, start + rows), role="source_idx")
        role_records.append({"role": role, "membership_sha256": digest, "row_order_sha256": digest,
            "usage_sha256": canonical_fingerprint({"role": role, "contract": RUN, "membership": digest})})
    source = Path("D:/w/k1-mean2-single-ema-500k-20261013/experiments/pcqm_k1_mean2_single_ema_500k")
    old_record, k1 = read_recipe(source / "training_recipe_single_ema.json")
    _, gp = read_recipe(ROOT / "experiments/family_topology_t4_100k/training_recipe_gptrans_local.json")
    policy = CalibrationPolicy(MANIFEST_SHA, role_records[0]["usage_sha256"],
        source_order_sha256(torch.arange(16384)), 128, 128)
    common = dict(dataset=DatasetSelection(DATASET.split("/")[1] + "/manifest.json", MANIFEST_SHA, True),
        train_rows=500000, development_rows=50000, optimizer_steps=234360,
        target_mean_eV=5.413434769952119, target_std_eV=1.176728647311213,
        checkpoint_every=128, finite_check_every=50, sampler="torch-pass-randperm")
    configs = {"k1_clean_ema": replace(k1, **common, clean_bn=policy),
        "gptrans_local": replace(gp, **common, schedule=ScheduleSpec("warmup_cosine_epoch", 60,
            minimum_lr=1e-6, warmup_units=4))}
    for arm, recipe in configs.items():
        record = build_family_recipe(INTERFACES[recipe.family], configuration=recipe,
            addon="gptrans_degree_bond_local" if arm == "gptrans_local" else None,
            source_idx_sha256=old_record["acceptance_requirements"]["source_idx_sha256"],
            target_sha256=old_record["acceptance_requirements"]["target_sha256"],
            train_role_identity=role_records[0]["usage_sha256"],
            development_role_identity=role_records[1]["usage_sha256"])
        write(HERE / ("training_recipe_" + arm + ".json"), record)
    write(HERE / "role_plan.json", {"roles": role_records, "train_range": [0, 500000],
        "development_range": [500000, 550000], "development_selection_used": True,
        "calibration": policy.to_dict(), "official_validation_consumed": False, "test_consumed": False})
    write(HERE / "normalization_source.json", json.loads((source / "normalization_source.json").read_bytes()))
    write(HERE / "config.json", {"run_id": RUN, "account": "nothingnessvoid", "title": TITLE,
        "dataset": DATASET, "source_dataset": SOURCE_DATASET, "dataset_manifest_sha256": MANIFEST_SHA,
        "allocation_wall_limit_seconds": 42900, "cleanup_reserve_seconds": 180,
        "accelerator": "NvidiaTeslaT4", "device_count": 2, "cpu_cores": 4, "memory_gib": 30,
        "train_rows": 500000, "development_rows": 50000, "epochs": 60})
    policy_record = json.loads((ROOT / "research_memory/policies/pcqm-k1-colab-execution-profile.1.json").read_bytes())
    policy_record.update(policy_id=RUN, status="candidate", comparability_selector={"scientific_contract": RUN},
        created_from_source_digest=sha256_file(HERE / "protocol.md"))
    policy_record["action_rule"]["action"] = "RUN_PAIRED_500K"
    policy_record["cost_model"]["assumptions"] = ["11h55m wall and23.8333 assigned T4-device hours are ceilings, not observations."]
    write(ROOT / "research_memory/policies" / (RUN + ".1.json"), policy_record)


def draft(initial_root):
    if (HERE / "rml").exists():
        raise ValueError("Prospective records exist; reconcile rather than redraft")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    roles = json.loads((HERE / "role_plan.json").read_bytes())["roles"]
    template = json.loads((ROOT / "experiments/family_topology_t4_100k/experiment_spec.json").read_bytes())
    arms, bindings, entries = [], [], []
    priors = {"k1_clean_ema": "pcqm-k1-bn-mechanism-a100-20261008",
        "gptrans_local": "TB-family-topology-t4-100k-gptrans_local-s42-terminal-20261014"}
    for device, name in enumerate(("k1_clean_ema", "gptrans_local")):
        _, recipe = read_recipe(HERE / ("training_recipe_" + name + ".json"))
        arm = deepcopy(template["arms"][device])
        arm.update(arm_id=name, scientific_role="candidate", addons=[], addon_semantics="baseline")
        b = family_arm_bindings(recipe)
        arm["base"] = b["base"]
        arm["initialization"] = {"kind": "frozen_state", "seed": recipe.seed, "state_sha256": recipe.initialization_sha256}
        arm["data"]["dataset"]["sha256"] = MANIFEST_SHA
        arm["data"]["split"] = {"name": "fixed500k-next50k", "version": "1", "sha256": canonical_fingerprint({"roles": roles})}
        arm["data"]["roles"], arm["data"]["feature_sha256"] = deepcopy(roles), MANIFEST_SHA
        arm["training"] = {"recipe": {"name": "shared-family-training", "version": "1",
            "sha256": normalized_source_sha256(HERE / ("training_recipe_" + name + ".json"))}, "overrides": {},
            **{key: b[key] for key in ("objective", "sampler", "transform")}}
        if name == "gptrans_local":
            arm.update(addon_semantics="ordered", addons=[{"name": "gptrans_degree_bond_local", "version": "1",
                "config": {}, "source_sha256": normalized_source_sha256(ROOT / "src/molgap/gptrans_bond_local.py")}])
        arms.append(arm)
        tid = "TB-clean-k1-gptrans-local-500k-" + name + "-s42"
        cost_id = "cost-" + tid + "-planned"
        plan = json.loads((ROOT / "experiments/family_100k_operator_profile/plan_k1_mean2.json").read_bytes())
        t = plan["trajectory"]
        t.update(trajectory_id=tid, family_id="clean-k1-gptrans-local-500k", question=
            "Does explicit Clean-EMA selection improve the single-forward K1 package at500K?" if device == 0 else
            "Does the reviewed degree/true-bond local GPTrans100K package retain its500K benefit?")
        h = t["hypothesis"]
        h.update(hypothesis_id="H-" + tid, supporting_evidence_ids=[priors[name]],
            observed_deficiency="Raw inference BN and local-bond scale transfer are unresolved by prior500K endpoint custody.",
            alternative_explanations=["Calibration-only benefit", "EMA/regularization tradeoff", "Diminishing local-bond transfer", "Seed/runtime variation"],
            changed_mechanism="Independent per-pass Clean EMA selection" if device == 0 else "Transfer registered degree/bond-local recipe from100K to500K",
            cheapest_falsifier="One bounded user-authorized500K candidate allocation", expected_native_cost_ref=cost_id,
            decision_changed_if_positive="Nominate separate qualified comparison; no automatic adoption",
            decision_changed_if_negative="Close or retain partial ambiguity with attribution; no automatic retry",
            related_closed_family_ids=["k1-clean-second-view", "gptrans-pair-norm", "gptrans-t-centered-logits-k1pair"])
        t["state_at_start"].update(source_commit=commit, source_config_identity=canonical_fingerprint(arm),
            contract_refs=[REL + "/protocol.md", REL + "/config.json", REL + "/evidence_review.md"],
            reference_ids=[], parent_trajectory_ids=[], prior_evidence_ids=[priors[name]],
            role_snapshot_refs=[REL + "/role_plan.json"], budget_snapshot_ref=REL + "/config.json")
        t["actions"] = [{"action_id": "A001", "type": "candidate_500k_training",
            "run_ids": [RUN + ":" + name + ":downstream"], "attempt_ids": [name + "-v1"],
            "source_commit": commit, "evidence_refs": [REL + "/config.json"], "cost_event_ids": [cost_id]}]
        t["result"] = {"evidence_ids": [], "evidence_refs": []}
        t["decision"] = {"outcome": "ACTIVE", "decision_ref": REL + "/decision.md", "next_allowed_actions": ["A001"], "reopen_conditions": []}
        cost = plan["costs"][0]
        cost.update(cost_event_id=cost_id, trajectory_id=tid, run_id=t["actions"][0]["run_ids"][0],
            attempt_id=name + "-v1", evidence_ref=REL + "/protocol.md", category="training")
        for metric in ("wall_hours", "device_hours"):
            cost["measurement"][metric] = {"value": 42900 / 3600, "status": "estimated"}
        plan["decision_state"].update(known_trajectory_ids=[], known_evidence_ids=[priors[name]], active_reference_ids=[],
            available_actions=["RUN_PAIRED_500K", "NO_TRAIN"], chosen_action="RUN_PAIRED_500K", policy_id=RUN,
            source_commit=commit, state_timestamp=datetime.now(timezone.utc).isoformat(),
            role_snapshot_refs=[REL + "/role_plan.json"], budget_snapshot_ref=REL + "/config.json")
        path = HERE / ("plan_" + name + ".json")
        write(path, plan)
        bindings.append({"arm_id": name, "trajectory_id": tid, "plan_spec_ref": path.relative_to(ROOT).as_posix(),
            "plan_spec_sha256": sha256_file(path), "output": REL + "/rml/" + name})
        initial = initial_root / ("k1_mean2.pt" if device == 0 else "gptrans_local.pt")
        inspect_frozen_state_artifact(initial, expected_state_sha256=recipe.initialization_sha256)
        entries.append({"arm_id": name, "device": device, "recipe": REL + "/training_recipe_" + name + ".json", "initial_state": str(initial)})
    spec = ExperimentSpec({"schema_version": SCHEMA_VERSION_V2, "experiment_id": RUN, "logical_run_id": RUN,
        "arms": arms, "platform": template["platform"], "prospective": {"arms": bindings},
        "evidence": template["evidence"], "terminal_protocol": TERMINAL_PROTOCOL})
    write(HERE / "experiment_spec.json", spec.to_dict())
    write(HERE / "preparation_plan.json", {"format": "molgap-family-training-preparation-v1", "spec_identity": spec.identity,
        "source_files": [REL + "/" + name for name in ("config.json", "protocol.md", "evidence_review.md", "role_plan.json", "normalization_source.json")], "arms": entries})
    print(json.dumps({"status": "DRAFT_ONLY", "source_commit": commit, "spec_identity": spec.identity}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("recipes", "draft"))
    p.add_argument("--initial-root", type=Path)
    args = p.parse_args()
    recipes() if args.phase == "recipes" else draft(args.initial_root)
