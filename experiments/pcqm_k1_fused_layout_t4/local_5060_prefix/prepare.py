"""Thin local binding over K1 drafts, source packaging and prospective planning."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from molgap.experiment_package import build_experiment_source_package
from molgap.experiment_prospective import plan_prospective
from molgap.experiment_source_inventory import SHARED_SOURCE_FILES
from molgap.experiment_spec import ExperimentSpec
from molgap.k1_pair_preparation import build_inputs
from molgap.k1_screen_training import validate_recipe
from molgap.kaggle_workflow import stage_acceptance_inputs
from molgap.research_memory.policy import validate_policy
from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file

REL = "experiments/pcqm_k1_fused_layout_t4/local_5060_prefix"
RUN = "molgap-k1-audited-local5060-100k-10ep-s42-20261011"
ARMS = ("reference", "fused_layout")


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def prepare(root, initial, *, draft):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    folder = root / REL
    records = build_inputs(root, source_commit=commit, initial_state=initial,
        state_timestamp=datetime.now(timezone.utc).isoformat(), relative_dir=REL, logical_run_id=RUN,
        reference_arm_id="reference", reference_addon=None, candidate_arm_id="fused_layout",
        candidate_addon="k1_fused_layout", family_id="k1-local-audited-prefix",
        policy_id=RUN, allocation_wall_limit_seconds=3600,
        candidate_objective={"base_loss": "normalized-gap-l1", "forward_passes": 1,
            "optimizer_updates_per_batch": 1, "consistency_penalty": False, "batchnorm_updates_per_batch": 1},
        supporting_evidence_ids=["pcqm-k1-local-speed-10ep-fused_layout-20261011"],
        question="Does audited K1 execution improve original-default5060 speed without an early Gap regression?",
        changed_mechanism="Native AdamW/original transfers/scalar reads versus fused AdamW/CPU layout/audited transfers and metrics",
        hypothesis_overrides={"observed_deficiency": "Earlier local10ep control disabled native foreach and did not access development",
            "cheapest_falsifier": "Authorized10epoch prefixes, same fixed100K/50K and40epoch schedule, on oneRTX5060",
            "alternative_explanations": ["Native foreach closes optimizer gain", "Graphics activity changes timing", "Fused rounding affects the early trajectory"],
            "decision_changed_if_positive": "Retain local execution evidence only; separate complete/native quality still needed",
            "decision_changed_if_negative": "Do not adopt audited execution on the strength of synthetic tests"})
    spec_record = records["experiment_spec.json"]
    spec_record["platform"].update(name="local", accelerator="NVIDIA GeForce RTX 5060", device_count=1)
    for arm in spec_record["arms"]:
        name = arm["arm_id"]
        recipe = records[f"training_recipe_{name}.json"]
        recipe["local_execution_prefix"] = {"epochs": 10, "accelerator": "NVIDIA GeForce RTX 5060",
            "original_execution": name == "reference", "purpose": "execution-quality-prefix"}
        recipe["acceptance_requirements"].update(epochs=10, optimizer_steps=7810, sample_presentations=999680)
        validate_recipe(recipe, mode=name)
        arm["training"]["recipe"]["sha256"] = canonical_fingerprint(recipe)
        # Both arms retain the exact same supervised objective, not a copied two-view identity.
        if name == "fused_layout":
            reference = next(a for a in spec_record["arms"] if a["arm_id"] == "reference")
            arm["training"]["objective"] = reference["training"]["objective"].copy()
        if draft:
            write(folder / f"training_recipe_{name}.json", recipe)
    if draft:
        print("Local recipes drafted; commit executable source and recipes before binding.")
        return
    policy = records["policy.json"]
    policy.update(status="approved", required_observable_fields=["user_release_validated"],
        action_rule={"field": "user_release_validated", "operator": "eq", "threshold": 1, "action": "RUN_DIAGNOSTIC"})
    policy["approval"] = {"approved_by": "user", "approved_at": "2026-10-11", "authority_ref": REL + "/protocol.md"}
    policy["cost_model"]["assumptions"] = ["One sharedRTX5060,3600s ceiling; no T4/A100 quota conversion; native times measured separately"]
    validate_policy(policy)
    for arm in spec_record["arms"]:
        name = arm["arm_id"]
        plan = records[f"training_plan_{name}.json"]
        trajectory = plan["trajectory"]
        trajectory["state_at_start"]["source_config_identity"] = canonical_fingerprint(arm)
        trajectory["comparison_class"] = "CONTEXT_ONLY"
        trajectory["comparison_readiness_ref"] = REL + "/protocol.md"
        trajectory["comparison_blockers"] = ["10epoch prefix, not complete40", "One seed and nonexclusiveRTX5060"]
        trajectory["reference_bundle_id"] = None
        trajectory["actions"][0]["type"] = "user_authorized_local_execution_quality_prefix"
        trajectory["decision"]["next_allowed_actions"] = ["A001"]
        plan["decision_state"].update(available_actions=["NO_TRAIN", "RUN_DIAGNOSTIC"], chosen_action="RUN_DIAGNOSTIC")
        cost = plan["costs"][0]
        cost.update(platform="local", hardware="NVIDIA GeForce RTX 5060; shared single-device prefix")
        for key in ("device_hours", "wall_hours"):
            cost["measurement"][key] = {"status": "estimated", "value": 0.5}
        cost["measurement"]["queue_hours"] = {"status": "not_applicable", "value": None}
    for binding in spec_record["prospective"]["arms"]:
        binding["plan_spec_sha256"] = canonical_fingerprint(records[f"training_plan_{binding['arm_id']}.json"])
    spec = ExperimentSpec(spec_record)
    acceptance = records["family_acceptance_plan.json"]
    acceptance["spec_identity"] = spec.identity
    for arm in acceptance["arms"]:
        name = arm["arm_id"]
        arm["expected"] = records[f"training_recipe_{name}.json"]["acceptance_requirements"]
        arm["contract"]["sha256"] = spec_record["arms"][ARMS.index(name)]["training"]["recipe"]["sha256"]
    records["role_plan.json"].update(dataset="pcqm4mv2-ogb-fixed-100k-v1",
        development_usage="per-epoch internal development, selected-prefix and epoch10 diagnostics; not independent holdout")
    for name in ("policy.json", "role_plan.json", "experiment_spec.json", "family_acceptance_plan.json",
                 "training_plan_reference.json", "training_plan_fused_layout.json"):
        write(folder / name, records[name])
    write(root / f"research_memory/policies/{RUN}-v1.json", policy)
    stage = root / "platforms/_records/local/staging" / RUN
    package = stage / "package"
    files = [*SHARED_SOURCE_FILES, "src/molgap/k1_local_prefix.py", "src/molgap/router.py",
             *(REL + f"/training_recipe_{arm}.json" for arm in ARMS)]
    manifest = build_experiment_source_package(spec, root, files, package)
    receipt, code = plan_prospective(spec, root, execution_source_commit=commit)
    atomic_json(folder / "plan_receipt.json", receipt)
    if code:
        raise RuntimeError("Local prospective publication requires reconciliation")
    frozen = stage / "frozen"
    frozen.mkdir()
    with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            path = frozen / member.name
            path.resolve().relative_to(frozen.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
    staged = stage / "inputs"
    acceptance_binding = stage_acceptance_inputs(spec, root, REL + "/family_acceptance_plan.json", staged)
    shutil.copyfile(initial, staged / "initial_state.pt")
    hashes, jobs = {}, []
    for binding in spec_record["prospective"]["arms"]:
        name = binding["arm_id"]
        target = staged / "prospective" / name / "trajectory.json"
        target.parent.mkdir(parents=True)
        shutil.copyfile(root / binding["output"] / "trajectory.json", target)
        hashes[name] = sha256_file(target)
        jobs.append({"arm_id": name, "device": 0, "recipe": REL + f"/training_recipe_{name}.json",
            "initial_state": "initial_state.pt", "trajectory_id": binding["trajectory_id"]})
    config = {"package": str(package), "package_identity": manifest["package_identity"], "source_root": str(frozen),
        "staged_root": str(staged), "input_root": "D:/文档/molgap/data/pcqm_fixed_100k_v1", "jobs": jobs,
        "prospective_sha256": hashes, "allocation_seconds": 3600, "output": str(stage / "run"),
        "acceptance_plan": acceptance_binding["plan_path"], "acceptance_plan_sha256": acceptance_binding["plan_sha256"]}
    atomic_json(stage / "config.json", config)
    atomic_json(folder / "release.json", {"source_commit": commit, "config": str(stage / "config.json"),
        "config_sha256": sha256_file(stage / "config.json"), "source": manifest, "submitted_remote": False})
    print(json.dumps({"config": str(stage / "config.json"), "source_root": str(frozen), "output": config["output"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    prepare(args.repo_root.resolve(), args.initial_state.resolve(), draft=args.draft)
