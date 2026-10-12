"""Question-specific configuration over shared preparation; never trains/submits."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from molgap.experiment_execution import build_family_recipe
from molgap.experiment_family_workflow import tensor_digest
from molgap.experiment_launch import canonical_json
from molgap.experiment_source_inventory import SHARED_SOURCE_FILES
from molgap.experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL
from molgap.family_training_data import DatasetSelection
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_runner import INTERFACES, family_arm_bindings
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()


def reference(name, digest):
    return {"name": name, "version": "1", "sha256": digest}


def write_json(path, value):
    # Draft/generated inputs only; canonical prospective records use the planner.
    atomic_json(path, value)


def draft(initial, normalization_checkpoint):
    import torch
    if (HERE / "rml").exists():
        raise ValueError("Published prospective records exist; reconcile instead of rewriting")
    c = json.loads((HERE / "config.json").read_text())
    inspect_frozen_state_artifact(initial, expected_state_sha256=c["initialization_sha256"])
    expected = "a152ff13719ce3e6bd817eb49b6da2c204e752b6f7787f5991f6c3206794bd03"
    if sha256_file(normalization_checkpoint) != expected:
        raise ValueError("Accepted same500K normalization source changed")
    retained = torch.load(normalization_checkpoint, map_location="cpu", weights_only=True)["selected"]
    if (retained["mean"] != c["target_mean_eV"] or retained["std"] != c["target_std_eV"] or
            retained["binding"]["config"]["cpu_accepted_dataset_manifest_sha256"] != c["dataset_manifest_sha256"]):
        raise ValueError("Frozen train normalization does not match retained identity")
    payload = retained["payload"]
    source, target = payload["source_idx"], payload["target_eV"]
    if not torch.equal(source.long(), torch.arange(c["train_rows"], c["train_rows"] + c["development_rows"])):
        raise ValueError("Retained development identity changed")
    write_json(HERE / "normalization_source.json", {
        "origin_kernel": "nothingnessvoid/molgap-k1-single-ema-500k-t4-20261010",
        "origin_artifact": "k1_single_ema_500k/worker/reference/epoch_15.pt",
        "sha256": expected, "manifest_sha256": c["dataset_manifest_sha256"],
        "mean_eV": retained["mean"], "sample_std_eV": retained["std"],
        "usage": "transport accepted all500K train statistics; no trained weights reused",
    })
    policy_id = c["run_id"]
    policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-colab-execution-profile.1.json").read_text())
    policy.update(policy_id=policy_id, status="candidate",
                  comparability_selector={"scientific_contract": c["run_id"]},
                  created_from_source_digest=sha256_file(HERE / "protocol.md"))
    policy["action_rule"]["action"] = "RUN_PAIRED_500K"
    policy["cost_model"]["assumptions"] = ["Nine wall-hours and18 assigned T4 device-hours are ceilings, not observations."]
    policy_path = ROOT / "research_memory/policies" / (policy_id + ".1.json")
    write_json(policy_path, policy)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    roles = []
    for role, indices in (("train", torch.arange(c["train_rows"])), ("development", source)):
        digest = tensor_digest(indices, role="source_idx")
        usage = canonical_fingerprint({"role": role, "contract": c["run_id"], "membership": digest})
        roles.append(dict(role=role, membership_sha256=digest, row_order_sha256=digest, usage_sha256=usage))
    arms, bindings, preparation = [], [], []
    for device, (arm_id, mode, selection, decay) in enumerate((
            ("mean2", "mean2", "best-development-live", None),
            ("single_ema", "reference", "best-development-ema", c["ema_decay"]))):
        recipe = FamilyRecipe("neural_atom_k1", mode, c["seed"], c["initialization_sha256"],
            DatasetSelection(c["dataset"].split("/")[1] + "/manifest.json", c["dataset_manifest_sha256"], True),
            c["train_rows"], c["development_rows"], c["train_rows"] // c["batch_size"] * c["epochs"],
            c["learning_rate"], c["weight_decay"],
            ScheduleSpec(c["schedule"], c["epochs"], minimum_lr=c["minimum_lr"]),
            selection, c["sampler"], c["checkpoint_every"], c["target_mean_eV"], c["target_std_eV"],
            ema_decay=decay, batch_size=c["batch_size"], gradient_clip=c["gradient_clip"],
            finite_check_every=c["finite_check_every"], torch_version=c["torch_version"])
        record = build_family_recipe(INTERFACES[recipe.family],
            addon="k1_two_pass_mean" if mode == "mean2" else None, configuration=recipe,
            source_idx_sha256=tensor_digest(source, role="source_idx"),
            target_sha256=tensor_digest(target.float(), role="target"),
            train_role_identity=roles[0]["usage_sha256"], development_role_identity=roles[1]["usage_sha256"])
        recipe_rel = REL + "/training_recipe_" + arm_id + ".json"
        write_json(ROOT / recipe_rel, record)
        b = family_arm_bindings(recipe)
        arm = {"arm_id": arm_id, "scientific_role": "reference" if device == 0 else "candidate",
            "family": {"name": recipe.family, "version": INTERFACES[recipe.family][1]}, "base": b["base"],
            "initialization": {"kind": "frozen_state", "seed": c["seed"], "state_sha256": c["initialization_sha256"]},
            "data": {"dataset": reference("pcqm4mv2", c["dataset_manifest_sha256"]),
                "split": reference("fixed500k-next50k", canonical_fingerprint({"roles": roles})), "roles": deepcopy(roles),
                "feature_schema": "ogb-atom9-bond3-rwse16-v1", "feature_sha256": c["dataset_manifest_sha256"],
                "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": reference("shared-family-training", normalized_source_sha256(ROOT / recipe_rel)),
                "overrides": {}, **{k: b[k] for k in ("objective", "sampler", "transform")}},
            "addons": [], "addon_semantics": "baseline"}
        if mode == "mean2":
            arm.update(addon_semantics="ordered", addons=[{"name": "k1_two_pass_mean", "version": "1", "config": {},
                "source_sha256": normalized_source_sha256(ROOT / "src/molgap/k1_screen_training.py")}])
        arms.append(arm)
        tid = "TB-k1-mean2-single-ema-500k-20261013-" + arm_id
        cost_id = "cost-k1-mean2-single-ema-500k-20261013-" + arm_id + "-planned"
        action = {"action_id": "A001", "type": "paired_500k_training",
                  "run_ids": [c["run_id"] + ":" + arm_id + ":downstream"],
                  "attempt_ids": [c["run_id"] + "-v1"], "source_commit": commit,
                  "evidence_refs": [REL + "/config.json"], "cost_event_ids": [cost_id]}
        trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": tid, "record_mode": "prospective",
            "track": "B", "owner": "desktop", "family_id": "k1-mean2-single-ema-500k",
            "question": "Does single-forward EMA0.9999 improve500K cost/accuracy versus fresh mean2?",
            "hypothesis": {"hypothesis_id": "H-" + tid, "supporting_evidence_ids": [
                "pcqm-k1-clean-second-100k-s42-v1-mean2-terminal"],
                "observed_deficiency": "100K signal and execution-only savings do not establish500K recipe benefit.",
                "alternative_explanations": ["EMA lag or BN mismatch", "Loss of second-view regularization", "Seed/runtime variation"],
                "changed_mechanism": "Two-forward live mean2 versus one-forward stepwise EMA0.9999",
                "cheapest_falsifier": "One same-allocation500K pair under nine wall-hours",
                "decision_changed_if_positive": "Nominate a separate reviewed cost/accuracy decision, never automatically adopt",
                "decision_changed_if_negative": "Retain negative or partial ambiguity, no automatic continuation",
                "expected_native_cost_ref": cost_id, "related_closed_family_ids": ["k1-clean-second-view", "k1-t4-cost-quality"]},
            "state_at_start": {"source_commit": commit, "source_config_identity": canonical_fingerprint(arm),
                "contract_refs": [REL + "/protocol.md", REL + "/config.json", REL + "/evidence_review.md"],
                "reference_ids": [], "parent_trajectory_ids": [], "prior_evidence_ids": [
                    "pcqm-k1-clean-second-100k-s42-v1-mean2-terminal"],
                "role_snapshot_refs": [REL + "/protocol.md"], "budget_snapshot_ref": REL + "/config.json"},
            "actions": [action], "result": {"evidence_ids": [], "evidence_refs": []},
            "decision": {"outcome": "ACTIVE", "decision_ref": REL + "/decision.md",
                "next_allowed_actions": ["A001"], "reopen_conditions": []}}
        cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": tid,
            "action_id": "A001", "run_id": action["run_ids"][0], "attempt_id": action["attempt_ids"][0],
            "platform": "kaggle", "hardware": "Tesla T4", "category": "training", "evidence_ref": REL + "/protocol.md",
            "measurement": {k: {"value": 9.0 if k in ("wall_hours", "device_hours") else None,
                "status": "estimated" if k in ("wall_hours", "device_hours") else "measurement_missing"}
                for k in ("wall_hours", "device_hours", "cpu_hours", "queue_hours")}}
        plan = {"trajectory": trajectory, "costs": [cost], "decision_state": {
            "known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
            "available_actions": ["RUN_PAIRED_500K", "NO_TRAIN"], "chosen_action": "RUN_PAIRED_500K",
            "policy_id": policy_id, "policy_version": "1", "budget_snapshot_ref": REL + "/config.json",
            "role_snapshot_refs": [REL + "/protocol.md"], "source_commit": commit,
            "state_timestamp": datetime.now(timezone.utc).isoformat()}}
        plan_rel = REL + "/plan_" + arm_id + ".json"
        write_json(ROOT / plan_rel, plan)
        bindings.append({"arm_id": arm_id, "trajectory_id": tid, "plan_spec_ref": plan_rel,
                         "plan_spec_sha256": sha256_file(ROOT / plan_rel), "output": REL + "/rml/" + arm_id})
        preparation.append({"arm_id": arm_id, "device": device, "recipe": recipe_rel, "initial_state": str(initial.resolve())})
    spec = ExperimentSpec({"schema_version": SCHEMA_VERSION_V2, "experiment_id": c["run_id"],
        "logical_run_id": c["run_id"], "arms": arms,
        "platform": {k: c[k] for k in ("accelerator", "device_count", "cpu_cores", "memory_gib")}
                    | {"name": "kaggle", "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": {"arms": bindings, "same_run_replay": {"reference_arm_id": "mean2", "candidate_arm_ids": ["single_ema"]}},
        "evidence": {"policy": reference("molgap-v5", sha256_file(ROOT / "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md")),
                     "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": TERMINAL_PROTOCOL})
    atomic_json(HERE / "experiment_spec.json", spec.to_dict())
    return spec, {"format": "molgap-family-training-preparation-v1", "spec_identity": spec.identity,
        "source_files": [REL + "/config.json", *[REL + "/training_recipe_" + a["arm_id"] + ".json" for a in arms]],
        "arms": preparation}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--normalization-checkpoint", type=Path, required=True)
    p.add_argument("--stage", type=Path)
    args = p.parse_args()
    spec, plan = draft(args.initial, args.normalization_checkpoint)
    if args.stage is None:
        print(json.dumps({"status": "DRAFT_ONLY", "spec_identity": spec.identity}))
        return
    from molgap.experiment_workflow import prepare_family_training
    result = prepare_family_training(spec, ROOT, plan, args.stage)
    if result["status"] != "SOURCE_INPUTS_PREPARED":
        print(json.dumps(result))
        raise SystemExit(1)
    # Platform metadata over the shared prepared worker; no custom training loop.
    from molgap.kaggle_workflow import _metadata, freeze_inputs
    from molgap.experiment_launch import publish_immutable_bytes
    from molgap.experiment_preflight import check_release_inputs
    c = json.loads((HERE / "config.json").read_text())
    source = args.stage / "inputs"
    kernel = args.stage / "kernel"
    kernel.mkdir()
    metadata = _metadata({"account": c["account"], "kernel": c["account"] + "/" + c["title"].lower().replace(" ", "-"),
        "title": c["title"], "datasets": [c["source_dataset"], c["dataset"]],
        "source_dataset": c["source_dataset"], "accelerator": c["accelerator"]})
    publish_immutable_bytes(source / "dataset-metadata.json", canonical_json({"id": c["source_dataset"],
        "title": "K1 Mean2 Single EMA 500K Source", "licenses": [{"name": "CC0-1.0"}]}).encode())
    publish_immutable_bytes(kernel / "kernel-metadata.json", canonical_json(metadata).encode())
    manifest = json.loads((args.stage / "package/package_manifest.json").read_text())
    launch = {"format": "molgap-execution-launch-v1", "spec_identity": spec.identity,
        "expected_package_identity": manifest["package_identity"], "expected_source_archive_sha256": manifest["archive_sha256"],
        "account": c["account"], "run_reference": metadata["id"], "dataset_sources": metadata["dataset_sources"],
        "accelerator": c["accelerator"], "device_count": c["device_count"], "jobs": result["jobs"],
        "allocation_wall_limit_seconds": c["allocation_wall_limit_seconds"], "cleanup_reserve_seconds": c["cleanup_reserve_seconds"]}
    stage = {"input_root": source, "kernel_dir": kernel, "launch_path": source / "experiment_launch.json",
             "entry_path": kernel / "run.py", "launch": launch,
             "entry_text": (ROOT / "platforms/kaggle/run_experiment.py").read_text()}
    trajectories = {b["arm_id"]: ROOT / b["output"] / "trajectory.json" for b in spec.to_dict()["prospective"]["arms"]}
    freeze_inputs(stage, trajectories)
    release = check_release_inputs(spec, args.stage / "package", expected_package_identity=manifest["package_identity"],
        recipe_files={a["arm_id"]: a["recipe"] for a in plan["arms"]},
        initial_states={a["arm_id"]: source / "initial_states" / (a["arm_id"] + ".pt") for a in plan["arms"]},
        required_modules=["molgap.family_training_runner", "molgap.experiment_training_worker", "molgap.pcqm_wedge"],
        input_root=source, entry_script=kernel / "run.py", launch_config=stage["launch_path"],
        kernel_metadata=kernel / "kernel-metadata.json")
    write_json(args.stage / "release_report.json", release)
    print(json.dumps({"status": release["status"], "errors": release["errors"], "stage": str(args.stage)}))
    if release["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
