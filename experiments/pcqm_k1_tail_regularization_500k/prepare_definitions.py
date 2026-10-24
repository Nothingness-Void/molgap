"""Build this question's declarations with the shared family recipe owner."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

import torch

from molgap.experiment_execution import build_family_recipe
from molgap.family_training_data import DatasetSelection
from molgap.family_training_kernel import load_family_model
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_runner import family_arm_bindings
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import atomic_torch_save, sha256_file
from molgap.experiment_family_workflow import _publish as atomic_json
from molgap.v4_runtime import state_dict_sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
RUN = "pcqm-k1-tail-regularization-500k-s42"
ARMS = ["tail_wd1e3", "tail_wd1e2"]
PRIMARY = Path("D:/文档/molgap")
PARENT = Path("D:/molgap-cleanup-retained-20261017/final-retirement/objects/2e9cee13ebbf67031470e89acaf46d551238f937255fed601eb0922b93fdad92")
PREDICTIONS = PRIMARY / "experiments/pcqm_k1_consistency_ablation_500k/submission_kaggle3_v1/terminal_inspection_20261009/evidence/stages/k1_pretrained_consistency/best_predictions.pt"
MANIFEST_SHA = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"


def main():
    assert sha256_file(PARENT) == PARENT.name
    assert sha256_file(PREDICTIONS) == "57463e4e578db6310eadfd2b46d9c57f50d0ef7e2aa12c3f883ace72e7265f55"
    parent = torch.load(PARENT, map_location="cpu", weights_only=True)
    predictions = torch.load(PREDICTIONS, map_location="cpu", weights_only=True)
    assert parent["epoch"] == 48 and parent["contract"]["parameters"] == 3658817
    state_sha = state_dict_sha256(parent["model"])
    initial = Path("D:/molgap-artifacts/initializations/k1-tail-regularization-500k/selected_epoch49.pt")
    initial.parent.mkdir(parents=True, exist_ok=True)
    if initial.exists():
        assert state_dict_sha256(torch.load(initial, map_location="cpu", weights_only=True)["model_state"]) == state_sha
    else:
        atomic_torch_save(initial, {"model_state": parent["model"], "state_sha256": state_sha,
            "reference_development": {"prediction_eV": predictions["prediction"],
                "target_eV": predictions["target"], "source_idx": predictions["source_idx"],
                "mae_eV": predictions["mae_eV"]}})
    manifest = PRIMARY / "experiments/pcqm_gptrans_noisy_nodes_500k/results/data_manifest.json"
    assert sha256_file(manifest) == MANIFEST_SHA
    staged = Path("D:/molgap-data/pcqm_fixed_500k_v1/manifest.json")
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists():
        assert sha256_file(staged) == MANIFEST_SHA
    else:
        shutil.copyfile(manifest, staged)
    timestamp = datetime.now(timezone.utc).isoformat()
    authority = {"format": "molgap-execution-authority-v1", "user_authorized": True,
        "authorized_at": timestamp, "account": "nothingnessvoid", "arms": ARMS, "seeds": [42],
        "allocated_t4_hours_ceiling": 12, "allocation_wall_limit_seconds": 21600,
        "same_run_reference_arm_id": ARMS[0],
        "scope": "User approved 500K20pass warm-start pair, WD1e-3 versus1e-2; one T4x2 job; shutdown after actual training starts. No full/protected roles/retry/adoption.",
        "reference_semantics": "Two new tail recipes; moderate WD candidate is the within-job comparator, not a retrained original baseline."}
    atomic_json(HERE / "authority.json", authority)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    role_entries = []
    for role, (start, stop) in {"train": (0, 500000), "development": (500000, 550000)}.items():
        ids = torch.arange(start, stop, dtype=torch.int64)
        membership = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        usage = canonical_fingerprint({"role": role, "logical_run_id": RUN, "source_idx": [start, stop],
            "use": "gradient_and_fixed_cohort_diagnostics" if role == "train" else "initial_reproduction_and_live_selection_and_shadow_ema"})
        role_entries.append({"role": role, "membership_sha256": membership,
            "row_order_sha256": membership, "usage_sha256": usage})
    source_sha = hashlib.sha256(predictions["source_idx"].numpy().tobytes()).hexdigest()
    target_sha = hashlib.sha256(predictions["target"].numpy().tobytes()).hexdigest()
    recipe = FamilyRecipe("neural_atom_k1", "reference", 42, state_sha,
        DatasetSelection("pcqm_fixed_500k_v1/manifest.json", MANIFEST_SHA, True),
        500000, 50000, 78120, 1e-4, 1e-3, ScheduleSpec("cosine_epoch", 20, minimum_lr=1e-5),
        "best-development-live", "torch-pass-randperm", 128,
        float(torch.tensor(parent["mean"], dtype=torch.float32)),
        float(torch.tensor(parent["std"], dtype=torch.float32)),
        ema_decay=.999, torch_version="2.11.0", resume_chunks=True,
        development_reports=("ema",))
    model, initialization = load_family_model(recipe, initial)
    assert sum(p.numel() for p in model.parameters()) == 3658817
    record = build_family_recipe(("neural_atom_k1", "3"), addon=None, configuration=recipe,
        source_idx_sha256=source_sha, target_sha256=target_sha,
        train_role_identity=role_entries[0]["usage_sha256"],
        development_role_identity=role_entries[1]["usage_sha256"],
        logging={"mode": "epoch", "interval": None}, resume_keep_count=2)
    record["initial_development_check"] = {"prediction_tolerance_eV": 1e-4, "mae_tolerance_eV": 1e-6}
    atomic_json(HERE / "base_recipe.json", record)
    recipe_sha = sha256_file(HERE / "base_recipe.json")
    bindings = family_arm_bindings(recipe)
    # Reuse only the unchanged V5 policy declaration, not another experiment's arms.
    evidence = json.loads((PRIMARY / "experiments/pcqm_structural_bias_norm_100k/k1/experiment_spec.json").read_text(encoding="utf-8"))["evidence"]
    platform = {"name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2,
        "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True}
    arm = {"arm_id": ARMS[0], "family": {"name": "neural_atom_k1", "version": "3"},
        "scientific_role": "reference", "base": bindings["base"], "addons": [], "addon_semantics": "baseline",
        "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": state_sha},
        "data": {"dataset": {"name": "pcqm4mv2", "version": "1", "sha256": MANIFEST_SHA},
            "split": {"name": "fixed500k-next50k", "version": "1", "sha256": canonical_fingerprint(manifest_value["roles"])},
            "feature_schema": "ogb-atom9-bond3-rwse16-v1", "feature_sha256": MANIFEST_SHA,
            "roles": role_entries, "target": "pcqm4mv2-gap-eV-direct"},
        "training": {"recipe": {"name": "shared-family-training", "version": "1", "sha256": recipe_sha},
            **{key: bindings[key] for key in ("sampler", "transform", "objective")}, "overrides": {}}}
    atomic_json(HERE / "data_declaration.json", {"schema_version": "molgap-experiment-spec-v2",
        "experiment_id": "pcqm_k1_tail_regularization_500k", "logical_run_id": RUN, "arms": [arm],
        "platform": platform, "evidence": evidence, "terminal_protocol": "molgap-experiment-terminal-descriptor-v1",
        "prospective": {"arms": [{"arm_id": ARMS[0], "trajectory_id": "declaration-only-" + ARMS[0],
            "plan_spec_ref": REL + "/authority.json", "plan_spec_sha256": sha256_file(HERE / "authority.json"),
            "output": REL + "/declaration_only"}]}})
    atomic_json(HERE / "target_manifest.json", {"development_target_sha256": target_sha,
        "target": "pcqm4mv2-gap-eV-direct", "target_encoding": "little-endian-float32-contiguous-raw-bytes"})
    atomic_json(HERE / "initialization_provenance.json", {"parent_checkpoint_sha256": PARENT.name,
        "parent_epoch_zero_based": 48, "parent_optimizer_steps": 191394, "parent_presentations": 24498432,
        "parent_source_sha256": parent["source_sha256"], "parent_contract": parent["contract"],
        "initialization": initialization, "initial_state_local_path": initial.as_posix(),
        "parent_predictions_sha256": sha256_file(PREDICTIONS), "parent_mae_eV": predictions["mae_eV"],
        "parent_transform": {"mean": parent["mean"], "std": parent["std"]},
        "execution_transform": {"mean": recipe.target_mean_eV, "std": recipe.target_std_eV},
        "transform_note": "Explicit FP32 scalar encoding preserves the source's FP32 tensor arithmetic; native initial prediction gate verifies equivalence.",
        "optimizer_reset": True, "original_optimizer_resume": False,
        "parent_missing_qualifications_inherited": "No parent strict replay/cost exclusions are repaired by this new pair."})
    atomic_json(HERE / "role_plan.json", {"planned": True, "train": manifest_value["roles"]["train"],
        "development": manifest_value["roles"]["development"], "development_previously_consumed": True,
        "role_identities": role_entries, "official_validation": "sealed", "test": "sealed",
        "geometry_model_inputs": False, "train_metric": "stochastic normalized objective, not clean fit",
        "fixed_train_followup": "Same frozen16384TRAIN cohort at start/selected/final, using retained artifacts after completion; no new training."})
    value = {"format": "molgap-scientific-experiment-v1", "experiment_id": "pcqm_k1_tail_regularization_500k",
        "logical_run_id": RUN, "attempt_id": "v1", "machine_profile": "D:/molgap-machine-profile.json",
        "timestamp": timestamp, "monitor_paths": [REL + "/STATUS.md"], "authority_ref": REL + "/authority.json",
        "role_ref": REL + "/role_plan.json", "decision_ref": REL + "/decision.md", "seeds": [42],
        "question": {"policy_id": RUN, "action": "RUN_PAIRED_500K_TAIL",
            "question": "Does stronger late decoupled weight decay improve K1 generalization from the same selected500K predictor?",
            "deficiency": "Accepted49-to60 fixed clean TRAIN improves0.538meV while development worsens0.787meV; additional exposure alone is not supported.",
            "changed_mechanism": "late-decoupled-weight-decay",
            "supporting_evidence": ["pcqm-k1-clean-fit-generalization-500k-20261009"],
            "alternatives": ["Both decay levels insufficient", "Strong decay underfits", "BN state dominates", "One-seed variability"],
            "closed_families": ["K1 late weight averaging", "K1 consistency500K under material gate", "blind slot-width expansion"],
            "positive_decision": "Nominate this warm-start recipe only after valid paired evidence; no automatic full/adoption.",
            "negative_decision": "Close this tail recipe; do not add epochs or retry seeds blindly.", "reference_ids": [], "prior_trajectories": []},
        "profile": {"owner": "desktop", "track": "B", "platform": "kaggle", "hardware": "Tesla T4 FP32",
            "cheapest_falsifier": "Two20pass warm-start recipes; no from-scratch baseline or new model module.",
            "estimated_device_hours": 4.5, "estimated_wall_hours": 4.5},
        "platform": {"platform": platform, "kaggle": {"account": "nothingnessvoid",
            "kernel": "nothingnessvoid/molgap-k1-tail-wd-500k-s42", "title": "MolGap K1 Tail WD 500K S42",
            "datasets": ["nothingnessvoid/k1-tail-wd-500k-src", "nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1"],
            "source_dataset": "nothingnessvoid/k1-tail-wd-500k-src", "accelerator": "NvidiaTeslaT4",
            "allocation_wall_limit_seconds": 21600, "cleanup_reserve_seconds": 180}},
        "same_run": {"reference_arm_id": ARMS[0], "target_manifest_ref": REL + "/target_manifest.json"},
        "shared": {"model": {"family": "neural_atom_k1", "version": "3", "addons": []},
            "recipe_ref": REL + "/base_recipe.json", "data_ref": {"spec_ref": REL + "/data_declaration.json", "arm_id": ARMS[0]},
            "initial_state_ref": initial.as_posix(), "training": {}, "purpose": "optimizer_comparison"},
        "arms": [{"arm_id": ARMS[0]}, {"arm_id": ARMS[1], "training": {"weight_decay": .01}}]}
    atomic_json(HERE / "experiment.json", value)
    policy = json.loads((PRIMARY / "research_memory/policies/pcqm-k1-late-weight-average.1.json").read_text(encoding="utf-8"))
    policy.update(policy_id=RUN, comparability_selector={"scientific_contract": RUN},
        created_from_source_digest=sha256_file(HERE / "protocol.md"),
        required_observable_fields=["user_requested_scope_reviewed"],
        action_rule={"action": "RUN_PAIRED_500K_TAIL", "field": "user_requested_scope_reviewed", "operator": "eq", "threshold": 1})
    atomic_json(ROOT / f"research_memory/policies/{RUN}.1.json", policy)
    print(json.dumps({"status": "DEFINITIONS_WRITTEN", "initial_state_sha256": state_sha,
        "initial_state_bytes": initial.stat().st_size, "model_parameters": 3658817}))


if __name__ == "__main__":
    main()
