"""Setup initial states, training recipes, prospective plans, and freeze comparison declarations for K1 norm ablation."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import torch

from molgap.execution_comparison import (
    ExecutionQuestion, ExecutionProfile, freeze_comparison_declarations, build_execution_proposal,
)
from molgap.experiment_catalog import FAMILIES
from molgap.experiment_execution import build_family_recipe
from molgap.family_training_data import DatasetSelection
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_runner import INTERFACES, family_arm_bindings, read_recipe
from molgap.k1_ada_rmsnorm import make_k1_ada_rmsnorm
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file
from molgap.v4_runtime import (
    inspect_frozen_state_artifact, normalized_source_sha256,
    read_frozen_state_artifact, state_dict_sha256,
)

from molgap.constants import REPO_ROOT as ROOT

HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
STORAGE = ROOT / "platforms/_records/kaggle/staging/pcqm_k1_norm_ablation_100k"


def setup_initial_states():
    initial_dir = STORAGE / "initial_states"
    initial_dir.mkdir(parents=True, exist_ok=True)

    # 1. k1_single_bn
    bn_dest = initial_dir / "k1_single_bn.pt"
    src_bn = Path("D:/w/family-topology-t4-100k/platforms/_records/kaggle/staging/family_topology_t4_100k/release002/source_dataset/initial_states/k1_mean2.pt")
    if not bn_dest.exists() or sha256_file(bn_dest) != sha256_file(src_bn):
        shutil.copyfile(src_bn, bn_dest)
    inspect_frozen_state_artifact(bn_dest, expected_state_sha256=CONFIG["arms"]["k1_single_bn"]["initialization_sha256"])

    # 2. k1_ada_rmsnorm
    ada_dest = initial_dir / "k1_ada_rmsnorm.pt"
    if not ada_dest.exists():
        bn_state, _ = read_frozen_state_artifact(bn_dest, expected_state_sha256=CONFIG["arms"]["k1_single_bn"]["initialization_sha256"])
        ada_model = make_k1_ada_rmsnorm()
        ada_state = ada_model.state_dict()
        for k in ada_state:
            if k in bn_state:
                ada_state[k].copy_(bn_state[k])
            elif k.endswith(".alpha"):
                ada_state[k].zero_()
            elif k.endswith(".beta"):
                ada_state[k].fill_(1.0)
            else:
                raise ValueError("Unexpected key: " + k)
        assert state_dict_sha256(ada_state) == CONFIG["arms"]["k1_ada_rmsnorm"]["initialization_sha256"]
        atomic_torch_save(ada_dest, ada_state)
    inspect_frozen_state_artifact(ada_dest, expected_state_sha256=CONFIG["arms"]["k1_ada_rmsnorm"]["initialization_sha256"])
    print("Initial states verified successfully.")


def setup_recipes():
    roles = json.loads((HERE / "role_plan.json").read_text(encoding="utf-8"))["roles"]
    source_idx_sha256 = roles[1]["row_order_sha256"]  # development role row order = 3270b300...
    target_sha256 = "a1c0b711815225f0bccee905419d124174e1bf01c76877f5c661b063ee170137"

    ds = DatasetSelection("pcqm4mv2-ogb-fixed-100k-v1/manifest.json", CONFIG["dataset_manifest_sha256"], True)

    for arm_id, arm in CONFIG["arms"].items():
        recipe = FamilyRecipe(
            arm["family"], arm["mode"], CONFIG["seed"], arm["initialization_sha256"],
            ds, CONFIG["train_rows"], CONFIG["development_rows"],
            CONFIG["train_rows"] // 128 * arm["epochs"],
            arm["learning_rate"], arm["weight_decay"],
            ScheduleSpec(arm["schedule"], arm["epochs"], warmup_units=arm["warmup_units"]),
            arm["selection"], arm["sampler"], 128,
            arm["target_mean_eV"], arm["target_std_eV"],
            ema_decay=arm["ema_decay"], torch_version=CONFIG["torch_version"],
        )
        record = build_family_recipe(
            INTERFACES[recipe.family], addon=arm["addon"], configuration=recipe,
            source_idx_sha256=source_idx_sha256, target_sha256=target_sha256,
            train_role_identity=roles[0]["usage_sha256"],
            development_role_identity=roles[1]["usage_sha256"],
        )
        recipe_path = HERE / f"training_recipe_{arm_id}.json"
        recipe_path.write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
        print(f"Saved {recipe_path.name}")


def setup_plans_and_declarations():
    roles = json.loads((HERE / "role_plan.json").read_text(encoding="utf-8"))["roles"]
    prior = "TB-family-topology-t4-100k-k1_mean2-s42-terminal-20261014"
    ref = lambda name, digest: {"name": name, "version": "1", "sha256": digest}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    arms_spec = []
    bindings = []
    jobs = []
    entries = []

    question = ExecutionQuestion(
        policy_id=CONFIG["run_id"],
        action="RUN_NORMALIZATION_ABLATION",
        question="Does replacing K1 BatchNorm with AdaRMSNorm eliminate EMA running-stat degradation without degrading single-forward PCQM4Mv2 Gap MAE?",
        deficiency="K1 BatchNorm suffers from EMA running-mean/var drift and requires Clean-BN recalibration.",
        changed_mechanism="Replace 18 BatchNorm1d layers in 9 LocalGPSBlocks with dimension-matched AdaRMSNorm (alpha=0, beta=1 init).",
        supporting_evidence=(prior,),
        alternatives=("LayerNorm substitution", "Static RMSNorm without scale/shift adaptation", "Frozen BatchNorm"),
        closed_families=("family-topology-t4-execution", "family-100k-operator-profile"),
        positive_decision="Adopt AdaRMSNorm as standard normalization layer for K1 and future PPGT architectures if non-inferiority ΔMAE >= -0.5 meV is achieved.",
        negative_decision="Retain BatchNorm baseline and investigate alternative norm/residual formulations.",
        reference_ids=(),
        prior_trajectories=("TB-family-topology-t4-100k-k1_mean2-s42",),
    )

    profile = ExecutionProfile(
        owner="desktop",
        track="B",
        platform="kaggle1",
        hardware="Tesla_T4x2_shared_allocation",
        cheapest_falsifier="one matched seed42 T4x2 execution diagnostic comparing single-forward BN vs AdaRMSNorm",
        estimated_device_hours=8.0,
        estimated_wall_hours=4.0,
    )

    for device_idx, (arm_id, arm) in enumerate(CONFIG["arms"].items()):
        recipe_ref = REL + "/training_recipe_" + arm_id + ".json"
        record, recipe = read_recipe(ROOT / recipe_ref)
        b = family_arm_bindings(recipe)
        module = "k1_screen_training.py" if arm["addon"] is None else "k1_ada_rmsnorm.py"
        arm_entry = {
            "arm_id": arm_id,
            "scientific_role": "reference" if arm_id == "k1_single_bn" else "candidate",
            "family": {"name": recipe.family, "version": INTERFACES[recipe.family][1]},
            "base": b["base"],
            "initialization": {"kind": "frozen_state", "seed": recipe.seed, "state_sha256": recipe.initialization_sha256},
            "data": {
                "dataset": ref("pcqm4mv2", CONFIG["dataset_manifest_sha256"]),
                "split": ref("fixed100k-next50k", canonical_fingerprint({"roles": roles})),
                "roles": copy.deepcopy(roles),
                "feature_schema": FAMILIES[INTERFACES[recipe.family]].feature_schema,
                "feature_sha256": CONFIG["dataset_manifest_sha256"],
                "target": "pcqm4mv2-gap-eV-direct",
            },
            "training": {
                "recipe": ref("shared-family-training", normalized_source_sha256(ROOT / recipe_ref)),
                "overrides": {},
                **{k: b[k] for k in ("objective", "sampler", "transform")},
            },
            "addons": [] if arm["addon"] is None else [{
                "name": arm["addon"], "version": "1", "config": {},
                "source_sha256": normalized_source_sha256(ROOT / "src/molgap" / module),
            }],
            "addon_semantics": "ordered" if arm["addon"] else "baseline",
        }
        arms_spec.append(arm_entry)

        tid = f"TB-pcqm-k1-norm-ablation-100k-s42-{arm_id}"
        plan_data = build_execution_proposal(
            question=question,
            arm=arm_entry,
            trajectory_id=tid,
            source_commit=commit,
            run=CONFIG["run_id"],
            version="v1",
            contract_ref=REL + "/config.json",
            contract_refs=[REL + "/protocol.md", REL + "/config.json", REL + "/evidence_review.md"],
            authority_ref=REL + "/config.json",
            role_ref=REL + "/role_plan.json",
            decision_ref=REL + "/decision.md",
            timestamp=datetime.now(timezone.utc).isoformat(),
            profile=profile,
        )
        # Update attempt IDs for same_run_replay compliance
        plan_data["trajectory"]["actions"][0]["attempt_ids"] = [f"{CONFIG['run_id']}-v1"]
        plan_data["costs"][0]["attempt_id"] = f"{CONFIG['run_id']}-v1"

        plan_ref = REL + f"/plan_{arm_id}.json"
        atomic_json(ROOT / plan_ref, plan_data)
        plan_sha = sha256_file(ROOT / plan_ref)
        bindings.append({
            "arm_id": arm_id,
            "trajectory_id": tid,
            "plan_spec_ref": plan_ref,
            "plan_spec_sha256": plan_sha,
            "output": REL + "/rml/" + arm_id,
        })
        jobs.append({
            "arm_id": arm_id,
            "device": device_idx,
            "recipe": recipe_ref,
            "initial_state": str(STORAGE / "initial_states" / f"{arm_id}.pt"),
        })
        entries.append({
            "arm_id": arm_id,
            "adapter": "k1-session-v1",
            "contract": {
                "path": recipe_ref,
                "sha256": normalized_source_sha256(ROOT / recipe_ref),
            },
            "expected": record["acceptance_requirements"],
            "target_manifest": {
                "path": "experiments/pcqm_k1_slot_width96/reference_binding/target_manifest.json",
                "sha256": "9c5b2f4a161cc3d6539cfa7bca89da37773276b201d775b1ab88300ac4c9de10",
            },
        })

    freeze_comparison_declarations(
        ROOT,
        base=REL,
        experiment_id=CONFIG["run_id"],
        logical_run_id=CONFIG["run_id"],
        arms=arms_spec,
        bindings=bindings,
        jobs=jobs,
        entries=entries,
        plan_format="molgap-family-same-run-acceptance-plan-v1",
        authority_ref=REL + "/authority.json",
        source_extras=[REL + "/config.json", REL + "/protocol.md", REL + "/evidence_review.md", REL + "/role_plan.json"],
        account=CONFIG["account"],
        kernel=CONFIG["kernel"],
        title=CONFIG["title"],
        source_dataset=CONFIG["source_dataset"],
        data_datasets=[CONFIG["dataset"]],
        same_run_replay={"reference_arm_id": "k1_single_bn", "candidate_arm_ids": ["k1_ada_rmsnorm"]},
        dataset_manifests={
            "k1_single_bn": "data/pcqm_fixed_100k_v1/manifest.json",
            "k1_ada_rmsnorm": "data/pcqm_fixed_100k_v1/manifest.json",
        },
    )
    print("Envelopes frozen successfully.")


if __name__ == "__main__":
    setup_initial_states()
    setup_recipes()
    setup_plans_and_declarations()
