"""Author local declarations with existing owners; no publication or execution."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import struct


from molgap.experiment_execution import build_addon_declaration, build_family_recipe
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_recipe import ScheduleSpec
from molgap.family_training_runner import family_arm_bindings, read_recipe, validate_screen_recipe
from molgap.training.state import CalibrationPolicy
from molgap.v4_runtime import inspect_frozen_state_artifact

REL = "experiments/pcqm_k1_initialization_forward_500k"
ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / REL
TAIL = ROOT / "experiments/pcqm_k1_tail_regularization_500k"
TIMESTAMP = "2026-10-26T00:00:00+09:00"
STATES = {
    "scratch": ("D:/文档/molgap/experiments/pcqm_k1_flag/qualification/initial_state.pt",
        "7b088bad6359804309a1cd98792cc4ef99f2fcc3051b9e2f5a37916c3c83c242",
        "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd"),
    "pretrained": ("D:/molgap-cleanup-retained-20261017/final-retirement/objects/9656064f7bb0256881da4bc063e90de135171abf517a486bf6202d774694fdba",
        "9656064f7bb0256881da4bc063e90de135171abf517a486bf6202d774694fdba",
        "7458f5c4776208da1715d39cbce66537727ff8054b30da0ea07d82d0e94181f4"),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    _, old = read_recipe(TAIL / "base_recipe.json")
    template = read(TAIL / "data_declaration.json")
    source_arm = template["arms"][0]
    roles = {r["role"]: r for r in source_arm["data"]["roles"]}
    manifest = Path("D:/molgap-data/pcqm_fixed_500k_v1/manifest.json")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != old.dataset.manifest_sha256:
        raise ValueError("Frozen 500K manifest changed")
    for path, file_sha, state_sha in STATES.values():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != file_sha:
            raise ValueError("Initialization file changed: " + path)
        inspect_frozen_state_artifact(Path(path), expected_state_sha256=state_sha)
    target = read(TAIL / "target_manifest.json")
    target["training_float32_target_sha256"] = "cef6fbb56a546b1808b2a42e94b8a08354d23928512be23de9f1c8b8f66f3676"
    write(REL + "/target_manifest.json", target)
    role_plan = read(TAIL / "role_plan.json")
    role_plan.update(fixed_train_followup="Only separately authorized retained-artifact analysis after completion.",
        clean_bn_calibration={"role": "train", "source_idx_start": 0, "source_idx_stop": 16384,
                              "batches": 128, "batch_size": 128},
        development_usage="Per-pass Clean-EMA selection, live reporting, and within-slot paired terminal comparison.",
        protocol_ref=REL + "/protocol.md", actual_consumption="pending")
    write(REL + "/role_plan.json", role_plan)
    calibration = CalibrationPolicy(input_identity=old.dataset.manifest_sha256,
        training_role_identity=roles["train"]["usage_sha256"],
        source_order_sha256=hashlib.sha256(struct.pack("<16384q", *range(16384))).hexdigest(), batches=128)
    for slot, addon in (("single", None), ("mean2", "k1_two_pass_mean")):
        slot_rel = REL + "/" + slot
        run = "pcqm-k1-init-forward-500k-" + slot + "-s42"
        arms = []
        for name, (initial, _, state_sha) in STATES.items():
            recipe = replace(old, mode="reference" if addon is None else "mean2",
                initialization_sha256=state_sha, optimizer_steps=117180, learning_rate=4e-4,
                weight_decay=1e-5, schedule=ScheduleSpec("cosine_epoch", 30, 1e-6),
                selection="best-development-clean-ema", development_reports=("live",),
                clean_bn=calibration, resume_chunks=True)
            record = build_family_recipe(("neural_atom_k1", "3"), addon=addon,
                configuration=recipe, source_idx_sha256=roles["development"]["row_order_sha256"],
                target_sha256=target["development_target_sha256"],
                train_role_identity=roles["train"]["usage_sha256"],
                development_role_identity=roles["development"]["usage_sha256"],
                logging={"mode": "epoch", "interval": None}, resume_keep_count=2)
            recipe_ref = slot_rel + "/" + name + "/recipe.json"
            recipe_sha = write(recipe_ref, record)
            arm = deepcopy(source_arm)
            arm.update(arm_id=name, scientific_role="reference" if name == "scratch" else "candidate",
                addon_semantics="baseline" if addon is None else "ordered",
                addons=[] if addon is None else [build_addon_declaration(("neural_atom_k1", "3"), addon, repo_root=ROOT)],
                initialization={"kind": "frozen_state", "seed": 42, "state_sha256": state_sha})
            binding = family_arm_bindings(recipe)
            arm["base"] = binding["base"]
            arm["training"].update({k: binding[k] for k in ("sampler", "transform", "objective")})
            arm["training"]["recipe"]["sha256"] = recipe_sha
            declaration = deepcopy(template)
            declaration.update(experiment_id="pcqm_k1_initialization_forward_500k", logical_run_id=run, arms=[arm])
            declaration["prospective"] = {"arms": [{"arm_id": name,
                "trajectory_id": "declaration-only-" + slot + "-" + name,
                "plan_spec_ref": slot_rel + "/authority.json", "plan_spec_sha256": "0" * 64,
                "output": slot_rel + "/declaration_only/" + name}]}
            arms.append({"arm_id": name, "recipe_ref": recipe_ref,
                "data_ref": {"spec_ref": slot_rel + "/" + name + "/data_declaration.json", "arm_id": name},
                "initial_state_ref": initial})
            # Authority is written below; bind its exact bytes before writing data.
            arm["training"]["recipe"]["name"] = "shared-family-training"
            arms[-1]["_declaration"] = declaration
            validate_screen_recipe(ExperimentSpec(declaration), name, record)
        authority = {"format": "molgap-execution-authority-v1", "user_authorized": True,
            "authorized_at": TIMESTAMP, "account": "nothingnessvoid", "arms": ["scratch", "pretrained"],
            "seeds": [42], "same_run_reference_arm_id": "scratch", "allocated_t4_hours_ceiling": 18,
            "allocation_wall_limit_seconds": 32400, "cleanup_reserve_seconds": 180,
            "automatic_retry": False, "protocol_ref": REL + "/protocol.md",
            "scope": "One of two authorized Kaggle1 T4x2 slots: " + slot + "; 500K30 passes per arm; initialization comparison only. Total two-slot ceiling36 allocated T4 hours; no automatic retry, full training, protected roles or adoption.",
            "reference_semantics": "Scratch is a new explicitly released within-slot comparator; retained100K pretrained initialization changes only initial model state. Cross-slot comparisons are descriptive auxiliary evidence."}
        authority_sha = write(slot_rel + "/authority.json", authority)
        for arm in arms:
            declaration = arm.pop("_declaration")
            declaration["prospective"]["arms"][0]["plan_spec_sha256"] = authority_sha
            write(arm["data_ref"]["spec_ref"], declaration)
        experiment = read(TAIL / "experiment.json")
        experiment.update(experiment_id="pcqm_k1_initialization_forward_500k", logical_run_id=run,
            authority_ref=slot_rel + "/authority.json", decision_ref=REL + "/protocol.md",
            role_ref=REL + "/role_plan.json", arms=arms, timestamp=TIMESTAMP,
            monitor_paths=[REL + "/STATUS.md"],
            same_run={"reference_arm_id": "scratch", "target_manifest_ref": REL + "/target_manifest.json"},
            shared={"model": {"family": "neural_atom_k1", "version": "3", "addons": [] if addon is None else [addon]},
                    "purpose": "initialization_comparison", "training": {}})
        kaggle = experiment["platform"]["kaggle"]
        source_dataset = "nothingnessvoid/k1-init-500k-" + slot + "-src"
        kaggle.update(allocation_wall_limit_seconds=32400, cleanup_reserve_seconds=180,
            source_dataset=source_dataset, kernel="nothingnessvoid/molgap-k1-init-500k-" + slot + "-s42",
            title="MolGap K1 Init 500K " + slot.title() + " S42",
            datasets=[source_dataset, "nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1"])
        experiment["profile"].update(estimated_device_hours=18, estimated_wall_hours=9,
            cheapest_falsifier="Matched scratch/pretrained pair within one forward mode at fixed500K30 passes; terminal acceptance before any successor.")
        experiment["question"].update(action="RUN_INITIALIZATION_PAIR_500K", policy_id=run,
            question="Does retained100K pretrained initialization improve fixed500K K1 " + slot + " development versus scratch?",
            changed_mechanism="retained-pretrained-initialization",
            deficiency="Retained100K pretraining has no strict matched500K initialization comparison under this forward mode.",
            alternatives=["Initial representation changes optimization", "Development transfer fails", "Single-seed training variability"],
            closed_families=["K1 tail regularization500K", "K1 consistency500K"],
            positive_decision="Nomination only after >=1meV gain and paired95%CI excludes zero within this slot; no automatic adoption.",
            negative_decision="Close or retain inconclusive evidence under the frozen contract; no automatic retry or scale-up.")
        experiment["question"]["supporting_evidence"] = ["pcqm-k1-scale-attribution-20261025-attempt002"]
        write(slot_rel + "/experiment.json", experiment)
        policy = read(ROOT / "research_memory/policies/pcqm-k1-tail-regularization-500k-s42.1.json")
        policy.update(policy_id=run, created_from_source_digest=hashlib.sha256((HERE / "protocol.md").read_bytes()).hexdigest())
        policy["comparability_selector"]["scientific_contract"] = run
        policy["action_rule"]["action"] = "RUN_INITIALIZATION_PAIR_500K"
        write("research_memory/policies/" + run + ".1.json", policy)
    write(REL + "/initialization_provenance.json", read(ROOT / "experiments/pcqm_k1_500k_bottleneck_diagnostic/input_authority/initialization_provenance.snapshot.json"))
    print("Authored two scientific inputs and four recipe/data declarations; no prospective publication or execution.")


if __name__ == "__main__":
    main()
