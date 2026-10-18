"""Bounded mature-family workflow qualification, not a model comparison.

This adapter binds existing recipes, inspectors and allocation controls. It
does not own a training loop, optimizer, sampler, submitter or data cache.
"""
from pathlib import Path

from .experiment_family_workflow import _artifact_path, _json
from .training_reproducibility import sha256_file

PLAN_FORMAT = "molgap-family-workflow-trial-plan-v1"
EXPERIMENT_ID = "family-workflow-trial-100k"
INTERFACES = {"k1": ("neural_atom_k1", "3"), "gptrans": ("gptrans_t", "2")}
FIXED_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
WALL_SECONDS = 1800


def is_trial(spec):
    return spec.to_dict()["experiment_id"] == EXPERIMENT_ID


def plan_path(spec):
    if not is_trial(spec):
        raise ValueError("Not a family workflow trial")
    parents = {str(Path(b["plan_spec_ref"]).parent.parent).replace("\\", "/")
               for b in spec.to_dict()["prospective"]["arms"]}
    if len(parents) != 1:
        raise ValueError("Workflow trial needs one prospective owner")
    return parents.pop() + "/acceptance_plan.json"


def _pinned(root, pointer):
    if type(pointer) is not dict or set(pointer) != {"path", "sha256"}:
        raise ValueError("Expected a pinned trial record")
    path = _artifact_path(Path(root).absolute(), pointer["path"])
    if path.stat().st_size > 1024 ** 2 or sha256_file(path) != pointer["sha256"]:
        raise ValueError("Workflow trial metadata identity changed")
    return path


def check_plan(spec, root, plan):
    """Validate a train-only execution endpoint without inventing a reference."""
    from .family_training_runner import read_recipe, validate_screen_recipe
    declaration = spec.to_dict()
    if (not is_trial(spec) or type(plan) is not dict or
            set(plan) != {"format", "spec_identity", "authority", "arms"} or
            plan["format"] != PLAN_FORMAT or plan["spec_identity"] != spec.identity):
        raise ValueError("Workflow trial plan/Spec differs")
    platform = declaration["platform"]
    if (platform["name"] != "kaggle" or platform["device_count"] != 2 or
            platform["accelerator"] != "NvidiaTeslaT4"):
        raise ValueError("Workflow trial requires the authorized two T4 allocation")
    authority = _json(_pinned(root, plan["authority"]))
    required = {"purpose": "WORKFLOW_EXECUTION_TRIAL", "user_authorized": True,
        "arms": list(INTERFACES), "seeds": [42], "wall_ceiling_seconds": WALL_SECONDS,
        "physical_batch_size": 128, "optimizer_steps_per_arm": 32,
        "account": "kaseichou", "allocated_devices": 2,
        "allocated_t4_hours_ceiling": 1, "checkpoint_reserve_seconds": 60,
        "automatic_successor_authorized": False, "development_role_access": False,
        "protected_role_access": False}
    for key, value in required.items():
        if (authority.get(key) != value or
                (isinstance(value, (bool, int)) and type(authority.get(key)) is not type(value))):
            raise ValueError("Workflow trial authority or budget differs: " + key)
    arms = declaration["arms"]
    if ([a["arm_id"] for a in arms] != list(INTERFACES) or
            type(plan["arms"]) is not list or
            [e.get("arm_id") for e in plan["arms"]] != list(INTERFACES)):
        raise ValueError("Workflow trial must bind exactly K1 and GPTrans")
    for arm, entry in zip(arms, plan["arms"]):
        name = arm["arm_id"]
        if (set(entry) != {"arm_id", "adapter", "contract", "expected"} or
                (arm["family"]["name"], arm["family"]["version"]) != INTERFACES[name] or
                arm["addons"] or [r["role"] for r in arm["data"]["roles"]] != ["train"]):
            raise ValueError("Workflow trial family or role exceeds its authorization")
        if entry["adapter"] != ("k1-session-v1" if name == "k1" else "gptrans-session-v1"):
            raise ValueError("Wrong owning session inspector")
        record, recipe = read_recipe(_pinned(root, entry["contract"]))
        validate_screen_recipe(spec, name, record)
        if (recipe.interface != INTERFACES[name] or recipe.mode != "reference" or
                recipe.seed != 42 or recipe.batch_size != 128 or recipe.train_rows != 100000 or
                recipe.optimizer_steps != 32 or recipe.checkpoint_every != 8 or
                recipe.dataset.development or recipe.development_rows != 0 or
                recipe.dataset.manifest_sha256 != FIXED_SHA256 or
                recipe.input_sidecar is not None or recipe.clean_bn is not None or
                recipe.selection != ("final-live" if name == "k1" else "final-ema") or
                record.get("logging") != {"mode": "batches", "interval": 1} or
                "profiling" in record or entry["expected"] != record["acceptance_requirements"]):
            raise ValueError("Workflow trial recipe, endpoint or selected data changed")
    return {"status": "ACCEPTANCE_INPUTS_AVAILABLE", "spec_identity": spec.identity,
        "arms": [{"arm_id": a, "status": "AVAILABLE", "blockers": []} for a in INTERFACES],
        "comparison_scope": "EXECUTION_ONLY", "scientific_acceptance": "NOT_APPLICABLE",
        "replay_readiness": "NOT_APPLICABLE", "trainer_execution": "NOT_VERIFIED"}


def execution_extension(spec, source_root):
    from .experiment_execution import ExecutionExtension
    plan = _json(_artifact_path(Path(source_root).absolute(), plan_path(spec)))
    check_plan(spec, source_root, plan)
    return ExecutionExtension(wall_ceiling_seconds=WALL_SECONDS,
        configuration={"format": PLAN_FORMAT, "checkpoint_reserve_seconds": 60})
