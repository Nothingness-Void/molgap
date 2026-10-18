"""Release two new reference screens, never a fabricated matched comparison.

Existing recipe, package, role and Session owners retain their checks. This
adapter only binds their small metadata and the explicitly authorized budget.
"""
from pathlib import Path

from .common.wire import canonical_fingerprint
from .experiment_family_workflow import _artifact_path, _json
from .research_memory.trace import file_digest


PLAN_FORMAT = "molgap-modern2d-formal-reference-plan-v1"
WALL_CEILING_SECONDS = 16200
CHECKPOINT_RESERVE_SECONDS = 600


def is_formal_screen(spec):
    from .modern2d_training import INTERFACES
    arms = spec.to_dict()["arms"]
    return (len(arms) == 2 and
            {(a["family"]["name"], a["family"]["version"]) for a in arms} == set(INTERFACES))


def plan_path(spec):
    """Locate the single owner from frozen prospective declarations."""
    parents = {str(Path(a["plan_spec_ref"]).parent.parent).replace("\\", "/")
               for a in spec.to_dict()["prospective"]["arms"]}
    if not is_formal_screen(spec) or len(parents) != 1:
        raise ValueError("Formal screen requires one two-arm declaration owner")
    return parents.pop() + "/acceptance_plan.json"


def _pinned(root, pointer):
    if type(pointer) is not dict or set(pointer) != {"path", "sha256"}:
        raise ValueError("Formal release requires a pinned metadata pointer")
    path = _artifact_path(Path(root).absolute(), pointer["path"])
    if path.stat().st_size > 1024 ** 2 or file_digest(path) != pointer["sha256"]:
        raise ValueError("Formal release metadata changed or exceeds the size ceiling")
    return path


def check_plan(spec, root, plan):
    """No payload/model IO and no accepted-reference or superiority claim."""
    from .family_training_runner import read_recipe, validate_screen_recipe
    from .modern2d_training import AUTHORITY_FORMAT, INTERFACES, FORMAL_STEPS, PROFILE
    from .modern2d_feature_binding import indexed_transform_identity, validate_indexed_feature_binding
    from .experiment_family_artifacts import artifact_adapter
    from .data.indexed_inputs import AcceptedIndexedInputPermit
    root = Path(root).absolute()
    declaration = spec.to_dict()
    if (not is_formal_screen(spec) or type(plan) is not dict or
            set(plan) != {"format", "spec_identity", "authority", "inputs", "arms"} or
            plan["format"] != PLAN_FORMAT or plan["spec_identity"] != spec.identity):
        raise ValueError("Formal reference plan/Spec mismatch")
    authority = _json(_pinned(root, plan["authority"]))
    metadata = _json(_pinned(root, plan["inputs"]))
    arm_ids = [a["arm_id"] for a in declaration["arms"]]
    gpu = authority.get("gpu", {})
    if (authority.get("format") != AUTHORITY_FORMAT or authority.get("owner") != "server" or
            authority.get("user_authorized") is not True or
            authority.get("purpose") != "FORMAL_100K_TRAINING" or
            authority.get("interfaces") != [list(item) for item in INTERFACES] or
            authority.get("arms") != arm_ids or authority.get("roles") != ["train_100k", "common_dev"] or
            authority.get("seeds") != [42] or authority.get("physical_batch_size") != 128 or
            authority.get("horizon_passes") != 60 or authority.get("horizon_steps") != FORMAL_STEPS or
            authority.get("allocated_t4_hours_ceiling") != 9 or
            authority.get("scientific_training_authorized") is not True or
            any(authority.get(k) is not False for k in ("protected_role_access_authorized",
                "extra_seed_authorized", "automatic_successor_authorized", "scale_500k_authorized",
                "full_training_authorized")) or
            gpu.get("allocated_devices") != 2 or gpu.get("accelerator") != "NvidiaTeslaT4" or
            gpu.get("wall_ceiling_seconds") != WALL_CEILING_SECONDS or
            gpu.get("allocated_t4_hours_ceiling") != 9 or
            declaration["platform"]["name"] != "kaggle" or
            declaration["platform"]["accelerator"] != "NvidiaTeslaT4" or
            declaration["platform"]["device_count"] != 2):
        raise ValueError("Formal two-arm authority, roles or allocation changed")
    if (metadata.get("format") != "molgap-modern2d-formal-input-metadata-v1" or
            metadata.get("scientific_training_authorized") is not True or
            type(plan["arms"]) is not list or [e.get("arm_id") for e in plan["arms"]] != arm_ids or
            set(metadata.get("models", {})) != set(arm_ids)):
        raise ValueError("Formal release must cover both independent reference arms")
    binding_path = _pinned(_pinned(root, plan["inputs"]).parent, {
        k: metadata["training_input"][k] for k in ("path", "sha256")})
    binding = _json(binding_path)
    if (binding.get("format") != "molgap-indexed-training-input-v1" or
            binding.get("scope") != "ACCEPTED_INDEXED_INPUT_V1"):
        raise ValueError("Formal release requires the accepted indexed binding")
    permits = {}
    for name, role in (("train", "train_100k"), ("development", "common_dev")):
        item = binding[name]
        permit = AcceptedIndexedInputPermit(**item["permit"])
        permit.verify(item["authority"], item["role_history"])
        if permit.role != role:
            raise ValueError("Formal role permit changed")
        permits[name] = permit
    if (permits["train"].acceptance_sha256 != permits["development"].acceptance_sha256 or
            permits["train"].version_identity != permits["development"].version_identity):
        raise ValueError("Formal roles do not share accepted data identity")
    for arm, entry in zip(declaration["arms"], plan["arms"]):
        if set(entry) != {"arm_id", "adapter", "expected", "contract", "target_manifest"}:
            raise ValueError("Unsupported formal reference arm fields")
        family = arm["family"]["name"], arm["family"]["version"]
        if artifact_adapter(entry["adapter"], family).protocol != "family-session":
            raise ValueError("Formal screen must retain the existing Session output contract")
        contract_path = _pinned(root, entry["contract"])
        record, recipe = read_recipe(contract_path)
        validate_screen_recipe(spec, arm["arm_id"], record)
        if (recipe.execution_policy["name"] != PROFILE or
                recipe.formal_training.phase != "train" or record["acceptance_requirements"] != entry["expected"] or
                entry["contract"]["sha256"] != arm["training"]["recipe"]["sha256"] or
                recipe.dataset.manifest_sha256 != file_digest(binding_path) or
                recipe.learning_rate != .001 or recipe.weight_decay != 1e-5 or recipe.gradient_clip != 1 or
                recipe.schedule.kind != "four_stage_warmup_cosine_step" or
                recipe.schedule.total_units != FORMAL_STEPS or recipe.schedule.warmup_units != 4 or
                recipe.schedule.minimum_lr != 1e-6 or recipe.schedule.reference_epochs != 60 or
                recipe.torch_version != "2.7.1"):
            raise ValueError("Formal prospective recipe or input identity changed")
        model = metadata["models"][arm["arm_id"]]
        if model["family"] != list(family) or model["model_config"] != record["model_config"]:
            raise ValueError("Formal model/config identity differs from input derivation")
        for name, role in (("train", "train_100k"), ("development", "common_dev")):
            feature = model["feature_bindings"][role]
            validate_indexed_feature_binding(feature, feature)
            if (feature["base_dataset_manifest_sha"] != permits[name].manifest_sha256 or
                    feature["role_history_sha"] != permits[name].role_history_sha256 or
                    feature["transform_code_sha"] != indexed_transform_identity(family) or
                    feature["transform_config_sha"] != canonical_fingerprint(record["model_config"]) or
                    canonical_fingerprint(feature) != model["feature_binding_sha256"][role]):
                raise ValueError("Formal feature/source/role derivation changed")
        target = _json(_pinned(root, entry["target_manifest"]))
        if (target.get("development_target_sha256") != entry["expected"]["target_sha256"] or
                target.get("target_encoding") != "little-endian-float64-contiguous-raw-bytes" or
                entry["expected"]["source_idx_sha256"] != model["feature_bindings"]["common_dev"]["source_row_sha"]):
            raise ValueError("Formal target encoding or row-order identity changed")
    return {"status": "ACCEPTANCE_INPUTS_AVAILABLE", "spec_identity": spec.identity,
        "reference_status": "NEW_REFERENCES_NOT_ACCEPTED", "scientific_acceptance": "NOT_EVALUATED",
        "comparison_scope": "CONTEXT_ONLY", "noReplay": True,
        "arms": [{"arm_id": name, "status": "AVAILABLE", "blockers": []} for name in arm_ids],
        "trainer_execution": "NOT_VERIFIED"}


def build_formal_context(*, spec, job, source_root, staged_root, input_root, run_context):
    from .modern2d_training import Modern2DFormalTrainingContext
    from .data.indexed_inputs import AcceptedIndexedInputPermit
    from .family_training_data import resolve_pinned_manifest
    from .family_training_runner import read_recipe
    plan = _json(_artifact_path(Path(source_root).absolute(), plan_path(spec)))
    check_plan(spec, source_root, plan)
    authority_path = _pinned(source_root, plan["authority"])
    authority = _json(authority_path)
    if (authority.get("run_reference") != run_context.run_reference or
            authority.get("logical_run_id") != run_context.logical_run_id or
            authority["gpu"]["owner"] != run_context.account):
        raise ValueError("Formal platform account/run differs from human authority")
    record, recipe = read_recipe(_artifact_path(Path(source_root).absolute(), job["recipe"]))
    binding = _json(resolve_pinned_manifest(input_root, recipe.dataset))
    metadata = _json(_pinned(source_root, plan["inputs"]))
    model = metadata["models"][job["arm_id"]]
    context = Modern2DFormalTrainingContext(run_context=run_context, recipe_identity=recipe.identity,
        authority_sha256=plan["authority"]["sha256"],
        train_permit=AcceptedIndexedInputPermit(**binding["train"]["permit"]),
        development_permit=AcceptedIndexedInputPermit(**binding["development"]["permit"]),
        train_feature_binding_sha256=model["feature_binding_sha256"]["train_100k"],
        development_feature_binding_sha256=model["feature_binding_sha256"]["common_dev"])
    return context, authority_path


def execution_extension(spec, source_root):
    from .experiment_execution import ExecutionExtension
    plan = _json(_artifact_path(Path(source_root).absolute(), plan_path(spec)))
    check_plan(spec, source_root, plan)
    return ExecutionExtension(wall_ceiling_seconds=WALL_CEILING_SECONDS,
        configuration={"format": PLAN_FORMAT, "checkpoint_reserve_seconds": CHECKPOINT_RESERVE_SECONDS})
