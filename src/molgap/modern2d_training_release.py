"""Release two new reference screens, never a fabricated matched comparison.

Existing recipe, package, role and Session owners retain their checks. This
adapter only binds their small metadata and the explicitly authorized budget.
"""
from pathlib import Path
import math

from .common.wire import canonical_fingerprint
from .experiment_family_workflow import _artifact_path, _json
from .research_memory.trace import file_digest


PLAN_FORMAT = "molgap-modern2d-formal-reference-plan-v1"
CONFIGURED_PLAN_FORMAT = "molgap-modern2d-configured-plan-v1"
CONFIGURED_AUTHORITY_FORMAT = "molgap-modern2d-configured-authority-v1"
WALL_CEILING_SECONDS = 16200
CHECKPOINT_RESERVE_SECONDS = 600
BUDGET_FORMAT = "molgap-modern2d-retry-budget-v1"


def validate_retry_budget(record):
    """Charge the entire observed account increment, never an invented bill.

    Kaggle quota is an account observation. Charging twice its complete settled
    increment is conservative for the observed T4x2 allocation; unrelated
    consumption, if any, is charged too rather than subtracted speculatively.
    """
    from datetime import datetime
    if (record.get("format") != BUDGET_FORMAT or record.get("retry_authorized") is not True or
            record.get("aggregate_t4_hours_ceiling") != 9 or
            record.get("allocated_devices") != 2 or
            record.get("prior_attempt") != {"kernel": "kaseichou/molgap-modern2d-formal-100k-v1",
                "kernel_id": 139364627, "version": 1, "status": "ERROR"}):
        raise ValueError("Retry budget requires the separately authorized exact failed allocation")
    snapshots = record["quota_observations"]
    before, after, confirmed = (snapshots[k] for k in ("before", "after", "confirmed"))
    def seconds(item):
        value = item["seconds"] + item.get("nanos", 0) / 1e9
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid quota observation")
        return value
    windows = [s["quota"]["quotaRefreshTime"] for s in (before, after, confirmed)]
    totals = [seconds(s["quota"]["gpuQuota"]["totalTimeAllowed"]) for s in (before, after, confirmed)]
    stamps = [datetime.fromisoformat(s["observed_at"].replace("Z", "+00:00")) for s in (before, after, confirmed)]
    used = [seconds(s["quota"]["gpuQuota"]["timeUsed"]) for s in (before, after, confirmed)]
    if (len(set(windows)) != 1 or len(set(totals)) != 1 or totals[0] <= 0 or
            any(s.tzinfo is None for s in stamps) or not stamps[0] < stamps[1] < stamps[2] or
            (stamps[2] - stamps[1]).total_seconds() < 30 or
            used[1] != used[2] or not 0 < used[1] - used[0] <= 600 or
            any(seconds(s["quota"]["gpuQuota"]["timeReserved"]) != 0 for s in (before, after, confirmed))):
        raise ValueError("Quota window/reset/reservation/settlement is not reconciled")
    debit = math.ceil(used[1] - used[0])
    if (record.get("prior_debit_wall_seconds") != debit or
            record.get("prior_debit_allocated_t4_hours") != 2 * debit / 3600 or
            record.get("budget_attribution") != "entire-settled-account-increment-charged-at-two-devices" or
            record.get("per_job_full_billing_status") != "NOT_INFERRED_FROM_ACCOUNT_QUOTA"):
        raise ValueError("Prior allocation debit differs from the conservative account observation")
    return debit


def is_formal_screen(spec):
    from .modern2d_training import INTERFACES
    arms = spec.to_dict()["arms"]
    return (1 <= len(arms) <= 2 and all(
        (a["family"]["name"], a["family"]["version"]) in INTERFACES for a in arms))


def validate_configured_authority(authority, declaration, recipes=None):
    """Bind new runs explicitly; historical campaign permission is never inherited."""
    from .modern2d_training import INTERFACES
    arms, platform = declaration["arms"], declaration["platform"]
    gpu = authority.get("gpu", {})
    wall, budget = gpu.get("wall_ceiling_seconds"), authority.get("allocated_t4_hours_ceiling")
    if (authority.get("format") != CONFIGURED_AUTHORITY_FORMAT or authority.get("owner") != "server"
            or authority.get("user_authorized") is not True
            or authority.get("scientific_training_authorized") is not True
            or authority.get("purpose") != "CONFIGURED_SCREEN"
            or authority.get("logical_run_id") != declaration["logical_run_id"]
            or authority.get("arms") != [a["arm_id"] for a in arms]
            or authority.get("interfaces") != [[a["family"]["name"], a["family"]["version"]] for a in arms]
            or not 1 <= len(arms) <= 2
            or any((a["family"]["name"], a["family"]["version"]) not in INTERFACES for a in arms)
            or authority.get("seeds") != [42] or authority.get("physical_batch_size") != 128
            or authority.get("roles") not in (["train_100k", "common_dev"], ["train_500k", "common_dev"])
            or authority.get("scale_500k_authorized") is not (authority.get("roles") == ["train_500k", "common_dev"])
            or any(authority.get(k) is not False for k in ("protected_role_access_authorized",
                "extra_seed_authorized", "automatic_successor_authorized", "full_training_authorized"))
            or platform["name"] != "kaggle" or platform["accelerator"] != "NvidiaTeslaT4"
            or platform["device_count"] != 2 or gpu.get("allocated_devices") != 2
            or gpu.get("accelerator") != "NvidiaTeslaT4"
            or type(wall) is not int or not 660 <= wall <= 43200
            or type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0
            or gpu.get("allocated_t4_hours_ceiling") != budget or 2 * wall / 3600 > budget
            or not isinstance(gpu.get("owner"), str) or not gpu["owner"]
            or not isinstance(authority.get("run_reference"), str)
            or not authority["run_reference"].startswith(gpu["owner"] + "/")
            or (len(arms) == 1 and not authority.get("single_arm_reason"))):
        raise ValueError("Configured authority does not cover the exact roles, arms and full allocation")
    if recipes is not None:
        expected = {name: {"horizon_passes": recipe.formal_training.horizon_passes,
            "horizon_steps": recipe.formal_training.horizon_steps} for name, recipe in recipes.items()}
        if authority.get("recipes") != expected:
            raise ValueError("Configured authority recipe/horizon binding changed")
        for recipe in recipes.values():
            if (not recipe.formal_training.configured or recipe.formal_training.phase != "train"
                    or recipe.dataset.manifest.startswith("synthetic/")
                    or recipe.train_rows != (100000 if authority["roles"][0] == "train_100k" else 500000)
                    or not recipe.dataset.development or recipe.development_rows != 50000):
                raise ValueError("Configured recipe differs from authorized scale/development role")
    return wall


def plan_path(spec):
    """Locate the single owner from frozen prospective declarations."""
    parents = {str(Path(a["plan_spec_ref"]).parent.parent).replace("\\", "/")
               for a in spec.to_dict()["prospective"]["arms"]}
    if not is_formal_screen(spec) or len(parents) != 1:
        raise ValueError("Modern2D screen requires one shared declaration owner")
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
    from .modern2d_feature_binding import indexed_transform_identity, validate_indexed_feature_binding, transform_configuration
    from .experiment_family_artifacts import artifact_adapter
    from .data.indexed_inputs import AcceptedIndexedInputPermit
    root = Path(root).absolute()
    declaration = spec.to_dict()
    configured = type(plan) is dict and plan.get("format") == CONFIGURED_PLAN_FORMAT
    if (not is_formal_screen(spec) or type(plan) is not dict or
            set(plan) not in ({"format", "spec_identity", "authority", "inputs", "arms"},
                {"format", "spec_identity", "authority", "inputs", "arms", "budget_reconciliation"}) or
            plan["format"] not in {PLAN_FORMAT, CONFIGURED_PLAN_FORMAT} or plan["spec_identity"] != spec.identity):
        raise ValueError("Formal reference plan/Spec mismatch")
    authority = _json(_pinned(root, plan["authority"]))
    if "budget_reconciliation" in plan:
        validate_retry_budget(_json(_pinned(root, plan["budget_reconciliation"])))
        if authority.get("retry_authorized") is not True or authority.get("budget_reconciliation") != plan["budget_reconciliation"]:
            raise ValueError("Retry authority and reconciled prior debit differ")
    metadata = _json(_pinned(root, plan["inputs"]))
    arm_ids = [a["arm_id"] for a in declaration["arms"]]
    gpu = authority.get("gpu", {})
    if configured:
        if "budget_reconciliation" in plan:
            raise ValueError("Configured runs cannot inherit historical retry budgets")
        validate_configured_authority(authority, declaration)
    elif (len(declaration["arms"]) != 2 or
            [(a["family"]["name"], a["family"]["version"]) for a in declaration["arms"]] != list(INTERFACES) or
            authority.get("format") != AUTHORITY_FORMAT or authority.get("owner") != "server" or
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
    permits, recipes = {}, {}
    train_role = authority["roles"][0]
    for name, role in (("train", train_role), ("development", "common_dev")):
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
        recipes[arm["arm_id"]] = recipe
        validate_screen_recipe(spec, arm["arm_id"], record)
        if (recipe.execution_policy["name"] != PROFILE or
                recipe.formal_training.phase != "train" or record["acceptance_requirements"] != entry["expected"] or
                entry["contract"]["sha256"] != arm["training"]["recipe"]["sha256"] or
                recipe.dataset.manifest_sha256 != file_digest(binding_path) or
                recipe.formal_training.configured != configured):
            raise ValueError("Formal prospective recipe or input identity changed")
        if not configured and (recipe.learning_rate != .001 or recipe.weight_decay != 1e-5 or recipe.gradient_clip != 1 or
                recipe.schedule.kind != "four_stage_warmup_cosine_step" or
                recipe.schedule.total_units != FORMAL_STEPS or recipe.schedule.warmup_units != 4 or
                recipe.schedule.minimum_lr != 1e-6 or recipe.schedule.reference_epochs != 60 or
                recipe.torch_version != "2.7.1"):
            raise ValueError("Formal prospective recipe or input identity changed")
        model = metadata["models"][arm["arm_id"]]
        if model["family"] != list(family) or model["model_config"] != record["model_config"]:
            raise ValueError("Formal model/config identity differs from input derivation")
        if configured and (metadata.get("target_mean_eV") != recipe.target_mean_eV
                or metadata.get("target_std_eV") != recipe.target_std_eV
                or arm["data"]["feature_sha256"] != canonical_fingerprint(model["feature_binding_sha256"])):
            raise ValueError("Configured normalization or feature identity differs from Spec")
        for name, role in (("train", train_role), ("development", "common_dev")):
            feature = model["feature_bindings"][role]
            validate_indexed_feature_binding(feature, feature)
            if (feature["base_dataset_manifest_sha"] != permits[name].manifest_sha256 or
                    feature["role"] != role or feature["scope"] != "ACCEPTED_INDEXED_INPUT_V1" or
                    feature["role_history_sha"] != permits[name].role_history_sha256 or
                    feature["transform_code_sha"] != indexed_transform_identity(family, feature_only=configured) or
                    feature["transform_config_sha"] != canonical_fingerprint(
                        transform_configuration(family, record["model_config"]) if configured else record["model_config"]) or
                    canonical_fingerprint(feature) != model["feature_binding_sha256"][role]):
                raise ValueError("Formal feature/source/role derivation changed")
            if configured:
                role_entry = next(r for r in arm["data"]["roles"] if r["role"] == name)
                if role_entry["row_order_sha256"] != feature["source_row_sha"]:
                    raise ValueError("Configured feature row order differs from Spec")
        target = _json(_pinned(root, entry["target_manifest"]))
        if (target.get("development_target_sha256") != entry["expected"]["target_sha256"] or
                target.get("target_encoding") != "little-endian-float64-contiguous-raw-bytes" or
                entry["expected"]["source_idx_sha256"] != model["feature_bindings"]["common_dev"]["source_row_sha"]):
            raise ValueError("Formal target encoding or row-order identity changed")
    if configured:
        validate_configured_authority(authority, declaration, recipes)
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
        train_feature_binding_sha256=model["feature_binding_sha256"][authority["roles"][0]],
        development_feature_binding_sha256=model["feature_binding_sha256"]["common_dev"],
        **({"configured": True, "max_wall_seconds": authority["gpu"]["wall_ceiling_seconds"]}
           if plan["format"] == CONFIGURED_PLAN_FORMAT else {}))
    return context, authority_path


def execution_extension(spec, source_root):
    from .experiment_execution import ExecutionExtension
    plan = _json(_artifact_path(Path(source_root).absolute(), plan_path(spec)))
    check_plan(spec, source_root, plan)
    configuration = {"format": plan["format"], "checkpoint_reserve_seconds": CHECKPOINT_RESERVE_SECONDS}
    if "budget_reconciliation" in plan:
        configuration["prior_budget_wall_seconds"] = validate_retry_budget(
            _json(_pinned(source_root, plan["budget_reconciliation"])))
    wall = (_json(_pinned(source_root, plan["authority"]))["gpu"]["wall_ceiling_seconds"]
            if plan["format"] == CONFIGURED_PLAN_FORMAT else WALL_CEILING_SECONDS)
    return ExecutionExtension(wall_ceiling_seconds=wall, configuration=configuration)
