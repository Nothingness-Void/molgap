"""Release checks for the original/shared GPTrans-T execution diagnostic.

This adapter declares one matched execution question.  It deliberately keeps
the old V4 owner and the shared family owner separate: the former is used for
the original arm and the latter for the optimized arm.  Training, packaging,
submission and terminal evidence remain owned by their existing modules.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess

from .experiment_family_workflow import _artifact_path, _json
from .training_reproducibility import sha256_file
from .execution_comparison import (safe_comparison_base, run_version as _run_version,
    comparison_base as shared_comparison_base, ExecutionQuestion, execution_proposal,
    freeze_comparison_declarations)


FORMAT = "molgap-gptrans-infrastructure-comparison-plan-v1"
BASE = "experiments/gptrans_infrastructure_t4_100k"
ACCOUNT = "nvoid912"
EXPERIMENT_ID = "gptrans-infrastructure-t4-100k"
LOGICAL_RUN = "gptrans-infrastructure-t4-100k-s42-v1"
KERNEL = "nvoid912/molgap-gptrans-infrastructure-dual-100k-v1"
TITLE = "MolGap GPTrans Infrastructure Dual 100K V1"
SOURCE_DATASET = "nvoid912/molgap-gptrans-infrastructure-source-v1"
WALL_SECONDS = 7 * 3600
ALLOCATED_T4_HOURS = 14

SEED = 42
TRAIN_ROWS = 100_000
DEVELOPMENT_ROWS = 50_000
EPOCHS = 60
OPTIMIZER_STEPS = 46_860
SAMPLE_PRESENTATIONS = 5_998_080
PARAMETER_COUNT = 5_246_817
PHYSICAL_BATCH = 128
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 0.05
GRADIENT_CLIP = 1.0
EMA_DECAY = 0.9999
TARGET_MEAN = 5.338300266634226
TARGET_STD = 1.2750902063230574
MINIMUM_LR = 1.0e-6
WARMUP_EPOCHS = 4
CHECKPOINT_EVERY = 781
TORCH_VERSION = "2.7.1"

INITIAL_STATE_PATH = "platforms/_records/kaggle/packages/gptrans_t_v4_source_814d104/initial_state.pt"
INITIAL_STATE_SHA256 = "8988db8659c6c7e2b27401312f43684215c34cd8d69309aed7ce946ee9cb1ec6"
INITIAL_ARTIFACT_SHA256 = "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"
MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
ARCHITECTURE_SHA256 = "04edcb6f928617d1142c624d071b7d7accb15c92d2f66e3473ed666ca7e5b2b2"
LEGACY_OBJECTIVE_SHA256 = "7bb2059a0ca03c37928f0924218564f63808ac02c7fa9bda89a0547c79a4ffe4"
LEGACY_TRANSFORM_SHA256 = "9462cf73ea022b030675a7f18f2e7682eda6f90b7a42b47affc2f0f4071353be"
SUPPORTING_PRIOR = "pcqm-gptrans-t-v4-100k-reference-s42"
TARGET_MANIFEST = "experiments/pcqm_gptrans_cycle_pair_100k/gpu/target_manifest.json"

PROFILE = {
    "warmup_steps": 120,
    "active_steps": 16,
    "record_shapes": False,
    "profile_memory": False,
    "with_stack": False,
    "row_limit": 80,
    "module_paths": [],
}
LOGGING = {"mode": "epoch", "interval": None}
_MISSING = object()


def _safe_base(value, *, require_under_base=True):
    return safe_comparison_base(value, base=BASE, require_under_base=require_under_base)


def comparison_base(spec):
    return shared_comparison_base(spec, experiment_id='gptrans-infrastructure-t4-100k', base=BASE)


def _pin(root, relative):
    """Return a strict repository-relative file pointer."""
    if type(relative) is not str:
        raise ValueError("Expected repository-relative pinned path")
    path = _artifact_path(Path(root).absolute(), relative)
    return {"path": relative, "sha256": sha256_file(path)}


def _load_pin(root, pointer):
    if type(pointer) is not dict or set(pointer) != {"path", "sha256"}:
        raise ValueError("Expected pinned execution input")
    path = _artifact_path(Path(root).absolute(), pointer["path"])
    if sha256_file(path) != pointer["sha256"]:
        raise ValueError("Execution comparison input changed")
    return path


def _target_hashes(root):
    target = _json(_artifact_path(Path(root).absolute(), TARGET_MANIFEST))
    if (target.get("rows") != DEVELOPMENT_ROWS
            or target.get("target_encoding") != "little-endian-float64-contiguous-raw-bytes"):
        raise ValueError("Accepted GPTrans target manifest changed")
    source_idx = target.get("development_source_idx_sha256")
    target_sha = target.get("development_target_sha256")
    if (type(source_idx) is not str or len(source_idx) != 64
            or type(target_sha) is not str or len(target_sha) != 64):
        raise ValueError("Accepted GPTrans target identity is incomplete")
    return source_idx, target_sha


def _inspect_initial(path, expected_state_sha256=INITIAL_STATE_SHA256):
    """Validate the portable state and its frozen parameter-count identity."""
    from .v4_runtime import read_frozen_state_artifact

    state, metadata = read_frozen_state_artifact(path, expected_state_sha256=expected_state_sha256)
    if sum(value.numel() for value in state.values()) != PARAMETER_COUNT:
        raise ValueError("GPTrans initialization parameter count changed")
    return metadata


def _default_shared_recipe(*, state_sha256, manifest_sha256=MANIFEST_SHA256):
    from .family_training_data import DatasetSelection
    from .family_training_recipe import FamilyRecipe, ScheduleSpec

    return FamilyRecipe(
        "gptrans_t", "reference", SEED, state_sha256,
        DatasetSelection("fixed100k/manifest.json", manifest_sha256, True),
        TRAIN_ROWS, DEVELOPMENT_ROWS, OPTIMIZER_STEPS,
        LEARNING_RATE, WEIGHT_DECAY,
        ScheduleSpec("warmup_cosine_epoch", EPOCHS, minimum_lr=MINIMUM_LR,
                      warmup_units=WARMUP_EPOCHS),
        "best-development-ema", "torch-pass-randperm", CHECKPOINT_EVERY,
        TARGET_MEAN, TARGET_STD, ema_decay=EMA_DECAY,
        gradient_clip=GRADIENT_CLIP, torch_version=TORCH_VERSION,
    )


def freeze_comparison(root, *, account=ACCOUNT, base=BASE, run_version="v1"):
    """Freeze a fresh prospective declaration through existing owners.

    The accepted initialization and target identities are only read and
    pinned.  This function never recreates or mutates frozen model assets.
    """
    import torch
    from .experiment_spec import FAMILIES
    from .experiment_family_workflow import tensor_digest
    from .experiment_execution import build_family_recipe
    from .family_training_runner import family_arm_bindings
    from .gptrans_screen_workflow import build_screen_recipe as legacy_recipe
    from .pcqm_gptrans_v4 import _scientific_fields
    from .research_memory.trace import atomic_write, json_bytes
    from .screen_policy import canonical_fingerprint
    from .v4_runtime import normalized_source_sha256

    root = Path(root).resolve()
    base = _safe_base(base)
    run_version = _run_version(run_version)
    if base == BASE and run_version != "v1":
        raise ValueError("The frozen v1 comparison cannot be overwritten")
    here = root / base
    from .gptrans_execution_diagnostic import read_diagnostic
    diagnostic = read_diagnostic(root, base)
    dynamic = ("experiment_spec.json", "acceptance_plan.json", "workflow_plan.json",
               "original", "optimized")
    if any((here / name).exists() for name in dynamic):
        raise ValueError("Comparison already frozen; reconcile before another attempt")
    if type(account) is not str or not account.strip():
        raise ValueError("A nonempty platform account is required")
    authority_ref = base + "/authority.json"
    role_ref = base + "/role_plan.json"
    protocol_ref = base + "/protocol.md"
    decision_ref = base + "/decision.md"
    for reference in (authority_ref, role_ref, protocol_ref, decision_ref):
        _artifact_path(root, reference)
    authority = _json(root / authority_ref)
    if set(authority) != {"user_authorized", "scope", "arms", "seeds",
                           "wall_ceiling_seconds", "allocated_t4_hours_ceiling",
                           "official_validation_allowed", "source"}:
        raise ValueError("GPTrans authority fields changed")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    initial = root / INITIAL_STATE_PATH
    if not initial.is_file() or sha256_file(initial) != INITIAL_ARTIFACT_SHA256:
        raise ValueError("Frozen GPTrans initialization artifact is missing or changed")
    _inspect_initial(initial)
    idx_sha, target_sha = _target_hashes(root)
    train_sha = tensor_digest(torch.arange(TRAIN_ROWS), role="source_idx")
    usage = {role: canonical_fingerprint({"role": role, "plan": sha256_file(root / role_ref)})
             for role in ("train", "development")}
    manifest_sha = MANIFEST_SHA256
    old = legacy_recipe("reference", source_idx_sha256=idx_sha, target_sha256=target_sha)
    # The native reference arm alone publishes ten-epoch recovery chunks.  It
    # is an execution durability switch, not a scientific recipe change.
    old["execution_only_chunking"] = True
    shared_recipe = _default_shared_recipe(state_sha256=INITIAL_STATE_SHA256,
                                            manifest_sha256=manifest_sha)
    shared = build_family_recipe(("gptrans_t", "2"), configuration=shared_recipe,
        source_idx_sha256=idx_sha, target_sha256=target_sha,
        train_role_identity=usage["train"], development_role_identity=usage["development"])
    old["profiling"] = dict(PROFILE)
    old["logging"] = dict(LOGGING)
    shared["profiling"] = dict(PROFILE)
    shared["logging"] = dict(LOGGING)
    atomic_json = lambda path, value: atomic_write(path, json_bytes(value))

    def ref(name, version, digest):
        return {"name": name, "version": version, "sha256": digest}

    old_base = ref("gptrans_t_frozen_core", "1", ARCHITECTURE_SHA256)
    shared_base = family_arm_bindings(shared_recipe)["base"]
    target_feature = _scientific_fields()["feature_fingerprint"]
    train_contract = base + "/original/contract.json"
    shared_contract = base + "/optimized/contract.json"
    atomic_json(root / train_contract, old)
    atomic_json(root / shared_contract, shared)
    train_recipe_sha = normalized_source_sha256(root / train_contract)
    shared_recipe_sha = normalized_source_sha256(root / shared_contract)
    arms, bindings, jobs, entries = [], [], [], []
    run = "gptrans-infrastructure-t4-100k-s42-" + run_version
    kernel = account + "/molgap-gptrans-infrastructure-dual-100k-" + run_version
    title = "MolGap GPTrans Infrastructure Dual 100K " + run_version.upper()
    source_dataset = account + "/molgap-gptrans-infrastructure-source-" + run_version
    if diagnostic is not None:
        label = "Drift Attribution" if diagnostic["kind"] == "attribution" else "Execution Repeat"
        title = "MolGap GPTrans " + label + " V1"
        kernel = account + "/" + title.lower().replace(" ", "-")
        source_dataset = kernel + "-source"
    timestamp = datetime.now(timezone.utc).isoformat()
    for device, (arm_id, family_version, record, base_binding, role, contract_ref) in enumerate((
        ("original", "1", old, old_base, "reference", train_contract),
        ("optimized", "2", shared, shared_base, "candidate", shared_contract),
    )):
        family = {"name": "gptrans_t", "version": family_version}
        contract = FAMILIES[("gptrans_t", family_version)]
        sampler = (ref("seed-plus-epoch-global-randperm-v1", "1", _scientific_fields()["row_order_fingerprint"])
                   if family_version == "1" else family_arm_bindings(shared_recipe)["sampler"])
        transform = (ref("fixed-train-100k-mean-sample-std", "1", LEGACY_TRANSFORM_SHA256)
                     if family_version == "1" else family_arm_bindings(shared_recipe)["transform"])
        objective = ref("normalized-gap-l1", "1", LEGACY_OBJECTIVE_SHA256 if family_version == "1"
                        else family_arm_bindings(shared_recipe)["objective"]["sha256"])
        arm = {"arm_id": arm_id, "scientific_role": role, "family": family,
            "base": base_binding,
            "initialization": {"kind": "frozen_state", "seed": SEED, "state_sha256": INITIAL_STATE_SHA256},
            "data": {"dataset": ref("pcqm4mv2", "1", MANIFEST_SHA256),
                "split": ref("fixed100k-internal50k", "1", MANIFEST_SHA256),
                "roles": [{"role": name, "membership_sha256": digest,
                           "row_order_sha256": digest, "usage_sha256": usage[name]}
                          for name, digest in (("train", train_sha), ("development", idx_sha))],
                "feature_schema": contract.feature_schema, "feature_sha256": target_feature,
                "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": ref(contract.recipe, "1", train_recipe_sha if family_version == "1" else shared_recipe_sha),
                "overrides": {}, "objective": objective, "sampler": sampler, "transform": transform},
            "addons": [], "addon_semantics": "baseline"}
        arms.append(arm)
        trajectory = f"TC-gptrans-infrastructure-{arm_id}-100k-s42"
        if run_version != "v1":
            trajectory += "-" + run_version
        cost = "cost-" + trajectory + "-planned"
        plan_ref = base + "/" + arm_id + "/plan_input.json"
        changed = ("retained original GPTrans-T V4 execution"
                   if arm_id == "original" else "latest shared GPTrans-T execution refinement")
        question = "Does the latest shared GPTrans-T execution refinement change fixed-100K T4 execution cost at matched exposure?"
        if diagnostic is not None:
            question = ("Where do forward, gradients, optimizer, EMA and fill policy first diverge under matched input/RNG?"
                if diagnostic["kind"] == "attribution" else
                "Does the GPTrans update speed benefit repeat with common CPU/loader settings and exchanged T4 assignment?")
        proposal = execution_proposal(
            question=ExecutionQuestion("gptrans-infrastructure-execution" if diagnostic is None else "gptrans-execution-followup", "RUN_GPTRANS_EXECUTION_DIAGNOSTIC",
                question, "The latest GPTrans-T shared execution refinement lacks a matched native original profiler comparison",
                changed, (SUPPORTING_PRIOR,), ("CPU contention", "T4 device variability", "Profiler instrumentation overhead"),
                ("gptrans-t-v4-reference",), "retain measured execution evidence without scientific promotion",
                "record no execution gain or regression without an extra seed", reference_ids=(SUPPORTING_PRIOR,)),
            arm=arm, trajectory_id=trajectory, source_commit=source_commit, run=run, version=run_version,
            contract_ref=contract_ref, contract_refs=[contract_ref, authority_ref, role_ref, protocol_ref],
            authority_ref=authority_ref, role_ref=role_ref, decision_ref=decision_ref, timestamp=timestamp,
            estimated_device_hours=authority["allocated_t4_hours_ceiling"] / 2,
            estimated_wall_hours=3.5 if diagnostic is None else authority["wall_ceiling_seconds"] / 3600)
        atomic_json(root / plan_ref, proposal)
        bindings.append({"arm_id": arm_id, "trajectory_id": trajectory, "plan_spec_ref": plan_ref,
            "plan_spec_sha256": sha256_file(root / plan_ref), "output": base + "/" + arm_id + "/rml"})
        jobs.append({"arm_id": arm_id, "device": device, "recipe": contract_ref, "initial_state": str(initial)})
        entries.append({"arm_id": arm_id, "contract": {"path": contract_ref, "sha256": sha256_file(root / contract_ref)},
            "expected": record["acceptance_requirements"],
            "initial_state": {"path": INITIAL_STATE_PATH, "sha256": INITIAL_ARTIFACT_SHA256}})
    result = freeze_comparison_declarations(root, base=base, experiment_id=EXPERIMENT_ID,
        logical_run_id=LOGICAL_RUN if run_version == "v1" else run,
        arms=arms, bindings=bindings, jobs=jobs, entries=entries, plan_format=FORMAT,
        authority_ref=authority_ref, source_extras=[authority_ref, role_ref, protocol_ref, decision_ref, TARGET_MANIFEST,
            *([base + "/diagnostic.json"] if diagnostic is not None else [])],
        account=account, kernel=kernel, title=title, source_dataset=source_dataset,
        data_datasets=[account + "/pcqm4mv2-ogb-fixed-100k-v1"])
    return {**result, "initial_state": str(initial)}


def is_comparison(spec):
    return spec.to_dict().get("experiment_id") == EXPERIMENT_ID


def _check_authority(root, pointer, *, diagnostic=None):
    authority = _json(_load_pin(root, pointer))
    expected = {"user_authorized", "scope", "arms", "seeds", "wall_ceiling_seconds",
                "allocated_t4_hours_ceiling", "official_validation_allowed", "source"}
    if set(authority) != expected or authority.get("user_authorized") is not True:
        raise ValueError("GPTrans execution authority fields changed")
    if (authority.get("scope") != "execution-diagnostic-only"
            or authority.get("arms") != ["original", "optimized"]
            or authority.get("seeds") != [42]
            or authority.get("wall_ceiling_seconds") != (WALL_SECONDS if diagnostic is None else diagnostic["wall_ceiling_seconds"])
            or authority.get("allocated_t4_hours_ceiling") != (ALLOCATED_T4_HOURS if diagnostic is None else diagnostic["allocated_t4_hours_ceiling"])
            or authority.get("official_validation_allowed") is not False):
        raise ValueError("GPTrans execution diagnostic scope changed")
    return authority


def _check_profile(record):
    if record.get("profiling") != PROFILE:
        raise ValueError("Both GPTrans arms require the same bounded profile window")
    if record.get("logging") != LOGGING:
        raise ValueError("GPTrans epoch logging policy changed")


def _check_initial(root, pointer, expected_state_sha256, *, initial_states=None, arm_id=None):
    # During remote preflight the immutable launch input is staged outside the
    # source tree; only the no-override declaration path is resolved from root.
    path = (Path(initial_states[arm_id]) if initial_states is not None
            else _load_pin(root, pointer))
    if not path.is_file() or sha256_file(path) != pointer["sha256"]:
        raise ValueError("Staged GPTrans initialization bytes changed")
    if pointer["path"] != INITIAL_STATE_PATH or pointer["sha256"] != INITIAL_ARTIFACT_SHA256:
        raise ValueError("GPTrans initialization artifact is not the accepted V4 state")
    _inspect_initial(path, expected_state_sha256)
    return path


def _check_scientific_fields(record):
    from .pcqm_gptrans_v4 import _scientific_fields
    expected = _scientific_fields("reference")
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError("GPTrans frozen scientific field changed: " + key)
    fixed = {"mode": "reference", "epochs": EPOCHS, "train_rows": TRAIN_ROWS,
             "development_rows": DEVELOPMENT_ROWS, "physical_batch_per_device": PHYSICAL_BATCH,
             "optimizer_steps_per_epoch": OPTIMIZER_STEPS // EPOCHS,
             "seed42_initial_model_sha256": INITIAL_STATE_SHA256,
             "seed42_initial_state_artifact_sha256": INITIAL_ARTIFACT_SHA256,
             "manifest_sha256": MANIFEST_SHA256,
             "development_role_identity": "pcqm4mv2-ogb-fixed-100k-v1:development-100000-150000"}
    for key, value in fixed.items():
        if record.get(key) != value:
            raise ValueError("GPTrans frozen executable field changed: " + key)
    expected_requirements = {"epochs": EPOCHS, "optimizer_steps": OPTIMIZER_STEPS,
        "sample_presentations": SAMPLE_PRESENTATIONS, "development_rows": DEVELOPMENT_ROWS,
        "precision": "fp32"}
    requirements = record.get("acceptance_requirements")
    if (type(requirements) is not dict
            or set(requirements) != {"epochs", "optimizer_steps", "sample_presentations", "development_rows",
                                     "source_idx_sha256", "target_sha256", "precision"}):
        raise ValueError("GPTrans acceptance requirements are incomplete")
    for key, value in expected_requirements.items():
        if requirements.get(key) != value:
            raise ValueError("GPTrans exposure differs from the frozen reference")
    if record.get("development_role_identity") != "pcqm4mv2-ogb-fixed-100k-v1:development-100000-150000":
        raise ValueError("GPTrans development role differs from fixed V4 rows")


def _check_shared_recipe(recipe):
    if (recipe.family != "gptrans_t" or recipe.mode != "reference" or recipe.seed != SEED
            or recipe.initialization_sha256 != INITIAL_STATE_SHA256
            or recipe.dataset.manifest_sha256 != MANIFEST_SHA256 or not recipe.dataset.development
            or recipe.train_rows != TRAIN_ROWS or recipe.development_rows != DEVELOPMENT_ROWS
            or recipe.optimizer_steps != OPTIMIZER_STEPS or recipe.learning_rate != LEARNING_RATE
            or recipe.weight_decay != WEIGHT_DECAY or recipe.gradient_clip != GRADIENT_CLIP
            or recipe.ema_decay != EMA_DECAY or recipe.selection != "best-development-ema"
            or recipe.sampler != "torch-pass-randperm" or recipe.checkpoint_every != CHECKPOINT_EVERY
            or recipe.target_mean_eV != TARGET_MEAN or recipe.target_std_eV != TARGET_STD
            or recipe.torch_version != TORCH_VERSION or recipe.batch_size != PHYSICAL_BATCH
            or recipe.finite_check_every != 50 or recipe.drop_last is not True
            or recipe.input_sidecar is not None):
        raise ValueError("Shared GPTrans differs from the frozen original scientific recipe")
    schedule = recipe.schedule
    if (schedule.kind != "warmup_cosine_epoch" or schedule.total_units != EPOCHS
            or schedule.warmup_units != WARMUP_EPOCHS or schedule.minimum_lr != MINIMUM_LR):
        raise ValueError("Shared GPTrans schedule differs from the frozen original")


def check_plan(spec, root, plan, *, initial_states=None):
    """Validate exact execution inputs without constructing a model or running data."""
    from .family_training_runner import (family_arm_bindings, read_recipe,
                                         validate_screen_recipe as check_shared)
    from .gptrans_screen_workflow import validate_screen_recipe as check_legacy
    from .pcqm_gptrans_v4 import _scientific_fields

    root = Path(root).absolute()
    declaration = spec.to_dict()
    if (not is_comparison(spec) or set(plan) != {"format", "spec_identity", "authority", "arms"}
            or plan["format"] != FORMAT or plan["spec_identity"] != spec.identity):
        raise ValueError("Unsupported GPTrans execution comparison")
    base = comparison_base(spec)
    from .gptrans_execution_diagnostic import read_diagnostic
    diagnostic = read_diagnostic(root, base)
    if base != BASE and diagnostic is None:
        raise ValueError("GPTrans comparison must use the frozen declaration root")
    if base == BASE and diagnostic is not None:
        raise ValueError("Preserve the frozen full100K comparison")
    if type(plan.get("authority")) is not dict or plan["authority"].get("path") != base + "/authority.json":
        raise ValueError("Comparison authority path is not bound to the GPTrans declaration")
    authority = _check_authority(root, plan["authority"], diagnostic=diagnostic)
    if diagnostic is not None:
        expected = {"path": base + "/diagnostic.json", "sha256": sha256_file(root / base / "diagnostic.json")}
        if not isinstance(authority["source"], dict) or authority["source"].get("diagnostic") != expected:
            raise ValueError("Diagnostic scope is not pinned by its authority")
    platform = declaration["platform"]
    if (platform["name"] != "kaggle" or platform["accelerator"] != "NvidiaTeslaT4"
            or platform["device_count"] != 2 or platform["atomic_checkpoints"] is not True
            or platform["retrievable_chunks"] is not True):
        raise ValueError("GPTrans comparison requires its declared T4x2 durable allocation")
    arms = declaration["arms"]
    if ([arm["arm_id"] for arm in arms] != ["original", "optimized"]
            or [arm["family"] for arm in arms] != [{"name": "gptrans_t", "version": "1"},
                                                      {"name": "gptrans_t", "version": "2"}]
            or any(arm["addons"] or arm["training"]["overrides"] for arm in arms)
            or "same_run_replay" in declaration["prospective"]):
        raise ValueError("GPTrans comparison requires original/shared reference interfaces without addons or Replay")
    if arms[0]["initialization"] != arms[1]["initialization"] or arms[0]["data"] != arms[1]["data"]:
        raise ValueError("GPTrans comparison input, role or initialization identity differs")
    if arms[0]["base"] != {"name": "gptrans_t_frozen_core", "version": "1", "sha256": ARCHITECTURE_SHA256}:
        raise ValueError("Original GPTrans base identity changed")
    entries = plan["arms"]
    if type(entries) is not list or [entry.get("arm_id") for entry in entries] != ["original", "optimized"]:
        raise ValueError("Comparison plan must bind both GPTrans arms")
    records = []
    recipes = []
    for arm, entry in zip(arms, entries):
        if set(entry) != {"arm_id", "contract", "expected", "initial_state"}:
            raise ValueError("Unexpected GPTrans execution comparison fields")
        expected_contract = base + "/" + arm["arm_id"] + "/contract.json"
        if (type(entry["contract"]) is not dict or entry["contract"].get("path") != expected_contract):
            raise ValueError("GPTrans contract path is not bound to the declaration round")
        contract_path = _load_pin(root, entry["contract"])
        record = _json(contract_path)
        if record.get("acceptance_requirements") != entry["expected"]:
            raise ValueError("GPTrans endpoint exposure changed")
        _check_profile(record)
        if arm["arm_id"] == "original":
            if record.get("execution_only_chunking") is not True:
                raise ValueError("Native original GPTrans arm must enable execution-only chunks")
            _check_scientific_fields(record)
            check_legacy(spec, arm["arm_id"], record)
        else:
            if "execution_only_chunking" in record:
                raise ValueError("Execution-only chunks are restricted to the native original arm")
            check_shared(spec, arm["arm_id"], record)
            _, recipe = read_recipe(contract_path)
            _check_shared_recipe(recipe)
            if arm["base"] != {"name": "gptrans_t", "version": "1",
                                "sha256": family_arm_bindings(recipe)["base"]["sha256"]}:
                raise ValueError("Shared GPTrans substrate identity changed")
            recipes.append(recipe)
        records.append(record)
        _check_initial(root, entry["initial_state"], INITIAL_STATE_SHA256,
                       initial_states=initial_states, arm_id=arm["arm_id"])
    legacy, shared = records
    if len(recipes) != 1:
        raise ValueError("Shared GPTrans recipe is missing")
    recipe = recipes[0]
    comparable = ("optimizer_steps", "sample_presentations", "development_rows",
                  "precision", "source_idx_sha256", "target_sha256")
    if any(legacy["acceptance_requirements"][key] != shared["acceptance_requirements"][key]
           for key in comparable):
        raise ValueError("Original/shared GPTrans exposure or target identity differs")
    expected_idx, expected_target = _target_hashes(root)
    for record in records:
        requirements = record["acceptance_requirements"]
        if (requirements["source_idx_sha256"] != expected_idx
                or requirements["target_sha256"] != expected_target):
            raise ValueError("GPTrans acceptance target identity differs from the accepted target manifest")
    old_fields = _scientific_fields("reference")
    if legacy.get("row_order_fingerprint") != old_fields["row_order_fingerprint"]:
        raise ValueError("Original GPTrans sampler row order changed")
    expected_legacy_training = {
        "objective": {"name": "normalized-gap-l1", "version": "1",
                      "sha256": LEGACY_OBJECTIVE_SHA256},
        "sampler": {"name": "seed-plus-epoch-global-randperm-v1", "version": "1",
                    "sha256": old_fields["row_order_fingerprint"]},
        "transform": {"name": "fixed-train-100k-mean-sample-std", "version": "1",
                       "sha256": LEGACY_TRANSFORM_SHA256},
    }
    if any(arms[0]["training"][key] != value for key, value in expected_legacy_training.items()):
        raise ValueError("Original GPTrans training binding changed")
    expected_bindings = family_arm_bindings(recipe)
    optimized_arm = arms[1]
    for key in ("sampler", "transform", "objective"):
        if optimized_arm["training"][key] != expected_bindings[key]:
            raise ValueError("Shared GPTrans training binding changed: " + key)
    if (legacy["acceptance_requirements"]["optimizer_steps"] != OPTIMIZER_STEPS
            or legacy["acceptance_requirements"]["sample_presentations"] != SAMPLE_PRESENTATIONS):
        raise ValueError("GPTrans comparison exposure changed")
    if legacy.get("profiling") != shared.get("profiling") or legacy.get("profiling") != PROFILE:
        raise ValueError("GPTrans profiler window is not paired")
    return {"status": "ACCEPTANCE_INPUTS_AVAILABLE", "spec_identity": spec.identity,
        "scientific_acceptance": "NOT_EVALUATED",
        "replay_readiness": "NOT_APPLICABLE_EXECUTION_DIAGNOSTIC",
        "arms": [{"arm_id": arm["arm_id"], "status": "AVAILABLE", "blockers": []}
                 for arm in arms]}
