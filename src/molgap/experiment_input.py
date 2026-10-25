"""Human scientific configuration compiled through existing declaration owners.

This resolver never synthesizes hypotheses, authorizations or accepted references.
It freezes normal JSON through existing canonical parsers; old --spec is intact.
"""
from pathlib import Path
from time import perf_counter
import json
import math
import re
from uuid import uuid4

from .common.io import _safe_local, publish_immutable_bytes
from .common.paths import MachinePaths, validate_relative_path
from .common.wire import canonical_fingerprint
from .common.spec import _object, _identifier, _unique_object
from .experiment_spec import ExperimentSpec
from .research_memory.paths import repo_local_path


def _read(path):
    path = Path(path).absolute()
    _safe_local(path)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def resolve_experiment_input(path, repo_root):
    """Resolve references, not host defaults; no RML publication or network IO."""
    root = Path(repo_root).absolute()
    _safe_local(root)
    value = _read(path)
    if type(value) is dict and value.get("format") == "molgap-scientific-experiment-v1":
        existing = _existing_scientific_input(value, root)
        return existing if existing is not None else _compile_scientific_input(path, root, value)
    _object(value,"format spec_ref workflow_ref machine_profile authority_ref monitor_paths attempt_id","experiment input")
    if value["format"] != "molgap-experiment-input-v1":
        raise ValueError("Unknown editable experiment input")
    _identifier(value["attempt_id"],"attempt_id")
    for name in ("spec_ref","workflow_ref","authority_ref"):
        validate_relative_path(value[name])
    spec = ExperimentSpec(_read(repo_local_path(root,value["spec_ref"])))
    plan = _read(repo_local_path(root,value["workflow_ref"]))
    if type(plan) is not dict or plan.get("spec_identity") != spec.identity:
        raise ValueError("Editable input workflow differs from its scientific declaration")
    if (type(value["monitor_paths"]) is not list or not value["monitor_paths"] or
            any(type(p) is not str for p in value["monitor_paths"]) or
            len(value["monitor_paths"]) != len(set(value["monitor_paths"]))):
        raise ValueError("Existing owner monitor pointers must be explicit and unique")
    for pointer in value["monitor_paths"]:
        validate_relative_path(pointer)
    paths = MachinePaths.from_file(value["machine_profile"]).outside_checkout(root)
    authority = repo_local_path(root,value["authority_ref"])
    _safe_local(authority)
    if not authority.is_file():
        raise ValueError("Existing authority reference is missing; no authorization is inferred")
    # New preparations use one tracked distribution boundary. Preserve explicit
    # scientific/authority records from the old declaration, never old per-model
    # Python lists. The old --prepared and --spec paths keep their old formats.
    if "source_files" in plan:
        if "source_profile" in plan or "source_records" in plan:
            raise ValueError("Ambiguous legacy and runtime source fields")
        if type(plan["source_files"]) is not list or any(type(p) is not str for p in plan["source_files"]):
            raise ValueError("Invalid legacy source list")
        records = [p for p in plan.pop("source_files") if not p.endswith(".py")]
        plan.update(source_profile="runtime-v1",source_records=records)
    if (plan.get("source_profile") != "runtime-v1" or type(plan.get("source_records")) is not list
            or any(type(p) is not str for p in plan["source_records"])):
        raise ValueError("New editable workflow requires the fixed runtime source profile")
    plan["source_records"] = sorted(set([*plan["source_records"],value["authority_ref"]]))
    return value,spec,plan,paths


def _existing_scientific_input(value, root):
    """Identify a frozen run before touching any development-time input."""
    for key in ("experiment_id", "logical_run_id", "attempt_id"):
        _identifier(value.get(key), key)
    paths = MachinePaths.from_file(value["machine_profile"]).outside_checkout(root)
    frozen = paths.run_directory(value["logical_run_id"], value["attempt_id"]) / "inputs"
    _safe_local(frozen)
    if not frozen.exists() or not any(frozen.iterdir()):
        return None
    # An incomplete original freeze cannot be safely replaced by today's HEAD.
    binding = _read(frozen / "input_binding.json")
    if binding.get("format") != value["format"] or binding.get("input_identity") != canonical_fingerprint(value):
        raise ValueError("Frozen scientific input changed; create an explicit new run instead of overwriting")
    try:
        spec = ExperimentSpec(_read(frozen / "experiment_spec.json"))
        plan = _read(frozen / "workflow_plan.json")
    except FileNotFoundError as error:
        raise ValueError("Original frozen inputs are incomplete; reconcile the original freeze") from error
    _check_frozen_binding(binding, value, spec, plan)
    return value, spec, plan, paths


def _check_frozen_binding(binding, value, spec, plan):
    declaration = spec.to_dict()
    if (binding.get("format") != value["format"] or binding.get("input_identity") != canonical_fingerprint(value)
            or binding.get("spec_identity") != spec.identity
            or binding.get("workflow_identity") != canonical_fingerprint(plan)
            or binding.get("authority_ref") != _authority_reference(value, plan)
            or binding.get("monitor_paths") != value["monitor_paths"]
            or plan.get("spec_identity") != spec.identity
            or (value["format"] == "molgap-scientific-experiment-v1" and (
                declaration["experiment_id"] != value["experiment_id"]
                or declaration["logical_run_id"] != value["logical_run_id"]))):
        raise ValueError("Original frozen run binding changed; refusing overwrite")
    if "source_commit" in binding:
        _commit(binding["source_commit"])


def _authority_reference(value, plan):
    """Resolve inline authority from frozen generated pins, never today's HEAD."""
    if "authority_ref" in value:
        return value["authority_ref"]
    from .experiment_launch import canonical_json
    import hashlib
    digest = hashlib.sha256(canonical_json(value["authority"]).encode()).hexdigest()
    matches = [name for name, pin in plan.get("generated_records", {}).items()
               if name.endswith("/authority.json") and pin == digest]
    if len(matches) != 1:
        raise ValueError("Frozen inline authority binding is missing or ambiguous")
    validate_relative_path(matches[0])
    return matches[0]


def _commit(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None:
        raise ValueError("Missing or invalid original source commit")
    return value


def _bound_source_commit(run, spec, root):
    """Read new pins, or recover legacy pins from original retained artifacts."""
    binding = _read(run / "inputs/input_binding.json")
    if "source_commit" in binding:
        return _commit(binding["source_commit"])
    package = run / "prepared/package"
    _safe_local(package)
    if package.exists():
        from .experiment_package import verify_experiment_source_package
        manifest = verify_experiment_source_package(package)
        if manifest["spec_identity"] != spec.identity:
            raise ValueError("Original source package differs from frozen Spec")
        return _commit(manifest["source_commit"])
    from .research_memory.trace import file_digest
    commits = set()
    for entry in spec.to_dict()["prospective"]["arms"]:
        path = repo_local_path(root, entry["plan_spec_ref"])
        if file_digest(path) != entry["plan_spec_sha256"]:
            raise ValueError("Original prospective input bytes changed")
        commits.add(_commit(_read(path)["trajectory"]["state_at_start"]["source_commit"]))
    if len(commits) != 1:
        raise ValueError("Original prospective inputs do not bind one source commit")
    return commits.pop()


def _read_scientific_recipe(path, family, authority):
    """Reissue only a repair probe's source pin; retained input bytes stay frozen."""
    from copy import deepcopy
    from .family_training_runner import read_recipe, _recipe_configuration
    from .modern2d_feature_binding import indexed_transform_identity
    from .experiment_execution import MODERN2D_SCIENTIFIC_INPUT_ABI, scientific_input_abi

    try:
        return read_recipe(path)
    except ValueError as error:
        repair = (authority.get("purpose") == "CONFIGURED_EXECUTION_REPAIR"
                  and authority.get("scientific_training_authorized") is False)
        fresh_screen = (authority.get("purpose") in {"CONFIGURED_SCREEN", "CONFIGURED_TRANSFER_STUDY"}
                  and authority.get("scientific_training_authorized") is True
                  and authority.get("source_reissue_authorized") is True)
        if (not (repair or fresh_screen)
                or scientific_input_abi(family) != MODERN2D_SCIENTIFIC_INPUT_ABI
                or str(error) != "Execution policy differs from the accelerated family default"):
            raise
        record = _read(path)
        rebound = deepcopy(record)
        policy = rebound.get("execution_policy")
        if not isinstance(policy, dict) or "feature_transform_source_sha256" not in policy:
            raise ValueError("Execution repair requires a frozen source-transform pin")
        policy["feature_transform_source_sha256"] = indexed_transform_identity(family,
            feature_only=rebound.get("formal_training", {}).get("purpose") == "configured-screen")
        return record, _recipe_configuration(rebound)


def _compile_scientific_input(path, root, value):
    """Compile declarations, not permission, reference acceptance or trainer code.

    The existing recipe fixes data, sampler, selection and runtime. Catalog
    addons and optimizer parameters are editable; Modern2D additionally binds
    an explicit pass horizon to its independent authority.
    Derived JSON is explicitly separate from committed executable source.
    """
    from copy import deepcopy
    from .modern2d_training import Modern2DFormalTrainingProfile
    from dataclasses import replace
    import hashlib
    import math
    from .common.declarations import ExecutionQuestion, ExecutionProfile, build_execution_proposal
    from .comparison_readiness import assess_comparison_prelaunch, validate_reference_bundle, validate_target_transform_asset
    from .experiment_execution import (MODERN2D_SCIENTIFIC_INPUT_ABI,
        SUPPORTED_SCIENTIFIC_INPUT_ABIS, build_addon_declaration, build_family_recipe,
        scientific_input_abi, training_adapter)
    from .experiment_launch import canonical_json
    from .experiment_source_inventory import runtime_source_files
    from .experiment_spec import FAMILIES, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL
    from .family_training_runner import read_recipe, family_arm_bindings
    from .v4_bundle import _assert_clean_paths, _payload, committed_payloads
    from .v4_runtime import inspect_frozen_state_artifact
    from .research_memory.trace import file_digest

    optional = {"shared", "same_run"} & set(value)
    declaration_fields = []
    for name in ("authority", "platform"):
        selected = {name, name + "_ref"} & set(value)
        if len(selected) != 1 or (name in selected and type(value[name]) is not dict):
            raise ValueError("Supply exactly one explicit " + name + " or " + name + "_ref")
        declaration_fields.extend(selected)
    _object(value, "format experiment_id logical_run_id attempt_id machine_profile monitor_paths "
            "timestamp question profile role_ref decision_ref arms seeds " +
            " ".join([*optional, *declaration_fields]), "scientific input")
    from .common.declarations import run_version
    run_version(value["attempt_id"])
    for key in ("experiment_id", "logical_run_id", "attempt_id"):
        _identifier(value[key], key)
    if value["seeds"] != [42] or any(type(s) is not int for s in value["seeds"]):
        raise ValueError("Scientific input supports one paired seed 42; no seed expansion")
    if (type(value["monitor_paths"]) is not list or not value["monitor_paths"] or
            len(value["monitor_paths"]) != len(set(value["monitor_paths"]))):
        raise ValueError("Existing owner monitor pointers must be explicit and unique")
    for pointer in value["monitor_paths"]:
        validate_relative_path(pointer)
    paths = MachinePaths.from_file(value["machine_profile"]).outside_checkout(root)
    records = set()
    def local(name, *, packaged=True):
        validate_relative_path(name)
        result = repo_local_path(root, name)
        _safe_local(result)
        if not result.is_file():
            raise FileNotFoundError(name)
        if packaged:
            records.add(name)
        return result
    def pin(name, *, packaged=True):
        return {"path": name, "sha256": file_digest(local(name, packaged=packaged))}
    def retained(pointer):
        _object(pointer, "path sha256", "retained reference")
        if pin(pointer["path"]) != pointer:
            raise ValueError("Retained reference bytes changed")
        return _read(local(pointer["path"]))

    config_path = Path(path).absolute()
    _safe_local(config_path)
    config_name = config_path.relative_to(root).as_posix()
    local(config_name)
    input_sha = hashlib.sha256(_payload(config_path)).hexdigest()
    from .experiment_package import _git
    initial_commit = _commit(_git(root, "rev-parse", "HEAD").decode().strip())
    prefix = f"experiments/{value['experiment_id']}/generated/" + canonical_fingerprint({
        "input": input_sha, "commit": initial_commit})
    authority_ref = value.get("authority_ref", prefix + "/authority.json")
    authority = _read(local(authority_ref)) if "authority_ref" in value else deepcopy(value["authority"])
    transfer_study = authority.get("purpose") == "CONFIGURED_TRANSFER_STUDY"
    if transfer_study:
        transfer_provenance = retained(authority["transfer_study_ref"])
        for source in transfer_provenance["arms"].values():
            retained(source["source_contract"])
    if "execution_qualification_ref" in authority:
        retained(authority["execution_qualification_ref"] )
    if authority.get("research_design", {}).get("campaign_ref"):
        local(authority["research_design"]["campaign_ref"])
    for key in ("role_ref", "decision_ref"):
        if not local(value[key]).read_bytes().strip():
            raise ValueError("Scientific input requires retained role and decision content")
    platform = _read(local(value["platform_ref"])) if "platform_ref" in value else deepcopy(value["platform"])
    _object(platform, "platform kaggle", "platform input")
    question = ExecutionQuestion(**value["question"])
    from .research_memory.policy import load_policy_registry
    policies = [p for p in load_policy_registry(root) if p["policy_id"] == question.policy_id]
    if len(policies) != 1:
        raise ValueError("Scientific policy ID must identify one registered version")
    policy_version = policies[0]["version"]
    profile = ExecutionProfile(**value["profile"])
    if platform["platform"]["name"] != "kaggle" or profile.platform != "kaggle":
        raise ValueError("Scientific preparation requires the existing Kaggle workflow adapter")
    if type(value["arms"]) is not list or not 1 <= len(value["arms"]) <= 2:
        raise ValueError("One T4x2 slot supports one or two independent arms")
    arm_override_fields = {"arm_id", "model", "recipe_ref", "data_ref", "initial_state_ref", "preset_ref",
                           "training", "reference_ref", "purpose"}
    for item in value["arms"]:
        if type(item) is not dict or "arm_id" not in item or set(item) - arm_override_fields:
            raise ValueError("Scientific arm entries must be objects with a supported arm_id override")
        _identifier(item["arm_id"], "arm_id")
    shared = value.get("shared", {})
    if type(shared) is not dict or set(shared) - {
            "model", "recipe_ref", "data_ref", "initial_state_ref", "training", "reference_ref", "purpose", "preset_ref"}:
        raise ValueError("Unsupported shared scientific arm fields")
    items = [{**deepcopy(shared), **item} for item in value["arms"]]
    arm_ids = [a["arm_id"] for a in items]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("Duplicate scientific arm")
    budget = authority.get("allocated_t4_hours_ceiling")
    if (authority.get("user_authorized") is not True or type(budget) not in (int, float)
            or not math.isfinite(budget) or budget <= 0 or authority.get("arms") != arm_ids
            or authority.get("seeds") != [42]):
        raise ValueError("Existing authority does not cover exactly these arms, seed and positive ceiling")
    presets = {}
    for index, item in enumerate(items):
        if "preset_ref" in item:
            from .experiment_presets import expand_baseline_preset
            items[index], presets[item["arm_id"]] = expand_baseline_preset(item, read_local=local)
    pair = value.get("same_run")
    if transfer_study and pair is not None:
        raise ValueError("Transfer study cannot declare a strict same-run architecture pair")
    if pair is not None:
        pair = deepcopy(pair)
        if type(pair) is not dict:
            raise ValueError("Same-run declaration must be an object")
        if "target_manifest_ref" not in pair:
            targets = [presets[arm]["target_manifest"] for arm in arm_ids if arm in presets]
            if len(targets) != len(items) or any(target != targets[0] for target in targets):
                raise ValueError("Same-run target requires one explicit pin shared by every arm preset")
            pair["target_manifest_ref"] = targets[0]["path"]
        if (type(pair) is not dict
                or set(pair) - {"reference_arm_id", "target_manifest_ref", "minimum_material_gain_eV"}
                or {"reference_arm_id", "target_manifest_ref"} - set(pair)):
            raise ValueError("same_run: expected reference_arm_id, target_manifest_ref and optional minimum_material_gain_eV")
        if "minimum_material_gain_eV" in pair:
            gate = pair["minimum_material_gain_eV"]
            if (type(gate) not in (int, float) or isinstance(gate, bool)
                    or not math.isfinite(gate) or gate < 0):
                raise ValueError("same_run.minimum_material_gain_eV must be a finite non-negative number")
        if len(items) != 2 or pair["reference_arm_id"] not in arm_ids:
            raise ValueError("Same-run comparison requires one reference and one candidate")
        if authority.get("same_run_reference_arm_id") != pair["reference_arm_id"]:
            raise ValueError("Authority must explicitly release the same-run reference arm")
        target_pin = pin(pair["target_manifest_ref"])
        if any(preset["target_manifest"] != target_pin for preset in presets.values()):
            raise ValueError("Same-run target differs from its frozen baseline preset")
    compiled, data_specs = [], set()
    modern_inputs = {}
    modern_comparison_assets = {}
    modern_retained_references = {}
    input_abis = [scientific_input_abi((a["model"]["family"], a["model"]["version"]))
                  for a in items]
    modern = all(abi == MODERN2D_SCIENTIFIC_INPUT_ABI for abi in input_abis)
    if not modern and any(abi == MODERN2D_SCIENTIFIC_INPUT_ABI for abi in input_abis):
        raise ValueError("Modern2D indexed inputs require a separate workflow from mature fixed inputs")
    for item in items:
        preset = presets.get(item["arm_id"])
        _object(item, "arm_id model recipe_ref data_ref initial_state_ref training purpose" +
                (" reference_ref" if modern or pair is None else ""), "scientific arm")
        _identifier(item["arm_id"], "arm_id")
        _object(item["model"], "family version addons", "model")
        family = (item["model"]["family"], item["model"]["version"])
        if scientific_input_abi(family) not in SUPPORTED_SCIENTIFIC_INPUT_ABIS:
            raise ValueError("Scientific compiler requires a reviewed catalog scientific_input_abi")
        source_record, source_baseline = _read_scientific_recipe(local(item["recipe_ref"]), family, authority)
        # A retained Modern2D candidate keeps the indexed recipe as
        # provenance, but its scientific baseline is the contract that was
        # actually accepted and registered.  This matters for a first
        # configured bootstrap whose authorized horizon differs from the
        # original formal/indexed plan (for example 20 versus 60 passes).
        record, baseline = source_record, source_baseline
        accepted_bundle = None
        accepted_bundle_pointer = None
        retained_bundle_declared = False
        if (modern and pair is None and isinstance(item.get("reference_ref"), dict)
                and "bundle_ref" in item["reference_ref"]):
            from .comparison_readiness import validate_reference_bundle
            from .evidence_pointers import validate_release_reference_bundle_evidence
            from .family_pair_acceptance import verify_session_reference
            accepted_bundle_pointer = item["reference_ref"]["bundle_ref"]
            _object(accepted_bundle_pointer, "path sha256", "reference_ref.bundle_ref")
            accepted_bundle = verify_session_reference(
                root, retained(accepted_bundle_pointer))
            accepted_bundle = validate_release_reference_bundle_evidence(
                accepted_bundle, repo_root=root,
                reference_bundle_path=repo_local_path(root, accepted_bundle_pointer["path"]))
            accepted_bundle = validate_reference_bundle(accepted_bundle)
            bundle_record, bundle_baseline = _read_scientific_recipe(
                local(accepted_bundle["contract_ref"]), family, authority)
            if (bundle_baseline.interface != family or bundle_baseline.seed != source_baseline.seed
                    or bundle_baseline.dataset.manifest_sha256 != source_baseline.dataset.manifest_sha256
                    or bundle_baseline.initialization_sha256 != source_baseline.initialization_sha256
                    or bundle_record.get("model_config") != source_record.get("model_config")
                    or bundle_record.get("train_role_identity") != source_record.get("train_role_identity")
                    or bundle_record.get("development_role_identity") != source_record.get("development_role_identity")
                    or accepted_bundle["prediction_manifest"]["source_idx_sha256"] !=
                       source_record["acceptance_requirements"]["source_idx_sha256"]
                    or accepted_bundle["prediction_manifest"]["target_sha256"] !=
                       source_record["acceptance_requirements"]["target_sha256"]):
                raise ValueError("Accepted Modern2D bundle differs from indexed data/model provenance")
            record, baseline = bundle_record, bundle_baseline
            retained_bundle_declared = True
        if preset is not None and baseline.initialization_sha256 != preset["initialization_state_sha256"]:
            raise ValueError("Preset initialization differs from its frozen recipe")
        if baseline.interface != family or baseline.seed != 42 or not baseline.dataset.development:
            raise ValueError("Frozen screen recipe does not match model, seed or development contract")
        _object(item["data_ref"], "spec_ref arm_id", "data_ref")
        data_specs.add(item["data_ref"]["spec_ref"])
        source_spec = ExperimentSpec(_read(local(item["data_ref"]["spec_ref"])))
        source = source_spec.to_dict()
        previous = next((a for a in source["arms"] if a["arm_id"] == item["data_ref"]["arm_id"]), None)
        recipe_source_pins = {hashlib.sha256(_payload(local(item["recipe_ref"]))).hexdigest()}
        if accepted_bundle is not None:
            recipe_source_pins.add(file_digest(local(accepted_bundle["contract_ref"])))
        if (previous is None or previous["family"] != {"name": family[0], "version": family[1]}
                or previous["training"]["recipe"]["sha256"] not in recipe_source_pins
                or previous["data"]["dataset"]["sha256"] != source_baseline.dataset.manifest_sha256
                or previous["initialization"] != {"kind": "frozen_state", "seed": 42,
                                                   "state_sha256": source_baseline.initialization_sha256}
                or training_adapter(previous).mode(previous) != source_baseline.mode
                or (source_baseline.mode == "composed" and list(source_baseline.addons) != previous["addons"])):
            raise ValueError("Frozen data declaration, initialization and reference recipe differ")
        roles = {r["role"]: r for r in previous["data"]["roles"]}
        if set(roles) != {"train", "development"} or any(
                source_record[r + "_role_identity"] != roles[r]["usage_sha256"]
                or record[r + "_role_identity"] != roles[r]["usage_sha256"] for r in roles):
            raise ValueError("Frozen data role usage differs from reference recipe")
        if type(item["training"]) is not dict or set(item["training"]) - ({"learning_rate", "weight_decay", "horizon_passes", "execution_mode", "profiling", "horizon_steps", "evaluation_every_steps"} if modern else {"learning_rate", "weight_decay"}):
            raise ValueError("Unsupported training parameter; use learning_rate/weight_decay or Modern2D horizon_passes")
        if type(item["model"]["addons"]) is not list:
            raise ValueError("Model addons must be an ordered declaration list")
        addons = []
        for addon in item["model"]["addons"]:
            if type(addon) is str:
                addon = {"name": addon}
            if type(addon) is not dict or "name" not in addon or set(addon) - {"name", "version", "config"}:
                raise ValueError("Addon requires a catalog name and optional version/config")
            addons.append(build_addon_declaration(family, addon["name"], repo_root=root,
                version=addon.get("version", "1"), config=addon.get("config")))
        arm = deepcopy(previous)
        is_reference = pair is not None and item["arm_id"] == pair["reference_arm_id"]
        arm.update(arm_id=item["arm_id"], scientific_role="reference" if is_reference else "candidate", addons=addons,
                   addon_semantics="ordered" if addons else "baseline")
        if is_reference and addons != previous["addons"]:
            raise ValueError("Same-run reference must retain its frozen recipe and addons")
        if is_reference and modern and set(item["training"]) - {
                "horizon_passes", "execution_mode", "profiling"}:
            raise ValueError("Same-run reference only permits its shared horizon and execution fields")
        if is_reference and not modern and item["training"]:
            raise ValueError("Same-run reference must retain its frozen recipe and addons")
        mode = training_adapter(arm).mode(arm)
        edits = dict(item["training"])
        profiling_config = edits.pop("profiling", None)
        if profiling_config is not None:
            from .family_execution_profile import ProfileConfig
            ProfileConfig.from_dict(profiling_config)
        if modern:
            passes = edits.pop("horizon_passes", baseline.formal_training.horizon_passes)
            exact_steps = edits.pop("horizon_steps", None)
            evaluation_every = edits.pop("evaluation_every_steps", None)
            if (exact_steps is None) != (evaluation_every is None) or (exact_steps is not None and not transfer_study):
                raise ValueError("Explicit step clock requires an authorized transfer study")
            execution_mode = edits.pop("execution_mode", "original")
            if type(passes) is not int or passes < 1:
                raise ValueError("Modern2D horizon_passes must be a positive integer")
            steps = passes * (baseline.train_rows // baseline.batch_size) if exact_steps is None else exact_steps
            if type(steps) is not int or steps < 1:
                raise ValueError("Frozen step horizon must be a positive integer")
            physical_passes = passes if exact_steps is None else (steps + baseline.train_rows // baseline.batch_size - 1) // (baseline.train_rows // baseline.batch_size)
            edits.update(optimizer_steps=steps,
                schedule=replace(baseline.schedule, total_units=steps, reference_epochs=passes),
                formal_training=Modern2DFormalTrainingProfile(purpose="configured-screen",
                    horizon_passes=physical_passes, horizon_steps=steps, execution_mode=execution_mode,
                    evaluation_every_steps=evaluation_every))
        recipe = replace(baseline, mode=mode, addons=tuple(addons) if mode == "composed" else (), **edits)
        if (modern and authority.get("purpose") not in {
                "CONFIGURED_EXECUTION_PROFILE", "CONFIGURED_EXECUTION_REPAIR"}
                and item["purpose"] == "architecture_comparison"
                and recipe.selection != "best-development-live"):
            raise ValueError("Modern2D architecture comparison requires best-development-live selection")
        bindings = family_arm_bindings(recipe)
        arm["base"] = bindings["base"]
        arm["training"].update({k: bindings[k] for k in ("sampler", "transform", "objective")})
        names = tuple(a["name"] for a in addons)
        selected = names if len(names) > 1 else (names[0] if names else None)
        requirements = record["acceptance_requirements"]
        if preset is not None and preset["target"]["development_target_sha256"] != requirements["target_sha256"]:
            raise ValueError("Preset target manifest differs from its frozen recipe")
        kwargs = dict(configuration=recipe, source_idx_sha256=requirements["source_idx_sha256"],
            target_sha256=requirements["target_sha256"], train_role_identity=record["train_role_identity"],
            development_role_identity=record["development_role_identity"], logging=record.get("logging"))
        if mode == "composed":
            from .family_training_runner import build_screen_recipe
            new_record = build_screen_recipe(mode, family=family, **kwargs)
        else:
            new_record = build_family_recipe(family, addon=selected, **kwargs)
        if "profiling" in record:
            new_record["profiling"] = deepcopy(record["profiling"])
        if profiling_config is not None:
            new_record["profiling"] = deepcopy(profiling_config)
        initial = Path(item["initial_state_ref"])
        initial = initial if initial.is_absolute() else local(item["initial_state_ref"], packaged=False)
        _safe_local(initial.absolute())
        inspect_frozen_state_artifact(initial, expected_state_sha256=recipe.initialization_sha256)
        reference = None
        if modern:
            diagnostic = authority.get("purpose") in {
                "CONFIGURED_EXECUTION_PROFILE", "CONFIGURED_EXECUTION_REPAIR"}
            declared_reference_ref = item.get("reference_ref")
            retained_bundle_declared = (isinstance(declared_reference_ref, dict)
                                         and "bundle_ref" in declared_reference_ref)
            if pair is None and not diagnostic and not retained_bundle_declared and not transfer_study:
                raise ValueError("Modern2D scientific training requires a same-run or retained frozen reference")
            if item["purpose"] == "context_only" and not diagnostic:
                raise ValueError("Modern2D scientific training cannot use context_only")
            if diagnostic and item["purpose"] != "context_only":
                raise ValueError("Modern2D execution diagnostics require context_only")
            if transfer_study and (item["purpose"] != "transfer_study" or retained_bundle_declared):
                raise ValueError("Transfer study must retain noncausal purpose and explicit source provenance")
            # Modern2D always carries an explicit provenance anchor. In a
            # same-run pair this is the retained indexed input/target source;
            # a later scientific candidate may additionally bind an accepted
            # Session reference bundle without retraining that baseline.
            reference_ref = item["reference_ref"]
            if (type(reference_ref) is not dict
                    or {"plan_ref", "arm_id"} - set(reference_ref)
                    or set(reference_ref) - {"plan_ref", "arm_id", "bundle_ref"}):
                raise ValueError("reference_ref: expected plan_ref arm_id and optional bundle_ref")
            if "bundle_ref" in reference_ref:
                _object(reference_ref["bundle_ref"], "path sha256", "reference_ref.bundle_ref")
            reference_plan = _read(local(item["reference_ref"]["plan_ref"]))
            if (type(reference_plan) is not dict
                    or reference_plan.get("spec_identity") != source_spec.identity):
                raise ValueError("Retained reference context is not bound to the selected baseline data declaration")
            reference = next((e for e in reference_plan["arms"]
                              if e["arm_id"] == item["reference_ref"]["arm_id"]), None)
            if reference is None:
                raise ValueError("Existing retained reference arm is missing; no baseline is trained")
            if not retained_bundle_declared and reference["contract"] != pin(item["recipe_ref"]):
                raise ValueError("Retained reference is not bound to the frozen baseline recipe")
            from .modern2d_training_release import PLAN_FORMAT, CONFIGURED_PLAN_FORMAT
            if reference_plan.get("format") not in {PLAN_FORMAT, CONFIGURED_PLAN_FORMAT}:
                raise ValueError("Modern2D requires an existing pinned indexed input plan")
            metadata = retained(reference_plan["inputs"])
            input_ref = (metadata["training_input"]["path"] if
                         reference_plan.get("format") == CONFIGURED_PLAN_FORMAT else
                         (Path(reference_plan["inputs"]["path"]).parent /
                          metadata["training_input"]["path"]).as_posix())
            input_pin = pin(input_ref)
            if (input_pin["sha256"] != recipe.dataset.manifest_sha256
                    or input_pin["sha256"] != metadata["training_input"]["sha256"]):
                raise ValueError("Modern2D accepted binding differs from selected recipe")
            target = retained(reference["target_manifest"])
            if (target["development_source_idx_sha256"] != requirements["source_idx_sha256"]
                    or target["development_target_sha256"] != requirements["target_sha256"]):
                raise ValueError("Modern2D targets differ from retained declaration")
            source_model = deepcopy(metadata["models"][item["reference_ref"]["arm_id"]])
            if source_model["family"] != list(family) or source_model["model_config"] != record["model_config"]:
                raise ValueError("Modern2D source feature config differs from retained recipe")
            from .modern2d_feature_binding import indexed_transform_identity, transform_configuration, validate_indexed_feature_binding
            for role, feature in source_model["feature_bindings"].items():
                if canonical_fingerprint(feature) != source_model["feature_binding_sha256"][role]:
                    raise ValueError("Modern2D source feature binding changed")
                validate_indexed_feature_binding(feature, feature)
                feature["transform_code_sha"] = indexed_transform_identity(family, feature_only=True)
                feature["transform_config_sha"] = canonical_fingerprint(transform_configuration(family, recipe.model_config))
                source_model["feature_binding_sha256"][role] = canonical_fingerprint(feature)
            arm["data"]["feature_sha256"] = canonical_fingerprint(source_model["feature_binding_sha256"])
            from .modern2d_training_release import comparison_identity
            modern_identity, target_transform, contract_identity = comparison_identity(
                arm, recipe, baseline, new_record, metadata, target)
            modern_inputs[item["arm_id"]] = (metadata, input_ref, source_model, recipe, modern_identity)
            modern_comparison_assets[item["arm_id"]] = (target_transform, contract_identity)
            retained_scientific = pair is None and not diagnostic and not transfer_study and "bundle_ref" in reference_ref
            if retained_scientific:
                if item["purpose"] != "architecture_comparison":
                    raise ValueError("Retained Modern2D references support architecture_comparison only")
                from .comparison_readiness import (
                    STRICT_IDENTITY_FIELDS, validate_reference_bundle,
                    validate_role_applicability_plan, validate_target_transform_asset,
                    validate_trace_plan, validate_runtime_qualification_plan)
                bundle_pointer = reference_ref["bundle_ref"]
                bundle = accepted_bundle
                if bundle is None or accepted_bundle_pointer != bundle_pointer:
                    from .evidence_pointers import validate_release_reference_bundle_evidence
                    from .family_pair_acceptance import verify_session_reference
                    bundle = verify_session_reference(root, retained(bundle_pointer))
                    bundle = validate_release_reference_bundle_evidence(
                        bundle, repo_root=root,
                        reference_bundle_path=repo_local_path(root, bundle_pointer["path"]))
                    bundle = validate_reference_bundle(bundle)
                if tuple(question.reference_ids) != (bundle["reference_id"],):
                    raise ValueError("Retained Modern2D reference ID is not frozen in the question")
                # Bootstrap publication may relocate/rewrite compact JSON
                # pointers. Compare the immutable recipe/target identities,
                # not paths or formatting bytes.
                prediction = bundle["prediction_manifest"]
                if (prediction["source_idx_sha256"] != requirements["source_idx_sha256"]
                        or prediction["target_sha256"] != requirements["target_sha256"]):
                    raise ValueError("Retained Modern2D bundle target differs from indexed provenance")
                # The accepted bundle's compact row/target sidecars are a
                # separate publication from the indexed development target
                # manifest. Match their semantic pins and row count rather
                # than comparing paths or serialized bytes.
                row_manifest = _read(local(bundle["row_manifest_ref"]))
                target_manifest = _read(local(bundle["target_manifest_ref"]))
                if (row_manifest.get("source_idx_sha256") != prediction["source_idx_sha256"]
                        or row_manifest.get("rows") != prediction["row_count"]
                        or target_manifest.get("target_sha256") != prediction["target_sha256"]
                        or target_manifest.get("rows") != prediction["row_count"]
                        or target_manifest.get("unit") != "eV"):
                    raise ValueError("Retained Modern2D bundle row/target sidecars differ from indexed provenance")
                bundle_asset = validate_target_transform_asset(_read(local(bundle["target_transform_asset_ref"])))
                if bundle_asset != target_transform:
                    raise ValueError("Retained Modern2D target transform differs from frozen recipe/input")
                reference_identity = bundle["comparison_identity"]
                if set(reference_identity) != set(STRICT_IDENTITY_FIELDS):
                    raise ValueError("Retained Modern2D reference identity is incomplete")
                changed = sorted(field for field in STRICT_IDENTITY_FIELDS
                                 if modern_identity[field] != reference_identity[field])
                if changed != ["architecture_config_identity"]:
                    raise ValueError("Retained Modern2D candidate differs outside architecture intervention")
                # The accepted bundle carries the complete prospective
                # comparison declaration from the original transaction.  The
                # indexed plan remains provenance for recipe/data/target
                # identity and may be the formal (non-configured) plan.
                frozen_comparison = bundle.get("comparison_plan")
                if (type(frozen_comparison) is not dict
                        or frozen_comparison.get("experiment_purpose") != "architecture_comparison"):
                    raise ValueError("Retained Modern2D reference lacks its frozen comparison plan")
                if (frozen_comparison.get("declared_intervention_fields") != ["architecture_config_identity"]
                        or not isinstance(frozen_comparison.get("intervention_group_id"), str)
                        or not frozen_comparison["intervention_group_id"].strip()):
                    raise ValueError("Retained Modern2D comparison intervention differs from the accepted plan")
                frozen_identities = frozen_comparison.get("identities")
                if (type(frozen_identities) is not dict
                        or frozen_identities.get(reference["arm_id"]) != reference_identity
                        or frozen_comparison.get("target_transform_asset") != bundle_asset
                        or frozen_comparison.get("contract_identity") != contract_identity):
                    raise ValueError("Retained Modern2D frozen comparison is not bound to its accepted reference")
                # The bootstrap arm has no candidate prelaunch yet. Reuse its
                # frozen role/trace/runtime declarations and assess a fresh
                # candidate prelaunch against the accepted bundle below.
                role_plan = validate_role_applicability_plan(
                    frozen_comparison.get("role_applicability_plan"),
                    experiment_purpose="architecture_comparison")
                trace_plan = validate_trace_plan(
                    frozen_comparison.get("trace_plan"),
                    experiment_purpose="architecture_comparison", ema_enabled=False)
                runtime_plan = validate_runtime_qualification_plan(
                    frozen_comparison.get("runtime_qualification_plan"))
                context = {"reference_id": bundle["reference_id"],
                           "experiment_purpose": "architecture_comparison",
                           "role_applicability_plan": role_plan,
                           "trace_plan": trace_plan,
                           "runtime_qualification_plan": runtime_plan}
                if "minimum_material_gain_eV" in frozen_comparison:
                    gate = frozen_comparison["minimum_material_gain_eV"]
                    if (type(gate) not in (int, float) or isinstance(gate, bool)
                            or not math.isfinite(gate) or gate < 0):
                        raise ValueError("Modern2D material-gain gate must be finite and non-negative")
                    context["minimum_material_gain_eV"] = gate
                modern_retained_references[item["arm_id"]] = {
                    "bundle": bundle, "bundle_ref": bundle_pointer,
                    "context": context, "identity": modern_identity,
                    "target_transform": bundle_asset,
                    "contract_identity": contract_identity,
                    "changed": changed}
        if pair is not None:
            if preset is not None and target_pin != preset["target_manifest"]:
                raise ValueError("Same-run target differs from its frozen baseline preset")
            if not is_reference and recipe == baseline:
                raise ValueError("Same-run candidate must declare a scientific change")
            target = _read(local(pair["target_manifest_ref"]))
            if target.get("development_target_sha256") != requirements["target_sha256"]:
                raise ValueError("Same-run target manifest differs from the frozen recipe")
            compiled.append((item, arm, new_record, reference, None, None, None, [],
                             str(initial.absolute()), source["evidence"]))
            continue
        if modern and item["arm_id"] in modern_retained_references:
            retained_reference = modern_retained_references[item["arm_id"]]
            compiled.append((item, arm, new_record, reference,
                             retained_reference["bundle"], retained_reference["context"],
                             retained_reference["identity"], retained_reference["changed"],
                             str(initial.absolute()), source["evidence"]))
            continue
        _object(item["reference_ref"], "plan_ref arm_id", "reference_ref")
        reference_plan = _read(local(item["reference_ref"]["plan_ref"]))
        if type(reference_plan) is not dict or reference_plan.get("spec_identity") != source_spec.identity:
            raise ValueError("Retained reference context is not bound to the selected baseline data declaration")
        reference = next((e for e in reference_plan["arms"] if e["arm_id"] == item["reference_ref"]["arm_id"]), None)
        if reference is None:
            raise ValueError("Existing retained reference arm is missing; no baseline is trained")
        if (preset is not None and
                (reference["target_manifest"] if modern else reference["reference_artifacts"]["target_manifest"])
                != preset["target_manifest"]):
            raise ValueError("Retained target differs from its frozen baseline preset")
        if reference["contract"] != pin(item["recipe_ref"]):
            raise ValueError("Retained reference is not bound to the frozen baseline recipe")
        if modern:
            compiled.append((item, arm, new_record, reference, None, None, None, [], str(initial.absolute()), source["evidence"]))
            continue
        bundle = validate_reference_bundle(retained(reference["reference_bundle"]))
        context = retained(reference["comparison_prelaunch"])
        for pointer in reference["reference_artifacts"].values():
            if pin(pointer["path"], packaged=False) != pointer:
                raise ValueError("Retained reference artifact bytes changed")
        identity = deepcopy(bundle["comparison_identity"])
        transform = validate_target_transform_asset(retained(reference["reference_artifacts"]["target_transform_asset"]))
        if (transform["mean"] != baseline.target_mean_eV or transform["std"] != baseline.target_std_eV
                or transform["ddof"] != 1 or transform["variance_convention"] != "sample"
                or transform["asset_id"] != identity["target_transform_identity"]
                or transform["asset_sha256"] != identity["target_transform_asset_sha256"]):
            raise ValueError("Retained target transform differs from the frozen reference normalization")
        required_identity = {"seed": 42, "precision": "fp32", "physical_batch_per_device": 128,
            "gradient_accumulation_steps": 1, "optimizer_steps": requirements["optimizer_steps"],
            "sample_presentations": requirements["sample_presentations"], "ema_enabled": baseline.ema_decay is not None,
            "evaluation_weight_source": baseline.selection_weights}
        if bundle["contract_ref"] != item["recipe_ref"] or any(identity[k] != v for k, v in required_identity.items()):
            raise ValueError("Retained comparison identity is incompatible with the shared reference recipe")
        if any(arm["training"][k] != previous["training"][k] for k in ("sampler", "transform")):
            raise ValueError("Scientific compilation cannot silently change frozen sampler or normalization")
        changed = []
        if (recipe.learning_rate, recipe.weight_decay) != (baseline.learning_rate, baseline.weight_decay):
            identity["optimizer_identity"] = canonical_fingerprint({"reference": identity["optimizer_identity"],
                "learning_rate": recipe.learning_rate, "weight_decay": recipe.weight_decay})
            changed.append("optimizer_identity")
        if arm["base"] != previous["base"] or addons != previous["addons"]:
            identity["architecture_config_identity"] = canonical_fingerprint({"base": arm["base"], "addons": addons})
            changed.append("architecture_config_identity")
        # Reference/composed use the identical default normalized-Gap update.
        # The recipe owner's mode-tagged digest is still preserved in the Spec;
        # that label alone is not a changed scientific loss. Never alias an
        # unverified historical digest or a mode with a different objective.
        default_loss_alias = (baseline.mode in {"reference", "composed"} and recipe.mode in {"reference", "composed"}
            and previous["training"]["objective"] == family_arm_bindings(baseline)["objective"])
        if arm["training"]["objective"] != previous["training"]["objective"] and not default_loss_alias:
            identity["loss_identity"] = arm["training"]["objective"]["sha256"]
            changed.append("loss_identity")
        if not changed:
            raise ValueError("Scientific candidate has no change from the retained reference")
        compiled.append((item, arm, new_record, reference, bundle, context, identity, changed, str(initial.absolute()), source["evidence"]))

    comparison = None
    if modern and pair is not None:
        from .comparison_readiness import INTERVENTION_FIELDS_BY_PURPOSE, STRICT_IDENTITY_FIELDS
        horizons = {(entry[3].formal_training.horizon_passes,
                     entry[3].formal_training.horizon_steps)
                    for entry in modern_inputs.values()}
        if len(horizons) != 1:
            raise ValueError("Same-run Modern2D reference and candidate must share one explicit horizon")
        pair_items = {entry[0]["arm_id"]: entry[0] for entry in compiled}
        purposes = {item["purpose"] for item in pair_items.values()}
        if purposes != {"architecture_comparison"}:
            raise ValueError("Modern2D same-run scientific training requires architecture_comparison")
        declared = ["architecture_config_identity"]
        if set(declared) - INTERVENTION_FIELDS_BY_PURPOSE["architecture_comparison"]:
            raise ValueError("Modern2D comparison intervention is outside the architecture scope")
        reference_id = pair["reference_arm_id"]
        reference_identity = modern_inputs[reference_id][4]
        identities = {arm_id: modern_inputs[arm_id][4] for arm_id in arm_ids}
        if set(reference_identity) != set(STRICT_IDENTITY_FIELDS) or any(
                set(identity) != set(STRICT_IDENTITY_FIELDS) for identity in identities.values()):
            raise ValueError("Modern2D comparison identity must account for every strict field")
        for arm_id, identity in identities.items():
            if arm_id == reference_id:
                continue
            differences = sorted(field for field in STRICT_IDENTITY_FIELDS
                                 if identity[field] != reference_identity[field])
            if differences != declared:
                raise ValueError("Modern2D candidate differs outside the declared architecture intervention: "
                                 + str(differences))
        comparison = {"experiment_purpose": "architecture_comparison",
            "intervention_group_id": question.changed_mechanism,
            "declared_intervention_fields": declared,
                "identities": identities,
                "target_transform_asset": modern_comparison_assets[reference_id][0],
                "role_applicability_plan": {
                    "training_membership": "applicable", "prediction_input": "applicable",
                    "labels_read": "applicable", "metric_computed": "applicable",
                    "selection_used": "applicable", "external_submission": "not_applicable"},
                "trace_plan": {
                    "optimizer_step": True, "sample_presentations": True,
                    "epoch_or_pass": True, "learning_rate": True,
                    "live_train_metric": True, "live_dev_metric": True,
                    "ema_dev_metric": False, "checkpoint_identity": True},
                 "runtime_qualification_plan": {
                     "status": "declared", "runtime_certificate_required": True,
                     "qualification_scope": modern_inputs[reference_id][4]["runtime_certificate_scope"]},
                "contract_identity": modern_comparison_assets[reference_id][1]}
        if pair is not None and "minimum_material_gain_eV" in pair:
            comparison["minimum_material_gain_eV"] = pair["minimum_material_gain_eV"]
    elif modern and modern_retained_references:
        from .comparison_readiness import INTERVENTION_FIELDS_BY_PURPOSE, STRICT_IDENTITY_FIELDS
        if len(modern_retained_references) != len(compiled):
            raise ValueError("Modern2D scientific arms must all use the same retained reference")
        retained_ids = {item["bundle"]["reference_id"] for item in modern_retained_references.values()}
        retained_pointers = {canonical_json(item["bundle_ref"]) for item in modern_retained_references.values()}
        if len(retained_ids) != 1 or len(retained_pointers) != 1:
            raise ValueError("Modern2D scientific arms must share one retained reference bundle")
        first = next(iter(modern_retained_references.values()))
        declared = ["architecture_config_identity"]
        if set(declared) - INTERVENTION_FIELDS_BY_PURPOSE["architecture_comparison"]:
            raise ValueError("Modern2D comparison intervention is outside the architecture scope")
        identities = {arm_id: item["identity"] for arm_id, item in modern_retained_references.items()}
        reference_identity = first["bundle"]["comparison_identity"]
        if (set(reference_identity) != set(STRICT_IDENTITY_FIELDS)
                or any(set(identity) != set(STRICT_IDENTITY_FIELDS) for identity in identities.values())):
            raise ValueError("Modern2D retained comparison identity must account for every strict field")
        comparison = {"experiment_purpose": "architecture_comparison",
            "intervention_group_id": question.changed_mechanism,
            "declared_intervention_fields": declared,
            "identities": identities,
            "reference_id": first["bundle"]["reference_id"],
            "reference_bundle": first["bundle_ref"],
            "reference_identity": reference_identity,
            "target_transform_asset": first["target_transform"],
            "role_applicability_plan": first["context"]["role_applicability_plan"],
            "trace_plan": first["context"]["trace_plan"],
                 "runtime_qualification_plan": first["context"]["runtime_qualification_plan"],
            "contract_identity": first["contract_identity"]}
        if "minimum_material_gain_eV" in first["context"]:
            comparison["minimum_material_gain_eV"] = first["context"]["minimum_material_gain_eV"]

    # Commit only human/source inputs. Never require committing machine envelopes.
    # The selected data/roles are copied into this package's canonical Spec.
    # A baseline experiment_spec.json is a committed compilation input, not a
    # second package sidecar; preserve the existing reserved-name boundary.
    selected_sources = sorted(set(runtime_source_files(root, sorted(records - data_specs))) | data_specs)
    commit = _assert_clean_paths(root, selected_sources)
    if commit != initial_commit:
        raise ValueError("Scientific source HEAD changed during compilation")
    committed = committed_payloads(root, commit, selected_sources)
    if any(_payload(root / n) != committed[n] for n in selected_sources):
        raise ValueError("Scientific source differs from committed bytes")
    generated, arms, prospective, acceptance, jobs = {}, [], [], [], []
    def derive(name, payload):
        raw = canonical_json(payload).encode()
        generated[name] = raw
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
    if "authority" in value:
        derive(authority_ref, authority)
    for device, (item, arm, recipe, reference, bundle, context, identity, changed, initial, evidence) in enumerate(compiled):
        arm_id = arm["arm_id"]
        recipe_pin = derive(prefix + "/" + arm_id + "/recipe.json", recipe)
        arm["training"]["recipe"] = {"name": FAMILIES[(arm["family"]["name"], arm["family"]["version"])].recipe,
                                     "version": "1", "sha256": recipe_pin["sha256"]}
        retained_modern = modern and pair is None and bundle is not None
        prelaunch = None if (pair is not None or (modern and not retained_modern)) else assess_comparison_prelaunch(candidate_id=arm_id,
            candidate_plan={"comparison_identity": identity, "source_config_status": "frozen", "source_commit_or_archive": commit},
            reference_id=bundle["reference_id"], reference_bundle=bundle, experiment_purpose=item["purpose"],
            intervention_group_id=question.changed_mechanism, declared_intervention_fields=changed,
            **{k: context[k] for k in ("role_applicability_plan", "trace_plan", "runtime_qualification_plan")})
        if prelaunch is not None and not prelaunch["prelaunch_ready"]:
            raise ValueError("Scientific change is not a comparable declared intervention")
        comparison_pin = (derive(prefix + "/" + arm_id + "/comparison_prelaunch.json", prelaunch)
                          if prelaunch is not None else None)
        attempt_suffix = "" if value["attempt_id"] == "v1" else "-" + value["attempt_id"]
        tid = "T-" + value["logical_run_id"] + attempt_suffix + "-" + arm_id
        proposal = build_execution_proposal(question=question, arm=arm, trajectory_id=tid, source_commit=commit,
            run=value["logical_run_id"], version=value["attempt_id"], contract_ref=recipe_pin["path"],
            contract_refs=[recipe_pin["path"], item["data_ref"]["spec_ref"], config_name] +
                ([item["reference_ref"]["bundle_ref"]["path"]] if retained_modern else []),
            authority_ref=authority_ref, role_ref=value["role_ref"], decision_ref=value["decision_ref"],
            timestamp=value["timestamp"], profile=profile,
            action_type=("execution_diagnostic" if authority.get("purpose") in {
                "CONFIGURED_EXECUTION_PROFILE", "CONFIGURED_EXECUTION_REPAIR"} else "training"))
        proposal["decision_state"]["policy_version"] = policy_version
        if pair is not None:
            attempt = value["logical_run_id"] + "-" + value["attempt_id"]
            proposal["trajectory"]["actions"][0]["attempt_ids"] = [attempt]
            for cost in proposal["costs"]:
                cost["attempt_id"] = attempt
        proposal_pin = derive(prefix + "/" + arm_id + "/plan_input.json", proposal)
        prospective.append({"arm_id": arm_id, "trajectory_id": tid, "plan_spec_ref": proposal_pin["path"],
            "plan_spec_sha256": proposal_pin["sha256"],
            "output": f"experiments/{value['experiment_id']}/runs/{value['logical_run_id']}{attempt_suffix}/prospective/{arm_id}"})
        arms.append(arm)
        if not modern:
            entry = {"arm_id": arm_id, "adapter": training_adapter(arm).artifact_adapter,
                "expected": recipe["acceptance_requirements"], "contract": recipe_pin,
                **({"target_manifest": target_pin} if pair is not None else {
                    "comparison_prelaunch": comparison_pin, "reference_bundle": reference["reference_bundle"],
                    "reference_artifacts": reference["reference_artifacts"]})}
        else:
            entry = {"arm_id": arm_id, "adapter": training_adapter(arm).artifact_adapter,
                "expected": recipe["acceptance_requirements"], "contract": recipe_pin,
                "target_manifest": reference["target_manifest"]}
            if retained_modern:
                entry.update(comparison_prelaunch=comparison_pin,
                    reference_bundle=item["reference_ref"]["bundle_ref"])
                if reference.get("reference_artifacts") is not None:
                    entry["reference_artifacts"] = reference["reference_artifacts"]
        acceptance.append(entry)
        jobs.append({"arm_id": arm_id, "device": device, "recipe": recipe_pin["path"], "initial_state": initial})
    if any(c[-1] != compiled[0][-1] for c in compiled):
        raise ValueError("Scientific arms must share the existing V5 evidence policy")
    spec = ExperimentSpec({"schema_version": SCHEMA_VERSION_V2, "experiment_id": value["experiment_id"],
        "logical_run_id": value["logical_run_id"], "arms": arms, "platform": platform["platform"],
        "prospective": {"arms": prospective, **({"same_run_replay": {
            "reference_arm_id": pair["reference_arm_id"],
            "candidate_arm_ids": [a for a in arm_ids if a != pair["reference_arm_id"]]}} if pair is not None else {})},
        "evidence": evidence, "terminal_protocol": TERMINAL_PROTOCOL})
    acceptance_value = {"format": ("molgap-family-same-run-acceptance-plan-v1" if pair is not None else "molgap-family-acceptance-plan-v1"), "spec_identity": spec.identity, "arms": acceptance}
    if comparison is not None:
        acceptance_value["comparison"] = comparison
    if modern:
        from .modern2d_training_release import CONFIGURED_PLAN_FORMAT, validate_configured_authority
        validate_configured_authority(authority, spec.to_dict(), {name: item[3] for name, item in modern_inputs.items()})
        if (authority["run_reference"] != platform["kaggle"]["kernel"]
                or authority["gpu"]["owner"] != platform["kaggle"]["account"]):
            raise ValueError("Configured platform differs from human authority")
        # Family anchors may retain identical accepted manifest bytes at
        # different repository paths. Data identity is content, not that alias.
        shared_fields = ("format", "scientific_training_authorized", "target_mean_eV", "target_std_eV")
        if (len({pin(item[1])["sha256"] for item in modern_inputs.values()}) != 1
                or len({canonical_fingerprint({key: item[0][key] for key in shared_fields})
                        for item in modern_inputs.values()}) != 1):
            raise ValueError("Modern2D arms must share one accepted indexed input binding")
        metadata, input_ref, _, _, _ = next(iter(modern_inputs.values()))
        metadata = {key: deepcopy(metadata[key]) for key in shared_fields}
        # An accepted input is retained source, not compiler-generated JSON.
        # Pin its original bytes so pretty-printed accepted manifests need not
        # be canonicalized or admitted through a weaker package validator.
        metadata["training_input"] = pin(input_ref)
        metadata["models"] = {name: item[2] for name, item in modern_inputs.items()}
        metadata["authority"] = pin(authority_ref)
        metadata_pin = derive(prefix + "/training_inputs.json", metadata)
        acceptance_value.update(format=CONFIGURED_PLAN_FORMAT, authority=pin(authority_ref), inputs=metadata_pin)
        if transfer_study:
            acceptance_value["transfer"] = pin(authority["transfer_study_ref"]["path"])
    acceptance_pin = derive(prefix + "/acceptance_plan.json", acceptance_value)
    plan = {"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity, "arms": jobs,
        "acceptance_plan": acceptance_pin["path"], "kaggle": platform["kaggle"], "source_profile": "runtime-v1",
        "source_records": sorted(records - data_specs), "generated_records": {n: hashlib.sha256(b).hexdigest() for n, b in generated.items()},
        "configuration_source": {"path": config_name, "sha256": input_sha, "compiler": value["format"]}}
    # Validate every schema before making immutable, regenerable metadata visible.
    from .kaggle_workflow import validate_plan
    validate_plan(spec, plan["kaggle"])
    if (_assert_clean_paths(root, selected_sources) != commit or
            any(_payload(root / n) != committed[n] for n in selected_sources)):
        raise ValueError("Scientific source HEAD changed during compilation")
    for name, raw in generated.items():
        publish_immutable_bytes(repo_local_path(root, name), raw)
    return value, spec, plan, paths


def freeze_experiment_input(path, repo_root):
    """Idempotent local freeze; one run/attempt directory outside the worktree."""
    value,spec,plan,paths = resolve_experiment_input(path,repo_root)
    return _freeze(value,spec,plan,paths,repo_root=repo_root)


def locate_run(profile, repo_root, run_id, attempt_id, *, _verification=None):
    """Read the original frozen run by ID; never create a run or infer artifacts."""
    from .experiment_launch import read_launch_receipt
    from .experiment_package import _PackageVerification, verify_experiment_source_package
    from .research_memory.trace import file_digest
    paths=MachinePaths.from_file(profile).outside_checkout(repo_root)
    run=paths.run_directory(run_id,attempt_id)
    inputs=run/"inputs"
    binding=_read(inputs/"input_binding.json")
    spec=ExperimentSpec(_read(inputs/"experiment_spec.json"))
    plan=_read(inputs/"workflow_plan.json")
    if type(binding) is not dict or type(plan) is not dict:
        raise ValueError("Original frozen run records must be objects")
    from .common.spec import _digest
    if binding.get("format") not in {"molgap-experiment-input-v1", "molgap-scientific-experiment-v1"}:
        raise ValueError("Unknown original frozen input binding")
    _digest(binding.get("input_identity"), "original input identity")
    validate_relative_path(binding.get("authority_ref"))
    monitors=binding.get("monitor_paths")
    if (type(monitors) is not list or not monitors or any(type(p) is not str for p in monitors)
            or len(monitors) != len(set(monitors))):
        raise ValueError("Original frozen monitor bindings changed")
    for pointer in monitors:
        validate_relative_path(pointer)
    if (spec.to_dict()["logical_run_id"] != run_id or spec.identity != binding.get("spec_identity")
            or plan.get("spec_identity") != spec.identity
            or canonical_fingerprint(plan) != binding.get("workflow_identity")):
        raise ValueError("Original frozen run binding changed")
    package=run/"prepared/package"
    verification=_verification if _verification is not None else _PackageVerification()
    manifest=(verify_experiment_source_package(package, _verification=verification)
              if package.exists() else None)
    if manifest and manifest["spec_identity"] != spec.identity:
        raise ValueError("Run package differs from original frozen input")
    if "source_commit" in binding:
        source_commit=_commit(binding["source_commit"])
        if manifest and manifest["source_commit"] != source_commit:
            raise ValueError("Run package differs from original bound source")
    receipt_path=run/"operations/launch_receipt.json"
    _safe_local(receipt_path)
    if receipt_path.exists() and manifest is None:
        raise ValueError("A receipt without its original package cannot be resolved")
    receipt=(read_launch_receipt(receipt_path,spec,package,
        expected_package_identity=manifest["package_identity"], _verification=verification)
        if receipt_path.exists() else None)
    files={}
    for name in ("inputs/input_binding.json","inputs/experiment_spec.json","inputs/workflow_plan.json",
                 "prepared/workflow_report.json","prepared/release_report.json",
                 "operations/submission_state.json","operations/launch_receipt.json"):
        path=run/name
        _safe_local(path)
        files[name]={"path":str(path),"sha256":file_digest(path)} if path.is_file() else None
    return {"status":"RUN_LOCATED_ONLY","run_directory":str(run),"spec_identity":spec.identity,
            "package_identity":manifest["package_identity"] if manifest else None,
            "receipt":receipt,"files":files,"network_accessed":False,
            "scientific_acceptance":"NOT_EVALUATED"}


def _freeze(value,spec,plan,paths,*,repo_root=None):
    from .experiment_launch import canonical_json
    run = paths.run_directory(spec.to_dict()["logical_run_id"],value["attempt_id"])
    binding = {"format":value["format"],"input_identity":canonical_fingerprint(value),
               "spec_identity":spec.identity,"workflow_identity":canonical_fingerprint(plan),
               "authority_ref":_authority_reference(value,plan),"monitor_paths":value["monitor_paths"]}
    frozen = run / "inputs"
    _safe_local(frozen / "input_binding.json")
    if (frozen / "input_binding.json").exists():
        original = _read(frozen / "input_binding.json")
        if original.get("source_commit") is not None:
            binding["source_commit"] = original["source_commit"]
        _check_frozen_binding(original, value, spec, plan)
    elif value["format"] == "molgap-scientific-experiment-v1":
        from .experiment_package import _git
        commit = _commit(_git(Path(repo_root), "rev-parse", "HEAD").decode().strip())
        prefix = f"experiments/{value['experiment_id']}/generated/" + canonical_fingerprint({
            "input": plan["configuration_source"]["sha256"], "commit": commit}) + "/"
        if not plan["generated_records"] or any(not n.startswith(prefix) for n in plan["generated_records"]):
            raise ValueError("Source HEAD changed between compilation and first freeze")
        binding["source_commit"] = commit
    # Binding first: a partial freeze can be retried with identical bytes only.
    publish_immutable_bytes(frozen/"input_binding.json",canonical_json(binding).encode())
    publish_immutable_bytes(frozen/"experiment_spec.json",spec.to_json().encode())
    publish_immutable_bytes(frozen/"workflow_plan.json",canonical_json(plan).encode())
    return {"status":"INPUTS_FROZEN","run_directory":str(run),"spec_identity":spec.identity,
            "submitted":False,"scientific_acceptance":"NOT_EVALUATED"}


def prepare_experiment_input(path, repo_root):
    """Recover regenerable pre-publication staging; never retry ambiguous effects."""
    from .experiment_package import verify_experiment_source_package
    from .experiment_workflow import prepare_workflow
    started=perf_counter()
    value,spec,plan,paths=resolve_experiment_input(path,repo_root)
    frozen=_freeze(value,spec,plan,paths,repo_root=repo_root)
    run=Path(frozen["run_directory"]);prepared=run/"prepared"
    retained=None
    if prepared.exists():
        _safe_local(prepared)
        report=_read(prepared/"workflow_report.json") if (prepared/"workflow_report.json").is_file() else {}
        if report.get("status") != "PREPARED_FOR_PLATFORM":
            _safe_local(run/"operations")
            safe_local_retry=(report.get("format")=="molgap-workflow-preparation-v1"
                and report.get("spec_identity")==spec.identity
                and report.get("submitted") is False and report.get("prospective_published") is False
                and report.get("publication_intent") is not True
                and report.get("status") in {"ERROR_BEFORE_PUBLICATION","BLOCKED"}
                and not (run/"operations").exists())
            if not safe_local_retry:
                return {"status":"RECONCILIATION_REQUIRED","prepared":str(prepared),"submitted":False,
                        "reason":"Published/unknown preparation or platform effects require reconciliation"}
            retained=run/"local_preparation_failures"/("prepared-"+uuid4().hex)
            _safe_local(retained)
            if prepared.resolve().parent != run.resolve() or not retained.resolve().is_relative_to(run.resolve()):
                raise ValueError("Local preparation retention escaped its bound run")
        else:
            manifest=verify_experiment_source_package(prepared/"package")
            if manifest["spec_identity"] != spec.identity or manifest["package_identity"] != report.get("package_identity"):
                raise ValueError("Existing prepared package/input binding changed")
            binding=_read(run/"inputs/input_binding.json")
            if "source_commit" in binding and binding["source_commit"] != manifest["source_commit"]:
                raise ValueError("Existing prepared package differs from the original bound source")
            result={"status":"PREPARED_FOR_PLATFORM","reused_preparation":True,
                    "package_identity":manifest["package_identity"],
                    "source_commit":manifest.get("source_commit")}
    if not prepared.exists() or retained is not None:
        if value["format"] == "molgap-scientific-experiment-v1":
            from .experiment_package import _git
            try:
                original_commit=_bound_source_commit(run,spec,Path(repo_root).absolute())
            except (ValueError, KeyError, OSError):
                return {"status":"RECONCILIATION_REQUIRED","prepared":str(prepared),"submitted":False,
                    "reason":"Original source binding is unavailable; reconcile the original run without recompiling"}
            current_commit=_git(Path(repo_root),"rev-parse","HEAD").decode().strip()
            if current_commit != original_commit:
                return {"status":"RECONCILIATION_REQUIRED","prepared":str(prepared),"submitted":False,
                    "source_commit":original_commit,
                    "reason":"Unpublished preparation requires the original bound source checkout; current HEAD is not substituted"}
        if retained is not None:
            retained.parent.mkdir(parents=True,exist_ok=True)
            prepared.rename(retained)  # Preserve every byte; only regenerable staging is replaced.
        result=prepare_workflow(spec,Path(repo_root).absolute(),plan,prepared,
                                source_cache=paths.artifact_root/"source-packages/runtime-cache")
    result = {**result,"prepared":str(prepared),"operations":str(run/"operations"),
              "authority_ref":_authority_reference(value,plan),"monitor_paths":value["monitor_paths"],
              "local_call_seconds":perf_counter()-started,"submitted":False}
    if result["status"] == "PREPARED_FOR_PLATFORM":
        _observe_preparation(run, spec, result)
    return result


def _observe_preparation(run, spec, result):
    """Retain measured local work, without rewriting frozen bytes or the journal."""
    from datetime import datetime, timezone
    from .experiment_launch import canonical_json
    observation = {"format":"molgap-local-preparation-observation-v1",
        "observed_at":datetime.now(timezone.utc).isoformat(),
        "status":result["status"], "spec_identity":spec.identity,
        "package_identity":result.get("package_identity"),
        "source_commit":result.get("source_commit"),
        "reused_preparation":result.get("reused_preparation",False),
        "local_call_seconds":result["local_call_seconds"]}
    path = run / "local_observations" / ("preparation-" + uuid4().hex + ".json")
    try:
        publish_immutable_bytes(path, canonical_json(observation).encode())
    except (OSError, ValueError):
        # An auxiliary observation failure is not loss of the durable package.
        result.update(local_observation_status="MISSING",
                      local_observation_reason="Local timing observation could not be retained")
    else:
        result.update(local_observation_status="RETAINED", local_observation=str(path))


def local_preparation_error(error,stage):
    """Finite local diagnostics, never arbitrary exception/SDK request text."""
    known={
        "Missing frozen initialization file":("arms.initial_state","Frozen initialization file is missing"),
        "--execute requires explicit credentials before preparation":("credentials","Explicit credentials are required for --execute"),
        "Editable input already owns the immutable run/authority/monitor bindings":("experiment","Do not override frozen run/authority/monitor bindings"),
        "Existing authority reference is missing; no authorization is inferred":("authority_ref","Existing authority file is missing"),
        "Editable input workflow differs from its scientific declaration":("workflow_ref","Workflow identity differs from the scientific declaration"),
        "Unsupported training parameter; only learning_rate and weight_decay are editable":("arms.training","Only learning_rate and weight_decay are supported in this compiler"),
        "Scientific input supports one paired seed 42; no seed expansion":("seeds","This compiler supports exactly seed 42"),
        "Existing authority does not cover exactly these arms, seed and positive ceiling":("authority_ref","Existing authority must cover the declared arms, seed and positive cost ceiling"),
        "Frozen data declaration, initialization and reference recipe differ":("arms.data_ref","Frozen data, initialization and reference recipe must match"),
        "Scientific candidate has no change from the retained reference":("arms","No scientific change from the retained reference was declared"),
        "Scientific change is not a comparable declared intervention":("arms.purpose","Declared purpose does not isolate the scientific change"),
        "Scientific compiler requires a reviewed catalog scientific_input_abi":("arms.model","Family execution must declare a supported scientific input ABI"),
        "Retained reference context is not bound to the selected baseline data declaration":("arms.data_ref","Select the baseline declaration bound by the retained reference context"),
        "Frozen scientific input changed; create an explicit new run instead of overwriting":("logical_run_id","Scientific input changed; create an explicit new run rather than overwrite this run"),
        "Original frozen inputs are incomplete; reconcile the original freeze":("run.inputs","Original freeze is incomplete; reconcile it without recompiling under a new HEAD"),
    }
    field,reason=known.get(str(error),("configuration","Local validation failed; check configuration fields and referenced files"))
    if isinstance(error,FileNotFoundError):
        field,reason="referenced_file","A required local file is missing"
    return {"status":"BLOCKED","error_type":type(error).__name__,"stage":stage,"field":field,"reason":reason}
