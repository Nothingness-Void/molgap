"""Human scientific configuration compiled through existing declaration owners.

This resolver never synthesizes hypotheses, authorizations or accepted references.
It freezes normal JSON through existing canonical parsers; old --spec is intact.
"""
from pathlib import Path
from time import perf_counter
import json
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
        return _compile_scientific_input(path, root, value)
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


def _compile_scientific_input(path, root, value):
    """Compile declarations, not permission, reference acceptance or trainer code.

    The existing recipe fixes data, sampler, budget, selection and runtime. Only
    catalog addons and typed optimizer parameters are editable in this version.
    Derived JSON is explicitly separate from committed executable source.
    """
    from copy import deepcopy
    from dataclasses import replace
    import hashlib
    import math
    from .common.declarations import ExecutionQuestion, ExecutionProfile, build_execution_proposal
    from .comparison_readiness import assess_comparison_prelaunch, validate_reference_bundle, validate_target_transform_asset
    from .experiment_execution import build_addon_declaration, build_family_recipe, training_adapter
    from .experiment_launch import canonical_json
    from .experiment_source_inventory import runtime_source_files
    from .experiment_spec import FAMILIES, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL
    from .family_training_runner import read_recipe, family_arm_bindings
    from .v4_bundle import _assert_clean_paths, _payload, committed_payloads
    from .v4_runtime import inspect_frozen_state_artifact
    from .research_memory.trace import file_digest

    _object(value, "format experiment_id logical_run_id attempt_id machine_profile authority_ref monitor_paths "
            "timestamp question profile platform_ref role_ref decision_ref arms seeds", "scientific input")
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
    authority = _read(local(value["authority_ref"]))
    local(value["role_ref"]); local(value["decision_ref"])
    platform = _read(local(value["platform_ref"]))
    _object(platform, "platform kaggle", "platform input")
    question = ExecutionQuestion(**value["question"])
    profile = ExecutionProfile(**value["profile"])
    if platform["platform"]["name"] != "kaggle" or profile.platform != "kaggle":
        raise ValueError("Scientific preparation requires the existing Kaggle workflow adapter")
    if type(value["arms"]) is not list or not 1 <= len(value["arms"]) <= 2:
        raise ValueError("One T4x2 slot supports one or two independent arms")
    arm_ids = [a["arm_id"] for a in value["arms"]]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("Duplicate scientific arm")
    budget = authority.get("allocated_t4_hours_ceiling")
    if (authority.get("user_authorized") is not True or type(budget) not in (int, float)
            or not math.isfinite(budget) or budget <= 0 or authority.get("arms") != arm_ids
            or authority.get("seeds") != [42]):
        raise ValueError("Existing authority does not cover exactly these arms, seed and positive ceiling")
    compiled, data_specs = [], set()
    for item in value["arms"]:
        _object(item, "arm_id model recipe_ref data_ref initial_state_ref training reference_ref purpose", "scientific arm")
        _identifier(item["arm_id"], "arm_id")
        _object(item["model"], "family version addons", "model")
        family = (item["model"]["family"], item["model"]["version"])
        if family not in {("neural_atom_k1", "3"), ("gptrans_t", "2")}:
            raise ValueError("Scientific compiler supports reviewed shared K1/3 and GPTrans/2 only")
        record, baseline = read_recipe(local(item["recipe_ref"]))
        if baseline.interface != family or baseline.seed != 42 or not baseline.dataset.development:
            raise ValueError("Frozen screen recipe does not match model, seed or development contract")
        _object(item["data_ref"], "spec_ref arm_id", "data_ref")
        data_specs.add(item["data_ref"]["spec_ref"])
        source = ExperimentSpec(_read(local(item["data_ref"]["spec_ref"]))).to_dict()
        previous = next((a for a in source["arms"] if a["arm_id"] == item["data_ref"]["arm_id"]), None)
        if (previous is None or previous["family"] != {"name": family[0], "version": family[1]}
                or previous["training"]["recipe"]["sha256"] != hashlib.sha256(_payload(local(item["recipe_ref"]))).hexdigest()
                or previous["data"]["dataset"]["sha256"] != baseline.dataset.manifest_sha256
                or previous["initialization"] != {"kind": "frozen_state", "seed": 42,
                                                   "state_sha256": baseline.initialization_sha256}
                or training_adapter(previous).mode(previous) != baseline.mode
                or (baseline.mode == "composed" and list(baseline.addons) != previous["addons"])):
            raise ValueError("Frozen data declaration, initialization and reference recipe differ")
        roles = {r["role"]: r for r in previous["data"]["roles"]}
        if set(roles) != {"train", "development"} or any(
                record[r + "_role_identity"] != roles[r]["usage_sha256"] for r in roles):
            raise ValueError("Frozen data role usage differs from reference recipe")
        if type(item["training"]) is not dict or set(item["training"]) - {"learning_rate", "weight_decay"}:
            raise ValueError("Unsupported training parameter; only learning_rate and weight_decay are editable")
        if type(item["model"]["addons"]) is not list:
            raise ValueError("Model addons must be an ordered declaration list")
        addons = []
        for addon in item["model"]["addons"]:
            _object(addon, "name version config", "addon")
            addons.append(build_addon_declaration(family, addon["name"], repo_root=root,
                version=addon["version"], config=addon["config"]))
        arm = deepcopy(previous)
        arm.update(arm_id=item["arm_id"], scientific_role="candidate", addons=addons,
                   addon_semantics="ordered" if addons else "baseline")
        mode = training_adapter(arm).mode(arm)
        recipe = replace(baseline, mode=mode, addons=tuple(addons) if mode == "composed" else (), **item["training"])
        bindings = family_arm_bindings(recipe)
        arm["base"] = bindings["base"]
        arm["training"].update({k: bindings[k] for k in ("sampler", "transform", "objective")})
        names = tuple(a["name"] for a in addons)
        selected = names if len(names) > 1 else (names[0] if names else None)
        requirements = record["acceptance_requirements"]
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
        initial = Path(item["initial_state_ref"])
        initial = initial if initial.is_absolute() else local(item["initial_state_ref"], packaged=False)
        _safe_local(initial.absolute())
        inspect_frozen_state_artifact(initial, expected_state_sha256=recipe.initialization_sha256)
        _object(item["reference_ref"], "plan_ref arm_id", "reference_ref")
        reference_plan = _read(local(item["reference_ref"]["plan_ref"]))
        reference = next((e for e in reference_plan["arms"] if e["arm_id"] == item["reference_ref"]["arm_id"]), None)
        if reference is None:
            raise ValueError("Existing retained reference arm is missing; no baseline is trained")
        if reference["contract"] != pin(item["recipe_ref"]):
            raise ValueError("Retained reference is not bound to the frozen baseline recipe")
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
        if arm["training"]["objective"] != previous["training"]["objective"]:
            identity["loss_identity"] = arm["training"]["objective"]["sha256"]
            changed.append("loss_identity")
        if not changed:
            raise ValueError("Scientific candidate has no change from the retained reference")
        compiled.append((item, arm, new_record, reference, bundle, context, identity, changed, str(initial.absolute()), source["evidence"]))

    # Commit only human/source inputs. Never require committing machine envelopes.
    # The selected data/roles are copied into this package's canonical Spec.
    # A baseline experiment_spec.json is a committed compilation input, not a
    # second package sidecar; preserve the existing reserved-name boundary.
    selected_sources = sorted(set(runtime_source_files(root, sorted(records - data_specs))) | data_specs)
    commit = _assert_clean_paths(root, selected_sources)
    committed = committed_payloads(root, commit, selected_sources)
    if any(_payload(root / n) != committed[n] for n in selected_sources):
        raise ValueError("Scientific source differs from committed bytes")
    input_sha = hashlib.sha256(_payload(config_path)).hexdigest()
    prefix = f"experiments/{value['experiment_id']}/generated/" + canonical_fingerprint({"input": input_sha, "commit": commit})
    generated, arms, prospective, acceptance, jobs = {}, [], [], [], []
    def derive(name, payload):
        raw = canonical_json(payload).encode()
        generated[name] = raw
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
    for device, (item, arm, recipe, reference, bundle, context, identity, changed, initial, evidence) in enumerate(compiled):
        arm_id = arm["arm_id"]
        recipe_pin = derive(prefix + "/" + arm_id + "/recipe.json", recipe)
        arm["training"]["recipe"] = {"name": FAMILIES[(arm["family"]["name"], arm["family"]["version"])].recipe,
                                     "version": "1", "sha256": recipe_pin["sha256"]}
        prelaunch = assess_comparison_prelaunch(candidate_id=arm_id,
            candidate_plan={"comparison_identity": identity, "source_config_status": "frozen", "source_commit_or_archive": commit},
            reference_id=bundle["reference_id"], reference_bundle=bundle, experiment_purpose=item["purpose"],
            intervention_group_id=question.changed_mechanism, declared_intervention_fields=changed,
            **{k: context[k] for k in ("role_applicability_plan", "trace_plan", "runtime_qualification_plan")})
        if not prelaunch["prelaunch_ready"]:
            raise ValueError("Scientific change is not a comparable declared intervention")
        comparison_pin = derive(prefix + "/" + arm_id + "/comparison_prelaunch.json", prelaunch)
        tid = "T-" + value["logical_run_id"] + "-" + arm_id
        proposal = build_execution_proposal(question=question, arm=arm, trajectory_id=tid, source_commit=commit,
            run=value["logical_run_id"], version="v1", contract_ref=recipe_pin["path"],
            contract_refs=[recipe_pin["path"], item["data_ref"]["spec_ref"], config_name],
            authority_ref=value["authority_ref"], role_ref=value["role_ref"], decision_ref=value["decision_ref"],
            timestamp=value["timestamp"], profile=profile, action_type="training")
        proposal_pin = derive(prefix + "/" + arm_id + "/plan_input.json", proposal)
        prospective.append({"arm_id": arm_id, "trajectory_id": tid, "plan_spec_ref": proposal_pin["path"],
            "plan_spec_sha256": proposal_pin["sha256"],
            "output": f"experiments/{value['experiment_id']}/runs/{value['logical_run_id']}/prospective/{arm_id}"})
        arms.append(arm)
        acceptance.append({"arm_id": arm_id, "adapter": training_adapter(arm).artifact_adapter,
            "expected": recipe["acceptance_requirements"], "contract": recipe_pin,
            "comparison_prelaunch": comparison_pin, "reference_bundle": reference["reference_bundle"],
            "reference_artifacts": reference["reference_artifacts"]})
        jobs.append({"arm_id": arm_id, "device": device, "recipe": recipe_pin["path"], "initial_state": initial})
    if any(c[-1] != compiled[0][-1] for c in compiled):
        raise ValueError("Scientific arms must share the existing V5 evidence policy")
    spec = ExperimentSpec({"schema_version": SCHEMA_VERSION_V2, "experiment_id": value["experiment_id"],
        "logical_run_id": value["logical_run_id"], "arms": arms, "platform": platform["platform"],
        "prospective": {"arms": prospective}, "evidence": evidence, "terminal_protocol": TERMINAL_PROTOCOL})
    acceptance_pin = derive(prefix + "/acceptance_plan.json", {"format": "molgap-family-acceptance-plan-v1",
        "spec_identity": spec.identity, "arms": acceptance})
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
    return _freeze(value,spec,plan,paths)


def locate_run(profile, repo_root, run_id, attempt_id):
    """Read the original frozen run by ID; never create a run or infer artifacts."""
    from .experiment_launch import read_launch_receipt
    from .experiment_package import verify_experiment_source_package
    from .research_memory.trace import file_digest
    paths=MachinePaths.from_file(profile).outside_checkout(repo_root)
    run=paths.run_directory(run_id,attempt_id)
    inputs=run/"inputs"
    binding=_read(inputs/"input_binding.json")
    spec=ExperimentSpec(_read(inputs/"experiment_spec.json"))
    plan=_read(inputs/"workflow_plan.json")
    if (spec.to_dict()["logical_run_id"] != run_id or spec.identity != binding.get("spec_identity")
            or canonical_fingerprint(plan) != binding.get("workflow_identity")):
        raise ValueError("Original frozen run binding changed")
    package=run/"prepared/package"
    manifest=verify_experiment_source_package(package) if package.exists() else None
    if manifest and manifest["spec_identity"] != spec.identity:
        raise ValueError("Run package differs from original frozen input")
    receipt_path=run/"operations/launch_receipt.json"
    _safe_local(receipt_path)
    if receipt_path.exists() and manifest is None:
        raise ValueError("A receipt without its original package cannot be resolved")
    receipt=(read_launch_receipt(receipt_path,spec,package,
        expected_package_identity=manifest["package_identity"]) if receipt_path.exists() else None)
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


def _freeze(value,spec,plan,paths):
    from .experiment_launch import canonical_json
    run = paths.run_directory(spec.to_dict()["logical_run_id"],value["attempt_id"])
    binding = {"format":value["format"],"input_identity":canonical_fingerprint(value),
               "spec_identity":spec.identity,"workflow_identity":canonical_fingerprint(plan),
               "authority_ref":value["authority_ref"],"monitor_paths":value["monitor_paths"]}
    frozen = run / "inputs"
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
    frozen=_freeze(value,spec,plan,paths)
    run=Path(frozen["run_directory"]);prepared=run/"prepared"
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
            retained.parent.mkdir(parents=True,exist_ok=True)
            prepared.rename(retained)  # Preserve every byte; only regenerable staging is replaced.
        else:
            manifest=verify_experiment_source_package(prepared/"package")
            if manifest["spec_identity"] != spec.identity or manifest["package_identity"] != report.get("package_identity"):
                raise ValueError("Existing prepared package/input binding changed")
            result={"status":"PREPARED_FOR_PLATFORM","reused_preparation":True}
    if not prepared.exists():
        result=prepare_workflow(spec,Path(repo_root).absolute(),plan,prepared,
                                source_cache=paths.artifact_root/"source-packages/runtime-cache")
    return {**result,"prepared":str(prepared),"operations":str(run/"operations"),
            "authority_ref":value["authority_ref"],"monitor_paths":value["monitor_paths"],
            "local_call_seconds":perf_counter()-started,"submitted":False}


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
        "Scientific compiler supports reviewed shared K1/3 and GPTrans/2 only":("arms.model","This compiler supports shared K1/3 and GPTrans/2; no new family is qualified"),
    }
    field,reason=known.get(str(error),("configuration","Local validation failed; check configuration fields and referenced files"))
    if isinstance(error,FileNotFoundError):
        field,reason="referenced_file","A required local file is missing"
    return {"status":"BLOCKED","error_type":type(error).__name__,"stage":stage,"field":field,"reason":reason}
