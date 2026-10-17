"""Freeze and package the human-authorized Modern2D scratch qualification only.

Freeze writes reviewable declarations; package preparation requires their commit.
Neither operation submits, publishes datasets, opens graphs or trains a model.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import subprocess

from .common.io import _safe_local, publish_immutable_bytes
from .common.wire import canonical_fingerprint
from .modern2d_qualification_runtime import _json, _ref, _sha, _under
from .research_memory.trace import json_bytes

BASE = "experiments/modern2d_model_integration/native_preflight"
ACCOUNT = "kaseichou"
KERNEL = ACCOUNT + "/molgap-modern2d-native-100k-v1"
SOURCE_DATASET = ACCOUNT + "/molgap-modern2d-qualification-source-v1"
DATASET = ACCOUNT + "/molgap-modern2d-100k-commondev-v1"
MOUNT = "molgap-modern2d-100k-commondev-v1"
POLICY_ID = "modern2d-native-qualification"
POLICY = "research_memory/policies/" + POLICY_ID + ".v1.json"
ARMS = {"ppgt_author_v2": ("ppgt_small", "2"), "esa_native_v3": ("esa_small", "3")}
PARAMETERS = {"ppgt_author_v2": 9536769, "esa_native_v3": 5015233}
ENTRY = "platforms/kaggle/run_modern2d_qualification.py"
# This supports the shared Session substrate only, never a Modern2D result.
SESSION_EVIDENCE = "pcqm-k1-infrastructure-optimized-100k-s42"


def _write(path, value):
    publish_immutable_bytes(path, json_bytes(value))
    return {"path": path.name, "sha256": _sha(path)}


def _authority(root):
    path = _under(root, BASE + "/authority.json")
    record = _json(path)
    if (record.get("purpose") != "EXECUTION_QUALIFICATION_ONLY" or record.get("user_authorized") is not True
            or record.get("arms") != list(ARMS) or record.get("seeds") != [42]
            or record.get("physical_batch_size") != 128
            or record.get("gpu") != {"platform": "kaggle2", "owner": ACCOUNT,
                "accelerator": "NvidiaTeslaT4", "allocated_devices": 2,
                "wall_ceiling_seconds": 1800, "allocated_t4_hours_ceiling": 1}
            or any(record.get(k) is not False for k in ("scientific_training_authorized",
                "protected_role_access_authorized", "extra_seed_authorized", "automatic_successor_authorized"))):
        raise ValueError("Exact new human qualification authority required")
    return path


def freeze_qualification_declarations(repo_root, input_metadata_path, input_metadata_sha256, *, init_root):
    """Freeze typed defaults, CPU initial states and independent prospective inputs.

    Metadata is Ampere's qualification_inputs.json, externally pinned by SHA256;
    its sibling training_input.json supplies the independently accepted permits.
    init_root must be fresh external storage (E: on this Windows host).
    Return the Spec identity and files the caller must commit before packaging.
    """
    import torch
    from .common.declarations import ExecutionQuestion, ExecutionProfile, build_execution_proposal
    from .experiment_execution import TRAINING_ADAPTERS, build_family_recipe
    from .experiment_spec import ExperimentSpec, FAMILIES, TERMINAL_PROTOCOL
    from .family_training_data import DatasetSelection
    from .family_training_recipe import FamilyRecipe, ScheduleSpec
    from .family_training_runner import family_arm_bindings
    from .modern2d_input_preparation import METADATA_FORMAT
    from .modern2d_qualification import QualificationProbe
    from .shared_model_adapter import resolve_model_factory
    from .training_reproducibility import assert_finite_state_dict, atomic_torch_save
    from .v4_runtime import model_state_sha256

    root, initial = Path(repo_root).absolute(), Path(init_root).absolute()
    _safe_local(root)
    _safe_local(initial)
    if (initial.exists() or initial.is_relative_to(root) or root.is_relative_to(initial)
            or (initial.drive and initial.drive.upper() != "E:")):
        raise ValueError("Initialization requires fresh external E: storage")
    metadata_path = Path(input_metadata_path).absolute()
    _safe_local(metadata_path)
    _ref(metadata_path.parent, {"path": metadata_path.name, "sha256": input_metadata_sha256})
    metadata = _json(metadata_path)
    pointer = metadata["training_input"]
    binding_path = _ref(metadata_path.parent, {k: pointer[k] for k in ("path", "sha256")})
    binding = _json(binding_path)
    if (metadata["format"] != METADATA_FORMAT or metadata["status"] != "INPUT_METADATA_PREPARED"
            or metadata["data_root"] != "indexed_assets" or binding["data_root"] != "indexed_assets"
            or metadata["scientific_training_authorized"] is not False
            or metadata["development_label_statistics_exposed"] is not False
            or set(metadata["models"]) != set(ARMS)
            or {k: v["rows"] for k, v in metadata["roles"].items()} != {"train_100k": 100000, "common_dev": 50000}
            or binding_path.stat().st_size != pointer["bytes"]):
        raise ValueError("Expected exact selected Modern2D input metadata")
    if pointer["path"] != "training_input.json" or binding["scope"] != "ACCEPTED_INDEXED_INPUT_V1":
        raise ValueError("Exact top-level accepted training_input.json binding required")
    from .data.indexed_inputs import AcceptedIndexedInputPermit
    for name, role in (("train", "train_100k"), ("development", "common_dev")):
        item = binding[name]
        permit = AcceptedIndexedInputPermit(**item["permit"])
        permit.verify(item["authority"], item["role_history"])
        if permit.role != role or permit.acceptance_sha256 != metadata["export_acceptance_sha256"]:
            raise ValueError("Metadata and accepted export permit differ")
    for arm_id, interface in ARMS.items():
        values = metadata["models"][arm_id]
        defaults = asdict(import_module(FAMILIES[interface].source_module).CONFIG_TYPE())
        if (tuple(values["family"]) != interface or
                canonical_fingerprint(values["model_config"]) != canonical_fingerprint(defaults)):
            raise ValueError("Qualification requires the prior typed defaults; no tuning")
    authority = _authority(root)
    here = _under(root, BASE)
    if (here / "preparation.json").exists() or (here / "experiment_spec.json").exists():
        raise ValueError("Declarations already frozen; preserve and reconcile their bytes")
    _write(here / "qualification_inputs.json", metadata)
    publish_immutable_bytes(here / "training_input.json", binding_path.read_bytes())
    authority_ref, role_ref = BASE + "/authority.json", BASE + "/role_plan.json"
    _write(root / role_ref, {"purpose": "EXECUTION_QUALIFICATION_ONLY", "roles_opened": ["train"],
        "development_metrics": False, "head_to_head_comparison": False,
        "scientific_tuning_claim": False,
        "scratch_optimizer_updates_per_model": 6, "metadata_sha256": input_metadata_sha256})
    _write(root / POLICY, {"schema": "molgap-policy-v1", "policy_id": POLICY_ID, "version": "v1",
        "policy_type": "research_action", "status": "candidate", "observation_point": None,
        "comparability_selector": {"scientific_contract": "modern2d-native-qualification-only-v1"},
        "required_observable_fields": ["user_authorized_qualification"],
        "created_from_source_digest": _sha(authority), "approval": {"authority_ref": authority_ref,
            "activation": "new-human-authorized-six-scratch-updates-only"},
        "action_rule": {"field": "user_authorized_qualification", "operator": "eq", "threshold": True,
            "action": "RUN_MODERN2D_QUALIFICATION"}, "promotion_rule": None, "early_stop_rule": None,
        "borderline_action": "DEFER", "cost_model": {"kind": "measured_only",
            "assumptions": ["entire T4x2 allocation including environment preparation and idle tail"]}})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    arms, prospective, jobs = [], [], []
    records = [authority_ref, role_ref, POLICY, BASE + "/qualification_inputs.json", BASE + "/training_input.json", ENTRY,
        "experiments/modern2d_model_integration/integration_attempt_report.json", BASE + "/cpu_acceptance.json"]
    initial.mkdir(parents=True, exist_ok=False)
    for device, (arm_id, interface) in enumerate(ARMS.items()):
        model_metadata = metadata["models"][arm_id]
        if tuple(model_metadata["family"]) != interface or torch.get_default_dtype() != torch.float32:
            raise ValueError("Pinned family and FP32 CPU defaults required")
        factory = resolve_model_factory(TRAINING_ADAPTERS[interface])
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(42)
            model = factory({"model_config": model_metadata["model_config"]})
        if sum(p.numel() for p in model.parameters()) != PARAMETERS[arm_id]:
            raise ValueError("Qualification default parameter count changed")
        assert_finite_state_dict(model.state_dict(), label="Qualification CPU initialization")
        state_sha = model_state_sha256(model)
        state_path = initial / (arm_id + ".pt")
        atomic_torch_save(state_path, model.state_dict())
        del model
        usage = canonical_fingerprint({"authority_sha256": _sha(authority), "role": "train",
            "metadata_sha256": input_metadata_sha256, "arm_id": arm_id})
        recipe = FamilyRecipe(interface[0], "reference", 42, state_sha,
            DatasetSelection(pointer["path"], pointer["sha256"], False), 100000, 0, 2, .001, .01,
            ScheduleSpec("cosine_step", 2), "final-live", "torch-pass-randperm", 1,
            metadata["target_mean_eV"], metadata["target_std_eV"], finite_check_every=1,
            torch_version="2.7.1", family_version=interface[1], model_config=model_metadata["model_config"],
            qualification=QualificationProbe())
        contract_ref = BASE + "/" + arm_id + "/contract.json"
        _write(root / contract_ref, build_family_recipe(interface, configuration=recipe, train_role_identity=usage))
        catalog, bindings = FAMILIES[interface], family_arm_bindings(recipe)
        feature = model_metadata["feature_bindings"]["train_100k"]
        feature_sha = model_metadata["feature_binding_sha256"]["train_100k"]
        if canonical_fingerprint(feature) != feature_sha:
            raise ValueError("Feature metadata digest differs")
        member_sha = metadata["roles"]["train_100k"]["source_idx_tensor_sha256"]
        ref = lambda name, sha: {"name": name, "version": "1", "sha256": sha}
        arm = {"arm_id": arm_id, "scientific_role": "ablation", "family": {"name": interface[0], "version": interface[1]},
            "base": bindings["base"], "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": state_sha},
            "data": {"dataset": ref("pcqm4mv2", pointer["sha256"]), "split": ref("accepted-indexed-train100k", pointer["sha256"]),
                "roles": [{"role": "train", "membership_sha256": member_sha, "row_order_sha256": member_sha, "usage_sha256": usage}],
                "feature_schema": catalog.feature_schema, "feature_sha256": feature_sha, "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": ref(catalog.recipe, _sha(root / contract_ref)), "overrides": {},
                **{k: bindings[k] for k in ("objective", "sampler", "transform")}}, "addons": [], "addon_semantics": "baseline"}
        trajectory = "TC-modern2d-qualification-" + arm_id + "-s42-v1"
        plan_ref = BASE + "/" + arm_id + "/plan_input.json"
        proposal = build_execution_proposal(question=ExecutionQuestion(POLICY_ID, "RUN_MODERN2D_QUALIFICATION",
            "Can this frozen interface complete numerical, resume and resource qualification?",
            "Native qualification is unestablished", "Versioned model execution only", (SESSION_EVIDENCE,),
            ("runtime incompatibility", "resource ceiling"), ("k1-infrastructure-execution",), "retain qualification evidence; no scientific release",
            "retain execution failure; no automatic successor"), arm=arm, trajectory_id=trajectory,
            source_commit=commit, run="modern2d-native-100k-s42-v1", version="v1", contract_ref=contract_ref,
            contract_refs=[contract_ref, authority_ref, role_ref,
                "experiments/modern2d_model_integration/integration_attempt_report.json", BASE + "/cpu_acceptance.json"],
            authority_ref=authority_ref, role_ref=role_ref,
            decision_ref=BASE + "/STATUS.md", timestamp=datetime.now(timezone.utc).isoformat(),
            profile=ExecutionProfile("server", "C", "kaggle2", "NvidiaTeslaT4x2_shared_allocation",
                "exactly six scratch updates per model; no development metrics", .5, .5))
        proposal["costs"][0]["category"] = "preflight"
        _write(root / plan_ref, proposal)
        prospective.append({"arm_id": arm_id, "trajectory_id": trajectory, "plan_spec_ref": plan_ref,
            "plan_spec_sha256": _sha(root / plan_ref), "output": BASE + "/" + arm_id + "/rml"})
        jobs.append({"arm_id": arm_id, "device": device, "recipe": contract_ref,
            "initial_state": {"path": state_path.name, "sha256": _sha(state_path)}})
        records.extend([contract_ref, plan_ref])
        arms.append(arm)
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2", "experiment_id": "modern2d-native-qualification",
        "logical_run_id": "modern2d-native-100k-s42-v1", "arms": arms, "prospective": {"arms": prospective},
        "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2, "cpu_cores": 4,
            "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
        "evidence": {"policy": ref("molgap-v5", canonical_fingerprint({"contract": "MOLGAP-COMMON-V5-FINAL"})),
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": TERMINAL_PROTOCOL})
    _write(here / "experiment_spec.json", spec.to_dict())
    _write(here / "preparation.json", {"spec_identity": spec.identity, "jobs": jobs, "source_records": records,
        "metadata_sha256": _sha(here / "qualification_inputs.json")})
    return {"spec_identity": spec.identity, "base": BASE, "initial_root": str(initial),
        "commit_required": [*records, BASE + "/experiment_spec.json", BASE + "/preparation.json"], "scope": "QUALIFICATION_ONLY"}


def prepare_package(repo_root, output_root, *, init_root, mounted_root, expected_spec_identity):
    """Package committed declarations and validate the separate qualification ABI.

    mounted_root contains the declared dataset in a flat or owner-namespaced
    layout, with training_input.json and indexed_assets. Direct plan_many
    publication never rebuilds RML derived files.
    """
    from .experiment_family_workflow import RunContext
    from .experiment_package import build_experiment_source_package, stage_source_inputs, read_packaged_text
    from .experiment_preflight import _unpack, check_release_inputs
    from .experiment_prospective import _prospective_inputs
    from .experiment_source_inventory import runtime_source_files
    from .experiment_spec import ExperimentSpec
    from .modern2d_qualification import Modern2DQualificationContext
    from .modern2d_qualification_runtime import LAUNCH_FORMAT, LAUNCH_FILENAME, validate_launch
    from .data.indexed_inputs import AcceptedIndexedInputPermit
    from .family_training_runner import read_recipe
    from .kaggle_mounts import resolve_dataset_mount
    from .research_memory.plan import plan_many

    root, output = Path(repo_root).absolute(), Path(output_root).absolute()
    _safe_local(output)
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Package preparation requires a fresh external output root")
    here = _under(root, BASE)
    spec = ExperimentSpec(_json(here / "experiment_spec.json"))
    prep = _json(here / "preparation.json")
    if spec.identity != expected_spec_identity or prep["spec_identity"] != spec.identity:
        raise ValueError("Frozen Spec identity differs")
    metadata = _json(_ref(here, {"path": "qualification_inputs.json", "sha256": prep["metadata_sha256"]}))
    binding = _json(_ref(here, {k: metadata["training_input"][k] for k in ("path", "sha256")}))
    authority = _authority(root)
    package = output / "package"
    sources = runtime_source_files(root, prep["source_records"])
    manifest = build_experiment_source_package(spec, root, sources, package)
    states = {j["arm_id"]: _under(Path(init_root).absolute(), j["initial_state"]["path"]) for j in prep["jobs"]}
    for job in prep["jobs"]:
        if _sha(states[job["arm_id"]]) != job["initial_state"]["sha256"]:
            raise ValueError("CPU initial state artifact changed")
    staged = stage_source_inputs(package, output / "inputs", states)
    publish_immutable_bytes(staged / "authority.json", authority.read_bytes())
    authority_pin = {"path": "authority.json", "sha256": _sha(staged / "authority.json")}
    asset_root = resolve_dataset_mount(Path(mounted_root).absolute(), DATASET,
        mount_name=MOUNT, marker="training_input.json")
    receipt = _json(_ref(asset_root / "indexed_assets", {"path": "acceptance.json", "sha256": metadata["export_acceptance_sha256"]}))
    jobs = []
    for job in prep["jobs"]:
        arm = job["arm_id"]
        _, recipe = read_recipe(root / job["recipe"])
        context = Modern2DQualificationContext(RunContext.for_training(spec, package,
            expected_package_identity=manifest["package_identity"], arm_id=arm, account=ACCOUNT, run_reference=KERNEL),
            recipe.identity, authority_pin["sha256"], AcceptedIndexedInputPermit(**binding["train"]["permit"]),
            AcceptedIndexedInputPermit(**binding["development"]["permit"]),
            metadata["models"][arm]["feature_binding_sha256"]["train_100k"])
        context.validate_recipe(recipe)
        jobs.append({"arm_id": arm, "device": job["device"], "context": _write(staged / (arm + "_context.json"), context.to_dict()),
            "recipe": {"path": job["recipe"], "sha256": _sha(root / job["recipe"])},
            "initial_state": {"path": "initial_states/" + arm + ".pt", "sha256": job["initial_state"]["sha256"]}})
    launch = {"format": LAUNCH_FORMAT, "spec_identity": spec.identity, "expected_package_identity": manifest["package_identity"],
        "expected_source_archive_sha256": manifest["archive_sha256"], "account": ACCOUNT, "run_reference": KERNEL,
        "max_wall_seconds": 1800, "authority": authority_pin, "jobs": jobs,
        "private_asset": {"dataset_source": DATASET, "mount": MOUNT, "data_root": "indexed_assets",
            "acceptance_sha256": metadata["export_acceptance_sha256"], "version_identity": binding["train"]["permit"]["version_identity"],
            "train_logical_content_sha256": receipt["roles"]["train_100k"]["logical_content_sha256"],
            "development_logical_content_sha256": receipt["roles"]["common_dev"]["logical_content_sha256"]}}
    launch_pin = _write(staged / LAUNCH_FILENAME, launch)
    entry = read_packaged_text(package, ENTRY)
    resolver_source = read_packaged_text(package, "src/molgap/kaggle_mounts.py")
    replacements = {
        "EXPECTED_LAUNCH_SHA256 = None": "EXPECTED_LAUNCH_SHA256 = " + repr(launch_pin["sha256"]),
        "EXPECTED_SOURCE_DATASET = None": "EXPECTED_SOURCE_DATASET = " + repr(SOURCE_DATASET),
        "MOUNT_RESOLVER_SOURCE = None": "MOUNT_RESOLVER_SOURCE = " + repr(resolver_source),
    }
    for marker in replacements:
        if entry.count(marker) != 1:
            raise ValueError("Qualification entrypoint pin marker changed: " + marker)
    for marker, replacement in replacements.items():
        entry = entry.replace(marker, replacement)
    publish_immutable_bytes(output / "run_modern2d_qualification.py",
        entry.encode("utf-8"))
    _write(output / "kernel-metadata.json", {"id": KERNEL, "title": "MolGap Modern2D Native 100K V1",
        "code_file": "run_modern2d_qualification.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [SOURCE_DATASET, DATASET], "competition_sources": [], "kernel_sources": []})
    source = output / "source"
    source.mkdir()
    _unpack(package, source)
    validate_launch(launch_path=staged / LAUNCH_FILENAME, expected_launch_sha256=launch_pin["sha256"],
        source_root=source, package_dir=package, mounted_root=mounted_root)
    release = check_release_inputs(spec, package, expected_package_identity=manifest["package_identity"],
        recipe_files={j["arm_id"]: j["recipe"] for j in prep["jobs"]},
        initial_states={arm: staged / "initial_states" / (arm + ".pt") for arm in ARMS},
        required_modules=["molgap.family_training_runner", "molgap.modern2d_qualification",
            "molgap.modern2d_qualification_runtime", "molgap.ppgt_author", "molgap.esa_native"],
        entry_script=output / "run_modern2d_qualification.py", input_root=staged, launch_config=None)
    _write(output / "release_report.json", release)
    if release["status"] != "LOCAL_RELEASE_INPUTS_VERIFIED":
        raise ValueError("Generic CPU release-input checks failed; retain release_report.json")
    plans, _ = _prospective_inputs(spec, root)
    batch = plan_many(root, plans)
    _write(output / "prospective_batch.json", batch)
    report = {"status": "QUALIFICATION_PACKAGE_PREPARED", "spec_identity": spec.identity,
        "package_identity": manifest["package_identity"], "source_commit": manifest["source_commit"],
        "source_archive_sha256": manifest["archive_sha256"], "qualification_launch_validated": True,
        "authority_sha256": authority_pin["sha256"], "input_metadata_sha256": prep["metadata_sha256"],
        "release_report_sha256": _sha(output / "release_report.json"),
        "launch_sha256": launch_pin["sha256"], "entry_sha256": _sha(output / "run_modern2d_qualification.py"),
        "kernel_metadata_sha256": _sha(output / "kernel-metadata.json"), "rml_rebuilt": False,
        "scientific_training_authorized": False, "submission_performed": False,
        "platform_requirement": "Kaggle2 NvidiaTeslaT4 x2; 1800 seconds including bootstrap; qualification entry only"}
    _write(output / "qualification_release.json", report)
    return report
