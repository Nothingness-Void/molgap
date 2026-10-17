"""Freeze and package the human-authorized Modern2D scratch qualification only.

Freeze writes reviewable declarations; package preparation requires their commit.
Neither operation submits, publishes datasets, opens graphs or trains a model.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import subprocess
import re

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


def _qualification_base(root, base):
    from .common.declarations import safe_comparison_base
    base = safe_comparison_base(base, base="experiments/modern2d_model_integration")
    path = _under(root, base)
    _safe_local(path)
    return path


def _qualification_refs(kernel, source_dataset):
    for value in (kernel, source_dataset):
        if (type(value) is not str or
                re.fullmatch(ACCOUNT + r"/[a-z0-9]+(?:-[a-z0-9]+)*", value) is None):
            raise ValueError("Qualification requires a safe kaseichou owner/reference")
    if kernel in {source_dataset, DATASET} or source_dataset == DATASET:
        raise ValueError("Qualification kernel, source and accepted data must be distinct")


def _retry_suffix(kernel, source_dataset):
    _qualification_refs(kernel, source_dataset)
    match = re.search(r"(?:^|-)v([2-9][0-9]*)$", kernel)
    if match is None:
        raise ValueError("Retry kernel and source dataset need the same version suffix, v2 or later")
    suffix = "v" + match.group(1)
    if re.search(r"(?:^|-)" + re.escape(suffix) + r"$", source_dataset) is None:
        raise ValueError("Retry kernel and source dataset need the same version suffix, v2 or later")
    return suffix


def _authority(root, base=BASE):
    _qualification_base(root, base)
    path = _under(root, base + "/authority.json")
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


def freeze_qualification_retry(repo_root, *, base, kernel, source_dataset,
                               authorization, previous_base=BASE):
    """Declare a versioned retry using frozen recipes/states, without models.

    authorization is the new human authorization text. The returned commit_required
    paths must be committed before prepare_package; init_root remains the V1 root.
    This operation does not publish prospective RML records or execute a job.
    """
    from .experiment_spec import ExperimentSpec

    from .common.declarations import safe_comparison_base

    root = Path(repo_root).absolute()
    _safe_local(root)
    base = safe_comparison_base(base, base="experiments/modern2d_model_integration")
    previous_base = safe_comparison_base(previous_base, base="experiments/modern2d_model_integration")
    here = _qualification_base(root, base)
    previous = _qualification_base(root, previous_base)
    suffix = _retry_suffix(kernel, source_dataset)
    if (here == previous or previous.is_relative_to(here)
            or kernel == KERNEL or source_dataset == SOURCE_DATASET):
        raise ValueError("Retry requires separate declaration, kernel and source identities")
    prior_authority = _json(_authority(root, previous_base))
    if (type(authorization) is not str or not authorization.strip() or
            authorization == prior_authority.get("authorization")):
        raise ValueError("Retry requires new explicit human authorization")
    if here.exists():
        raise ValueError("Retry declaration root already exists; preserve its bytes")
    original = ExperimentSpec(_json(previous / "experiment_spec.json"))
    declaration = deepcopy(original.to_dict())
    old_suffix = declaration["logical_run_id"].rsplit("-", 1)[-1]
    if old_suffix == suffix:
        raise ValueError("Retry must have a new logical run version")
    prep = deepcopy(_json(previous / "preparation.json"))
    if (prep["spec_identity"] != original.identity or
            [a["arm_id"] for a in declaration["arms"]] != list(ARMS) or
            [j["arm_id"] for j in prep["jobs"]] != list(ARMS)):
        raise ValueError("Retry requires the frozen two-arm preparation and Spec")
    metadata = _ref(previous, {"path": "qualification_inputs.json", "sha256": prep["metadata_sha256"]})
    training = _ref(previous, {k: _json(metadata)["training_input"][k] for k in ("path", "sha256")})
    source_records = list(prep["source_records"])
    if len(source_records) != len(set(source_records)):
        raise ValueError("Frozen preparation source records must be unique")
    policy_sources = [path for path in source_records
                      if path.startswith("research_memory/policies/") and path.endswith(".json")]
    if len(policy_sources) != 1:
        raise ValueError("Frozen preparation must bind exactly one policy")
    old_policy_ref = policy_sources[0]
    old_policy_path = _under(root, old_policy_ref)
    if not old_policy_path.is_file():
        raise ValueError("Frozen policy source is absent")
    for record in source_records:
        source = _under(root, record)
        if not source.is_file():
            raise ValueError("Frozen preparation source record is absent: " + record)
    jobs = deepcopy(prep["jobs"])
    old_recipe_paths = []
    for job in jobs:
        recipe_path = job.get("recipe")
        if type(recipe_path) is not str or not recipe_path.startswith(previous_base + "/"):
            raise ValueError("Frozen retry recipes must remain under the previous declaration base")
        recipe = _under(root, recipe_path)
        if not recipe.is_file() or recipe_path not in source_records:
            raise ValueError("Frozen retry recipe is not a committed source record")
        old_recipe_paths.append(recipe_path)
    authority_ref, role_ref = base + "/authority.json", base + "/role_plan.json"
    policy_id = "modern2d-native-qualification-retry-" + suffix
    policy_ref = "research_memory/policies/" + policy_id + ".v1.json"
    policy_path = _under(root, policy_ref)
    if policy_path.exists():
        raise ValueError("Retry policy already exists; preserve its bytes")
    run = "modern2d-native-100k-s42-" + suffix
    failed_refs = [previous_base + "/STATUS.md", previous_base + "/experiment_spec.json",
                   previous_base + "/results/decision_2026-10-18.md"]
    authority = deepcopy(prior_authority)
    authority.update(authorization=authorization, owner="server", run_reference=kernel,
        source_dataset=source_dataset, logical_run_id=run, scratch_optimizer_updates_per_model=6,
        previous_attempt_refs=failed_refs, previous_spec_identity=original.identity)
    _write(here / "authority.json", authority)
    _authority(root, base)
    def rebase_record(path):
        if path == old_policy_ref:
            return policy_ref
        if path == previous_base or path.startswith(previous_base + "/"):
            return base + path[len(previous_base):]
        return path

    # Copy every retained declaration byte, including the frozen contracts. The
    # new preparation points at those copied bytes so the retry package never
    # depends on an old-base path that could be edited later.
    for record in source_records:
        if (record == old_policy_ref or record.endswith("/plan_input.json")
                or record.endswith("/authority.json") or record.endswith("/STATUS.md")):
            continue
        target = rebase_record(record)
        if target == record:
            continue
        publish_immutable_bytes(_under(root, target), _under(root, record).read_bytes())
    for job, old_recipe in zip(jobs, old_recipe_paths):
        job["recipe"] = rebase_record(old_recipe)
    status = here / "STATUS.md"
    _safe_local(status)
    publish_immutable_bytes(status, (
        "# Bounded qualification retry\n\nDeclarations only; no submission or execution claimed.\n"
        "Authority: [authority.json](authority.json). Frozen V1 recipes and initial states reused.\n"
        "T4x2, 1800 seconds, six scratch updates/model, BS128, seed42; no formal training or protected roles.\n"
    ).encode("utf-8"))
    policy = deepcopy(_json(old_policy_path))
    policy.update(policy_id=policy_id, created_from_source_digest=_sha(here / "authority.json"))
    policy["approval"]["authority_ref"] = authority_ref
    _write(policy_path, policy)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    timestamp = datetime.now(timezone.utc).isoformat()
    prospective, new_plan_records = [], []
    for item in declaration["prospective"]["arms"]:
        old_plan_path = _ref(root, {"path": item["plan_spec_ref"], "sha256": item["plan_spec_sha256"]})
        proposal = deepcopy(_json(old_plan_path))
        old_policy_id = proposal["decision_state"]["policy_id"]
        old_tid = item["trajectory_id"]
        tid = "TC-modern2d-qualification-" + item["arm_id"] + "-s42-" + suffix
        old_run = declaration["logical_run_id"]

        def rebind(value):
            if isinstance(value, dict):
                return {k: commit if k == "source_commit" else
                        timestamp if k == "state_timestamp" else rebind(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rebind(v) for v in value]
            if isinstance(value, str):
                if value == old_policy_id:
                    value = policy_id
                elif value == old_policy_ref:
                    value = policy_ref
                elif value == previous_base or value.startswith(previous_base + "/"):
                    value = base + value[len(previous_base):]
                value = value.replace(old_tid, tid).replace(old_run, run)
                return value.replace(item["arm_id"] + "-" + old_suffix,
                                     item["arm_id"] + "-" + suffix)
            return value

        proposal = rebind(proposal)
        arm = next(a for a in declaration["arms"] if a["arm_id"] == item["arm_id"])
        proposal["trajectory"]["state_at_start"]["source_config_identity"] = canonical_fingerprint(arm)
        parent = _under(root, item["output"] + "/trajectory.json")
        if parent.is_file() and _json(parent).get("trajectory_id") == old_tid:
            proposal["trajectory"]["state_at_start"]["parent_trajectory_ids"] = [old_tid]
            proposal["trajectory"]["state_at_start"]["prior_trajectory_ids"] = [old_tid]
            proposal["decision_state"]["known_trajectory_ids"] = [old_tid]
        plan_ref = base + "/" + item["arm_id"] + "/plan_input.json"
        _write(root / plan_ref, proposal)
        prospective.append({**item, "trajectory_id": tid, "plan_spec_ref": plan_ref,
            "plan_spec_sha256": _sha(root / plan_ref), "output": base + "/" + item["arm_id"] + "/rml"})
        new_plan_records.append(plan_ref)
    declaration["logical_run_id"] = run
    declaration["prospective"]["arms"] = prospective
    spec = ExperimentSpec(declaration)
    _write(here / "experiment_spec.json", spec.to_dict())
    records = [rebase_record(record) for record in source_records
               if record != old_policy_ref and not record.endswith("/plan_input.json")]
    records.extend([policy_ref, ENTRY, *new_plan_records, base + "/STATUS.md"])
    prep.update(jobs=jobs, spec_identity=spec.identity,
                source_records=list(dict.fromkeys(records)))
    _write(here / "preparation.json", prep)
    return {"spec_identity": spec.identity, "base": base, "kernel": kernel,
        "source_dataset": source_dataset,
        "commit_required": [*prep["source_records"], base + "/experiment_spec.json",
                             base + "/preparation.json"],
        "scope": "QUALIFICATION_ONLY"}


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


def prepare_package(repo_root, output_root, *, init_root, mounted_root, expected_spec_identity,
                    base=BASE, kernel=KERNEL, source_dataset=SOURCE_DATASET):
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
    from .common.declarations import safe_comparison_base

    root, output = Path(repo_root).absolute(), Path(output_root).absolute()
    _safe_local(output)
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Package preparation requires a fresh external output root")
    base = safe_comparison_base(base, base="experiments/modern2d_model_integration")
    here = _qualification_base(root, base)
    _qualification_refs(kernel, source_dataset)
    is_retry = (base != BASE or kernel != KERNEL or source_dataset != SOURCE_DATASET)
    if is_retry:
        _retry_suffix(kernel, source_dataset)
    spec = ExperimentSpec(_json(here / "experiment_spec.json"))
    prep = _json(here / "preparation.json")
    if spec.identity != expected_spec_identity or prep["spec_identity"] != spec.identity:
        raise ValueError("Frozen Spec identity differs")
    metadata = _json(_ref(here, {"path": "qualification_inputs.json", "sha256": prep["metadata_sha256"]}))
    binding = _json(_ref(here, {k: metadata["training_input"][k] for k in ("path", "sha256")}))
    authority = _authority(root, base)
    authorization = _json(authority)
    if is_retry:
        if (authorization.get("owner") != "server" or
                authorization.get("run_reference") != kernel or
                authorization.get("source_dataset") != source_dataset or
                authorization.get("logical_run_id") != spec.to_dict()["logical_run_id"] or
                authorization.get("scratch_optimizer_updates_per_model") != 6):
            raise ValueError("Retry authority must bind the exact kernel, source dataset and run")
        from .v4_bundle import committed_payloads, _payload
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        spec_ref = base + "/experiment_spec.json"
        if committed_payloads(root, commit, [spec_ref])[spec_ref] != _payload(here / "experiment_spec.json"):
            raise ValueError("Retry Spec must be committed before packaging")
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
            expected_package_identity=manifest["package_identity"], arm_id=arm, account=ACCOUNT, run_reference=kernel),
            recipe.identity, authority_pin["sha256"], AcceptedIndexedInputPermit(**binding["train"]["permit"]),
            AcceptedIndexedInputPermit(**binding["development"]["permit"]),
            metadata["models"][arm]["feature_binding_sha256"]["train_100k"])
        context.validate_recipe(recipe)
        jobs.append({"arm_id": arm, "device": job["device"], "context": _write(staged / (arm + "_context.json"), context.to_dict()),
            "recipe": {"path": job["recipe"], "sha256": _sha(root / job["recipe"])},
            "initial_state": {"path": "initial_states/" + arm + ".pt", "sha256": job["initial_state"]["sha256"]}})
    launch = {"format": LAUNCH_FORMAT, "spec_identity": spec.identity, "expected_package_identity": manifest["package_identity"],
        "expected_source_archive_sha256": manifest["archive_sha256"], "account": ACCOUNT, "run_reference": kernel,
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
        "EXPECTED_SOURCE_DATASET = None": "EXPECTED_SOURCE_DATASET = " + repr(source_dataset),
        "MOUNT_RESOLVER_SOURCE = None": "MOUNT_RESOLVER_SOURCE = " + repr(resolver_source),
    }
    for marker in replacements:
        if entry.count(marker) != 1:
            raise ValueError("Qualification entrypoint pin marker changed: " + marker)
    for marker, replacement in replacements.items():
        entry = entry.replace(marker, replacement)
    publish_immutable_bytes(output / "run_modern2d_qualification.py",
        entry.encode("utf-8"))
    title = "MolGap Modern2D Native 100K " + kernel.rsplit("-", 1)[-1].upper()
    _write(output / "kernel-metadata.json", {"id": kernel, "title": title,
        "code_file": "run_modern2d_qualification.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [source_dataset, DATASET], "competition_sources": [], "kernel_sources": []})
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
