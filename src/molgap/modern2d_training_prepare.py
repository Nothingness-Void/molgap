"""Freeze metadata for two independently authorized Modern2D reference screens.

This adapter writes declarations only. The registered Recipe, Session, runtime-v1
workflow and Kaggle adapter own preparation and execution. No RML publication,
graph/target loading, initialization generation or scientific acceptance occurs.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .common.declarations import (
    ExecutionProfile, ExecutionQuestion, build_execution_proposal, safe_comparison_base,
)
from .common.io import _safe_local, publish_immutable_bytes
from .common.spec import _digest, _unique_object
from .common.wire import canonical_fingerprint
from .research_memory.paths import repo_local_path
from .research_memory.trace import file_digest, json_bytes


BASE = "experiments/modern2d_100k_formal_screen"
RUN = "modern2d-formal-100k-s42-v1"
ACCOUNT = "kaseichou"
KERNEL = ACCOUNT + "/molgap-modern2d-formal-100k-v1"
SOURCE_DATASET = ACCOUNT + "/molgap-modern2d-formal-source-v1"
DATASET = ACCOUNT + "/molgap-modern2d-100k-commondev-v1"
OLD_BASE = "experiments/modern2d_model_integration/native_preflight/retry_v4"
INITIAL_ROOT = "E:/MolGap/artifacts/modern2d-native-preflight/initial_states_v1"
ROLE_MANIFEST_ROOT = "E:/MolGap/artifacts/modern2d-native-preflight/data_roundtrip/indexed_assets/roles"
MACHINE_PROFILE = "E:/MolGap/machine-profile.json"
ARMS = {"ppgt_author_v2": ("ppgt_small", "3"), "esa_native_v3": ("esa_small", "4")}
MODEL_INTERFACES = {"ppgt_author_v2": ("ppgt_small", "2"), "esa_native_v3": ("esa_small", "3")}
POLICY_ID = "modern2d-formal-100k-reference-screen"
POLICY = "research_memory/policies/" + POLICY_ID + ".v1.json"
PLAN_FORMAT = "molgap-modern2d-formal-reference-plan-v1"
TARGET_ENCODING = "little-endian-float64-contiguous-raw-bytes"
# This evidence supports the shared Session substrate, not matched model scores.
SESSION_EVIDENCE = "pcqm-k1-infrastructure-optimized-100k-s42"


def _read_metadata(path):
    import json
    path = Path(path).absolute()
    _safe_local(path)
    if not path.is_file() or path.stat().st_size > 1024 ** 2:
        raise ValueError("Expected a small local metadata file: " + str(path))
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite metadata")))
    if type(value) is not dict:
        raise ValueError("Expected metadata object")
    return value, raw


def _source_snapshot(root, records):
    """Pin committed executable bytes, independently of the later records commit."""
    from .experiment_source_inventory import runtime_source_files
    from .v4_bundle import _assert_clean_paths, _payload, committed_payloads
    runtime = runtime_source_files(root)
    required = {"src/molgap/modern2d_training.py", "src/molgap/modern2d_training_release.py",
                "src/molgap/modern2d_training_prepare.py", "platforms/kaggle/run_experiment.py"}
    if not required <= set(runtime):
        raise ValueError("Commit the formal admission/runtime implementation before freezing")
    names = sorted(set(runtime) | set(records))
    commit = _assert_clean_paths(root, names)
    payloads = committed_payloads(root, commit, names)
    if any(_payload(repo_local_path(root, name)) != payloads[name] for name in names):
        raise ValueError("Formal source and prior records must match their committed bytes")
    return {"format": "molgap-modern2d-formal-source-identity-v1", "source_commit": commit,
        "source_profile": "runtime-v1", "runtime_files": runtime,
        "files": {name: hashlib.sha256(payloads[name]).hexdigest() for name in names}}


def _inputs(root, role_manifest_root):
    """Verify accepted role metadata without constructing an indexed dataset."""
    from .data.indexed_inputs import AcceptedIndexedInputPermit
    names = [OLD_BASE + "/" + name for name in (
        "qualification_inputs.json", "training_input.json", "preparation.json", "experiment_spec.json")]
    names += [OLD_BASE + "/" + arm + "/contract.json" for arm in ARMS]
    observed = {name: _read_metadata(repo_local_path(root, name)) for name in names}
    metadata, _ = observed[names[0]]
    binding, binding_raw = observed[names[1]]
    preparation, _ = observed[names[2]]
    old_spec, _ = observed[names[3]]
    if (metadata.get("format") != "molgap-modern2d-qualification-input-metadata-v1"
            or metadata.get("status") != "INPUT_METADATA_PREPARED"
            or metadata.get("scientific_training_authorized") is not False
            or metadata.get("development_label_statistics_exposed") is not False
            or set(metadata.get("models", {})) != set(ARMS)
            or metadata.get("normalization") != {"role": "train_100k", "dtype": "float64", "correction": 1, "std_floor": 1e-6}
            or metadata.get("data_root") != "indexed_assets"
            or metadata.get("training_input") != {"path": "training_input.json",
                "sha256": hashlib.sha256(binding_raw).hexdigest(), "bytes": len(binding_raw)}
            or set(binding) != {"format", "data_root", "scope", "train", "development"}
            or binding["format"] != "molgap-indexed-training-input-v1"
            or binding["scope"] != "ACCEPTED_INDEXED_INPUT_V1"
            or binding["data_root"] != "indexed_assets"
            or preparation.get("metadata_sha256") != hashlib.sha256(observed[names[0]][1]).hexdigest()
            or preparation.get("spec_identity") != canonical_fingerprint(old_spec)
            or [j["arm_id"] for j in preparation["jobs"]] != list(ARMS)
            or [a["arm_id"] for a in old_spec["arms"]] != list(ARMS)
            or set(metadata.get("roles", {})) != {"train_100k", "common_dev"}):
        raise ValueError("Frozen qualification metadata/input identity differs")
    manifests, roles = {}, {}
    for name, role, rows in (("train", "train_100k", 100000), ("development", "common_dev", 50000)):
        item = binding[name]
        if type(item) is not dict or set(item) != {"permit", "authority", "role_history"}:
            raise ValueError("Exact accepted indexed role binding required")
        permit = AcceptedIndexedInputPermit(**item["permit"])
        permit.verify(item["authority"], item["role_history"])
        manifest, raw = _read_metadata(Path(role_manifest_root) / (role + ".json"))
        if (permit.role != role or permit.acceptance_sha256 != metadata["export_acceptance_sha256"]
                or hashlib.sha256(raw).hexdigest() != permit.manifest_sha256
                or manifest.get("role") != role or manifest.get("rows") != rows
                or manifest.get("protected") is not False or manifest.get("pure_2d") is not True
                or manifest.get("status") != "complete"
                or manifest.get("ordering") != "ascending-source-index"
                or manifest.get("version_identity") != permit.version_identity
                or metadata["roles"][role] != {"rows": rows,
                    "source_idx_tensor_sha256": manifest.get("source_index_sha256")}):
            raise ValueError("Accepted role manifest/source-index identity differs")
        _digest(manifest["source_index_sha256"], role + " source indices")
        manifests[role], roles[role] = (manifest, raw), permit
    if roles["train_100k"].version_identity != roles["common_dev"].version_identity:
        raise ValueError("Accepted train/CommonDev versions differ")
    return names, observed, metadata, binding, manifests, preparation, old_spec


def _target_manifest(path, sha256, *, metadata, binding):
    """A whole-role raw target digest must come from independently retained proof."""
    _digest(sha256, "development target manifest SHA256")
    target, raw = _read_metadata(path)
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Development target manifest bytes changed")
    permit = binding["development"]["permit"]
    if target.get("schema") == "molgap-modern2d-common-dev-identity-v1":
        proof = target
        pinned = proof["pinned_metadata"]
        role, source = pinned["role_manifest"], proof["source_index_tensor"]
        tensor = proof["target_tensor"]
        if (proof.get("status") != "COMMON_DEV_IDENTITY_ACCEPTED"
                or proof.get("source", {}).get("selected_role") != "common_dev"
                or role["sha256"] != permit["manifest_sha256"] or role["rows"] != 50000
                or role["version_identity"] != permit["version_identity"]
                or pinned["root_acceptance"]["sha256"] != permit["acceptance_sha256"]
                or pinned["training_input"]["sha256"] != metadata["training_input"]["sha256"]
                or pinned["training_input"]["authority"] != binding["development"]["authority"]
                or pinned["training_input"]["role_history"] != binding["development"]["role_history"]
                or tensor["role"] != "common_dev" or tensor["proof_rows"] != 50000
                or tensor["encoding"] != "little-endian float64 raw bytes"
                or source["role"] != "common_dev" or source["proof_rows"] != 50000
                or source["encoding"] != "little-endian int64 raw bytes"
                or source["sha256"] != role["source_index_sha256"]
                or tensor["chunk_count"] != role["chunk_count"]
                or len(tensor["chunk_rows"]) != tensor["chunk_count"]
                or any(type(n) is not int or n < 1 for n in tensor["chunk_rows"])
                or sum(tensor["chunk_rows"]) != 50000
                or proof["execution_boundary"].get("role_manifest_and_chunk_hashes_verified") is not True
                or any(proof["execution_boundary"].get(key) is not False for key in (
                    "training_executed", "models_loaded", "cuda_used", "graphs_rebuilt",
                    "official_validation_read", "test_roles_read", "protected_roles_read",
                    "train_dataset_instantiated", "audit_label_statistics_exposed"))):
            raise ValueError("Accepted CommonDev identity proof/provenance differs")
        target = {"rows": tensor["proof_rows"], "development_target_sha256": tensor["sha256"],
            "development_source_idx_sha256": source["sha256"], "target_encoding": TARGET_ENCODING,
            "role_manifest_sha256": role["sha256"],
            "export_acceptance_sha256": pinned["root_acceptance"]["sha256"]}
    if (target.get("rows") != 50000 or target.get("target_encoding") != TARGET_ENCODING
            or target.get("development_source_idx_sha256") != metadata["roles"]["common_dev"]["source_idx_tensor_sha256"]
            or target.get("role_manifest_sha256") != permit["manifest_sha256"]
            or target.get("export_acceptance_sha256") != permit["acceptance_sha256"]):
        raise ValueError("Development target manifest lacks exact accepted Dev50K/source-index provenance")
    _digest(target.get("development_target_sha256"), "whole-role development target digest")
    return target, raw


def freeze_formal_screen(repo_root, *, authorization, development_target_manifest,
                         development_target_manifest_sha256, init_root=INITIAL_ROOT,
                         role_manifest_root=ROLE_MANIFEST_ROOT, machine_profile=MACHINE_PROFILE,
                         base=BASE, timestamp=None):
    """Write a fresh two-reference declaration set, returning files to commit.

    ``authorization`` is the human's quoted grant of sixty passes/aggregate9 T4h.
    The SHA-pinned input accepts Beauvoir's nested
    ``molgap-modern2d-common-dev-identity-v1`` proof, or a compact manifest with
    ``rows`` (50000),
    ``development_target_sha256``, ``development_source_idx_sha256``,
    ``target_encoding`` (raw little-endian float64), ``role_manifest_sha256``
    and ``export_acceptance_sha256``. Chunk hashes are never a substitute.

    Initial-state files are streamed only for their existing file SHA; no torch
    load, model construction, tensor access, preparation, submission or finalize
    is performed. The normal typed Recipe owner validates the configuration.
    """
    from .experiment_execution import build_family_recipe
    from .experiment_spec import ExperimentSpec, FAMILIES, TERMINAL_PROTOCOL
    from .family_training_data import DatasetSelection
    from .family_training_recipe import FamilyRecipe, ScheduleSpec
    from .family_training_runner import family_arm_bindings
    from .modern2d_feature_binding import indexed_transform_identity, validate_indexed_feature_binding
    from .modern2d_training import AUTHORITY_FORMAT, FORMAL_STEPS, Modern2DFormalTrainingProfile
    from .kaggle_workflow import validate_plan
    from .experiment_package import _allowlist
    from .research_memory.policy import validate_policy

    root = Path(repo_root).absolute()
    _safe_local(root)
    base = safe_comparison_base(base, base=BASE)
    here = repo_local_path(root, base)
    _safe_local(here)
    if here.exists():
        raise ValueError("Formal declarations already exist; preserve and reconcile their bytes")
    if type(authorization) is not str or not authorization.strip():
        raise ValueError("The human's quoted formal training authorization is required")
    initial = Path(init_root).absolute()
    _safe_local(initial)
    if initial.is_relative_to(root) or root.is_relative_to(initial):
        raise ValueError("Reuse initial states from external storage")
    stamp = timestamp or datetime.now(timezone.utc).isoformat()
    names, observed, old_metadata, binding, manifests, preparation, old_spec = _inputs(root, role_manifest_root)
    target, target_raw = _target_manifest(development_target_manifest,
        development_target_manifest_sha256, metadata=old_metadata, binding=binding)
    snapshot = _source_snapshot(root, names)
    commit = snapshot["source_commit"]
    outputs = {}

    def retain(name, value, *, raw=None):
        payload = json_bytes(value) if raw is None else raw
        if name in outputs:
            raise ValueError("Duplicate formal declaration path")
        outputs[name] = payload
        return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}

    authority_ref, role_ref = base + "/authority.json", base + "/role_plan.json"
    authority = retain(authority_ref, {"format": AUTHORITY_FORMAT, "owner": "server",
        "purpose": "FORMAL_100K_TRAINING", "user_authorized": True, "authorization": authorization,
        "arms": list(ARMS), "interfaces": [list(i) for i in ARMS.values()], "seeds": [42],
        "roles": ["train_100k", "common_dev"], "physical_batch_size": 128,
        "horizon_passes": 60, "horizon_steps": FORMAL_STEPS, "allocated_t4_hours_ceiling": 9,
        "scientific_training_authorized": True, "protected_role_access_authorized": False,
        "extra_seed_authorized": False, "automatic_successor_authorized": False,
        "scale_500k_authorized": False, "full_training_authorized": False,
        "run_reference": KERNEL, "source_dataset": SOURCE_DATASET, "logical_run_id": RUN,
        "source_commit": commit, "gpu": {"platform": "kaggle2", "owner": ACCOUNT,
            "accelerator": "NvidiaTeslaT4", "allocated_devices": 2,
            "wall_ceiling_seconds": 16200, "allocated_t4_hours_ceiling": 9}})
    source_pin = retain(base + "/source_identity.json", snapshot)
    training_input = retain(base + "/training_input.json", binding,
                            raw=observed[OLD_BASE + "/training_input.json"][1])
    target["identity_proof"] = retain(base + "/development_target_proof.json", None, raw=target_raw)
    target_pin = retain(base + "/target_manifest.json", target)
    role_pins = {role: retain(base + "/roles/" + role + ".json", value, raw=raw)
                 for role, (value, raw) in manifests.items()}
    retain(role_ref, {"purpose": "FORMAL_100K_TRAINING", "roles_opened": ["train_100k", "common_dev"],
        "uses": {"train_100k": "training-and-normalization", "common_dev": "selection-used"},
        "role_manifests": role_pins, "target_manifest": target_pin,
        "protected_role_access_authorized": False})
    policy = {"schema": "molgap-policy-v1", "policy_id": POLICY_ID, "version": "v1",
        "policy_type": "research_action", "status": "candidate", "observation_point": None,
        "comparability_selector": {"scientific_contract": "modern2d-v2-commondev-independent-reference-100k-v1"},
        "required_observable_fields": ["user_authorized_formal_screen"],
        "created_from_source_digest": authority["sha256"],
        "approval": {"authority_ref": authority_ref, "activation": "one seed42 two-arm60-pass allocation under9 T4h"},
        "action_rule": {"field": "user_authorized_formal_screen", "operator": "eq", "threshold": True,
            "action": "RUN_SCREEN60"}, "promotion_rule": None, "early_stop_rule": None,
        "borderline_action": "DEFER", "cost_model": {"kind": "measured_only", "assumptions": [
            "Charge the full T4x2 allocation including preparation, idle devices and tail; aggregate ceiling9 T4h",
            "V5 context-only/INCONCLUSIVE; no matched reference, Replay or superiority",
            "No STQS, uncalibrated early-stop rule,100K veto,500K/full or automatic successor"]}}
    validate_policy(policy)
    retain(POLICY, policy)
    retain(base + "/STATUS.md", None, raw=(
        "# Formal Modern2D reference declarations\n\n"
        "Declarations frozen; preparation, native admission and execution remain separate.\n"
        "[Authority](authority.json): two independent seed42 references,60 passes, BS128, FP32/TF32off, "
        "T4x2 at most4.5h wall/9 allocated T4-hours.\n"
        "CommonDev V2 context only; INCONCLUSIVE without a matched reference, noReplay or superiority.\n"
        "No500K, extra seeds, full training, protected roles or automatic successor.\n").encode())
    metadata = deepcopy(old_metadata)
    metadata.update(format="molgap-modern2d-formal-input-metadata-v1",
        scientific_training_authorized=True, authority=authority, source_identity=source_pin,
        qualification_metadata={"path": OLD_BASE + "/qualification_inputs.json",
            "sha256": hashlib.sha256(observed[OLD_BASE + "/qualification_inputs.json"][1]).hexdigest()})
    arms, prospective, jobs, acceptance = [], [], [], []
    ref = lambda name, sha: {"name": name, "version": "1", "sha256": sha}
    for device, (arm_id, interface) in enumerate(ARMS.items()):
        old_contract_ref = OLD_BASE + "/" + arm_id + "/contract.json"
        old_contract = observed[old_contract_ref][0]
        old_arm = next(a for a in old_spec["arms"] if a["arm_id"] == arm_id)
        old_job = next(j for j in preparation["jobs"] if j["arm_id"] == arm_id)
        model = metadata["models"][arm_id]
        old_interface = MODEL_INTERFACES[arm_id]
        state = old_job["initial_state"]
        state_path = initial / (arm_id + ".pt")
        _safe_local(state_path)
        _digest(state["sha256"], "retained initial-state file digest")
        if (model["family"] != list(old_interface) or old_contract["model_config"] != model["model_config"]
                or old_contract["family"] != old_interface[0] or old_contract["family_version"] != old_interface[1]
                or old_contract["seed"] != 42 or old_job["recipe"] != old_contract_ref
                or old_arm["family"] != {"name": old_interface[0], "version": old_interface[1]}
                or old_arm["initialization"] != {"kind": "frozen_state", "seed": 42,
                    "state_sha256": old_contract["initialization_sha256"]}
                or old_arm["training"]["recipe"]["sha256"] != hashlib.sha256(observed[old_contract_ref][1]).hexdigest()
                or state["path"] != state_path.name or file_digest(state_path) != state["sha256"]):
            raise ValueError("Reuse exact old config, seed42 tensor hash and initial-state file SHA")
        model["family"] = list(interface)
        for role, (manifest, _) in manifests.items():
            feature = model["feature_bindings"][role]
            if (canonical_fingerprint(feature) != model["feature_binding_sha256"][role]
                    or feature["base_dataset_manifest_sha"] != role_pins[role]["sha256"]
                    or feature["source_row_sha"] != manifest["source_index_sha256"]
                    or feature["source_topology_sha"] != canonical_fingerprint({"chunks": [
                        c["topology_sha256"] for c in manifest["graph_chunks"]]})
                    or feature["transform_config_sha"] != canonical_fingerprint(model["model_config"])
                    or feature["role_history_sha"] != binding["train" if role == "train_100k" else "development"]["permit"]["role_history_sha256"]):
                raise ValueError("Frozen feature binding differs from accepted role/config metadata")
            validate_indexed_feature_binding(feature, feature)
            feature["transform_code_sha"] = indexed_transform_identity(interface)
            feature["feature_schema_version"] = FAMILIES[interface].feature_schema
            model["feature_binding_sha256"][role] = validate_indexed_feature_binding(feature, feature)
        usage = {name: canonical_fingerprint({"authority_sha256": authority["sha256"], "role": name,
            "arm_id": arm_id, "permit": binding[name]["permit"]}) for name in ("train", "development")}
        recipe = FamilyRecipe(family=interface[0], family_version=interface[1], mode="reference", seed=42,
            initialization_sha256=old_contract["initialization_sha256"],
            dataset=DatasetSelection("training_input.json", training_input["sha256"], True),
            train_rows=100000, development_rows=50000, optimizer_steps=FORMAL_STEPS,
            learning_rate=.001, weight_decay=1e-5,
            schedule=ScheduleSpec("four_stage_warmup_cosine_step", FORMAL_STEPS, 1e-6, 4, 60),
            selection="best-development-live", sampler="torch-pass-randperm", checkpoint_every=128,
            target_mean_eV=metadata["target_mean_eV"], target_std_eV=metadata["target_std_eV"],
            torch_version="2.7.1", model_config=deepcopy(model["model_config"]),
            formal_training=Modern2DFormalTrainingProfile())
        contract = build_family_recipe(interface, configuration=recipe,
            source_idx_sha256=target["development_source_idx_sha256"], target_sha256=target["development_target_sha256"],
            train_role_identity=usage["train"], development_role_identity=usage["development"])
        contract_pin = retain(base + "/" + arm_id + "/contract.json", contract)
        bindings, catalog = family_arm_bindings(recipe), FAMILIES[interface]
        arm = {"arm_id": arm_id, "scientific_role": "reference", "family": {"name": interface[0], "version": interface[1]},
            "base": bindings["base"], "initialization": deepcopy(old_arm["initialization"]),
            "data": {"dataset": ref("pcqm4mv2", training_input["sha256"]),
                "split": ref("accepted-indexed-train100k-commondev-v2", training_input["sha256"]),
                "roles": [{"role": name, "membership_sha256": metadata["roles"][role]["source_idx_tensor_sha256"],
                    "row_order_sha256": metadata["roles"][role]["source_idx_tensor_sha256"], "usage_sha256": usage[name]}
                    for name, role in (("train", "train_100k"), ("development", "common_dev"))],
                "feature_schema": catalog.feature_schema,
                "feature_sha256": canonical_fingerprint(model["feature_binding_sha256"]), "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": ref(catalog.recipe, contract_pin["sha256"]), "overrides": {},
                **{key: bindings[key] for key in ("objective", "sampler", "transform")}},
            "addons": [], "addon_semantics": "baseline"}
        tid = "TC-modern2d-formal-100k-" + arm_id + "-s42-v1"
        proposal = build_execution_proposal(question=ExecutionQuestion(POLICY_ID, "RUN_SCREEN60",
            "Establish this unchanged model's independent V2 CommonDev100K reference",
            "No accepted matched reference exists for this formal contract", "Independent new reference screen; no model tuning",
            (SESSION_EVIDENCE,), ("runtime cost may exceed the ceiling", "unmatched historical contracts"),
            ("k1-infrastructure-execution",), "retain new reference evidence for review; context-only/INCONCLUSIVE, noReplay",
            "retain bounded evidence; no100K veto or automatic successor"), arm=arm, trajectory_id=tid,
            source_commit=commit, run=RUN, version="v1", contract_ref=contract_pin["path"],
            contract_refs=[contract_pin["path"], authority_ref, role_ref, source_pin["path"], target_pin["path"], POLICY],
            authority_ref=authority_ref, role_ref=role_ref, decision_ref=base + "/STATUS.md", timestamp=stamp,
            profile=ExecutionProfile("server", "C", "kaggle2", "NvidiaTeslaT4x2_shared_allocation",
                "One independent seed42 reference per device;60 passes within aggregate9 T4-hours", None, None),
            action_type="training")
        proposal_pin = retain(base + "/" + arm_id + "/plan_input.json", proposal)
        prospective.append({"arm_id": arm_id, "trajectory_id": tid, "plan_spec_ref": proposal_pin["path"],
            "plan_spec_sha256": proposal_pin["sha256"], "output": base + "/" + arm_id + "/rml"})
        acceptance.append({"arm_id": arm_id, "adapter": "modern2d-formal-session-v1",
            "expected": contract["acceptance_requirements"], "contract": contract_pin, "target_manifest": target_pin})
        jobs.append({"arm_id": arm_id, "device": device, "recipe": contract_pin["path"],
            "initial_state": state_path.as_posix()})
        arms.append(arm)
    metadata_pin = retain(base + "/training_inputs.json", metadata)
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2", "experiment_id": "modern2d-formal-100k-reference-screen",
        "logical_run_id": RUN, "arms": arms, "prospective": {"arms": prospective},
        "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2,
            "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
        "evidence": {"policy": ref("molgap-v5", canonical_fingerprint({"contract": "MOLGAP-COMMON-V5-FINAL"})),
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": TERMINAL_PROTOCOL})
    retain(base + "/acceptance_plan.json", {"format": PLAN_FORMAT, "spec_identity": spec.identity,
        "authority": authority, "inputs": metadata_pin, "arms": acceptance})
    spec_ref = base + "/experiment_spec.json"
    retain(spec_ref, spec.to_dict())
    # The package owner supplies the canonical Spec as its reserved sidecar.
    source_records = sorted(set(outputs) - {spec_ref})
    _allowlist(source_records)
    kaggle = {"account": ACCOUNT, "kernel": KERNEL, "title": "MolGap Modern2D Formal 100K V1",
        "datasets": [SOURCE_DATASET, DATASET], "source_dataset": SOURCE_DATASET, "accelerator": "NvidiaTeslaT4"}
    validate_plan(spec, kaggle)
    workflow_ref = base + "/workflow.json"
    retain(workflow_ref, {"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
        "arms": jobs, "acceptance_plan": base + "/acceptance_plan.json", "kaggle": kaggle,
        "source_profile": "runtime-v1", "source_records": source_records})
    input_pin = retain(base + "/experiment_input.json", {"format": "molgap-experiment-input-v1",
        "spec_ref": spec_ref, "workflow_ref": workflow_ref, "machine_profile": str(machine_profile),
        "authority_ref": authority_ref, "monitor_paths": [base + "/STATUS.md"], "attempt_id": "v1"})
    if _source_snapshot(root, names) != snapshot:
        raise ValueError("Committed source changed during formal declaration construction")
    # Check the entire write set first; even partial earlier freezes are retained.
    for name in outputs:
        path = repo_local_path(root, name)
        _safe_local(path)
        if path.exists():
            raise ValueError("Formal declaration already exists: " + name)
    for name, raw in outputs.items():
        publish_immutable_bytes(repo_local_path(root, name), raw)
    return {"status": "FORMAL_SCREEN_DECLARATIONS_FROZEN", "base": base, "spec_identity": spec.identity,
        "source_commit": commit, "source_identity": source_pin, "commit_required": sorted(outputs),
        "experiment_input": input_pin, "experiment_input_ref": input_pin["path"],
        "spec_ref": spec_ref, "workflow_ref": workflow_ref, "source_records": source_records,
        "comparison_scope": "CONTEXT_ONLY", "scientific_status": "INCONCLUSIVE", "noReplay": True,
        "prepared": False, "submitted": False, "rml_published": False}
