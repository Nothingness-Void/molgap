"""Registered shared K1/GPTrans entrypoints; platform skills own job operations.

Interface versions identify this executable API, not model generations or a new
scientific contract. Historical frozen trainers are not silently migrated.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import signal
import sys
import time

from .experiment_family_workflow import RunContext, _artifact_path, _check_context, _digest, _json, _text, tensor_digest
from .family_training_recipe import FamilyRecipe
from .family_training_policy import mode_policy
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_json, atomic_torch_save
from .v4_runtime import normalized_source_sha256, state_dict_sha256
from .experiment_execution import shared_training_adapters


INTERFACES = {name: adapter.family for name, adapter in shared_training_adapters().items()}
PROFILES = {name: adapter.artifact_adapter for name, adapter in shared_training_adapters().items()}
METADATA = {"acceptance_requirements", "train_role_identity", "development_role_identity"}
OPTIONAL_METADATA = {"profiling", "logging", "resume_keep_count", "trace_manifest_plan", "initial_development_check"}


def _recipe_configuration(record):
    if "initial_development_check" in record:
        check = record["initial_development_check"]
        if (type(check) is not dict or set(check) != {"prediction_tolerance_eV", "mae_tolerance_eV"}
                or any(type(v) not in (int, float) or not 0 < v <= 1e-4 for v in check.values())):
            raise ValueError("Invalid frozen initialization prediction check")
    if "trace_manifest_plan" in record:
        from .research_memory.schemas import validate_trace_manifest_plan
        validate_trace_manifest_plan(record["trace_manifest_plan"])
    if "resume_keep_count" in record and (type(record["resume_keep_count"]) is not int or record["resume_keep_count"] < 1):
        raise ValueError("Expected a positive explicit resume_keep_count")
    if "logging" in record:
        from .training_logging import TrainingLogPolicy
        if record["logging"] is None:
            raise ValueError("Explicit logging configuration must be an object")
        TrainingLogPolicy.from_dict(record["logging"])
    if "profiling" in record:
        from .family_execution_profile import ProfileConfig
        ProfileConfig.from_dict(record["profiling"])
    return FamilyRecipe.from_dict({key: value for key, value in record.items()
        if key not in METADATA | OPTIONAL_METADATA})


def family_arm_bindings(recipe):
    """Static source/sampler/transform/objective bindings for a new Spec arm."""
    base_files = ("gps.py", "qm9_neural_atom.py") if recipe.family == "neural_atom_k1" else ("gptrans.py", "pcqm_gptrans_v4.py")
    base = canonical_fingerprint({name: normalized_source_sha256(Path(__file__).with_name(name)) for name in base_files})
    return {"base": {"name": recipe.family, "version": "1", "sha256": base},
        "sampler": {"name": "frozen-family-pass-order", "version": "1", "sha256": canonical_fingerprint({
            "algorithm": recipe.sampler, "seed": recipe.seed, "rows": recipe.train_rows, "drop_last": recipe.drop_last})},
        "transform": {"name": "frozen-train-mean-sample-std", "version": "1", "sha256": canonical_fingerprint({
            "manifest_sha256": recipe.dataset.manifest_sha256, "mean_eV": recipe.target_mean_eV, "std_eV": recipe.target_std_eV})},
        "objective": {"name": "normalized-gap-l1", "version": "1", "sha256": canonical_fingerprint({
            "mode": mode_policy(recipe.family, recipe.mode).objective_name or recipe.mode,
            "family": recipe.family, "gradient_clip": recipe.gradient_clip})}}


def _requirements(recipe, source_idx_sha256, target_sha256):
    plan = recipe.loop_plan
    passes, offset = divmod(recipe.optimizer_steps, plan.steps_per_pass)
    per_pass = plan.steps_per_pass * recipe.batch_size if recipe.drop_last else recipe.train_rows
    if recipe.dataset.development:
        _digest(source_idx_sha256, "development.source_idx_sha256")
        _digest(target_sha256, "development.target_sha256")
    elif source_idx_sha256 is not None or target_sha256 is not None:
        raise ValueError("Final-only training cannot declare development hashes")
    return {"optimizer_steps": recipe.optimizer_steps,
        "sample_presentations": passes * per_pass + min(offset * recipe.batch_size, per_pass),
        "completed_passes": passes, "next_batch": offset, "development_rows": recipe.development_rows,
        "source_idx_sha256": source_idx_sha256, "target_sha256": target_sha256, "precision": "fp32"}


def build_screen_recipe(mode, *, configuration, family, source_idx_sha256=None,
                        target_sha256=None, train_role_identity=None, development_role_identity=None,
                        logging=None, resume_keep_count=None):
    from .training_logging import TrainingLogPolicy
    if type(configuration) is not FamilyRecipe or family != INTERFACES.get(configuration.family) or mode != configuration.mode:
        raise ValueError("Explicit matching shared family configuration is required")
    _text(train_role_identity, "train_role_identity")
    if configuration.clean_bn is not None and configuration.clean_bn.training_role_identity != train_role_identity:
        raise ValueError("Calibration training role differs from the frozen recipe")
    if configuration.dataset.development:
        _text(development_role_identity, "development_role_identity")
    elif development_role_identity is not None:
        raise ValueError("Final-only training has no development role")
    if resume_keep_count is not None and (type(resume_keep_count) is not int or resume_keep_count < 1):
        raise ValueError("Expected a positive explicit resume_keep_count")
    return {**configuration.to_dict(), "acceptance_requirements": _requirements(configuration, source_idx_sha256, target_sha256),
        "train_role_identity": train_role_identity, "development_role_identity": development_role_identity,
        "logging": TrainingLogPolicy.from_dict(logging).to_dict(),
        **({"resume_keep_count": resume_keep_count} if resume_keep_count is not None else {})}


def read_recipe(path):
    record = _json(Path(path))
    if not METADATA <= set(record):
        raise ValueError("Shared contract lacks frozen acceptance/role bindings")
    recipe = _recipe_configuration(record)
    requirements = record["acceptance_requirements"]
    if type(requirements) is not dict:
        raise ValueError("Invalid shared acceptance requirements")
    expected = _requirements(recipe, requirements.get("source_idx_sha256"), requirements.get("target_sha256"))
    if requirements != expected:
        raise ValueError("Acceptance counters differ from the executable recipe")
    _text(record["train_role_identity"], "train_role_identity")
    if recipe.dataset.development:
        _text(record["development_role_identity"], "development_role_identity")
    elif record["development_role_identity"] is not None:
        raise ValueError("Final-only recipe contains a development role")
    if recipe.clean_bn is not None and recipe.clean_bn.training_role_identity != record["train_role_identity"]:
        raise ValueError("Calibration training role differs from contract")
    return record, recipe


def validate_screen_recipe(spec, arm_id, record):
    from .experiment_execution import training_adapter
    arm = next(arm for arm in spec.to_dict()["arms"] if arm["arm_id"] == arm_id)
    recipe = _recipe_configuration(record)
    adapter = training_adapter(arm)
    if (adapter.family != INTERFACES.get(recipe.family) or adapter.mode(arm) != recipe.mode or
            arm["initialization"] != {"kind": "frozen_state", "seed": recipe.seed, "state_sha256": recipe.initialization_sha256} or
            arm["data"]["dataset"]["sha256"] != recipe.dataset.manifest_sha256):
        raise ValueError("Shared family/initialization/data binding differs from Spec")
    bindings = family_arm_bindings(recipe)
    if arm["base"] != bindings["base"] or any(arm["training"][key] != bindings[key] for key in ("sampler", "transform", "objective")):
        raise ValueError("Shared base/sampler/normalization/objective binding differs from Spec")
    roles = {role["role"]: role for role in arm["data"]["roles"]}
    if set(roles) != ({"train", "development"} if recipe.dataset.development else {"train"}):
        raise ValueError("Selected roles differ from the executable recipe")
    if any(record.get(role + "_role_identity") != value["usage_sha256"] for role, value in roles.items()):
        raise ValueError("Contract role usage identity differs from Spec")
    if recipe.mode == "composed":
        from .shared_model_adapter import _addon_hook
        if list(recipe.addons) != arm["addons"]:
            raise ValueError("Recipe addon order/configuration differs from Spec")
        for addon in recipe.addons:
            _addon_hook(adapter, arm, addon)
    elif recipe.mode != "reference":
        from .experiment_spec import ADDONS
        addon = arm["addons"][0]
        module = ADDONS[(addon["name"], addon["version"])].source_module
        if addon["source_sha256"] != normalized_source_sha256(Path(__file__).parent / (module.split(".")[-1] + ".py")):
            raise ValueError("Addon source hash differs from executed module")
    if record.get("acceptance_requirements") != _requirements(recipe,
            record.get("acceptance_requirements", {}).get("source_idx_sha256"),
            record.get("acceptance_requirements", {}).get("target_sha256")):
        raise ValueError("Shared acceptance requirements differ from recipe")
    return {"family": arm["family"], "mode": recipe.mode, "recipe": "frozen_recipe_verified"}


def _loaded_source(package_dir):
    from .experiment_package import verify_experiment_source_package
    verify_experiment_source_package(Path(package_dir))
    root = Path(__file__).resolve().parents[2]
    inventory = {item["path"]: item["sha256"] for item in _json(Path(package_dir) / "SOURCE_FILES.json")["files"]}
    for name, module in tuple(sys.modules.items()):
        if name != "molgap" and not name.startswith("molgap."):
            continue
        file = getattr(module, "__file__", None)
        if file is None:
            raise ValueError("Unbound MolGap module: " + name)
        path = Path(file).resolve()
        if not path.is_relative_to(root):
            raise ValueError("MolGap import escaped the frozen source root: " + name)
        relative = path.relative_to(root).as_posix()
        if inventory.get(relative) != normalized_source_sha256(path):
            raise ValueError("Executed MolGap source differs from package: " + relative)
    return root


def _assert_target_statistics(recipe, data):
    """Require the selected fixed-train labels to match the frozen transform."""
    observed = (float(data.target_mean_eV), float(data.target_std_eV))
    if recipe.family == "neural_atom_k1":
        # The accepted historical K1 transform is frozen as FP32 constants;
        # shared data loading intentionally computes its observation in FP64.
        import torch
        observed = tuple(float(torch.tensor(value, dtype=torch.float32).item())
                         for value in observed)
    expected = (float(recipe.target_mean_eV), float(recipe.target_std_eV))
    if observed != expected:
        raise ValueError(
            "Decoded train target statistics differ from frozen recipe: "
            f"observed={observed!r}, expected={expected!r}"
        )


def _bootstrap(*, spec, package_dir, expected_package_identity, arm_id, mode, recipe_path,
               initial_state_path, input_root, account, run_reference, trajectory_id, prospective_root=None,
               training_only=False, cost_root=None):
    import torch
    from .experiment_execution import validate_staged_trajectory
    from .family_training_data import load_training_data
    from .gptrans_execution_tuning import ExecutionTuning
    from .training_reproducibility import build_runtime_manifest, configure_fp32_determinism, validate_accelerator_request
    from .v4_runtime import inspect_frozen_state_artifact
    record, recipe = read_recipe(recipe_path)
    validate_screen_recipe(spec, arm_id, record)
    if recipe.mode != mode:
        raise ValueError("Executable mode differs from recipe")
    context = RunContext.for_training(spec, package_dir, expected_package_identity=expected_package_identity,
        arm_id=arm_id, account=account, run_reference=run_reference)
    if normalized_source_sha256(Path(recipe_path)) != context.training_recipe_sha256:
        raise ValueError("Recipe bytes differ from Spec")
    trajectory = validate_staged_trajectory(spec, arm_id, Path(prospective_root or input_root))
    if trajectory["trajectory_id"] != trajectory_id or trajectory["state_at_start"]["source_commit"] != context.source_commit:
        raise ValueError("Prospective trajectory/source mismatch")
    if cost_root is not None:
        from .family_training_outputs import _recover_commit
        from .experiment_launch import _safe_local
        root = Path(cost_root).absolute()
        _safe_local(root)
        root.mkdir(parents=True, exist_ok=True)
        _recover_commit(root, context)
        _begin_cost_interval(root, context, had_checkpoint=(root / "last_checkpoint.pt").is_file())
    initial = inspect_frozen_state_artifact(Path(initial_state_path), expected_state_sha256=recipe.initialization_sha256)
    # Data acceptance remains CPU work and precedes construction of a GPU model.
    selection = replace(recipe.dataset, development=False) if training_only else recipe.dataset
    data = load_training_data(Path(input_root), selection)
    if recipe.input_sidecar is not None:
        data = mode_policy(recipe.family, recipe.mode).attach_inputs(Path(input_root), recipe.input_sidecar, data)
    arm = next(arm for arm in spec.to_dict()["arms"] if arm["arm_id"] == arm_id)
    for role in arm["data"]["roles"]:
        if training_only and role["role"] != "train":
            continue
        start = data.train_source_start if role["role"] == "train" else data.development_source_start
        rows = data.train_rows if role["role"] == "train" else data.development_rows
        digest = tensor_digest(torch.arange(start, start + rows), role="source_idx")
        if role["row_order_sha256"] != digest or role["membership_sha256"] != digest:
            raise ValueError("Decoded role membership/order differs from Spec")
    if data.train_rows != recipe.train_rows or (not training_only and data.development_rows != recipe.development_rows):
        raise ValueError("Decoded view counts differ from recipe")
    _assert_target_statistics(recipe, data)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Native training requires one isolated observed accelerator per worker")
    validate_accelerator_request(spec.to_dict()["platform"]["accelerator"], torch.cuda.get_device_name(0),
        hip_runtime=torch.version.hip is not None)
    tuning = ExecutionTuning(independent_models=len(spec.to_dict()["arms"]),
        total_cpu_budget=spec.to_dict()["platform"]["cpu_cores"])
    torch.set_num_threads(tuning.cpu_threads)
    runtime = build_runtime_manifest(configure_fp32_determinism(recipe.seed))
    source_root = _loaded_source(package_dir)
    provenance = {"context": context.to_dict(), "recipe_identity": recipe.identity,
        "dataset_manifest_sha256": recipe.dataset.manifest_sha256, "initialization": initial,
        "runtime_fingerprint": runtime["runtime_fingerprint"], "tuning_identity": tuning.identity,
        "accelerator": torch.cuda.get_device_name(0)}
    return record, recipe, data, tuning, context, provenance, source_root


def _state_fingerprint(value):
    import torch
    def encode(item):
        if isinstance(item, torch.Tensor):
            return {"shape": list(item.shape), "dtype": str(item.dtype),
                "sha256": state_dict_sha256({"value": item})}
        if isinstance(item, dict):
            return {str(key): encode(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(child) for child in item]
        return item
    return canonical_fingerprint(encode(value))


def run_screen_preflight(**options):
    """Scratch repeat/resume plus a GPTrans safety-fill reference; never dev."""
    import torch
    from .k1_execution_qualification import (
        _execution_state_fingerprint, _run_execution_evaluation,
        _native_probe, _run_execution_shadow, _verify_execution_proof,
        _write_execution_proof)
    from .family_training_kernel import FamilyKernel
    from .family_training_loaders import FamilyLoaders
    from .family_training_session import FamilyTrainingSession
    from .training_reproducibility import configure_fp32_determinism
    began = time.perf_counter()
    output = Path(options.pop("output")).absolute()
    record, recipe, data, tuning, context, provenance, _ = _bootstrap(training_only=True, **options)
    # Preflight consumes training topology only, never development selection.
    probe = _probe_recipe(recipe)
    data = replace(data, development=None, development_rows=0, development_source_start=None)
    output.mkdir(parents=True, exist_ok=True)
    proofs = []
    profile_proof = None
    memory_fill_proof = None
    labels = ("repeat-a", "repeat-b", "resumed")
    if recipe.family == "gptrans_t":
        labels += ("fill-enabled",)
    for label in labels:
        configure_fp32_determinism(recipe.seed)
        kernel = FamilyKernel(probe, Path(options["initial_state_path"]), device="cuda")
        kernel._qualification_memory_fill = label == "fill-enabled"
        observations = []
        def observe_execution(boundary):
            observations.append({"boundary": boundary,
                "memory_fill_enabled": torch.utils.deterministic.fill_uninitialized_memory,
                "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled()})
        if label == "fill-enabled":
            kernel._qualification_execution_observer = observe_execution
        _loaded_source(options["package_dir"])
        def checkpoint(progress, state):
            payload = state
            if label == "fill-enabled":
                payload = {"format": "molgap-gptrans-memory-fill-reference-v1",
                    "context": context.to_dict(), "recipe_identity": probe.identity,
                    "runtime_fingerprint": provenance["runtime_fingerprint"],
                    "execution_role": "fill-enabled-reference", "observations": list(observations),
                    "state": state}
            atomic_torch_save(output / (label + ".pt"), payload)
        session = FamilyTrainingSession(kernel, FamilyLoaders(probe, data, tuning),
            checkpoint=checkpoint, selected=lambda *args: None, pass_finished=lambda *args: None)
        if label == "resumed":
            session.run(should_pause=lambda: True)
            retained = torch.load(output / (label + ".pt"), map_location="cpu", weights_only=True)
            atomic_torch_save(output / "resumed-first.pt", retained)
            kernel = FamilyKernel(probe, Path(options["initial_state_path"]), device="cuda")
            session = FamilyTrainingSession(kernel, FamilyLoaders(probe, data, tuning),
                checkpoint=checkpoint, selected=lambda *args: None, pass_finished=lambda *args: None)
            session.load_state_dict(retained)
        if label == "repeat-a" and "profiling" in record:
            from .family_execution_profile import FamilyExecutionProfiler, ProfileConfig
            profile_config = replace(ProfileConfig.from_dict(record["profiling"]), warmup_steps=0, active_steps=2)
            with FamilyExecutionProfiler(session, output / "profile_qualification", profile_config) as observed:
                session.run()
            if observed.report["status"] != "COMPLETE" or not observed.report["device_events_observed"]:
                raise ValueError("Native profiler did not capture the requested CUDA events")
            profile_proof = {"report": "profile_qualification/profile.json",
                "sha256": normalized_file_sha(output / "profile_qualification/profile.json")}
        else:
            session.run()
        retained = torch.load(output / (label + ".pt"), map_location="cpu", weights_only=True)
        state = retained["state"] if label == "fill-enabled" else retained
        proof = {"path": label + ".pt", "sha256": normalized_file_sha(output / (label + ".pt")),
            "state_fingerprint": _state_fingerprint(state)}
        if label == "fill-enabled":
            memory_fill_proof = proof
        else:
            proofs.append(proof)
    if len({proof["state_fingerprint"] for proof in proofs}) != 1:
        raise ValueError("Native optimizer/EMA/schedule/RNG repeat or serialized resume differs")
    if memory_fill_proof is not None:
        _verify_memory_fill_proof(output, probe, memory_fill_proof, proofs[0]["state_fingerprint"], provenance)
    execution_proof = None
    if recipe.k1_execution is not None:
        # Keep the candidate recipe identity in the candidate state, but run a
        # shadow with the historical backend under the same data/init/RNG.  The
        # comparison removes only the recipe identity inside the cursor.
        native_probe = _native_probe(recipe)
        native_state, native_path = _run_execution_shadow(
            recipe=native_probe, data=data, tuning=tuning, options=options, output=output)
        candidate_path = output / "repeat-b.pt"
        candidate_state = torch.load(candidate_path, map_location="cpu", weights_only=True)
        if _execution_state_fingerprint(candidate_state) != _execution_state_fingerprint(native_state):
            raise ValueError("K1 execution candidate/native optimizer state differs")
        candidate_eval, candidate_source = _run_execution_evaluation(
            recipe=probe, data=data, tuning=tuning, options=options,
            retained_state=candidate_state)
        native_eval, native_source = _run_execution_evaluation(
            recipe=native_probe, data=data, tuning=tuning, options=options,
            retained_state=native_state)
        if not torch.equal(candidate_source, native_source):
            raise ValueError("K1 execution evaluation fixtures differ")
        execution_proof = _write_execution_proof(
            output=output, probe=probe, candidate_state=candidate_state,
            candidate_path=candidate_path, native_state=native_state,
            native_path=native_path, candidate_eval=candidate_eval,
            candidate_eval_path=output / "execution-candidate-evaluation.pt",
            native_eval=native_eval,
            native_eval_path=output / "execution-native-evaluation.pt",
            source_idx=candidate_source,
            source_idx_path=output / "execution-evaluation-source-idx.pt")
        _verify_execution_proof(output, probe, execution_proof, required=True)
    resource = None
    if mode_policy(recipe.family, recipe.mode).qualify_resources is not None:
        resource = mode_policy(recipe.family, recipe.mode).qualify_resources(recipe, data, tuning, Path(options["initial_state_path"]))
        atomic_json(output / "resource_qualification.json", resource)
    atomic_json(output / "runtime_provenance.json", provenance)
    certificate = {"format": "molgap-runtime-certificate-v1", "status": "accepted",
        "provenance_sha256": canonical_fingerprint(provenance), "probe_recipe_identity": probe.identity,
        "proofs": proofs, "resume_origin": {"path": "resumed-first.pt",
            "sha256": normalized_file_sha(output / "resumed-first.pt")},
        "formal_training_exposure": 0, "scientific_acceptance": "NOT_EVALUATED"}
    if profile_proof is not None:
        certificate["profile_proof"] = profile_proof
    if memory_fill_proof is not None:
        certificate["memory_fill_proof"] = memory_fill_proof
    if execution_proof is not None:
        certificate["execution_proof"] = execution_proof
    if resource is not None:
        certificate["resource_qualification_sha256"] = normalized_file_sha(output / "resource_qualification.json")
    from .native_proof_storage import compact_certificate, retire_proof_sources
    from .research_memory.trace import atomic_write, json_bytes
    compact_certificate(output, certificate)
    atomic_write(output / "runtime_certificate.json", json_bytes(certificate))
    retire_proof_sources(output, certificate)
    atomic_json(output / "preflight_costs.json", {"context": context.to_dict(), "costs": [
        {"metric": "wall_seconds", "unit": "seconds", "value": time.perf_counter() - began,
            "status": "measured", "semantics": "process_wall", "hardware": provenance["accelerator"],
            "scope": "native qualification entry through proof publication"},
        {"metric": "device_seconds", "unit": "seconds", "value": None, "status": "missing",
            "semantics": "allocated_device", "hardware": provenance["accelerator"],
            "reason": "requires platform allocation accounting"}]})
    return certificate


def normalized_file_sha(path):
    # Artifact bytes are exact; only source text uses normalized line endings.
    from .training_reproducibility import sha256_file
    return sha256_file(Path(path))


def _probe_recipe(recipe):
    return replace(recipe, dataset=replace(recipe.dataset, development=False), development_rows=0,
        input_sidecar=replace(recipe.input_sidecar, development=False) if recipe.input_sidecar is not None else None,
        optimizer_steps=2, checkpoint_every=1,
        selection="final-clean-ema" if recipe.clean_bn is not None else "final-live")


def validate_screen_resume(*, spec, arm_id, manifest, bundle_root, artifacts, sidecars,
                           trajectory, context=None):
    """Recheck the actual shared checkpoint/trace before portable transport."""
    import torch
    from .family_training_session import validate_session_resume
    from .research_memory.trace import validate_canonical_trace
    record, recipe = read_recipe(Path(bundle_root) / "files/training_contract.json")
    validate_screen_recipe(spec, arm_id, record)
    saved = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=True)
    if context is not None:
        _check_context(saved.pop("context"), RunContext(**context))
    else:
        saved.pop("context")
    # The wire writer omits disabled EMA, as SessionOutputs.retained_state handles.
    # The session validator still rejects a missing EMA state for EMA-enabled recipes.
    saved.setdefault("ema", None)
    progress = validate_session_resume(recipe, saved)
    if progress.status == "complete" or progress.optimizer_step >= recipe.optimizer_steps:
        raise ValueError("Completed shared session cannot be resumed")
    trace = validate_canonical_trace(_json(artifacts["trace"]))
    observations = [r for r in trace["observations"] if r["event"] == "observation"]
    if len(observations) != progress.pass_index or any(r["optimizer_step"] > progress.optimizer_step for r in observations):
        raise ValueError("Resume trace/cursor mismatch")
    if trace["trajectory_id"] != trajectory["trajectory_id"]:
        raise ValueError("Resume trace/trajectory identity mismatch")
    return {"status": "SHARED_RESUME_VERIFIED", "optimizer_steps": progress.optimizer_step,
            "sample_presentations": progress.sample_presentations, "recipe_identity": recipe.identity}


def run_screen_resume_preflight(*, resume_output, **options):
    """Qualify initial repeat/resume plus two scratch next-steps from actual work."""
    import torch
    from .family_training_session import validate_session_resume
    result = run_screen_preflight(**options)
    output = Path(options["output"])
    bootstrap_options = {k: v for k, v in options.items() if k != "output"}
    _, recipe, data, tuning, context, _, _ = _bootstrap(**bootstrap_options)
    retained = torch.load(Path(resume_output) / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    _check_context(retained.pop("context"), context)
    retained.setdefault("ema", None)
    progress = validate_session_resume(recipe, retained)
    if progress.status == "complete":
        raise ValueError("Cannot resume a completed shared session")
    if mode_policy(recipe.family, recipe.mode).qualify_resources is not None:
        resource = _json(output / "resource_qualification.json")
        qualified = mode_policy(recipe.family, recipe.mode).refresh_resources(resource, progress.pass_index)
        atomic_json(output / "resource_qualification.json", qualified)
        cert = _json(output / "runtime_certificate.json")
        cert["resource_qualification_sha256"] = normalized_file_sha(output / "resource_qualification.json")
        atomic_json(output / "runtime_certificate.json", cert)
    proofs = _actual_resume_proofs(recipe, data, tuning, Path(options["initial_state_path"]),
                                  retained, output, device="cuda")
    actual = {"status": "accepted", "proofs": proofs,
        "formal_training_exposure": 0, "resume_step": progress.optimizer_step,
        "checkpoint_sha256": normalized_file_sha(Path(resume_output) / "last_checkpoint.pt")}
    from .native_proof_storage import compact_certificate, retire_proof_sources
    from .research_memory.trace import atomic_write, json_bytes
    compact_certificate(output, actual, actual_resume=True)
    atomic_write(output / "actual_resume_certificate.json", json_bytes(actual))
    retire_proof_sources(output, actual)
    return result


def _actual_resume_proofs(recipe, data, tuning, initial_state, retained, output, *, device):
    """Repeat one scratch update without lending the retained optimizer tensors."""
    import copy
    import torch
    from .family_training_kernel import FamilyKernel
    from .family_training_loaders import FamilyLoaders
    from .family_training_session import FamilyTrainingSession, validate_session_resume
    from .training_reproducibility import configure_fp32_determinism
    progress = validate_session_resume(recipe, retained)
    proofs = []
    for label in ("actual-resume-a", "actual-resume-b"):
        configure_fp32_determinism(recipe.seed)
        kernel = FamilyKernel(recipe, initial_state, device=device)
        session = FamilyTrainingSession(kernel, FamilyLoaders(recipe, data, tuning),
            checkpoint=lambda p, state: atomic_torch_save(output / (label + ".pt"), state),
            selected=lambda *args: None, pass_finished=lambda *args: None)
        # AdamW may retain CPU step tensors by reference during restore.
        session.load_state_dict(copy.deepcopy(retained))
        endpoint = session.run(should_pause=lambda: True)
        observed = torch.load(output / (label + ".pt"), map_location="cpu", weights_only=True)
        if endpoint.optimizer_step != progress.optimizer_step + 1:
            raise ValueError("Actual resume proof did not advance exactly one scratch update")
        proofs.append({"path": label + ".pt", "sha256": normalized_file_sha(output / (label + ".pt")),
                       "state_fingerprint": _state_fingerprint(observed)})
    if proofs[0]["state_fingerprint"] != proofs[1]["state_fingerprint"]:
        raise ValueError("Actual resume next-step repeats differ")
    return proofs


def _verify_profile_proof(root, probe, profile_proof, *, required):
    if required and profile_proof is None:
        raise ValueError("Native profiling proof is required by the frozen recipe")
    if profile_proof is None:
        return
    if (type(profile_proof) is not dict or set(profile_proof) != {"report", "sha256"}
            or profile_proof["report"] != "profile_qualification/profile.json"):
        raise ValueError("Malformed native profiling proof")
    profile_path = _artifact_path(root, profile_proof["report"])
    if normalized_file_sha(profile_path) != profile_proof["sha256"]:
        raise ValueError("Native profiler report changed")
    report = _json(profile_path)
    if (report.get("status") != "COMPLETE" or report.get("device_events_observed") is not True
            or report.get("captured_updates") != 2 or report.get("recipe_identity") != probe.identity
            or normalized_file_sha(_artifact_path(root, "profile_qualification/trace.json")) != report.get("trace_sha256")):
        raise ValueError("Native profiler event/recipe/trace evidence differs")


def _verify_memory_fill_proof(root, probe, proof, expected_fingerprint, provenance):
    """An observed reference artifact, not a summary boolean, gates fill removal."""
    from .family_training_session import validate_session_resume
    if (type(proof) is not dict or set(proof) != {"path", "sha256", "state_fingerprint"}
            or proof["path"] != "fill-enabled.pt"):
        raise ValueError("Missing native safety-fill reference proof")
    from .native_proof_storage import load_proof
    retained = load_proof(root, proof)
    if (type(retained) is not dict or set(retained) != {"format", "context", "recipe_identity",
            "runtime_fingerprint", "execution_role", "observations", "state"}
            or retained["format"] != "molgap-gptrans-memory-fill-reference-v1"
            or retained["context"] != provenance["context"]
            or retained["recipe_identity"] != probe.identity
            or retained["runtime_fingerprint"] != provenance["runtime_fingerprint"]
            or retained["execution_role"] != "fill-enabled-reference"):
        raise ValueError("Native safety-fill producer identity differs")
    observations = retained["observations"]
    boundaries = ["before_update", "after_update"] * probe.optimizer_steps
    if (type(observations) is not list or len(observations) != len(boundaries) or any(
            type(item) is not dict or set(item) != {"boundary", "memory_fill_enabled",
                "deterministic_algorithms_enabled", "warn_only"}
            or item["boundary"] != boundary or item["memory_fill_enabled"] is not True
            or item["deterministic_algorithms_enabled"] is not True or item["warn_only"] is not False
            for item, boundary in zip(observations, boundaries))):
        raise ValueError("Native safety-fill execution observations are incomplete")
    state = retained["state"]
    progress = validate_session_resume(probe, state)
    if progress.status != "complete" or len(state["rng_state"]["cuda"]) != 1:
        raise ValueError("Safety-fill reference is not a complete isolated-GPU proof")
    fingerprint = _state_fingerprint(state)
    if fingerprint != proof["state_fingerprint"] or fingerprint != expected_fingerprint:
        raise ValueError("Native safety-fill reference differs from accelerated execution")


def _verify_preflight(directory, recipe, provenance, *, profiling_required=False):
    from .family_training_session import validate_session_resume
    from .k1_execution_qualification import _verify_execution_proof
    root = Path(directory).absolute()
    saved = _json(_artifact_path(root, "runtime_provenance.json"))
    certificate = _json(_artifact_path(root, "runtime_certificate.json"))
    if saved != provenance or certificate.get("provenance_sha256") != canonical_fingerprint(provenance):
        raise ValueError("Native preflight source/recipe/data/runtime/device identity differs")
    if certificate.get("status") != "accepted":
        raise ValueError("Native preflight certificate is not accepted")
    if mode_policy(recipe.family, recipe.mode).qualify_resources is not None:
        resource_path = root / "resource_qualification.json"
        if certificate.get("resource_qualification_sha256") != normalized_file_sha(resource_path):
            raise ValueError("Missing or changed native resource qualification")
    probe = _probe_recipe(recipe)
    if (certificate.get("format") != "molgap-runtime-certificate-v1" or
            certificate.get("probe_recipe_identity") != probe.identity or certificate.get("formal_training_exposure") != 0):
        raise ValueError("Preflight probe identity differs")
    proofs = certificate.get("proofs")
    if type(proofs) is not list or [proof.get("path") for proof in proofs] != ["repeat-a.pt", "repeat-b.pt", "resumed.pt"]:
        raise ValueError("Missing native repeat/resume artifacts")
    origin = certificate.get("resume_origin")
    if type(origin) is not dict or set(origin) != {"path", "sha256"} or origin["path"] != "resumed-first.pt":
        raise ValueError("Missing serialized native resume origin")
    from .native_proof_storage import load_proof
    origin_state = load_proof(root, origin)
    origin_progress = validate_session_resume(probe, origin_state)
    if origin_progress.optimizer_step != 1 or origin_progress.status != "paused" or len(origin_state["rng_state"]["cuda"]) != 1:
        raise ValueError("Native resume origin is not an observed isolated-GPU scratch step")
    fingerprints = []
    for proof in proofs:
        state = load_proof(root, proof)
        if validate_session_resume(probe, state).status != "complete":
            raise ValueError("Incomplete native proof endpoint")
        if len(state["rng_state"]["cuda"]) != 1:
            raise ValueError("CPU-only proof cannot qualify native training")
        fingerprint = _state_fingerprint(state)
        if fingerprint != proof["state_fingerprint"]:
            raise ValueError("Native proof tensor identity mismatch")
        fingerprints.append(fingerprint)
    if len(set(fingerprints)) != 1:
        raise ValueError("Native repeat/resume proof differs")
    if recipe.family == "gptrans_t":
        _verify_memory_fill_proof(root, probe, certificate.get("memory_fill_proof"), fingerprints[0], provenance)
    _verify_execution_proof(root, probe, certificate.get("execution_proof"),
        required=recipe.k1_execution is not None,
        expected_candidate_sha256=proofs[1]["sha256"] if proofs else None)
    _verify_profile_proof(root, probe, certificate.get("profile_proof"), required=profiling_required)
    return certificate


def inspect_retained_native_qualification(directory, *, recipe_path, provenance, context,
        resume_checkpoint_sha256=None, resume_directory=None, inventory_only=False):
    """Inventory retained proof dependencies and reuse the native CPU recheck.

    Missing byte sizes stay unknown. This reads safe CPU tensors only; it does
    not run a model, recreate a probe or confer scientific/replay acceptance.
    Inventory-only callers verify dependency bytes without loading proof tensors.
    """
    from .experiment_launch import _safe_local
    from .research_memory.paths import repo_local_path
    # The selected source owner parses its own frozen recipe extensions.
    from .family_training_runner import read_recipe as read_owned_recipe
    root = Path(directory).absolute()
    _safe_local(root)
    record, recipe = read_owned_recipe(recipe_path)
    _check_context(provenance.get("context"), context)
    if (normalized_source_sha256(Path(recipe_path)) != context.training_recipe_sha256
            or provenance.get("recipe_identity") != recipe.identity):
        raise ValueError("Native qualification recipe differs from the frozen run")
    certificate = _json(_artifact_path(root, "runtime_certificate.json"))
    dependencies = {}
    from .native_proof_storage import retained_storage, proof_stream
    storages = {}
    native_storage = retained_storage(root, certificate)
    if native_storage is not None:
        storages[(str(root), native_storage["path"])] = native_storage

    def add(name, digest=None, *, base=root):
        # Keep the same local-path boundary for absent and present artifacts.
        if (type(name) is not str or "\\" in name or ":" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError("Invalid native proof dependency path")
        storage = next((item for (directory, _), item in storages.items()
                        if directory == str(base) and name in item["members"]), None)
        if storage is not None and name in storage["members"]:
            if digest != storage["members"][name]["sha256"]:
                raise ValueError("Native proof member hash differs from certificate")
            name, digest = storage["path"], storage["sha256"]
        path = repo_local_path(base, name)
        _safe_local(path)
        key = str(path)
        if key in dependencies and dependencies[key]["sha256"] not in {None, digest}:
            raise ValueError("Conflicting native proof dependency hashes")
        if digest is not None:
            _digest(digest, "native proof dependency.sha256")
        present = path.is_file()
        dependencies[key] = {"path": name, "local_path": str(path), "sha256": digest,
            "present": present, "size_bytes": path.stat().st_size if present else (
                storage["size_bytes"] if storage is not None and name == storage["path"] else None)}
        return dependencies[key]

    def walk(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                add(value["path"], value["sha256"])
            for key, child in value.items():
                if key != "proof_storage":
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    add("runtime_certificate.json")
    add("runtime_provenance.json")
    walk(certificate)
    if "resource_qualification_sha256" in certificate:
        add("resource_qualification.json", certificate["resource_qualification_sha256"])
    profile = certificate.get("profile_proof")
    if isinstance(profile, dict) and "report" in profile:
        report = add(profile["report"], profile.get("sha256"))
        add("profile_qualification/trace.json", _json(Path(report["local_path"])).get("trace_sha256")
            if report["present"] else None)
    resume_root = Path(resume_directory or root).absolute()
    actual = None
    if resume_checkpoint_sha256 is not None:
        _digest(resume_checkpoint_sha256, "resume checkpoint.sha256")
        resume_cert = add("actual_resume_certificate.json", base=resume_root)
        if resume_cert["present"]:
            actual = _json(Path(resume_cert["local_path"]))
            actual_storage = retained_storage(resume_root, actual, actual_resume=True)
            if actual_storage is not None:
                # Both packs may share a directory; add() selects by member.
                for proof in actual.get("proofs", []):
                    if proof["sha256"] != actual_storage["members"][proof["path"]]["sha256"]:
                        raise ValueError("Actual resume proof storage binding differs")
                storages[(str(resume_root), actual_storage["path"])] = actual_storage
            for proof in actual.get("proofs", []):
                add(proof["path"], proof["sha256"], base=resume_root)
    rows = [dependencies[name] for name in sorted(dependencies)]
    missing = [item["path"] for item in rows if not item["present"]]
    result = {"status": "BLOCKED", "dependencies": rows, "missing": missing,
        "known_local_bytes": sum(item["size_bytes"] or 0 for item in rows if item["present"]),
        "missing_size_bytes": (sum(item["size_bytes"] for item in rows if not item["present"])
            if all(item["size_bytes"] is not None for item in rows if not item["present"]) else None),
        "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_EVALUATED"}
    # Verify every available hash even while another dependency is missing.
    for item in rows:
        if item["present"]:
            item["verified_sha256"] = normalized_file_sha(Path(item["local_path"]))
            if item["sha256"] is not None and item["verified_sha256"] != item["sha256"]:
                raise ValueError("Native proof artifact hash mismatch: " + item["path"])
    if actual is not None and (
            actual.get("status") != "accepted" or actual.get("formal_training_exposure") != 0
            or actual.get("checkpoint_sha256") != resume_checkpoint_sha256
            or type(actual.get("resume_step")) is not int
            or [proof.get("path") for proof in actual.get("proofs", [])]
                != ["actual-resume-a.pt", "actual-resume-b.pt"]):
        raise ValueError("Actual retained-state native resume qualification differs")
    if missing:
        result["blockers"] = ["native_proof_dependencies_missing"]
        return result
    if inventory_only:
        # Legacy local migration has no archive pin in its immutable certificate.
        # Original per-file SHA values authenticate the reconstructed bytes.
        for (base, _), storage in storages.items():
            if storage["sha256"] is None:
                for name, item in storage["members"].items():
                    with proof_stream(base, name, item["sha256"], packed=True):
                        pass
        return {**result, "status": "DEPENDENCIES_VERIFIED", "blockers": []}
    _verify_preflight(root, recipe, provenance, profiling_required=record.get("profiling") is not None)
    if resume_checkpoint_sha256 is not None:
        from .family_training_session import validate_session_resume
        fingerprints = []
        for proof in actual["proofs"]:
            from .native_proof_storage import load_proof
            state = load_proof(resume_root, proof)
            progress = validate_session_resume(recipe, state)
            fingerprint = _state_fingerprint(state)
            if (progress.optimizer_step != actual["resume_step"] + 1
                    or len(state["rng_state"].get("cuda", [])) != 1
                    or fingerprint != proof.get("state_fingerprint")):
                raise ValueError("Actual native resume proof state/cursor differs")
            fingerprints.append(fingerprint)
        if len(set(fingerprints)) != 1:
            raise ValueError("Actual native resume next-step repeats differ")
    return {**result, "status": "VERIFIED", "blockers": []}


def _begin_cost_interval(root, context, *, had_checkpoint):
    """An unfinished invocation or unbound ledger makes cumulative cost unknown."""
    path, marker = root / "allocation_costs.json", root / "invocation_cost_state.json"
    prior = 0. if not had_checkpoint else None
    if path.is_file():
        record = _json(_artifact_path(root, path.name))
        _check_context(record.get("context"), context)
        value = record.get("wall_seconds")
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid retained native cost ledger")
        bound = record.get("checkpoint_sha256")
        if had_checkpoint and bound == normalized_file_sha(_artifact_path(root, "last_checkpoint.pt")):
            prior = value
        elif not had_checkpoint:
            raise ValueError("Retained cost ledger has no corresponding checkpoint")
    if marker.is_file():
        interval = _json(_artifact_path(root, marker.name))
        _check_context(interval.get("context"), context)
        if interval.get("status") not in {"running", "closed"}:
            raise ValueError("Invalid native cost invocation marker")
        if interval["status"] == "running":
            prior = None
    atomic_json(marker, {"context": context.to_dict(), "status": "running", "prior_wall_seconds": prior})
    return prior


def _finish_cost_interval(root, context, prior, elapsed):
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid invocation elapsed cost")
    if prior is not None and (type(prior) not in (int, float) or not math.isfinite(prior) or prior < 0):
        raise ValueError("Invalid prior invocation cost")
    total = prior + elapsed if prior is not None else None
    if total is not None and not math.isfinite(total):
        raise ValueError("Cumulative invocation cost overflow")
    atomic_json(root / "allocation_costs.json", {"context": context.to_dict(), "wall_seconds": total,
        "current_invocation_seconds": elapsed, "checkpoint_sha256": normalized_file_sha(_artifact_path(root, "last_checkpoint.pt"))})
    atomic_json(root / "invocation_cost_state.json", {"context": context.to_dict(), "status": "closed",
        "prior_wall_seconds": prior, "current_invocation_seconds": elapsed})
    return total


def _resume_trajectory(recipe, context, options):
    """Chunk retention is explicit recipe/policy, never an experiment-name lookup."""
    if not recipe.resume_chunks and mode_policy(recipe.family, recipe.mode).segment_passes is None:
        return None
    if options.get("prospective_root") is None:
        raise ValueError("Resume chunks require the verified staged prospective root")
    return _artifact_path(Path(options["prospective_root"]),
        "prospective/" + context.arm_id + "/trajectory.json").read_bytes()


def run_screen_arm(*, output, preflight_dir=None, target_identity=None, pause_after_step=None, **options):
    """Same callable for every scale/platform; same-budget resume only."""
    import torch
    if pause_after_step is not None and (type(pause_after_step) is not int or pause_after_step < 1):
        raise ValueError("pause_after_step must be a positive absolute optimizer step")
    from .family_training_kernel import FamilyKernel
    from .family_training_loaders import FamilyLoaders
    from .family_training_outputs import SessionOutputs
    from .family_training_session import FamilyTrainingSession
    began = time.perf_counter()
    root = Path(output).absolute()
    if (root / "output_manifest.json").is_file():
        # A completed retry only inspects immutable artifacts; no model/data is opened.
        from .experiment_family_workflow import inspect_output
        record, recipe = read_recipe(options["recipe_path"])
        validate_screen_recipe(options["spec"], options["arm_id"], record)
        context = RunContext.for_training(options["spec"], options["package_dir"],
            expected_package_identity=options["expected_package_identity"], arm_id=options["arm_id"],
            account=options["account"], run_reference=options["run_reference"])
        if normalized_source_sha256(Path(options["recipe_path"])) != context.training_recipe_sha256:
            raise ValueError("Recipe bytes differ from Spec")
        return inspect_output(root, context=context, expected=record["acceptance_requirements"], target_identity=target_identity)
    record, recipe, data, tuning, context, provenance, _ = _bootstrap(cost_root=root, **options)
    certificate = _verify_preflight(preflight_dir or output, recipe, provenance,
        profiling_required="profiling" in record)
    outputs = SessionOutputs(root, context, contract=options["recipe_path"], adapter=PROFILES[recipe.family],
        trajectory_id=options["trajectory_id"], train_role_identity=record["train_role_identity"],
        resume_trajectory=_resume_trajectory(recipe, context, options))
    had_checkpoint = (root / "last_checkpoint.pt").is_file()
    interval = _json(_artifact_path(root, "invocation_cost_state.json"))
    _check_context(interval["context"], context)
    prior_cost = interval["prior_wall_seconds"]
    kernel = FamilyKernel(recipe, Path(options["initial_state_path"]), device="cuda")
    _loaded_source(options["package_dir"])
    session = FamilyTrainingSession(kernel, FamilyLoaders(recipe, data, tuning),
        checkpoint=outputs.checkpoint, selected=outputs.selected, pass_finished=outputs.pass_finished,
        batch_finished=outputs.batch_finished, checkpoint_builder=outputs.checkpoint_from_builder)
    try:
        if had_checkpoint:
            if mode_policy(recipe.family, recipe.mode).qualify_resources is not None:
                actual = _json(_artifact_path(Path(preflight_dir or output), "actual_resume_certificate.json"))
                if (actual.get("status") != "accepted" or actual.get("formal_training_exposure") != 0 or
                        actual.get("checkpoint_sha256") != normalized_file_sha(root / "last_checkpoint.pt")):
                    raise ValueError("Actual retained-state native resume qualification is missing or changed")
            session.load_state_dict(outputs.retained_state())
        if pause_after_step is not None and not session.progress.optimizer_step < pause_after_step <= recipe.optimizer_steps:
            raise ValueError("Step pause must follow the retained cursor within the frozen training horizon")
        outputs.logger.start(session.progress)
    except Exception:
        session.close()
        raise
    pause = [False]
    previous_signal = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, lambda *_: pause.__setitem__(0, True))
        if not had_checkpoint and "initial_development_check" in record:
            # A warm start qualifies its retained predictor before any update.
            from .v4_runtime import torch_load_compat
            retained = torch_load_compat(Path(options["initial_state_path"]), map_location="cpu", weights_only=True)
            reference = retained["reference_development"]
            before = state_dict_sha256(kernel.model.state_dict())
            observed = kernel.evaluate(session.loaders, weights="live")
            if (not torch.equal(observed["source_idx"], reference["source_idx"])
                    or not torch.equal(observed["target_eV"], reference["target_eV"])):
                raise ValueError("Initial predictor development rows/targets differ")
            delta = float((observed["prediction_eV"] - reference["prediction_eV"]).abs().max())
            mae_delta = abs(observed["mae_eV"] - float(reference["mae_eV"]))
            check = record["initial_development_check"]
            if (delta > check["prediction_tolerance_eV"] or mae_delta > check["mae_tolerance_eV"]
                    or before != state_dict_sha256(kernel.model.state_dict())):
                raise ValueError("Initial predictor does not reproduce its retained prediction")
            atomic_json(root / "initial_development_check.json", {"context": context.to_dict(),
                "status": "VERIFIED", "weights": "live", "optimizer_steps": 0,
                "initialization_sha256": before, "development_rows": len(observed["source_idx"]),
                "mae_eV": observed["mae_eV"], "maximum_prediction_difference_eV": delta,
                "mae_difference_eV": mae_delta, "tolerances": check})
            print("INITIAL_DEVELOPMENT_VERIFIED " + str(observed["mae_eV"]), flush=True)
        # A completed physical ten-pass segment makes work externally retrievable.
        # The scientific horizon, scheduler and selected-state policy remain60.
        segment = mode_policy(recipe.family, recipe.mode).segment_passes
        bounded = segment is not None
        stop_pass = ((session.progress.pass_index // segment + 1) * segment) if bounded else None
        acknowledged_step = [session.progress.optimizer_step]
        def should_pause():
            # The loop calls this after every acknowledged update, before its
            # checkpoint callback refreshes session.progress.
            acknowledged_step[0] += 1
            return pause[0] or (pause_after_step is not None and acknowledged_step[0] >= pause_after_step) or (
                bounded and session.progress.pass_index >= stop_pass and session.progress.next_batch == 0)
        if "profiling" in record:
            from .family_execution_profile import FamilyExecutionProfiler, ProfileConfig
            profile_root = root / "profiles" / f"step_{session.progress.optimizer_step}_{time.time_ns()}"
            with FamilyExecutionProfiler(session, profile_root, ProfileConfig.from_dict(record["profiling"])) as observed:
                progress = session.run(should_pause=should_pause)
            atomic_json(root / "execution_profile_receipt.json", {"context": context.to_dict(),
                "report_path": (profile_root / "profile.json").relative_to(root).as_posix(),
                "report_sha256": normalized_file_sha(profile_root / "profile.json"),
                "trace_sha256": observed.report["trace_sha256"], "capture_status": observed.report["status"]})
        else:
            progress = session.run(should_pause=should_pause)
    finally:
        # run() closes itself, but admission/observer setup can fail before it.
        # The same idempotent owner also covers those in-process exits.
        try:
            session.close()
        finally:
            signal.signal(signal.SIGTERM, previous_signal)
    elapsed = time.perf_counter() - began
    total = _finish_cost_interval(root, context, prior_cost, elapsed)
    if progress.status != "complete":
        atomic_json(root / "pause_receipt.json", {"context": context.to_dict(), "status": "PAUSED",
            "optimizer_steps": progress.optimizer_step, "sample_presentations": progress.sample_presentations,
            "checkpoint_sha256": normalized_file_sha(root / "last_checkpoint.pt")})
        return {"status": "PAUSED", "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_EVALUATED"}
    hardware = provenance["accelerator"]
    costs = [{"metric": "wall_seconds", "unit": "seconds", "value": total,
        "status": "measured" if total is not None else "missing",
        "semantics": "process_wall", "hardware": hardware,
        "scope": "entry through acknowledged training endpoint; excludes terminal inspection/publication"},
        {"metric": "device_seconds", "unit": "seconds", "value": None, "status": "missing",
        "semantics": "allocated_device", "hardware": hardware, "reason": "requires platform allocation accounting"}]
    return outputs.complete(progress, runtime={"platform": context.platform, "account": context.account,
        "precision": "fp32", "source_commit": context.source_commit,
        "source_archive_sha256": context.source_archive_sha256,
        "runtime_certificate_id": canonical_fingerprint(certificate)}, costs=costs, target_identity=target_identity)
