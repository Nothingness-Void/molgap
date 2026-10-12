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
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_json, atomic_torch_save
from .v4_runtime import normalized_source_sha256, state_dict_sha256


INTERFACES = {"neural_atom_k1": ("neural_atom_k1", "3"), "gptrans_t": ("gptrans_t", "2")}
PROFILES = {"neural_atom_k1": "k1-session-v1", "gptrans_t": "gptrans-session-v1"}
METADATA = {"acceptance_requirements", "train_role_identity", "development_role_identity"}


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
            "mode": recipe.mode, "family": recipe.family, "gradient_clip": recipe.gradient_clip})}}


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
                        target_sha256=None, train_role_identity=None, development_role_identity=None):
    if type(configuration) is not FamilyRecipe or family != INTERFACES.get(configuration.family) or mode != configuration.mode:
        raise ValueError("Explicit matching shared family configuration is required")
    _text(train_role_identity, "train_role_identity")
    if configuration.dataset.development:
        _text(development_role_identity, "development_role_identity")
    elif development_role_identity is not None:
        raise ValueError("Final-only training has no development role")
    return {**configuration.to_dict(), "acceptance_requirements": _requirements(configuration, source_idx_sha256, target_sha256),
        "train_role_identity": train_role_identity, "development_role_identity": development_role_identity}


def read_recipe(path):
    record = _json(Path(path))
    if not METADATA <= set(record):
        raise ValueError("Shared contract lacks frozen acceptance/role bindings")
    recipe = FamilyRecipe.from_dict({key: value for key, value in record.items() if key not in METADATA})
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
    return record, recipe


def validate_screen_recipe(spec, arm_id, record):
    from .experiment_execution import training_adapter
    arm = next(arm for arm in spec.to_dict()["arms"] if arm["arm_id"] == arm_id)
    recipe = FamilyRecipe.from_dict({key: value for key, value in record.items() if key not in METADATA})
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
    modules = {"ssma": "k1_joint_aggregation.py", "mean2": "k1_screen_training.py",
        "mean2_clean_second": "k1_screen_training.py", "pair_prenorm": "gptrans_variants.py",
        "centered_logits": "gptrans_variants.py", "memory_value": "gptrans_memory.py", "memory_message": "gptrans_memory.py"}
    if recipe.mode != "reference" and arm["addons"][0]["source_sha256"] != normalized_source_sha256(Path(__file__).with_name(modules[recipe.mode])):
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


def _bootstrap(*, spec, package_dir, expected_package_identity, arm_id, mode, recipe_path,
               initial_state_path, input_root, account, run_reference, trajectory_id, prospective_root=None,
               training_only=False):
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
    initial = inspect_frozen_state_artifact(Path(initial_state_path), expected_state_sha256=recipe.initialization_sha256)
    # Data acceptance remains CPU work and precedes construction of a GPU model.
    selection = replace(recipe.dataset, development=False) if training_only else recipe.dataset
    data = load_training_data(Path(input_root), selection)
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
    """Three scratch executions: repeat and actual serialized mid-step resume."""
    import torch
    from .family_training_kernel import FamilyKernel
    from .family_training_loaders import FamilyLoaders
    from .family_training_session import FamilyTrainingSession
    from .training_reproducibility import configure_fp32_determinism
    began = time.perf_counter()
    output = Path(options.pop("output")).absolute()
    record, recipe, data, tuning, context, provenance, _ = _bootstrap(training_only=True, **options)
    # Preflight consumes training topology only, never development selection.
    probe = replace(recipe, dataset=replace(recipe.dataset, development=False), development_rows=0,
        optimizer_steps=2, checkpoint_every=1, selection="final-live")
    data = replace(data, development=None, development_rows=0, development_source_start=None)
    output.mkdir(parents=True, exist_ok=True)
    proofs = []
    for label in ("repeat-a", "repeat-b", "resumed"):
        configure_fp32_determinism(recipe.seed)
        kernel = FamilyKernel(probe, Path(options["initial_state_path"]), device="cuda")
        _loaded_source(options["package_dir"])
        def checkpoint(progress, state):
            atomic_torch_save(output / (label + ".pt"), state)
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
        session.run()
        state = torch.load(output / (label + ".pt"), map_location="cpu", weights_only=True)
        proofs.append({"path": label + ".pt", "sha256": normalized_file_sha(output / (label + ".pt")),
            "state_fingerprint": _state_fingerprint(state)})
    if len({proof["state_fingerprint"] for proof in proofs}) != 1:
        raise ValueError("Native optimizer/EMA/schedule/RNG repeat or serialized resume differs")
    atomic_json(output / "runtime_provenance.json", provenance)
    certificate = {"format": "molgap-runtime-certificate-v1", "status": "accepted",
        "provenance_sha256": canonical_fingerprint(provenance), "probe_recipe_identity": probe.identity,
        "proofs": proofs, "resume_origin": {"path": "resumed-first.pt",
            "sha256": normalized_file_sha(output / "resumed-first.pt")},
        "formal_training_exposure": 0, "scientific_acceptance": "NOT_EVALUATED"}
    atomic_json(output / "runtime_certificate.json", certificate)
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


def _verify_preflight(directory, recipe, provenance):
    import torch
    from .family_training_session import validate_session_resume
    root = Path(directory).absolute()
    saved = _json(_artifact_path(root, "runtime_provenance.json"))
    certificate = _json(_artifact_path(root, "runtime_certificate.json"))
    if saved != provenance or certificate.get("provenance_sha256") != canonical_fingerprint(provenance):
        raise ValueError("Native preflight source/recipe/data/runtime/device identity differs")
    probe = replace(recipe, dataset=replace(recipe.dataset, development=False), development_rows=0,
        optimizer_steps=2, checkpoint_every=1, selection="final-live")
    if (certificate.get("format") != "molgap-runtime-certificate-v1" or
            certificate.get("probe_recipe_identity") != probe.identity or certificate.get("formal_training_exposure") != 0):
        raise ValueError("Preflight probe identity differs")
    proofs = certificate.get("proofs")
    if type(proofs) is not list or [proof.get("path") for proof in proofs] != ["repeat-a.pt", "repeat-b.pt", "resumed.pt"]:
        raise ValueError("Missing native repeat/resume artifacts")
    origin = certificate.get("resume_origin")
    if type(origin) is not dict or set(origin) != {"path", "sha256"} or origin["path"] != "resumed-first.pt":
        raise ValueError("Missing serialized native resume origin")
    origin_path = _artifact_path(root, origin["path"])
    if normalized_file_sha(origin_path) != origin["sha256"]:
        raise ValueError("Native resume-origin hash mismatch")
    origin_state = torch.load(origin_path, map_location="cpu", weights_only=True)
    origin_progress = validate_session_resume(probe, origin_state)
    if origin_progress.optimizer_step != 1 or origin_progress.status != "paused" or len(origin_state["rng_state"]["cuda"]) != 1:
        raise ValueError("Native resume origin is not an observed isolated-GPU scratch step")
    fingerprints = []
    for proof in proofs:
        path = _artifact_path(root, proof["path"])
        if normalized_file_sha(path) != proof["sha256"]:
            raise ValueError("Native proof artifact hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
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
    return certificate


def run_screen_arm(*, output, preflight_dir=None, target_identity=None, **options):
    """Same callable for every scale/platform; same-budget resume only."""
    import torch
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
    record, recipe, data, tuning, context, provenance, _ = _bootstrap(**options)
    certificate = _verify_preflight(preflight_dir or output, recipe, provenance)
    outputs = SessionOutputs(root, context, contract=options["recipe_path"], adapter=PROFILES[recipe.family],
        trajectory_id=options["trajectory_id"], train_role_identity=record["train_role_identity"])
    kernel = FamilyKernel(recipe, Path(options["initial_state_path"]), device="cuda")
    _loaded_source(options["package_dir"])
    session = FamilyTrainingSession(kernel, FamilyLoaders(recipe, data, tuning),
        checkpoint=outputs.checkpoint, selected=outputs.selected, pass_finished=outputs.pass_finished)
    had_checkpoint = (root / "last_checkpoint.pt").is_file()
    try:
        if had_checkpoint:
            session.load_state_dict(outputs.retained_state())
    except Exception:
        session.close()
        raise
    pause = [False]
    previous_signal = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: pause.__setitem__(0, True))
    try:
        progress = session.run(should_pause=lambda: pause[0])
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
    cost_path = root / "allocation_costs.json"
    previous = _json(cost_path) if cost_path.is_file() else {"context": context.to_dict(),
        "wall_seconds": None if had_checkpoint else 0.}
    _check_context(previous["context"], context)
    if previous["wall_seconds"] is not None and (type(previous["wall_seconds"]) not in (int, float)
            or not math.isfinite(previous["wall_seconds"]) or previous["wall_seconds"] < 0):
        raise ValueError("Invalid retained native cost ledger")
    elapsed = time.perf_counter() - began
    total = previous["wall_seconds"] + elapsed if previous["wall_seconds"] is not None else None
    atomic_json(cost_path, {"context": context.to_dict(), "wall_seconds": total, "current_invocation_seconds": elapsed})
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
