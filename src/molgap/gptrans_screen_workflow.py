"""Spec-bound adapter for the owning, unchanged GPTrans fixed-100K V4 loop.

Legacy trainer files remain intact. The normalized bundle is a mechanical
translation, not scientific acceptance or a new resume loader.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import tarfile
import time

from . import pcqm_gptrans_v4 as owner
from .experiment_family_workflow import (
    EXPECTED, FamilyOutputSession, RunContext, inspect_output, write_output_manifest,
)
from .experiment_launch import _safe_local, publish_immutable_bytes
from .research_memory.trace import file_digest, json_bytes
from .v4_runtime import normalized_source_sha256, torch_load_compat


TRAIN_ROLE = "pcqm4mv2-ogb-fixed-100k-v1:training-0-100000"
DEV_ROLE = "pcqm4mv2-ogb-fixed-100k-v1:development-100000-150000"
VARIANTS = {"reference": None, "pair_prenorm": "gptrans_variants.py",
            "centered_logits": "gptrans_variants.py", "memory_value": "gptrans_memory.py",
            "memory_message": "gptrans_memory.py"}


def _screen_metric_semantics():
    return {"live_train_metric": {"metric": "MAE", "unit": "eV", "target": "Gap",
        "role_identity": TRAIN_ROLE, "weights": "live", "direction": "minimize",
        "timing": "online-pre-update; includes dropout"}, "live_dev_metric": None,
        "ema_dev_metric": {"metric": "MAE", "unit": "eV", "target": "Gap",
        "role_identity": DEV_ROLE, "weights": "ema", "direction": "minimize"}}


def build_screen_recipe(mode: str, *, source_idx_sha256: str, target_sha256: str) -> dict:
    """Freeze owning constants without copying a historical experiment JSON."""
    from .experiment_spec import _digest
    for name, value in (("source_idx_sha256", source_idx_sha256), ("target_sha256", target_sha256)):
        _digest(value, name)
    if mode not in VARIANTS:
        raise ValueError("Unsupported GPTrans screen variant")
    return {**owner._scientific_fields(), "mode": mode, "epochs": owner.EPOCHS,
        "train_rows": owner.TRAIN_ROWS, "development_rows": owner.DEVELOPMENT_ROWS,
        "physical_batch_per_device": owner.PHYSICAL_BATCH,
        "optimizer_steps_per_epoch": owner.BATCHES_PER_EPOCH,
        "seed42_initial_model_sha256": owner.EXPECTED_INITIAL_MODEL_SHA256,
        "seed42_initial_state_artifact_sha256": owner.EXPECTED_INITIAL_STATE_ARTIFACT_SHA256,
        "manifest_sha256": owner.MANIFEST_SHA256, "development_role_identity": DEV_ROLE,
        "metric_semantics": _screen_metric_semantics(),
        "acceptance_requirements": {"epochs": owner.EPOCHS,
            "optimizer_steps": owner.EPOCHS * owner.BATCHES_PER_EPOCH,
            "sample_presentations": owner.SAMPLE_PRESENTATIONS, "development_rows": owner.DEVELOPMENT_ROWS,
            "precision": "fp32", "source_idx_sha256": source_idx_sha256, "target_sha256": target_sha256}}


def _validate_recipe_fields(recipe):
    for key, value in owner._scientific_fields().items():
        if recipe.get(key) != value:
            raise ValueError("GPTrans frozen recipe changed: " + key)
    fixed = {"epochs": owner.EPOCHS, "train_rows": owner.TRAIN_ROWS,
        "development_rows": owner.DEVELOPMENT_ROWS, "physical_batch_per_device": owner.PHYSICAL_BATCH,
        "optimizer_steps_per_epoch": owner.BATCHES_PER_EPOCH,
        "seed42_initial_model_sha256": owner.EXPECTED_INITIAL_MODEL_SHA256,
        "seed42_initial_state_artifact_sha256": owner.EXPECTED_INITIAL_STATE_ARTIFACT_SHA256,
        "manifest_sha256": owner.MANIFEST_SHA256}
    for key, value in fixed.items():
        if recipe.get(key) != value:
            raise ValueError("GPTrans frozen recipe changed: " + key)
    requirements = recipe.get("acceptance_requirements")
    if not isinstance(requirements, dict) or set(requirements) != EXPECTED:
        raise ValueError("GPTrans requires frozen acceptance_requirements")
    for key, value in {"epochs": owner.EPOCHS,
                       "optimizer_steps": owner.BATCHES_PER_EPOCH * owner.EPOCHS,
                       "sample_presentations": owner.SAMPLE_PRESENTATIONS,
                       "development_rows": owner.DEVELOPMENT_ROWS, "precision": "fp32"}.items():
        if requirements[key] != value:
            raise ValueError("GPTrans acceptance exposure differs from executable recipe")
    if recipe.get("development_role_identity") != DEV_ROLE:
        raise ValueError("GPTrans development role differs from fixed V4 rows")


def validate_screen_recipe(spec, arm_id, recipe):
    """Static frozen recipe/arm check; no roles or model construction."""
    from .experiment_execution import training_adapter
    arm = next(a for a in spec.to_dict()["arms"] if a["arm_id"] == arm_id)
    mode = training_adapter(arm).mode(arm)
    _validate_recipe_fields(recipe)
    if recipe.get("mode") != mode:
        raise ValueError("GPTrans recipe mode differs from executable arm")
    if arm["initialization"] != {"kind": "frozen_state", "seed": owner.SEED, "state_sha256": owner.EXPECTED_INITIAL_MODEL_SHA256} or arm["training"]["overrides"]:
        raise ValueError("GPTrans executable initialization/overrides differ from Spec")
    if arm["training"]["sampler"]["sha256"] != recipe["row_order_fingerprint"]:
        raise ValueError("GPTrans Spec sampler differs from executable recipe")
    module = VARIANTS[mode]
    expected = [] if module is None else [{"name": mode, "version": "1", "config": {}, "source_sha256": normalized_source_sha256(Path(owner.__file__).with_name(module))}]
    if arm["addons"] != expected:
        raise ValueError("GPTrans executable addon differs from Spec")
    return {"family": arm["family"], "mode": mode, "recipe": "frozen_recipe_verified"}


def validate_screen_resume(spec, arm_id, manifest, bundle_root, artifacts,
                           sidecars, trajectory, context=None):
    """Validate a raw GPTrans prefix before the unchanged V4 loop resumes.

    GPTrans stores a zero-based historical trace and a NumPy RNG inside its
    checkpoint.  The shared transport loads that checkpoint with a restricted
    weights-only CPU unpickler; this hook binds the owner-specific epoch/trace
    contract and the retained runtime certificate without creating a new one.
    """
    from .experiment_execution import training_adapter
    from .experiment_family_workflow import _json
    from .experiment_resume import safe_cpu_torch_load
    from .training_reproducibility import validate_rng_state

    declaration = spec.to_dict()
    arm = next((item for item in declaration["arms"] if item["arm_id"] == arm_id), None)
    if arm is None:
        raise ValueError("GPTrans resume arm is absent from Spec")
    mode = training_adapter(arm).mode(arm)
    if mode not in VARIANTS:
        raise ValueError("GPTrans resume mode has no owning screen implementation")
    if context is not None and context.get("arm_id") != arm_id:
        raise ValueError("GPTrans resume context/arm mismatch")
    if not sidecars.get("context"):
        raise ValueError("GPTrans resume requires retained context evidence")
    if context is not None:
        for path in sidecars["context"]:
            payload = _json(path)
            observed = payload.get("context") if isinstance(payload.get("context"), dict) else payload
            if observed != context:
                raise ValueError("GPTrans resume context evidence changed")
    checkpoint = safe_cpu_torch_load(artifacts["checkpoint"])
    if checkpoint.get("format") != owner.CHECKPOINT_FORMAT:
        raise ValueError("GPTrans resume checkpoint format changed")
    if checkpoint.get("variant", "reference") != mode:
        raise ValueError("GPTrans resume checkpoint variant mismatch")
    required_checkpoint = {
        "format", "variant", "epoch", "model", "optimizer", "scheduler", "ema",
        "trace", "best_development_mae_eV", "best_epoch", "target_stats",
        "runtime_certificate_id", "source_archive_sha256", "rng_state",
        "scientific_fields", "ema_decay",
    }
    if not required_checkpoint <= set(checkpoint):
        raise ValueError("GPTrans resume checkpoint state is incomplete")
    if any(type(checkpoint[key]) is not dict or not checkpoint[key]
           for key in ("model", "optimizer", "scheduler", "ema", "target_stats")):
        raise ValueError("GPTrans resume checkpoint state mapping is incomplete")
    optimizer = checkpoint["optimizer"]
    scheduler = checkpoint["scheduler"]
    if (set(optimizer) != {"state", "param_groups"} or type(optimizer["state"]) is not dict
            or type(optimizer["param_groups"]) is not list or not optimizer["param_groups"]):
        raise ValueError("GPTrans resume optimizer state schema changed")
    if type(scheduler.get("epoch")) is not int:
        raise ValueError("GPTrans resume scheduler state schema changed")
    if type(checkpoint["rng_state"]) is not dict or not checkpoint["rng_state"]:
        raise ValueError("GPTrans resume checkpoint RNG state is incomplete")
    rng = checkpoint["rng_state"]
    try:
        validate_rng_state(rng, cuda_devices=1, label="GPTrans resume checkpoint")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GPTrans resume checkpoint RNG state schema changed: {exc}") from exc
    if checkpoint["scientific_fields"] != owner._scientific_fields(mode):
        raise ValueError("GPTrans resume checkpoint scientific contract changed")
    if checkpoint["ema_decay"] != owner._ema_decay(mode):
        raise ValueError("GPTrans resume checkpoint EMA decay changed")
    if (type(checkpoint["best_epoch"]) is not int or checkpoint["best_epoch"] < -1
            or not isinstance(checkpoint["best_development_mae_eV"], (int, float))
            or isinstance(checkpoint["best_development_mae_eV"], bool)
            or not math.isfinite(checkpoint["best_development_mae_eV"])):
        raise ValueError("GPTrans resume checkpoint selection state is invalid")
    certificate_id = checkpoint["runtime_certificate_id"]
    if (type(certificate_id) is not str or len(certificate_id) != 64
            or any(char not in "0123456789abcdef" for char in certificate_id)):
        raise ValueError("GPTrans resume checkpoint certificate identity is invalid")
    if context is not None and checkpoint.get("source_archive_sha256") != context["source_archive_sha256"]:
        raise ValueError("GPTrans resume checkpoint/source mismatch")
    trace = _json(artifacts["trace"])
    if trace.get("format") != owner.RUN_FORMAT or type(trace.get("rows")) is not list or not trace["rows"]:
        raise ValueError("GPTrans resume trace format changed")
    rows = trace["rows"]
    if checkpoint.get("trace") != rows or checkpoint.get("epoch") != len(rows) - 1:
        raise ValueError("GPTrans resume checkpoint/trace identity mismatch")
    if len(rows) >= owner.EPOCHS:
        raise ValueError("GPTrans resume bundle is already complete")
    for index, row in enumerate(rows):
        if type(row) is not dict or row.get("epoch") != index:
            raise ValueError("GPTrans resume trace is not a contiguous epoch prefix")
        if not {"train_mae_eV", "development_mae_eV", "best_development_mae_eV",
                "best_epoch", "learning_rate", "elapsed_seconds"} <= set(row):
            raise ValueError("GPTrans resume trace row is incomplete")
        if (row.get("optimizer_steps") != owner.BATCHES_PER_EPOCH
                or row.get("sample_presentations") != owner.BATCHES_PER_EPOCH * owner.PHYSICAL_BATCH):
            raise ValueError("GPTrans resume trace exposure changed")
    if checkpoint["best_epoch"] >= len(rows):
        raise ValueError("GPTrans resume checkpoint selection is outside the retained prefix")
    binding = checkpoint.get("family_output_binding")
    if binding is not None:
        if (type(binding) is not dict or binding.get("adapter") != "gptrans-v1"
                or binding.get("trainer") != "pcqm_gptrans_v4"
                or binding.get("variant") != mode):
            raise ValueError("GPTrans resume family output binding changed")
        if context is not None and binding.get("context") != context:
            raise ValueError("GPTrans resume family output context changed")
    if artifacts.get("selected_model") is None:
        if artifacts.get("predictions") is not None:
            raise ValueError("GPTrans predictions have no selected model")
        if checkpoint["best_epoch"] >= 0:
            raise ValueError("GPTrans resume checkpoint has a best epoch but no selected model")
    else:
        selected = safe_cpu_torch_load(artifacts["selected_model"])
        predictions = safe_cpu_torch_load(artifacts["predictions"])
        if (selected.get("format") != owner.RUN_FORMAT
                or type(selected.get("model")) is not dict or not selected["model"]
                or type(selected.get("model_config")) is not dict
                or type(selected.get("target_stats")) is not dict):
            raise ValueError("GPTrans selected model format changed")
        if (selected.get("variant", selected["model_config"].get("variant", mode)) != mode
                or selected["model_config"].get("variant", mode) != mode
                or type(selected.get("epoch")) is not int
                or not 0 <= selected["epoch"] <= rows[-1]["epoch"]):
            raise ValueError("GPTrans selected model is outside the retained prefix")
        if context is not None and selected.get("source_archive_sha256") != context["source_archive_sha256"]:
            raise ValueError("GPTrans selected model/source mismatch")
        if selected.get("source_commit") != (None if context is None else context["source_commit"]):
            raise ValueError("GPTrans selected model/source commit mismatch")
        if selected.get("runtime_certificate_id") != certificate_id:
            raise ValueError("GPTrans selected model/checkpoint certificate mismatch")
        if type(predictions) is not dict:
            raise ValueError("GPTrans development predictions are incomplete")
        if {"prediction_eV", "target_eV", "source_idx"} - set(predictions):
            raise ValueError("GPTrans development predictions are incomplete")
        if any(predictions.get(key) is not False
                for key in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")):
            raise ValueError("GPTrans selected predictions consume a protected role")
    certificate = None
    preflight = None
    provenance = None
    for path in sidecars.get("runtime", ()):
        payload = _json(path)
        if path.name == "runtime_certificate.json" or payload.get("format") == "molgap-runtime-certificate-v1":
            certificate = payload
        elif path.name == "preflight.json":
            preflight = payload
    for path in sidecars.get("provenance", ()):
        payload = _json(path)
        if payload.get("format") == "molgap-gptrans-runtime-provenance-v1":
            provenance = payload
    if certificate is not None and certificate.get("status") not in {"accepted", "ACCEPTED"}:
        raise ValueError("GPTrans retained runtime certificate is not accepted")
    if certificate is None:
        raise ValueError("GPTrans resume requires the retained runtime certificate")
    if owner.canonical_fingerprint(certificate) != certificate_id:
        raise ValueError("GPTrans checkpoint/runtime certificate identity mismatch")
    if preflight is None:
        raise ValueError("GPTrans resume requires the retained preflight record")
    if (preflight.get("accepted") is not True
            or preflight.get("runtime_certificate_id") != certificate_id
            or preflight.get("runtime_certificate") != certificate
            or preflight.get("variant", "reference") != mode
            or preflight.get("source_archive_sha256") != (None if context is None else context["source_archive_sha256"])
            or preflight.get("source_commit") != (None if context is None else context["source_commit"])):
        raise ValueError("GPTrans retained preflight/certificate identity mismatch")
    runtime_manifest = None
    for path in sidecars.get("runtime", ()):
        if path.name == "runtime_manifest.json":
            runtime_manifest = _json(path)
            if (runtime_manifest.get("runtime_fingerprint") != certificate.get("runtime_fingerprint")
                    or preflight.get("runtime_manifest") != runtime_manifest):
                raise ValueError("GPTrans retained runtime manifest/certificate mismatch")
    if runtime_manifest is None:
        raise ValueError("GPTrans resume requires the retained runtime manifest")
    if preflight is not None:
        if context is not None and preflight.get("source_archive_sha256") != context["source_archive_sha256"]:
            raise ValueError("GPTrans preflight/source mismatch")
        if context is not None and preflight.get("source_commit") != context["source_commit"]:
            raise ValueError("GPTrans preflight/source commit mismatch")
        if preflight.get("variant", "reference") != mode:
            raise ValueError("GPTrans preflight variant mismatch")
    return {"family": "gptrans_t", "version": "1", "mode": mode,
            "completed_epochs": len(rows),
            "cursor": {"epoch": len(rows), "next_batch": 0,
                       "optimizer_step": len(rows) * owner.BATCHES_PER_EPOCH,
                       "sample_presentations": len(rows) * owner.BATCHES_PER_EPOCH * owner.PHYSICAL_BATCH},
            "preflight": {"reuse_existing_certificate": certificate is not None,
                          "certificate_id": certificate_id,
                          "provenance_retained": provenance is not None,
                          "requires_new_diagnostic": False}}


def _read(path: Path) -> dict:
    from .experiment_family_workflow import _json
    _safe_local(Path(path).absolute())
    return _json(Path(path))


def run_screen_resume_preflight(*, spec, package_dir: Path, expected_package_identity: str,
                                arm_id: str, mode: str, recipe_path: Path,
                                initial_state_path: Path, input_root: Path, output: Path,
                                account: str, run_reference: str, trajectory_id: str,
                                resume_output: Path) -> dict:
    """Reuse an accepted GPTrans preflight for a restored native prefix.

    A resumed checkpoint carries the original certificate identity and the
    frozen V4 loop compares that identity before loading model state.  Running
    ``owner.run_preflight`` here would create a new certificate and make every
    valid resume fail.  This hook therefore performs only source/state/
    hardware/runtime compatibility checks and republishes the retained
    preflight bytes into the isolated directory consumed by ``run_screen_arm``.
    No model, dataset, or new diagnostic is executed.
    """
    import torch
    from .experiment_resume import safe_cpu_torch_load
    from .experiment_family_workflow import RunContext
    from .training_reproducibility import configure_fp32_determinism, build_runtime_manifest

    context, recipe = _validate_request(spec=spec, package_dir=package_dir,
        expected_package_identity=expected_package_identity, arm_id=arm_id, mode=mode,
        recipe_path=recipe_path, initial_state_path=initial_state_path, input_root=input_root,
        account=account, run_reference=run_reference, trajectory_id=trajectory_id)
    retained = Path(resume_output).absolute()
    destination = Path(output).absolute()
    _safe_local(retained)
    _safe_local(destination)
    if retained == destination or destination.is_relative_to(retained):
        raise ValueError("Resume preflight output must be separate from retained arm output")
    if not retained.is_dir():
        raise ValueError("Resume preflight requires the restored native arm output")
    checkpoint_path = retained / "last_checkpoint.pt"
    trace_path = retained / "trace.json"
    certificate_path = retained / "runtime_certificate.json"
    preflight_path = retained / "preflight.json"
    runtime_path = retained / "runtime_manifest.json"
    context_path = retained / "workflow_context.json"
    required = (checkpoint_path, trace_path, certificate_path, preflight_path, runtime_path,
                context_path)
    if any(not path.is_file() for path in required):
        raise ValueError("GPTrans resume preflight lacks retained runtime evidence")
    checkpoint = safe_cpu_torch_load(checkpoint_path)
    trace = _read(trace_path)
    certificate = _read(certificate_path)
    preflight = _read(preflight_path)
    runtime = _read(runtime_path)
    artifacts = {"checkpoint": checkpoint_path, "trace": trace_path,
                 "selected_model": (retained / "best_model.pt") if (retained / "best_model.pt").is_file() else None,
                 "predictions": (retained / "development_predictions.pt") if (retained / "development_predictions.pt").is_file() else None}
    sidecars = {"runtime": [runtime_path, certificate_path, preflight_path],
                "provenance": [], "context": [context_path], "cost_segments": []}
    # Reuse the same owner state gate used for exported bundles.  The runtime
    # worker has already restored exact bytes, so no transport manifest is
    # needed here; the trajectory ID is the original staged prospective ID.
    validate_screen_resume(spec, arm_id,
        {"trajectory": {"id": trajectory_id}}, retained, artifacts, sidecars,
        {"trajectory_id": trajectory_id}, context=context.to_dict())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPTrans resume preflight requires exactly one visible accelerator")
    hardware = torch.cuda.get_device_name(0)
    if "T4" not in hardware or certificate.get("accelerator") != hardware:
        raise RuntimeError("GPTrans resume hardware differs from the retained certificate")
    owner.validate_source_archive(Path(package_dir) / "source.tar.gz",
        context.source_archive_sha256, context.source_commit)
    determinism = owner.configure_fp32_determinism(owner.SEED)
    current_runtime = build_runtime_manifest(determinism)
    if (runtime.get("runtime_fingerprint") != certificate.get("runtime_fingerprint")
            or current_runtime.get("runtime_fingerprint") != certificate.get("runtime_fingerprint")):
        raise ValueError("GPTrans resume runtime fingerprint differs from the retained certificate")
    certificate_id = owner.canonical_fingerprint(certificate)
    if (preflight.get("accepted") is not True
            or preflight.get("runtime_certificate_id") != certificate_id
            or preflight.get("runtime_certificate") != certificate
            or preflight.get("runtime_manifest") != runtime
            or preflight.get("source_archive_sha256") != context.source_archive_sha256
            or preflight.get("source_commit") != context.source_commit
            or preflight.get("variant", "reference") != mode):
        raise ValueError("GPTrans retained preflight evidence is inconsistent")
    owner.validate_runtime_certificate(certificate, {
        **owner._scientific_fields(mode), "platform_id": certificate.get("platform_id"),
        "accelerator": certificate.get("accelerator"),
        "runtime_certificate_id": certificate_id})
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("runtime_manifest.json", "runtime_certificate.json", "preflight.json"):
        publish_immutable_bytes(destination / name, (retained / name).read_bytes())
    publish_immutable_bytes(destination / "workflow_context.json", json_bytes(context.to_dict()))
    return {**preflight, "resume_reused": True}


def _validate_request(*, spec, package_dir, expected_package_identity, arm_id, mode,
                      recipe_path, initial_state_path, input_root, account,
                      run_reference, trajectory_id):
    if mode not in VARIANTS:
        raise ValueError("Unsupported GPTrans screen variant")
    context = RunContext.for_training(spec, package_dir,
        expected_package_identity=expected_package_identity, arm_id=arm_id,
        account=account, run_reference=run_reference)
    declaration = spec.to_dict()
    arm = next(a for a in declaration["arms"] if a["arm_id"] == arm_id)
    if (context.family_name, context.family_version) != ("gptrans_t", "1"):
        raise ValueError("Expected GPTrans-T/1 screen family")
    if (arm["initialization"] != {"kind": "frozen_state", "seed": owner.SEED,
                                 "state_sha256": owner.EXPECTED_INITIAL_MODEL_SHA256}
            or arm["training"]["overrides"]):
        raise ValueError("GPTrans executable initialization/overrides differ from Spec")
    module = VARIANTS[mode]
    expected_addons = [] if module is None else [{"name": mode, "version": "1",
        "source_sha256": normalized_source_sha256(Path(owner.__file__).with_name(module)),
        "config": {}}]
    if arm["addons"] != expected_addons:
        raise ValueError("GPTrans executable variant/addons differ from Spec")
    recipe = _read(recipe_path)
    if file_digest(recipe_path) != context.training_recipe_sha256:
        raise ValueError("GPTrans recipe bytes differ from Spec")
    if recipe.get("mode") != mode:
        raise ValueError("GPTrans recipe mode differs from executable arm")
    # A caller-supplied recipe must also be present as those exact package bytes.
    with tarfile.open(Path(package_dir) / "source.tar.gz", "r:gz") as archive:
        if not any(member.isfile() and archive.extractfile(member).read() == Path(recipe_path).read_bytes()
                   for member in archive if member.name.endswith(".json")):
            raise ValueError("GPTrans recipe bytes absent from source package")
    _validate_recipe_fields(recipe)
    if arm["training"]["sampler"]["sha256"] != recipe["row_order_fingerprint"]:
        raise ValueError("GPTrans Spec sampler differs from executable recipe")
    if file_digest(initial_state_path) != owner.EXPECTED_INITIAL_STATE_ARTIFACT_SHA256:
        raise ValueError("GPTrans initial-state artifact changed")
    import torch
    state = torch.load(initial_state_path, map_location="cpu", weights_only=True)
    from .v4_runtime import state_dict_sha256
    if (state.get("format") != "molgap-gptrans-t-seed42-initial-state-v1" or
            state_dict_sha256(state.get("model_state", {})) != owner.EXPECTED_INITIAL_MODEL_SHA256
            or state.get("state_sha256") != owner.EXPECTED_INITIAL_MODEL_SHA256):
        raise ValueError("GPTrans initial tensor identity changed")
    bindings = declaration["prospective"].get("arms", [])
    binding = next((b for b in bindings if b["arm_id"] == arm_id), None)
    if binding is None or binding["trajectory_id"] != trajectory_id:
        raise ValueError("GPTrans requires matching Spec v2 prospective arm")
    from .experiment_execution import validate_staged_trajectory
    trajectory = validate_staged_trajectory(spec, arm_id, Path(input_root))
    from .research_memory.schemas import validate_trajectory
    validate_trajectory(trajectory)
    if (trajectory["trajectory_id"] != trajectory_id or trajectory["record_mode"] != "prospective"
            or trajectory["decision"]["outcome"] != "ACTIVE"
            or trajectory["state_at_start"]["source_config_identity"] != context.arm_identity
            or trajectory["state_at_start"]["source_commit"] != context.source_commit):
        raise ValueError("GPTrans prospective trajectory is not active or bound to this arm")
    return context, recipe


def _owner_kwargs(context, *, package_dir, mode, recipe, initial_state_path, input_root, output):
    root = Path(input_root)
    # Mounted source and graph datasets may be siblings. Inspect metadata only,
    # and let the owner verify all graph bytes after an unambiguous match.
    _safe_local(root.absolute())
    candidates = []
    for candidate in root.rglob("*manifest*.json"):
        _safe_local(candidate.absolute())
        if candidate.is_file() and file_digest(candidate) == owner.MANIFEST_SHA256:
            candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError("Expected exactly one pinned GPTrans fixed manifest")
    manifest = candidates[0]
    _safe_local(Path(output).absolute())
    Path(output).mkdir(parents=True, exist_ok=True)
    publish_immutable_bytes(Path(output).absolute() / "workflow_context.json", json_bytes(context.to_dict()))
    return dict(dataset_root=manifest.parent, manifest_path=manifest,
        source_archive=Path(package_dir) / "source.tar.gz",
        source_archive_sha256=context.source_archive_sha256, source_commit=context.source_commit,
        output=Path(output), platform_id=recipe.get("runtime_platform_id", context.platform),
        initial_state_path=Path(initial_state_path), variant=mode,
        runtime_calibration_fingerprint=recipe.get("runtime_calibration_fingerprint"))


def run_screen_preflight(*, spec, package_dir: Path, expected_package_identity: str,
                         arm_id: str, mode: str, recipe_path: Path, initial_state_path: Path,
                         input_root: Path, output: Path, account: str, run_reference: str,
                         trajectory_id: str) -> dict:
    context, recipe = _validate_request(spec=spec, package_dir=package_dir,
        expected_package_identity=expected_package_identity, arm_id=arm_id, mode=mode,
        recipe_path=recipe_path, initial_state_path=initial_state_path, input_root=input_root,
        account=account, run_reference=run_reference, trajectory_id=trajectory_id)
    return owner.run_preflight(**_owner_kwargs(context, package_dir=package_dir, mode=mode,
        recipe=recipe, initial_state_path=initial_state_path, input_root=input_root, output=output))


def _normalize(output: Path, context: RunContext, recipe_path: Path, trajectory_id: str,
               completion: dict, *, process_wall_seconds: float, certificate: dict,
               observed_costs: list | None = None) -> dict:
    import torch
    raw = Path(output)
    recipe = _read(recipe_path)
    if (completion.get("complete") is not True or completion.get("source_archive_sha256") != context.source_archive_sha256
            or completion.get("source_commit") != context.source_commit):
        raise ValueError("GPTrans completion/source identity mismatch")
    for field, name in (("checkpoint_sha256", "last_checkpoint.pt"),
                        ("best_model_sha256", "best_model.pt"),
                        ("development_predictions_sha256", "development_predictions.pt")):
        if completion.get(field) != file_digest(raw / name):
            raise ValueError("GPTrans retained output hash mismatch: " + field)
    selected = torch.load(raw / "best_model.pt", map_location="cpu", weights_only=True)
    predictions = torch.load(raw / "development_predictions.pt", map_location="cpu", weights_only=True)
    # The owning checkpoint is trusted executable output; its historical NumPy
    # RNG requires the owner loader before the shared tensor-safe translation.
    resume = torch_load_compat(raw / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    rows = _read(raw / "trace.json")["rows"]
    if resume.get("trace") != rows or resume.get("epoch") != len(rows) - 1:
        raise ValueError("GPTrans durable trace/checkpoint disagreement")
    if (resume.get("format") != owner.CHECKPOINT_FORMAT
            or resume.get("variant", "reference") != completion.get("variant", "reference")
            or resume.get("source_archive_sha256") != context.source_archive_sha256):
        raise ValueError("GPTrans resume/source/variant identity mismatch")
    for payload in (completion, predictions):
        if any(payload.get(key) is not False for key in
               ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")):
            raise ValueError("GPTrans protected roles are not declared unused")
    semantics = {"live_train_metric": {"metric": "MAE", "unit": "eV", "target": "Gap",
        "role_identity": TRAIN_ROLE, "weights": "live", "direction": "minimize",
        "timing": "online-pre-update; includes dropout"}, "live_dev_metric": None,
        "ema_dev_metric": {"metric": "MAE", "unit": "eV", "target": "Gap",
        "role_identity": recipe["development_role_identity"], "weights": "ema", "direction": "minimize"}}
    session = FamilyOutputSession(raw / "normalized", context, adapter="gptrans-v1",
        contract=recipe_path, trajectory_id=trajectory_id, metric_semantics=semantics)
    # An existing complete bundle must pass inspection; do not duplicate events.
    if (session.root / "output_manifest.json").exists():
        return inspect_output(session.root, context=context, expected=session.expected)
    if session.stage.recorder.record["observations"]:
        raise ValueError("Partial normalization exists; reconcile before retry")
    steps, samples = 0, 0
    for index, row in enumerate(rows):
        if (row["epoch"] != index or row["optimizer_steps"] != owner.BATCHES_PER_EPOCH
                or row["sample_presentations"] != owner.BATCHES_PER_EPOCH * owner.PHYSICAL_BATCH):
            raise ValueError("GPTrans trace exposure differs from owning V4 loop")
        steps += row["optimizer_steps"]
        samples += row["sample_presentations"]
        session.epoch_finished(epoch=index + 1, optimizer_step=steps, sample_presentations=samples,
            live_train_metric=row["train_mae_eV"], ema_dev_metric=row["development_mae_eV"],
            learning_rate=row["learning_rate"], wall_time_seconds=row["elapsed_seconds"])
    epoch = selected["epoch"] + 1
    if (selected["epoch"] != completion["best_epoch"] or
            selected["source_archive_sha256"] != context.source_archive_sha256):
        raise ValueError("GPTrans selected identity differs from completion")
    session.selected(model_state=selected["model"], epoch=epoch,
        optimizer_step=epoch * owner.BATCHES_PER_EPOCH, weights="ema",
        prediction_eV=predictions["prediction_eV"], target_eV=predictions["target_eV"],
        source_idx=predictions["source_idx"])
    session.checkpoint(model_state=resume["model"], optimizer_state=resume["optimizer"],
        scheduler_state=resume["scheduler"], ema_state=resume["ema"], rng_state=resume["rng_state"],
        cursor={"epoch": len(rows), "next_batch": 0,
                "sampler_order_sha256": recipe["row_order_fingerprint"]},
        optimizer_step=steps, sample_presentations=samples)
    hardware = certificate["accelerator"]
    from .screen_policy import canonical_fingerprint
    publish_immutable_bytes(session.root / "runtime_certificate.json", json_bytes(certificate))
    costs = observed_costs if observed_costs is not None else [{"metric": "wall_seconds", "unit": "seconds", "value": process_wall_seconds,
              "status": "measured", "semantics": "process_wall",
              "scope": "workflow_invocation_excludes_previous_resume_segments_bootstrap_queue",
              "hardware": hardware}, {"metric": "device_seconds", "unit": "seconds", "value": None,
              "status": "missing", "semantics": "allocated_device", "hardware": hardware,
              "reason": "Owning V4 output retained no allocated-device cost ledger"}]
    write_output_manifest(session.root, context, adapter="gptrans-v1",
        artifacts={"predictions": "development_predictions.pt", "selected_model": "selected_model.pt",
                   "resume": "last_checkpoint.pt", "trace": "canonical_trace.json", "contract": "training_contract.json"},
        progress={"epochs": len(rows), "optimizer_steps": steps, "sample_presentations": samples},
        runtime={"platform": context.platform, "account": context.account, "precision": "fp32",
                 "source_commit": context.source_commit, "source_archive_sha256": context.source_archive_sha256,
                 "runtime_certificate_id": canonical_fingerprint(certificate)}, costs=costs)
    return inspect_output(session.root, context=context, expected=session.expected)


def run_screen_arm(*, spec, package_dir: Path, expected_package_identity: str,
                   arm_id: str, mode: str, recipe_path: Path, initial_state_path: Path,
                   input_root: Path, output: Path, account: str, run_reference: str,
                   trajectory_id: str, preflight_dir: Path | None = None) -> dict:
    context, recipe = _validate_request(spec=spec, package_dir=package_dir,
        expected_package_identity=expected_package_identity, arm_id=arm_id, mode=mode,
        recipe_path=recipe_path, initial_state_path=initial_state_path, input_root=input_root,
        account=account, run_reference=run_reference, trajectory_id=trajectory_id)
    preflight = Path(preflight_dir or output)
    if _read(preflight / "workflow_context.json") != context.to_dict():
        raise ValueError("GPTrans preflight context differs from training arm")
    started = time.perf_counter()
    resumed = (Path(output) / "last_checkpoint.pt").exists()
    completion = owner.run_training(preflight_path=preflight / "preflight.json",
        **_owner_kwargs(context, package_dir=package_dir, mode=mode, recipe=recipe,
            initial_state_path=initial_state_path, input_root=input_root, output=output))
    seconds = time.perf_counter() - started
    certificate = _read(preflight / "runtime_certificate.json")
    hardware = certificate["accelerator"]
    # run_training enforces exactly one visible accelerator and its certificate.
    # This measures allocation during that call, not CUDA busy time or queue.
    costs = [{"metric": "wall_seconds", "unit": "seconds", "value": seconds,
              "status": "measured", "semantics": "process_wall", "hardware": hardware,
              "scope": "current_training_invocation_excludes_bootstrap_queue_previous_resume_segments"},
             {"metric": "device_seconds", "unit": "seconds", "value": None if resumed else seconds,
              "status": "missing" if resumed else "measured", "semantics": "allocated_device",
              "hardware": hardware, "scope": "fresh_complete_training_invocation_only",
              "reason": "Previous resume segments lack a retained allocated-device ledger" if resumed else None}]
    return _normalize(output, context, recipe_path, trajectory_id, completion,
                      process_wall_seconds=seconds, certificate=certificate, observed_costs=costs)
