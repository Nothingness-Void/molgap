"""Shared normalized-Gap trainer for genuinely new pure-2D graph families.

The owner fixes the scientific screen profile and owns the durable lifecycle.
An extension supplies a reviewed ``make_model(arm)`` function and optional
ordered ``apply_addon(model, config)`` hooks.  This module intentionally does
not attempt to represent arbitrary modalities or objectives; those remain
separate owners until a new bounded contract is reviewed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .experiment_family_workflow import (
    EXPECTED, FamilyOutputSession, RunContext, _finite_tree, _json, inspect_output,
    tensor_digest, tensor_safe_rng_state,
)
from .experiment_launch import publish_immutable_bytes
from .pcqm_graph_inputs import (
    FEATURE_SCHEMA, PHYSICAL_BATCH, ROLES, SAMPLER, SEED,
    load_graph_inputs, sampler_order_sha256,
)
from .research_memory.trace import file_digest, json_bytes
from .screen_policy import canonical_fingerprint
from .training_reproducibility import (
    atomic_json, atomic_torch_save, build_runtime_manifest,
    capture_rng_state, configure_fp32_determinism, restore_rng_state,
    sha256_file,
)
from .v4_runtime import inspect_frozen_state_artifact, torch_load_compat


RECIPE_FORMAT = "molgap-graph-gap-screen-recipe-v1"
PROFILE = "normalized-gap-graph-v1"
RECIPE = "graph_gap_screen_v1"
OBJECTIVE = "normalized-gap-l1"
OPTIMIZER = "AdamW"
SCHEDULER = "CosineAnnealingLR"
PRECISION = "fp32"
EMA = False
TRAIN_ROLE = "pcqm4mv2-fixed-topology-v1:train"
DEVELOPMENT_ROLE = "pcqm4mv2-fixed-topology-v1:development"

CONFIG_FIELDS = frozenset({
    "epochs", "learning_rate", "weight_decay", "target_mean_eV",
    "target_std_eV", "train_rows", "development_rows",
    "train_source_idx_sha256", "train_target_sha256", "manifest_sha256",
    "row_order_fingerprint",
})


def _digest(value, label):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _finite_number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")


def _validate_config(config: dict) -> dict:
    if type(config) is not dict or set(config) != CONFIG_FIELDS:
        raise ValueError("graph_gap_screen_v1 requires its complete bounded recipe_config")
    for name in ("epochs", "train_rows", "development_rows"):
        _positive_int(config[name], name)
    for name in ("learning_rate", "target_std_eV"):
        _finite_number(config[name], name, positive=True)
    _finite_number(config["weight_decay"], "weight_decay", nonnegative=True)
    _finite_number(config["target_mean_eV"], "target_mean_eV")
    for name in ("train_source_idx_sha256", "train_target_sha256", "manifest_sha256", "row_order_fingerprint"):
        _digest(config[name], name)
    if config["train_rows"] < PHYSICAL_BATCH or config["development_rows"] < PHYSICAL_BATCH:
        raise ValueError("Graph screen roles must each contain a physical batch")
    if config["train_rows"] // PHYSICAL_BATCH < 1:
        raise ValueError("Graph screen requires one complete train batch")
    return deepcopy(config)


def _metric_semantics() -> dict:
    return {
        "live_train_metric": {
            "metric": "MAE", "unit": "eV", "target": "Gap",
            "role_identity": TRAIN_ROLE, "weights": "live", "direction": "minimize",
            "timing": "online-pre-update; normalized-gap L1 converted to eV",
        },
        "live_dev_metric": {
            "metric": "MAE", "unit": "eV", "target": "Gap",
            "role_identity": DEVELOPMENT_ROLE, "weights": "live", "direction": "minimize",
        },
        "ema_dev_metric": None,
    }


def _acceptance(*, epochs: int, train_rows: int, development_rows: int,
                source_idx_sha256: str, target_sha256: str) -> dict:
    steps = epochs * (train_rows // PHYSICAL_BATCH)
    return {
        "epochs": epochs,
        "optimizer_steps": steps,
        "sample_presentations": epochs * (train_rows // PHYSICAL_BATCH) * PHYSICAL_BATCH,
        "development_rows": development_rows,
        "source_idx_sha256": source_idx_sha256,
        "target_sha256": target_sha256,
        "precision": PRECISION,
    }


def build_screen_recipe(mode: str, *, family: tuple[str, str],
                        source_idx_sha256: str, target_sha256: str,
                        recipe_config: dict | None = None) -> dict:
    """Freeze all generic owner constants before an ExperimentSpec is signed."""
    if type(mode) is not str or not mode:
        raise ValueError("Graph screen mode must be explicit")
    if type(family) not in (tuple, list) or len(family) != 2 or any(type(item) is not str or not item for item in family):
        raise ValueError("Graph screen family identity is invalid")
    _digest(source_idx_sha256, "source_idx_sha256")
    _digest(target_sha256, "target_sha256")
    config = _validate_config(recipe_config)
    if config["development_rows"] < PHYSICAL_BATCH:
        raise ValueError("Graph screen development role is too small")
    return {
        "format": RECIPE_FORMAT,
        "profile": PROFILE,
        "family": {"name": family[0], "version": family[1]},
        "mode": mode,
        "feature_schema": FEATURE_SCHEMA,
        "roles": list(ROLES),
        "recipe": RECIPE,
        "objective": OBJECTIVE,
        "sampler": SAMPLER,
        "transform": "train-mean-unbiased-std",
        "seed": SEED,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "precision": PRECISION,
        "ema": EMA,
        "optimizer": OPTIMIZER,
        "scheduler": SCHEDULER,
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "epochs": config["epochs"],
        "train_rows": config["train_rows"],
        "development_rows": config["development_rows"],
        "train_source_idx_sha256": config["train_source_idx_sha256"],
        "train_target_sha256": config["train_target_sha256"],
        "target_mean_eV": config["target_mean_eV"],
        "target_std_eV": config["target_std_eV"],
        "manifest_sha256": config["manifest_sha256"],
        "row_order_fingerprint": config["row_order_fingerprint"],
        "development_role_identity": DEVELOPMENT_ROLE,
        "metric_semantics": _metric_semantics(),
        "acceptance_requirements": _acceptance(
            epochs=config["epochs"], train_rows=config["train_rows"],
            development_rows=config["development_rows"],
            source_idx_sha256=source_idx_sha256, target_sha256=target_sha256),
    }


def _recipe_config(recipe: dict) -> dict:
    keys = {
        "epochs", "learning_rate", "weight_decay", "target_mean_eV", "target_std_eV",
        "train_rows", "development_rows", "train_source_idx_sha256",
        "train_target_sha256", "manifest_sha256", "row_order_fingerprint",
    }
    return {key: recipe[key] for key in keys}


def _validate_recipe_fields(recipe: dict) -> dict:
    if type(recipe) is not dict or recipe.get("format") != RECIPE_FORMAT:
        raise ValueError("Unsupported graph screen recipe")
    required = {
        "format", "profile", "family", "mode", "feature_schema", "roles", "recipe",
        "objective", "sampler", "transform", "seed", "physical_batch_per_device",
        "precision", "ema", "optimizer", "scheduler", "learning_rate", "weight_decay",
        "epochs", "train_rows", "development_rows", "train_source_idx_sha256",
        "train_target_sha256", "target_mean_eV", "target_std_eV", "manifest_sha256",
        "row_order_fingerprint", "development_role_identity", "metric_semantics",
        "acceptance_requirements",
    }
    if set(recipe) != required:
        raise ValueError("Graph screen recipe fields changed")
    if recipe["profile"] != PROFILE or recipe["feature_schema"] != FEATURE_SCHEMA or recipe["roles"] != list(ROLES) or recipe["recipe"] != RECIPE or recipe["objective"] != OBJECTIVE or recipe["sampler"] != SAMPLER or recipe["transform"] != "train-mean-unbiased-std" or recipe["seed"] != SEED or recipe["physical_batch_per_device"] != PHYSICAL_BATCH or recipe["precision"] != PRECISION or recipe["ema"] is not EMA or recipe["optimizer"] != OPTIMIZER or recipe["scheduler"] != SCHEDULER or recipe["development_role_identity"] != DEVELOPMENT_ROLE:
        raise ValueError("Graph screen scientific constants changed")
    config = _validate_config(_recipe_config(recipe))
    requirements = recipe["acceptance_requirements"]
    if type(requirements) is not dict or set(requirements) != EXPECTED:
        raise ValueError("Graph screen requires frozen acceptance_requirements")
    expected = _acceptance(
        epochs=config["epochs"], train_rows=config["train_rows"],
        development_rows=config["development_rows"],
        source_idx_sha256=requirements["source_idx_sha256"],
        target_sha256=requirements["target_sha256"],
    )
    if requirements != expected:
        raise ValueError("Graph screen acceptance exposure differs from recipe")
    if recipe["metric_semantics"] != _metric_semantics():
        raise ValueError("Graph screen metric semantics changed")
    return config


def validate_screen_recipe(spec, arm_id, recipe):
    """Validate the generic contract without opening graph data or tensors."""
    from .experiment_execution import training_adapter

    arm = next((item for item in spec.to_dict()["arms"] if item["arm_id"] == arm_id), None)
    if arm is None:
        raise ValueError("Graph screen arm is absent from Spec")
    adapter = training_adapter(arm)
    if adapter.module != "molgap.graph_screen_training" or adapter.model_factory is None:
        raise ValueError("Arm is not a shared graph training registration")
    config = _validate_recipe_fields(recipe)
    family = arm["family"]
    if recipe["family"] != family or recipe["mode"] != adapter.mode(arm):
        raise ValueError("Graph screen recipe family/mode differs from executable arm")
    if arm["initialization"]["kind"] != "frozen_state" or arm["initialization"]["seed"] != SEED or arm["training"]["overrides"]:
        raise ValueError("Graph screen requires frozen seed-42 initialization and no overrides")
    if arm["training"]["objective"]["name"] != OBJECTIVE or arm["training"]["sampler"]["name"] != SAMPLER or arm["training"]["transform"]["name"] != "train-mean-unbiased-std":
        raise ValueError("Graph screen Spec training contract differs from the owner")
    if arm["training"]["sampler"]["sha256"] != recipe["row_order_fingerprint"]:
        raise ValueError("Graph screen Spec sampler identity differs from recipe")
    return {"family": family, "mode": recipe["mode"], "recipe": "frozen_recipe_verified", "config": config}


def _state_cpu(model) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _tree_equal(left, right) -> bool:
    """Compare every durable field, including tensor dtypes and RNG bytes."""
    if torch.is_tensor(left):
        return (torch.is_tensor(right) and left.dtype == right.dtype and
                left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys() and
                all(_tree_equal(value, right[key]) for key, value in left.items()))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(_tree_equal(one, two) for one, two in zip(left, right)))
    return type(left) is type(right) and left == right


def _batch_digest(batch) -> str:
    digest = hashlib.sha256()
    for name in ("x", "edge_index", "edge_attr", "batch", "random_walk_pe", "y"):
        value = getattr(batch, name, None)
        if not torch.is_tensor(value):
            raise ValueError("Graph batch fixture is incomplete")
        digest.update(name.encode("ascii") + b"\0" + value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _device_batch(batch, device):
    return batch.to(device, non_blocking=(device.type == "cuda"))


def _forward_loss(model, batch, mean, std):
    from .shared_model_adapter import forward_normalized_gap
    prediction = forward_normalized_gap(model, batch)
    target = batch.y.reshape(-1).to(dtype=prediction.dtype)
    normalized_target = (target - mean) / std
    return (prediction - normalized_target).abs().mean(), prediction, target


def _evaluate(model, loader, mean_eV, std_eV, device):
    from .shared_model_adapter import forward_normalized_gap
    model.eval()
    predictions, targets, sources = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = _device_batch(batch, device)
            prediction = forward_normalized_gap(model, batch)
            target = batch.y.reshape(-1).to(dtype=prediction.dtype)
            source = __import__("molgap.pcqm_topology", fromlist=["graph_source_idx"]).graph_source_idx(batch).reshape(-1).to(torch.int64)
            predictions.append(prediction * std_eV + mean_eV)
            targets.append(target)
            sources.append(source)
    prediction_eV = torch.cat(predictions).detach().cpu()
    target_eV = torch.cat(targets).detach().cpu()
    source_idx = torch.cat(sources).detach().cpu()
    if len(torch.unique(source_idx)) != len(source_idx):
        raise ValueError("Development loader returned duplicate source rows")
    mae = float((prediction_eV.double() - target_eV.double()).abs().mean())
    return mae, prediction_eV, target_eV, source_idx


def _train_epoch(model, loader, optimizer, mean, std, device):
    model.train()
    total = 0.0
    rows = 0
    for batch in loader:
        batch = _device_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, target = _forward_loss(model, batch, mean, std)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Graph screen loss is nonfinite")
        loss.backward()
        optimizer.step()
        count = int(target.numel())
        total += float(loss.detach().cpu()) * count * float(std.detach().cpu())
        rows += count
    if rows == 0:
        raise RuntimeError("Graph screen consumed no complete train batch")
    return total / rows, rows


def _runtime_provenance(context, recipe_path, initial_state_path, runtime, mode):
    return {
        "format": "molgap-graph-screen-runtime-provenance-v1",
        "context": context.to_dict(),
        "recipe_sha256": file_digest(recipe_path),
        "initial_state_file_sha256": file_digest(initial_state_path),
        "initial_state_sha256": None,
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "mode": mode,
        "profile": PROFILE,
    }


def _make_provenance(context, recipe_path, initial_state_path, runtime, mode, state_sha256):
    value = _runtime_provenance(context, recipe_path, initial_state_path, runtime, mode)
    value["initial_state_sha256"] = state_sha256
    return value


def _write_json(path: Path, value: dict) -> None:
    atomic_json(Path(path), value)


def _preflight_certificate(runtime, provenance, architecture, *, mode, fixture_sha256):
    return {
        "format": "molgap-graph-screen-runtime-certificate-v1",
        "status": "accepted",
        "profile": PROFILE,
        "mode": mode,
        "precision": PRECISION,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "tail_batch_policy": "drop_last_train; development_tail_allowed",
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "software_fingerprint": runtime["installed_distributions_sha256"],
        "calibration_fixture_sha256": fixture_sha256,
        "calibration_checks_passed": True,
        "provenance_sha256": canonical_fingerprint(provenance),
        "architecture_sha256": canonical_fingerprint(architecture),
        "resume_roundtrip": architecture["resume_roundtrip"],
        "repeatability": architecture["repeatability"],
    }


def _validate_runtime_preflight(directory: Path, provenance: dict) -> dict:
    directory = Path(directory)
    try:
        saved = _json(directory / "runtime_provenance.json")
        runtime = _json(directory / "runtime_manifest.json")
        certificate = _json(directory / "runtime_certificate.json")
        architecture = _json(directory / "architecture_preflight.json")
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("Graph screen runtime preflight evidence is incomplete") from exc
    if saved != provenance or certificate.get("provenance_sha256") != canonical_fingerprint(provenance):
        raise ValueError("Graph screen runtime provenance changed")
    if (certificate.get("status") != "accepted" or certificate.get("profile") != PROFILE or
            certificate.get("runtime_fingerprint") != runtime.get("runtime_fingerprint") or
            certificate.get("architecture_sha256") != canonical_fingerprint(architecture) or
            architecture.get("accepted") is not True or
            architecture.get("repeatability", {}).get("accepted") is not True or
            architecture.get("resume_roundtrip", {}).get("accepted") is not True):
        raise ValueError("Graph screen runtime preflight is not qualified")
    return certificate


def _execution_device(spec):
    """Select the declared execution device without hiding a missing GPU."""
    platform_name = spec.to_dict()["platform"]["name"]
    if platform_name == "kaggle":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Shared Kaggle graph screen requires exactly one visible CUDA device")
        if "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Shared Kaggle graph screen requires an assigned T4")
        return torch.device("cuda:0")
    if platform_name == "local":
        return torch.device("cpu")
    raise ValueError("Shared graph owner has no execution policy for this platform")


def _request(*, spec, package_dir, expected_package_identity, arm_id, mode,
             recipe_path, initial_state_path, input_root, account, run_reference,
             trajectory_id):
    recipe = _json(Path(recipe_path))
    validate_screen_recipe(spec, arm_id, recipe)
    if recipe["mode"] != mode:
        raise ValueError("Graph screen mode differs from recipe")
    context = RunContext.for_training(
        spec, package_dir, expected_package_identity=expected_package_identity,
        arm_id=arm_id, account=account, run_reference=run_reference)
    from .experiment_execution import validate_staged_trajectory
    staged = validate_staged_trajectory(spec, arm_id, Path(input_root))
    if staged["trajectory_id"] != trajectory_id:
        raise ValueError("Graph screen prospective trajectory differs from request")
    arm = next(item for item in spec.to_dict()["arms"] if item["arm_id"] == arm_id)
    if file_digest(recipe_path) != context.training_recipe_sha256 or arm["initialization"]["state_sha256"] is None:
        raise ValueError("Graph screen recipe/initialization identity is not pinned")
    inspected = inspect_frozen_state_artifact(Path(initial_state_path), expected_state_sha256=arm["initialization"]["state_sha256"])
    if arm["initialization"]["seed"] != SEED:
        raise ValueError("Graph screen initialization seed differs from the owner")
    return context, recipe, arm, inspected


def _load_rng_for_restore(state: dict) -> dict:
    restored = deepcopy(state)
    numpy_state = restored.get("numpy")
    if isinstance(numpy_state, tuple) and len(numpy_state) == 5 and isinstance(numpy_state[1], list):
        import numpy as np
        restored["numpy"] = (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    return restored


def _checkpoint_payload(path: Path) -> dict:
    from .experiment_resume import safe_cpu_torch_load
    return safe_cpu_torch_load(Path(path))


def _retained_artifacts(root: Path) -> dict:
    return {"checkpoint": root / "last_checkpoint.pt",
            "trace": root / "canonical_trace.json",
            "selected_model": root / "selected_model.pt",
            "predictions": root / "development_predictions.pt"}


def _validate_optimizer_scheduler(checkpoint, recipe, arm):
    from .experiment_execution import training_adapter
    from .shared_model_adapter import build_model

    # A factory is the authority on parameter ordering and buffers. Preserve
    # caller RNG so a read-only resume inspection cannot alter later training.
    prior_rng = capture_rng_state()
    try:
        model = build_model(training_adapter(arm), arm)
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError as exc:
            raise ValueError("Graph screen resume model differs from its reviewed factory") from exc
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"],
                                     weight_decay=recipe["weight_decay"])
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe["epochs"], eta_min=0.0)
        expected_groups = optimizer.state_dict()["param_groups"]
        groups = checkpoint["optimizer"].get("param_groups")
        state = checkpoint["optimizer"].get("state")
        if type(groups) is not list or len(groups) != 1 or type(state) is not dict or not state:
            raise ValueError("Graph screen resume AdamW state is incomplete")
        group, expected = groups[0], expected_groups[0]
        if type(group) is not dict or set(group) != set(expected):
            raise ValueError("Graph screen resume AdamW group schema changed")
        if any(not _tree_equal(value, expected[key]) for key, value in group.items() if key != "lr"):
            raise ValueError("Graph screen resume AdamW frozen settings or parameter order changed")
        parameters = list(model.parameters())
        for index, moment in state.items():
            if (type(index) is not int or not 0 <= index < len(parameters) or
                    type(moment) is not dict or set(moment) != {"step", "exp_avg", "exp_avg_sq"}):
                raise ValueError("Graph screen resume AdamW moment schema changed")
            step = moment["step"]
            if (not torch.is_tensor(step) or step.ndim != 0 or step.dtype != torch.float32 or
                    not float(step).is_integer() or not 1 <= float(step) <= checkpoint["optimizer_step"]):
                raise ValueError("Graph screen resume AdamW parameter step is outside acknowledged exposure")
            for name in ("exp_avg", "exp_avg_sq"):
                value = moment[name]
                if (not torch.is_tensor(value) or value.shape != parameters[index].shape or
                        value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) or
                        (name == "exp_avg_sq" and bool((value < 0).any()))):
                    raise ValueError("Graph screen resume AdamW moment tensor changed")
        epoch = checkpoint["cursor"]["epoch"]
        scheduler = checkpoint["scheduler"]
        lr = recipe["learning_rate"] * (1 + math.cos(math.pi * epoch / recipe["epochs"])) / 2
        expected_scheduler = {"T_max": recipe["epochs"], "eta_min": 0.0,
                              "base_lrs": [recipe["learning_rate"]], "last_epoch": epoch,
                              "_step_count": epoch + 1}
        if any(scheduler.get(key) != value for key, value in expected_scheduler.items()):
            raise ValueError("Graph screen resume cosine scheduler differs from recipe/cursor")
        last_lr = scheduler.get("_last_lr")
        if (type(last_lr) is not list or len(last_lr) != 1 or
                not math.isclose(group["lr"], lr, rel_tol=1e-12, abs_tol=1e-15) or
                not math.isclose(last_lr[0], lr, rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError("Graph screen resume optimizer/scheduler learning rate changed")
    finally:
        restore_rng_state(prior_rng)


def _qualified_cuda_devices(runtime, platform):
    accelerator = runtime.get("accelerator")
    if accelerator is not None and type(accelerator) is not dict:
        raise ValueError("Graph screen resume accelerator manifest is invalid")
    count = 0 if accelerator is None else accelerator.get("device_count_visible")
    if type(count) is not int or count < 0:
        raise ValueError("Graph screen resume runtime lacks its visible CUDA count")
    if platform == "kaggle" and (accelerator is None or count != 1 or
            "T4" not in accelerator.get("name", "")):
        raise ValueError("Graph screen resume runtime requires one visible T4")
    if platform not in {"local", "kaggle"}:
        raise ValueError("Shared graph owner has no execution policy for this platform")
    return count


def _validate_selected_endpoint(checkpoint, observations, artifacts, root, context, requirements):
    """The trace-bound checkpoint pins the complete retained best endpoint."""
    selected_path = artifacts.get("selected_model")
    predictions_path = artifacts.get("predictions")
    if selected_path is None or predictions_path is None:
        raise ValueError("Graph screen resume requires its selected model and predictions")
    paths = [Path(selected_path), Path(predictions_path)]
    paths = [path if path.is_absolute() else root / path for path in paths]
    owner_state = checkpoint.get("owner_state")
    if type(owner_state) is not dict:
        raise ValueError("Graph screen resume owner state is missing")
    binding = owner_state.get("selected_endpoint")
    expected_binding = {"selected_model_sha256": file_digest(paths[0]),
                        "predictions_sha256": file_digest(paths[1])}
    if binding != expected_binding:
        raise ValueError("Graph screen selected endpoint differs from its acknowledged checkpoint")
    selected, predictions = (_checkpoint_payload(path) for path in paths)
    if selected.get("context") != context or predictions.get("context") != context:
        raise ValueError("Graph screen selected endpoint context changed")
    epoch, step = selected.get("epoch"), selected.get("optimizer_step")
    if (type(epoch) is not int or not 1 <= epoch <= len(observations) or
            type(step) is not int or selected.get("weights") != "live"):
        raise ValueError("Graph screen selected endpoint is outside its live trace")
    selected_row = observations[epoch - 1]
    if selected_row["epoch_or_pass"] != epoch or selected_row["optimizer_step"] != step:
        raise ValueError("Graph screen selected endpoint epoch/step differs from trace")
    model = selected.get("model")
    reference = checkpoint["model"]
    if (not isinstance(model, dict) or model.keys() != reference.keys() or
            any(not torch.is_tensor(value) or value.shape != reference[key].shape or
                value.dtype != reference[key].dtype or not bool(torch.isfinite(value).all())
                for key, value in model.items())):
        raise ValueError("Graph screen selected model state is incompatible with checkpoint")
    if epoch == checkpoint["cursor"]["epoch"] and not _tree_equal(model, reference):
        raise ValueError("Graph screen latest selected model differs from checkpoint")
    tensors = [predictions.get(key) for key in ("source_idx", "target_eV", "prediction_eV")]
    if any(not torch.is_tensor(value) or value.ndim != 1 or
           len(value) != requirements["development_rows"] for value in tensors):
        raise ValueError("Graph screen selected predictions have invalid shape")
    source, target, prediction = tensors
    if (source.dtype != torch.int64 or len(torch.unique(source)) != len(source) or
            target.dtype != torch.float32 or prediction.dtype != torch.float32 or
            not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(prediction).all()) or
            tensor_digest(source, role="source_idx") != requirements["source_idx_sha256"] or
            tensor_digest(target, role="target") != requirements["target_sha256"]):
        raise ValueError("Graph screen selected predictions changed fixed development identity")
    mae = float((prediction.double() - target.double()).abs().mean())
    for value in (selected.get("development_mae_eV"), selected_row["live_dev_metric"]):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not math.isclose(value, mae, rel_tol=1e-6, abs_tol=1e-8)):
            raise ValueError("Graph screen selected checkpoint/prediction/trace metric mismatch")
    best_row = min(observations, key=lambda row: row["live_dev_metric"])
    if selected_row is not best_row:
        raise ValueError("Graph screen selected endpoint is not the first minimum development MAE")


def validate_screen_resume(spec, arm_id, manifest, bundle_root, artifacts,
                           sidecars, trajectory, context=None):
    """Validate a complete-epoch generic checkpoint before owner resume."""
    from .training_reproducibility import assert_finite_state_dict, validate_rng_state
    from .research_memory.trace import validate_canonical_trace

    arm = next((item for item in spec.to_dict()["arms"] if item["arm_id"] == arm_id), None)
    if arm is None:
        raise ValueError("Graph screen resume arm is absent from Spec")
    if context is None or context.get("arm_id") != arm_id:
        raise ValueError("Graph screen resume context/arm mismatch")
    root = Path(bundle_root)
    checkpoint_path = Path(artifacts["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint = _checkpoint_payload(checkpoint_path)
    required = {"context", "model", "optimizer", "scheduler", "rng_state", "cursor", "optimizer_step", "sample_presentations", "owner_state"}
    if required != set(checkpoint) or any(not isinstance(checkpoint[key], dict) or not checkpoint[key] for key in ("model", "optimizer", "scheduler")):
        raise ValueError("Graph screen resume checkpoint is incomplete")
    if context is not None and checkpoint["context"] != context:
        raise ValueError("Graph screen resume checkpoint context changed")
    if not all(torch.is_tensor(value) for value in checkpoint["model"].values()):
        raise ValueError("Graph screen resume model must contain tensors only")
    assert_finite_state_dict(checkpoint["model"], label="Graph screen resume model")
    _finite_tree(checkpoint, "Graph screen resume")
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in checkpoint["model"].values()):
        raise ValueError("Graph screen resume model is not FP32")
    provenance = sidecars.get("provenance", ()) if isinstance(sidecars, dict) else ()
    recipe_paths = [Path(path) for path in provenance if Path(path).name == "training_contract.json"]
    if len(recipe_paths) != 1:
        raise ValueError("Graph screen resume requires one hash-pinned training_contract.json")
    recipe_path = recipe_paths[0]
    if context is not None and file_digest(recipe_path) != context["training_recipe_sha256"]:
        raise ValueError("Graph screen resume recipe hash differs from context")
    recipe = _json(recipe_path)
    validate_screen_recipe(spec, arm_id, recipe)
    rng = _load_rng_for_restore(checkpoint["rng_state"])
    runtime_paths = [Path(path) for path in sidecars.get("runtime", ())
                     if Path(path).name == "runtime_manifest.json"]
    runtime_path = runtime_paths[0] if len(runtime_paths) == 1 else root / "runtime_manifest.json"
    runtime = _json(runtime_path)
    cuda_devices = _qualified_cuda_devices(runtime, context["platform"])
    validate_rng_state(rng, cuda_devices=cuda_devices, label="Graph screen resume RNG")
    cursor = checkpoint["cursor"]
    if set(cursor) != {"epoch", "next_batch", "sampler_order_sha256"} or type(cursor["epoch"]) is not int or cursor["epoch"] < 1 or cursor["next_batch"] != 0:
        raise ValueError("Graph screen resume cursor is not an acknowledged epoch boundary")
    _digest(cursor["sampler_order_sha256"], "resume.sampler_order_sha256")
    if type(checkpoint["optimizer_step"]) is not int or checkpoint["optimizer_step"] <= 0 or type(checkpoint["sample_presentations"]) is not int or checkpoint["sample_presentations"] <= 0:
        raise ValueError("Graph screen resume exposure counters are invalid")
    trace_path = artifacts.get("trace") if isinstance(artifacts, dict) else None
    if trace_path is not None:
        trace_path = Path(trace_path)
        if not trace_path.is_absolute():
            trace_path = root / trace_path
        trace = _json(trace_path)
        trace = validate_canonical_trace(trace)
        if trace.get("metric_semantics") != _metric_semantics():
            raise ValueError("Graph screen resume trace metric semantics changed")
        observations = [row for row in trace.get("observations", ()) if row.get("event") == "observation"]
        if len(observations) != cursor["epoch"] or not observations or observations[-1]["optimizer_step"] != checkpoint["optimizer_step"] or observations[-1]["sample_presentations"] != checkpoint["sample_presentations"]:
            raise ValueError("Graph screen checkpoint is ahead of or behind its canonical trace")
        if ([row["epoch_or_pass"] for row in observations] != list(range(1, cursor["epoch"] + 1)) or
                trace["run_id"] != context["logical_run_id"] + ":" + arm_id + ":downstream" or
                (trajectory is not None and trace["trajectory_id"] != trajectory.get("trajectory_id")) or
                observations[-1]["checkpoint_identity"] != "sha256:" + file_digest(checkpoint_path)):
            raise ValueError("Graph screen resume trace does not bind its acknowledged checkpoint")
    else:
        raise ValueError("Graph screen resume requires its canonical trace")
    requirements = recipe["acceptance_requirements"]
    train_steps = recipe["train_rows"] // PHYSICAL_BATCH
    if cursor["epoch"] >= recipe["epochs"]:
        raise ValueError("Completed graph screen output cannot be resumed")
    if (checkpoint["optimizer_step"] != cursor["epoch"] * train_steps or checkpoint["sample_presentations"] != cursor["epoch"] * train_steps * PHYSICAL_BATCH or cursor["sampler_order_sha256"] != sampler_order_sha256(recipe["train_rows"], cursor["epoch"])):
        raise ValueError("Graph screen resume cursor differs from the hash-pinned recipe")
    if any(row["optimizer_step"] != (index + 1) * train_steps or
           row["sample_presentations"] != (index + 1) * train_steps * PHYSICAL_BATCH or
           row["ema_dev_metric"] is not None or row["live_dev_metric"] is None
           for index, row in enumerate(observations)):
        raise ValueError("Graph screen resume trace differs from recipe exposure or metric semantics")
    if any(not math.isclose(row["learning_rate"], recipe["learning_rate"] *
               (1 + math.cos(math.pi * index / recipe["epochs"])) / 2,
               rel_tol=1e-12, abs_tol=1e-15) for index, row in enumerate(observations)):
        raise ValueError("Graph screen resume trace learning rate differs from recipe")
    _validate_optimizer_scheduler(checkpoint, recipe, arm)
    _validate_selected_endpoint(checkpoint, observations, artifacts, root, context, requirements)
    if trajectory is not None and isinstance(trajectory, dict) and trajectory.get("trajectory_id") is None:
        raise ValueError("Graph screen resume trajectory is incomplete")
    for path in sidecars.get("context", ()) if isinstance(sidecars, dict) else ():
        observed = _json(Path(path))
        observed = observed.get("context", observed)
        if context is not None and observed != context:
            raise ValueError("Graph screen resume context evidence changed")
    return {"status": "RESUMABLE", "epoch": cursor["epoch"],
            "optimizer_step": checkpoint["optimizer_step"],
            "sample_presentations": checkpoint["sample_presentations"],
            "sampler_order_sha256": cursor["sampler_order_sha256"]}


def run_screen_preflight(*, spec, package_dir: Path, expected_package_identity: str,
                         arm_id: str, mode: str, recipe_path: Path, initial_state_path: Path,
                         input_root: Path, output: Path, account: str, run_reference: str,
                         trajectory_id: str, label_cache=None) -> dict:
    """Run a bounded CPU/GPU batch check without formal exposure counters."""
    if label_cache is not None:
        raise ValueError("Shared graph screen has no auxiliary label cache")
    context, recipe, arm, inspected = _request(
        spec=spec, package_dir=package_dir, expected_package_identity=expected_package_identity,
        arm_id=arm_id, mode=mode, recipe_path=recipe_path, initial_state_path=initial_state_path,
        input_root=input_root, account=account, run_reference=run_reference,
        trajectory_id=trajectory_id)
    inputs = load_graph_inputs(input_root, expected_manifest_sha256=recipe["manifest_sha256"])
    if inputs.role("train").rows != recipe["train_rows"] or inputs.role("development").rows != recipe["development_rows"]:
        raise ValueError("Graph screen role rows differ from the frozen recipe")
    if inputs.role("train").source_idx_sha256 != recipe["train_source_idx_sha256"] or inputs.role("train").target_sha256 != recipe["train_target_sha256"]:
        raise ValueError("Graph screen train row identity differs from recipe")
    if inputs.role("development").source_idx_sha256 != recipe["acceptance_requirements"]["source_idx_sha256"] or inputs.role("development").target_sha256 != recipe["acceptance_requirements"]["target_sha256"]:
        raise ValueError("Graph screen development row identity differs from recipe")
    mean_value = float(inputs.role("train").target_eV.mean())
    std_value = float(inputs.role("train").target_eV.std(unbiased=True))
    if not math.isclose(mean_value, recipe["target_mean_eV"], rel_tol=0.0, abs_tol=1e-12) or not math.isclose(std_value, recipe["target_std_eV"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Graph screen train target statistics differ from recipe")
    determinism = configure_fp32_determinism(SEED)
    runtime = build_runtime_manifest(determinism)
    device = _execution_device(spec)
    from .experiment_execution import training_adapter
    adapter = training_adapter(arm)
    batch = next(iter(inputs.train_loader(1))).to(device)
    fixture_sha256 = _batch_digest(batch)
    mean = torch.tensor(mean_value, device=device, dtype=torch.float32)
    std = torch.tensor(std_value, device=device, dtype=torch.float32)
    # Two identical fresh runs certify deterministic construction and one real
    # forward/backward/update without treating the diagnostic as experiment work.
    losses, states = [], []
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        configure_fp32_determinism(SEED)
        model = __import__("molgap.shared_model_adapter", fromlist=["build_model"]).build_model(adapter, arm, initial_state_path=Path(initial_state_path)).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = _forward_loss(model, batch, mean, std)
        loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
        states.append(_state_cpu(model))
    repeated = {"accepted": losses[0] == losses[1] and all(torch.equal(states[0][key], states[1][key]) for key in states[0]), "losses": losses, "state_sha256": canonical_fingerprint({key: value.tolist() for key, value in states[0].items()})}
    if not repeated["accepted"]:
        raise RuntimeError("Graph screen diagnostic is not numerically repeatable")
    # Exercise the exact durable state boundary used by the owner loop: one
    # scheduler step is saved, then the next optimizer/RNG step is compared to
    # an uninterrupted continuation.
    configure_fp32_determinism(SEED)
    continuous = __import__("molgap.shared_model_adapter", fromlist=["build_model"]).build_model(adapter, arm, initial_state_path=Path(initial_state_path)).to(device).train()
    continuous_optimizer = torch.optim.AdamW(continuous.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    continuous_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(continuous_optimizer, T_max=recipe["epochs"], eta_min=0.0)
    continuous_optimizer.zero_grad(set_to_none=True)
    first_loss, _, _ = _forward_loss(continuous, batch, mean, std)
    first_loss.backward(); continuous_optimizer.step(); continuous_scheduler.step()
    snapshot_path = output / "diagnostic_resume.pt"
    def diagnostic_state(model, optimizer, scheduler, epoch):
        return {"model": _state_cpu(model), "optimizer": deepcopy(optimizer.state_dict()),
                "scheduler": deepcopy(scheduler.state_dict()),
                "rng_state": tensor_safe_rng_state(capture_rng_state()),
                "cursor": {"epoch": epoch, "next_batch": 0,
                           "sampler_order_sha256": sampler_order_sha256(inputs.role("train").rows, epoch)},
                "optimizer_step": epoch, "sample_presentations": epoch * PHYSICAL_BATCH}
    atomic_torch_save(snapshot_path, diagnostic_state(continuous, continuous_optimizer, continuous_scheduler, 1))
    continuous_optimizer.zero_grad(set_to_none=True)
    second_loss, _, _ = _forward_loss(continuous, batch, mean, std)
    second_loss.backward(); continuous_optimizer.step(); continuous_scheduler.step()
    continuous_state = diagnostic_state(continuous, continuous_optimizer, continuous_scheduler, 2)
    restored = _checkpoint_payload(snapshot_path)
    configure_fp32_determinism(SEED)
    resumed = __import__("molgap.shared_model_adapter", fromlist=["build_model"]).build_model(adapter, arm, initial_state_path=Path(initial_state_path)).to(device).train()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=recipe["epochs"], eta_min=0.0)
    resumed.load_state_dict(restored["model"]); resumed_optimizer.load_state_dict(restored["optimizer"]); resumed_scheduler.load_state_dict(restored["scheduler"])
    restore_rng_state(_load_rng_for_restore(restored["rng_state"]))
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_loss, _, _ = _forward_loss(resumed, batch, mean, std)
    resumed_loss.backward(); resumed_optimizer.step(); resumed_scheduler.step()
    resumed_state = diagnostic_state(resumed, resumed_optimizer, resumed_scheduler, 2)
    compared = {key: _tree_equal(continuous_state[key], resumed_state[key]) for key in continuous_state}
    resume_roundtrip = {"accepted": float(second_loss.detach().cpu()) == float(resumed_loss.detach().cpu()) and all(compared.values()),
                        "continuous_loss": float(second_loss.detach().cpu()), "resumed_loss": float(resumed_loss.detach().cpu()),
                        "compared_fields": compared,
                        "scope": "saved model/AdamW/CosineAnnealingLR/RNG/cursor/exposure one-step continuation"}
    if not resume_roundtrip["accepted"]:
        raise RuntimeError(f"Graph screen diagnostic resume roundtrip is not repeatable: {resume_roundtrip}")
    architecture = {
        "accepted": True, "profile": PROFILE, "mode": mode,
        "physical_batch_per_device": PHYSICAL_BATCH, "formal_sample_presentations": 0,
        "fixture_sha256": fixture_sha256, "repeatability": repeated,
        "resume_roundtrip": resume_roundtrip,
        "geometry_used": False,
    }
    provenance = _make_provenance(context, recipe_path, initial_state_path, runtime, mode, arm["initialization"]["state_sha256"])
    certificate = _preflight_certificate(runtime, provenance, architecture, mode=mode, fixture_sha256=fixture_sha256)
    _write_json(output / "workflow_context.json", context.to_dict())
    _write_json(output / "runtime_provenance.json", provenance)
    _write_json(output / "runtime_manifest.json", runtime)
    _write_json(output / "architecture_preflight.json", architecture)
    _write_json(output / "runtime_certificate.json", certificate)
    _write_json(output / "diagnostic_cost.json", {"formal_sample_presentations": 0, "scope": "bounded preflight"})
    return certificate


def run_screen_resume_preflight(*, spec, package_dir: Path, expected_package_identity: str,
                                arm_id: str, mode: str, recipe_path: Path, initial_state_path: Path,
                                input_root: Path, output: Path, account: str, run_reference: str,
                                trajectory_id: str, resume_output: Path, label_cache=None) -> dict:
    """Reuse a retained checkpoint's qualification evidence before resuming."""
    if label_cache is not None:
        raise ValueError("Shared graph screen has no auxiliary label cache")
    context, recipe, arm, _ = _request(
        spec=spec, package_dir=package_dir, expected_package_identity=expected_package_identity,
        arm_id=arm_id, mode=mode, recipe_path=recipe_path, initial_state_path=initial_state_path,
        input_root=input_root, account=account, run_reference=run_reference,
        trajectory_id=trajectory_id)
    source = Path(resume_output).resolve()
    source_runtime = _json(source / "runtime_manifest.json")
    _execution_device(spec)
    current_runtime = build_runtime_manifest(configure_fp32_determinism(SEED))
    if current_runtime.get("runtime_fingerprint") != source_runtime.get("runtime_fingerprint"):
        raise ValueError("Recovery runtime differs from the qualified preflight runtime")
    provenance = _make_provenance(context, recipe_path, initial_state_path,
                                  source_runtime, mode, arm["initialization"]["state_sha256"])
    certificate = _validate_runtime_preflight(source, provenance)
    validate_screen_resume(spec, arm_id, None, source,
                           _retained_artifacts(source),
                           {"context": [source / "canonical_trace.json.context.json"],
                            "provenance": [source / "training_contract.json"]},
                           {"trajectory_id": trajectory_id}, context=context.to_dict())
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    for name in ("workflow_context.json", "runtime_provenance.json", "runtime_manifest.json", "runtime_certificate.json", "architecture_preflight.json", "diagnostic_cost.json"):
        publish_immutable_bytes(output / name, (source / name).read_bytes())
    return certificate


def run_screen_arm(*, spec, package_dir: Path, expected_package_identity: str,
                   arm_id: str, mode: str, recipe_path: Path, initial_state_path: Path,
                   input_root: Path, output: Path, account: str, run_reference: str,
                   trajectory_id: str, label_cache=None, preflight_dir: Path | None = None) -> dict:
    """Execute the fixed graph screen and emit the shared durable bundle."""
    if label_cache is not None:
        raise ValueError("Shared graph screen has no auxiliary label cache")
    context, recipe, arm, _ = _request(
        spec=spec, package_dir=package_dir, expected_package_identity=expected_package_identity,
        arm_id=arm_id, mode=mode, recipe_path=recipe_path, initial_state_path=initial_state_path,
        input_root=input_root, account=account, run_reference=run_reference,
        trajectory_id=trajectory_id)
    preflight = Path(preflight_dir or output).resolve()
    preflight_runtime = _json(preflight / "runtime_manifest.json")
    expected_provenance = _make_provenance(
        context, recipe_path, initial_state_path, preflight_runtime, mode,
        arm["initialization"]["state_sha256"])
    certificate = _validate_runtime_preflight(preflight, expected_provenance)
    inputs = load_graph_inputs(input_root, expected_manifest_sha256=recipe["manifest_sha256"])
    train = inputs.role("train"); development = inputs.role("development")
    requirements = recipe["acceptance_requirements"]
    if (train.rows != recipe["train_rows"] or development.rows != recipe["development_rows"] or
            train.source_idx_sha256 != recipe["train_source_idx_sha256"] or train.target_sha256 != recipe["train_target_sha256"] or
            development.source_idx_sha256 != requirements["source_idx_sha256"] or development.target_sha256 != requirements["target_sha256"]):
        raise ValueError("Graph screen staged role identity differs from recipe")
    mean_value = float(train.target_eV.mean()); std_value = float(train.target_eV.std(unbiased=True))
    if not math.isclose(mean_value, recipe["target_mean_eV"], rel_tol=0.0, abs_tol=1e-12) or not math.isclose(std_value, recipe["target_std_eV"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Graph screen target statistics differ from recipe")
    configure_fp32_determinism(SEED)
    runtime = _json(preflight / "runtime_manifest.json")
    device = _execution_device(spec)
    current_runtime = build_runtime_manifest(configure_fp32_determinism(SEED))
    if current_runtime.get("runtime_fingerprint") != runtime.get("runtime_fingerprint"):
        raise ValueError("Training runtime differs from the qualified preflight runtime")
    runtime = current_runtime
    from .experiment_execution import training_adapter
    from .shared_model_adapter import build_model
    adapter = training_adapter(arm)
    model = build_model(adapter, arm, initial_state_path=Path(initial_state_path)).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe["epochs"], eta_min=0.0)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output / "last_checkpoint.pt").exists():
        validate_screen_resume(spec, arm_id, None, output, _retained_artifacts(output),
            {"context": [output / "canonical_trace.json.context.json"],
             "provenance": [output / "training_contract.json"]},
            {"trajectory_id": trajectory_id}, context=context.to_dict())
    semantics = _metric_semantics()
    session = FamilyOutputSession(output, context, adapter="graph-screen-v1", contract=Path(recipe_path),
                                  trajectory_id=trajectory_id, metric_semantics=semantics)
    _write_json(session.root / "runtime_manifest.json", runtime)
    _write_json(session.root / "runtime_certificate.json", certificate)
    for name in ("workflow_context.json", "runtime_provenance.json",
                 "architecture_preflight.json", "diagnostic_cost.json"):
        publish_immutable_bytes(session.root / name, (preflight / name).read_bytes())
    start_epoch = 0
    checkpoint_path = session.root / "last_checkpoint.pt"
    if checkpoint_path.exists():
        saved = _checkpoint_payload(checkpoint_path)
        if saved.get("context") != context.to_dict():
            raise ValueError("Graph screen resume context changed")
        cursor = saved.get("cursor")
        if type(cursor) is not dict or set(cursor) != {"epoch", "next_batch", "sampler_order_sha256"} or cursor["next_batch"] != 0:
            raise ValueError("Graph screen resume cursor is incomplete")
        start_epoch = cursor["epoch"]
        if type(start_epoch) is not int or not 1 <= start_epoch <= recipe["epochs"] or cursor["sampler_order_sha256"] != sampler_order_sha256(train.rows, start_epoch):
            raise ValueError("Graph screen resume cursor differs from the frozen sampler")
        expected_steps = start_epoch * train.optimizer_steps
        expected_samples = start_epoch * train.exposed_rows
        if saved.get("optimizer_step") != expected_steps or saved.get("sample_presentations") != expected_samples:
            raise ValueError("Graph screen resume exposure differs from recipe")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng_state(_load_rng_for_restore(saved["rng_state"]))
    observations = [row for row in session.stage.recorder.record["observations"] if row.get("event") == "observation"]
    if len(observations) != start_epoch:
        raise ValueError("Graph screen checkpoint and canonical trace disagree")
    if checkpoint_path.exists() and observations:
        expected_checkpoint = "sha256:" + sha256_file(checkpoint_path)
        if observations[-1].get("checkpoint_identity") != expected_checkpoint:
            raise ValueError("Graph screen canonical trace does not identify the retained checkpoint")
    if not checkpoint_path.exists() and observations:
        raise ValueError("Graph screen trace exists without its durable checkpoint")
    best = math.inf
    selected_path = session.root / "selected_model.pt"
    if selected_path.exists():
        selected = torch_load_compat(selected_path, map_location="cpu", weights_only=True)
        best = float(selected.get("development_mae_eV", math.inf))
    mean = torch.tensor(mean_value, device=device, dtype=torch.float32)
    std = torch.tensor(std_value, device=device, dtype=torch.float32)
    for epoch in range(start_epoch + 1, recipe["epochs"] + 1):
        started = time.perf_counter()
        train_mae, rows = _train_epoch(model, inputs.train_loader(epoch), optimizer, mean, std, device)
        if rows != train.exposed_rows:
            raise RuntimeError("Graph screen optimizer exposure changed")
        dev_mae, prediction_eV, target_eV, source_idx = _evaluate(model, inputs.development_loader(), mean_value, std_value, device)
        step = epoch * train.optimizer_steps
        samples = epoch * train.exposed_rows
        if dev_mae < best:
            best = dev_mae
            session.selected(model_state=_state_cpu(model), epoch=epoch, optimizer_step=step,
                             weights="live", prediction_eV=prediction_eV,
                             target_eV=target_eV, source_idx=source_idx)
        observed_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        session.checkpoint(model_state=_state_cpu(model), optimizer_state=optimizer.state_dict(),
                           scheduler_state=scheduler.state_dict(), rng_state=capture_rng_state(),
                           cursor={"epoch": epoch, "next_batch": 0,
                                   "sampler_order_sha256": sampler_order_sha256(train.rows, epoch)},
                           optimizer_step=step, sample_presentations=samples,
                           owner_state={"selected_endpoint": {
                               "selected_model_sha256": file_digest(session.root / "selected_model.pt"),
                               "predictions_sha256": file_digest(session.root / "development_predictions.pt")}})
        session.epoch_finished(epoch=epoch, optimizer_step=step, sample_presentations=samples,
                               live_train_metric=train_mae, live_dev_metric=dev_mae,
                               ema_dev_metric=None, learning_rate=observed_lr,
                               checkpoint_identity="sha256:" + sha256_file(checkpoint_path),
                               wall_time_seconds=time.perf_counter() - started)
    costs = [{"metric": "wall_seconds", "unit": "seconds", "value": time.perf_counter() - session.started,
              "status": "measured", "semantics": "process_wall", "hardware": device.type},
             {"metric": "device_seconds", "unit": "seconds", "value": None,
              "status": "missing", "semantics": "allocated_device", "hardware": device.type,
              "reason": "Shared local owner retains no allocator ledger"}]
    result = session.complete(runtime={"platform": context.platform, "account": context.account,
        "precision": PRECISION, "source_commit": context.source_commit,
        "source_archive_sha256": context.source_archive_sha256,
        "runtime_certificate_id": canonical_fingerprint(certificate)}, hardware=device.type,
        observed_costs=costs)
    return result


__all__ = [
    "CONFIG_FIELDS", "EMA", "FEATURE_SCHEMA", "OBJECTIVE", "PHYSICAL_BATCH",
    "PROFILE", "RECIPE", "RECIPE_FORMAT", "SAMPLER", "SCHEDULER", "SEED",
    "build_screen_recipe", "run_screen_arm", "run_screen_preflight",
    "run_screen_resume_preflight", "validate_screen_recipe", "validate_screen_resume",
]
