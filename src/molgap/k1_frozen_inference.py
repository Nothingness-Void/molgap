"""Clean FP32 inference for accepted K1 states, with bounded native timing."""
from __future__ import annotations

import time
import math
from contextlib import contextmanager
from pathlib import Path

from .k1_screen_training import FORBIDDEN_MODEL_FIELDS, _forward
from .qm9_neural_atom import make_encoder


def read_session_k1_states(path: Path, *, expected_sha256: str,
                           expected_context: dict, recipe_path: Path,
                           expected_step: int, checkpoint_kind: str):
    """Bind saved Live/Raw/Clean views without constructing a training session.

    FamilyTrainingState owns restoration of the returned state. This translator
    checks outer identity, exposure and selected/resume bindings, not promotion.
    """
    import torch
    from .family_training_runner import read_recipe
    from .family_training_state import _state_identity
    from .training_reproducibility import sha256_file

    if checkpoint_kind not in {"selected", "final"}:
        raise ValueError("Unknown shared-session K1 checkpoint kind")
    if sha256_file(path) != expected_sha256:
        raise ValueError("Session checkpoint bytes differ")
    if sha256_file(recipe_path) != expected_context.get("training_recipe_sha256"):
        raise ValueError("Session recipe bytes differ")
    _, recipe = read_recipe(recipe_path)
    if (expected_context.get("family_name") != "neural_atom_k1" or
            expected_context.get("family_version") != "3" or
            recipe.family != "neural_atom_k1" or recipe.mode != "reference" or
            recipe.clean_bn is None or recipe.ema_decay is None or
            type(expected_step) is not int or expected_step <= 0 or
            expected_step % recipe.loop_plan.steps_per_pass):
        raise ValueError("Expected a complete-pass Clean K1 session")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("context") != expected_context or saved.get("optimizer_step") != expected_step:
        raise ValueError("Session checkpoint context/step differs")
    state = saved.get("owner_state")
    if (not isinstance(state, dict) or state.get("raw_revision") != expected_step or
            not isinstance(state.get("clean"), dict) or state["clean"].get("stale") is not False):
        raise ValueError("Session Clean state is absent, stale or differently stepped")
    clean = state["clean"]
    calibration = clean.get("calibration", {})
    if (calibration.get("raw_revision") != expected_step or
            calibration.get("policy_identity") != recipe.clean_bn.identity or
            calibration.get("raw_state_identity") != _state_identity(state["raw_ema"]) or
            calibration.get("clean_state_identity") != _state_identity(clean["model"])):
        raise ValueError("Session calibration identity differs")
    if checkpoint_kind == "selected":
        if (saved.get("weights") != "ema" or saved.get("selection") != recipe.selection or
                saved.get("epoch_or_pass") != expected_step / recipe.loop_plan.steps_per_pass or
                saved.get("calibration") != calibration or
                _state_identity(saved["model"]) != _state_identity(clean["model"])):
            raise ValueError("Selected Clean payload differs from its same-step state")
    elif (expected_step != recipe.optimizer_steps or
            saved.get("sample_presentations") != expected_step * recipe.batch_size or
            saved.get("cursor", {}).get("status") != "complete" or
            _state_identity(saved["model"]) != _state_identity(state["live"]) or
            _state_identity(saved["ema"]) != _state_identity(state["raw_ema"])):
        raise ValueError("Final session exposure or Live/Raw binding differs")
    return state, recipe


def load_native500k_k1(path: Path, *, expected_sha256: str,
                      expected_source_sha256: str, expected_epoch: int,
                      checkpoint_kind: str,
                      expected_arm: str = "k1_pretrained_consistency"):
    """Translate the retained native500K checkpoint through its K1 factory.

    The caller must use the accepted model source and bind the checkpoint bytes.
    This adapter supplies no role access or scientific comparison authority.
    """
    import torch
    from .training_reproducibility import sha256_file
    from .v4_runtime import torch_load_compat

    if expected_arm not in ("k1_pretrained_mean2", "k1_pretrained_consistency"):
        raise ValueError("Unknown native500K K1 arm")
    if checkpoint_kind not in ("selected", "final"):
        raise ValueError("Unknown native500K checkpoint kind")
    if sha256_file(path) != expected_sha256:
        raise ValueError("Native500K checkpoint bytes differ")
    state = torch_load_compat(path, map_location="cpu",
                              weights_only=checkpoint_kind == "selected")
    contract = state["contract"]
    required = {"arm": expected_arm, "parameters": 3658817,
                "benchmark_id": "pcqm-composed500k-dev50k-60pass-v1",
                "precision": "fp32", "geometry_used": False, "teacher_used": False}
    if any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("Native500K K1 contract differs")
    epoch = state["epoch"] if checkpoint_kind == "selected" else state["next_epoch"] - 1
    if epoch != expected_epoch or state["source_sha256"] != expected_source_sha256:
        raise ValueError("Native500K epoch/source differs")
    if checkpoint_kind == "final" and (state["next_batch_index"] != 0 or
            state["global_step"] != 234360 or state["next_epoch"] != 60):
        raise ValueError("Native500K final exposure differs")
    mean, std = float(state["mean"]), float(state["std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid native500K target transform")
    model = make_encoder("neural_atom_k1")
    model.load_state_dict(state["model"], strict=True)
    if sum(p.numel() for p in model.parameters()) != 3658817:
        raise ValueError("Native500K architecture differs")
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite native500K state")
    return model.eval().requires_grad_(False), {
        "mean": mean, "std": std, "epoch_zero_based": epoch,
        "checkpoint_kind": checkpoint_kind, "contract": contract,
    }


@contextmanager
def scale_slot_return(model, *, layers: tuple[int, ...], scale: float):
    """Temporarily intervene on frozen mixer returns, preserving local inputs."""
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("Slot intervention requires clean eval mode")
    if not layers or len(set(layers)) != len(layers) or any(type(layer) is not int or layer not in (3, 6, 9) for layer in layers):
        raise ValueError("Expected distinct K1 mixer layers")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Expected finite nonnegative slot scale")
    handles = []
    def intervene(module, inputs, output):
        if scale == 1:
            return output
        hidden = inputs[0]
        return hidden if scale == 0 else hidden + scale * (output - hidden)
    try:
        for layer in layers:
            mixer = model.neural_atom_mixers[str(layer)]
            if mixer.active_slots != 1:
                raise ValueError("Expected accepted single-slot K1")
            handles.append(mixer.register_forward_hook(intervene))
        yield
    finally:
        for handle in handles:
            handle.remove()


def load_selected_k1(path: Path, *, expected_epoch: int, expected_parameters: int):
    import torch

    selected = torch.load(path, map_location="cpu", weights_only=True)
    if selected["weights"] != "live" or selected["epoch"] != expected_epoch:
        raise ValueError("Selected epoch or weights differ from the accepted binding")
    model = make_encoder("neural_atom_k1")
    model.load_state_dict(selected["model"], strict=True)
    if sum(p.numel() for p in model.parameters()) != expected_parameters:
        raise ValueError("Accepted K1 architecture differs")
    return model.eval().requires_grad_(False), selected["context"]


def predict_clean(model, loader, *, mean: float, std: float, device: str,
                  deadline: float):
    """Time synchronized inference; loader and transfers are included in wall time."""
    import torch

    use_cuda = str(device).startswith("cuda")
    model.to(device)
    predictions, targets, sources = [], [], []
    forward_ms = 0.0
    start, cpu_start = time.perf_counter(), time.process_time()
    if use_cuda:
        torch.cuda.synchronize()
    with torch.inference_mode():
        for batch in loader:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Frozen inference wall budget exhausted")
            if any(field in batch for field in FORBIDDEN_MODEL_FIELDS):
                raise ValueError("Geometry field reached pure-2D K1")
            targets.append(batch.y.view(-1).cpu())
            sources.append(batch.source_idx.view(-1).long().cpu())
            batch = batch.to(device)
            if use_cuda:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            else:
                tick = time.perf_counter()
            prediction = _forward(model, batch) * std + mean
            if use_cuda:
                end.record()
                end.synchronize()
                forward_ms += begin.elapsed_time(end)
            else:
                forward_ms += (time.perf_counter() - tick) * 1000
            predictions.append(prediction.cpu())
    output = {"source_idx": torch.cat(sources), "target_eV": torch.cat(targets),
              "prediction_eV": torch.cat(predictions)}
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise ValueError("Nonfinite K1 inference artifact")
    return output, {"rows": len(output["source_idx"]),
                    "wall_seconds": time.perf_counter() - start,
                    "process_cpu_seconds": time.process_time() - cpu_start,
                    "forward_seconds": forward_ms / 1000,
                    "forward_scope": "synchronized model forward plus denormalization; excludes H2D/D2H and loader"}
