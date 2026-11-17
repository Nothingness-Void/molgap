"""Clean FP32 inference for accepted K1 states, with bounded native timing."""
from __future__ import annotations

import time
import math
from contextlib import contextmanager
from pathlib import Path

from .k1_screen_training import FORBIDDEN_MODEL_FIELDS, _forward
from .qm9_neural_atom import make_encoder


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
