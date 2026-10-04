"""Clean FP32 inference for accepted K1 states, with bounded native timing."""
from __future__ import annotations

import time
from pathlib import Path

from .k1_screen_training import FORBIDDEN_MODEL_FIELDS, _forward
from .qm9_neural_atom import make_encoder


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
