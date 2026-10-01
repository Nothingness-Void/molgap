"""Observational hooks on frozen K1 encoders; no training or interventions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool

from molgap.k1_screen_training import (
    FORBIDDEN_MODEL_FIELDS, _PackedGraphDatasetFactory, _forward,
    find_fixed_cache, sha256_file,
)
from molgap.qm9_neural_atom import make_encoder
from molgap.training_reproducibility import atomic_json, configure_fp32_determinism


SEED = 20261002
SAMPLE_ROWS = 2048
MEAN = 5.3383002281188965
STD = 1.275090217590332
TOLERANCE_EV = 1e-4


class Observations:
    """Read intermediate outputs without replacing any forward callable."""

    def __init__(self, model):
        self.pending = {}
        self.latest = {}
        self.handles = []
        for layer in (3, 6, 9):
            mixer = model.neural_atom_mixers[str(layer)]
            if mixer.active_slots != 1:
                raise RuntimeError("Only the frozen single-slot architecture is authorized")
            self.handles.append(mixer.register_forward_pre_hook(self.pre(layer)))
            for name in ("node_key", "slot_query", "slot_norm2", "return_projection"):
                self.handles.append(getattr(mixer, name).register_forward_hook(self.capture(layer, name)))
            self.handles.append(mixer.register_forward_hook(self.finish(layer)))

    def pre(self, layer):
        def observe(module, inputs):
            hidden, batch = inputs
            self.pending[layer] = {"hidden": hidden, "batch": batch}
        return observe

    def capture(self, layer, name):
        def observe(module, inputs, output):
            self.pending[layer][name] = output
        return observe

    def finish(self, layer):
        def observe(module, inputs, output):
            state = self.pending.pop(layer)
            batch = state["batch"]
            count = torch.bincount(batch)
            valid = torch.arange(state["node_key"].shape[1])[None, :] < count[:, None]
            logits = torch.einsum("kd,bnd->bkn", state["slot_query"], state["node_key"])
            logits = logits / math.sqrt(module.latent_channels)
            attention = logits.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)[:, 0, :]
            update = state["return_projection"]
            pooled_hidden = global_mean_pool(state["hidden"], batch)
            pooled_update = global_mean_pool(update, batch)
            recovered_u = global_add_pool(update, batch)
            projected_u = state["slot_norm2"][:, 0, :] @ module.return_projection.weight.T
            entropy = -(attention * attention.clamp_min(torch.finfo(attention.dtype).tiny).log()).sum(-1)
            hidden_norm = pooled_hidden.norm(dim=-1)
            update_norm = pooled_update.norm(dim=-1)
            identity = pooled_update - projected_u / count[:, None]
            values = {
                "nodes": count,
                "pooled_hidden_norm": hidden_norm,
                "pooled_update_norm": update_norm,
                "update_hidden_ratio": update_norm / hidden_norm.clamp_min(1e-30),
                "recovered_u_norm": recovered_u.norm(dim=-1),
                "projected_u_norm": projected_u.norm(dim=-1),
                "attention_entropy": entropy,
                "attention_effective_atoms": entropy.exp(),
                "attention_max": attention.max(-1).values,
                "attention_sum_abs_error": (attention.sum(-1) - 1).abs(),
                "u_recovery_max_abs_error": (recovered_u - projected_u).abs().max(-1).values,
                "mean_update_u_over_n_max_abs_error": identity.abs().max(-1).values,
                "mean_update_recovered_u_over_n_max_abs_error":
                    (pooled_update - recovered_u / count[:, None]).abs().max(-1).values,
                "forward_residual_max_abs_error":
                    global_add_pool((output - (state["hidden"] + update)).abs(), batch).max(-1).values,
            }
            self.latest[layer] = {key: value.detach().cpu().numpy() for key, value in values.items()}
        return observe

    def remove(self):
        for handle in self.handles:
            handle.remove()


def write_npz(path, arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def development_dataset(cache):
    root, manifest = find_fixed_cache(cache)
    parts, hashes = [], {}
    offset = 100000
    for item in manifest["geometry_shards"]:
        if item["role"] != "development":
            continue
        path = root / item["file"]
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise RuntimeError("Development shard hash changed")
        payload = _PackedGraphDatasetFactory.load(path)
        if len(payload) != item["rows"] or any(field in payload._data for field in FORBIDDEN_MODEL_FIELDS):
            raise RuntimeError("Development shard count/geometry stripping failed")
        expected = torch.arange(offset, offset + len(payload))
        if not torch.equal(payload._data.source_idx.view(-1).long(), expected):
            raise RuntimeError("Development source order changed")
        offset += len(payload)
        parts.append(payload)
        hashes[item["file"]] = digest
    if offset != 150000:
        raise RuntimeError("Development role length changed")
    return ConcatDataset(parts), hashes


def run_arm(name, width, artifact_root, loader, offsets, output, deadline):
    started, cpu_started = time.perf_counter(), time.process_time()
    checkpoint = artifact_root / "selected_model.pt"
    saved_path = artifact_root / "development_predictions.pt"
    saved = torch.load(saved_path, map_location="cpu", weights_only=True)
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if selected["context"] != saved["context"]:
        raise RuntimeError("Checkpoint/prediction context mismatch")
    if not torch.equal(saved["source_idx"].view(-1).long(), torch.arange(100000, 150000)):
        raise RuntimeError("Saved prediction row order changed")
    model = make_encoder("neural_atom_k1", hidden_channels=width)
    model.load_state_dict(selected["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != {192: 3658817, 256: 6035201}[width]:
        raise RuntimeError("Parameter identity changed")
    hooks = Observations(model)
    collected = {}
    predictions, targets, sources = [], [], []
    rows = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Global 600-second observational wall cutoff")
                if any(field in batch for field in FORBIDDEN_MODEL_FIELDS):
                    raise RuntimeError("Forbidden geometry reached model batch")
                predictions.append((_forward(model, batch) * STD + MEAN).numpy())
                targets.append(batch.y.view(-1).numpy())
                sources.append(batch.source_idx.view(-1).numpy())
                for layer, values in hooks.latest.items():
                    for key, value in values.items():
                        collected.setdefault(f"layer{layer}_{key}", []).append(value)
                rows += batch.num_graphs
                atomic_json(output / f"{name}_progress.json", {
                    "status": "running", "rows": rows, "wall_seconds": time.perf_counter() - started,
                    "process_cpu_seconds": time.process_time() - cpu_started,
                })
    finally:
        hooks.remove()
    arrays = {key: np.concatenate(values) for key, values in collected.items()}
    arrays.update(development_offset=offsets, source_idx=np.concatenate(sources),
                  prediction_eV=np.concatenate(predictions), target_eV=np.concatenate(targets))
    if not np.array_equal(arrays["source_idx"], offsets + 100000):
        raise RuntimeError("Inference source order changed")
    if not np.array_equal(arrays["target_eV"], saved["target_eV"].numpy()[offsets]):
        raise RuntimeError("Inference target identity changed")
    delta = np.abs(arrays["prediction_eV"] - saved["prediction_eV"].numpy()[offsets])
    finite = all(np.isfinite(value).all() for value in arrays.values())
    equivalent = finite and float(delta.max()) <= TOLERANCE_EV
    arrays["saved_prediction_eV"] = saved["prediction_eV"].numpy()[offsets]
    arrays["prediction_abs_delta_eV"] = delta
    write_npz(output / f"{name}_rows.npz", arrays)
    row_values = [{key: value[i].item() for key, value in arrays.items()} for i in range(rows)]
    atomic_json(output / f"{name}_rows.json", {"arm": name, "rows": row_values})
    report = {
        "arm": name, "status": "complete" if equivalent else "EQUIVALENCE_FAILED",
        "interpretation_permitted": equivalent, "rows": rows, "parameters": parameter_count,
        "checkpoint_sha256": sha256_file(checkpoint), "predictions_sha256": sha256_file(saved_path),
        "checkpoint_context": selected["context"], "checkpoint_epoch": selected["epoch"],
        "checkpoint_weights": selected["weights"], "strict_state_load": True,
        "prediction_max_abs_delta_eV": float(delta.max()), "prediction_mean_abs_delta_eV": float(delta.mean()),
        "equivalence_tolerance_eV": TOLERANCE_EV, "finite": finite,
        "wall_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu_started,
    }
    atomic_json(output / f"{name}_report.json", report)
    atomic_json(output / f"{name}_progress.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Frozen cpu_source/src directory used for PYTHONPATH")
    args = parser.parse_args()
    import molgap
    if not Path(molgap.__file__).resolve().is_relative_to(args.source_root.resolve()):
        raise RuntimeError("molgap was not imported from the frozen source root")
    args.output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    deadline = started + 600
    configure_fp32_determinism(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    offsets = np.sort(np.random.default_rng(SEED).choice(50000, SAMPLE_ROWS, replace=False))
    reports = []
    try:
        development, shard_hashes = development_dataset(args.cache)
        loader = DataLoader(Subset(development, offsets.tolist()), batch_size=64,
                            shuffle=False, num_workers=0, drop_last=False)
        for name, width, root in (("reference", 192, args.reference), ("candidate", 256, args.candidate)):
            report = run_arm(name, width, root, loader, offsets, args.output, deadline)
            reports.append(report)
            if not report["interpretation_permitted"]:
                break
        status = "complete" if len(reports) == 2 and all(r["interpretation_permitted"] for r in reports) else "EQUIVALENCE_FAILED"
        atomic_json(args.output / "measurement.json", {
            "status": status, "interpretation_permitted": status == "complete", "arms": reports,
            "seed": SEED, "sample_rows": SAMPLE_ROWS, "batch_size": 64, "workers": 0, "cpu_threads": 4,
            "sample_offsets_sha256": hashlib.sha256(offsets.astype("<i8").tobytes()).hexdigest(),
            "development_shard_sha256": shard_hashes, "model_geometry_input": False,
            "source_root": str(args.source_root.resolve()),
            "measure_sha256": sha256_file(Path(__file__)),
            "wall_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu_started,
            "runtime": {"torch": torch.__version__, "numpy": np.__version__},
        })
        return 0 if status == "complete" else 2
    except Exception as error:
        atomic_json(args.output / "measurement.json", {
            "status": "STOP_FOR_COST" if isinstance(error, TimeoutError) else "FAILED",
            "interpretation_permitted": False, "error": f"{type(error).__name__}: {error}", "arms": reports,
            "wall_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu_started,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
