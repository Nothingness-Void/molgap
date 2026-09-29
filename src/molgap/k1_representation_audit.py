"""Bounded frozen K1 representation audit; remote accelerator execution only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .pcqm_k1_cross_scale_diagnostic import (
    ARMS, TRANSFORM_SHA256, _accepted_development, _forward, _model, _payload,
)
from .pcqm_k1_causal_audit import _atomic_json, _descriptors
from .pcqm_k1_explainability import sha256_file
from .representation_diagnostics import exchange_summary, spectrum_summary

MODEL500 = "7ab0deb5f8ae4073e5a5447b437e5d0d5d661d119ea292c8bc2734e272841262"
PAYLOAD500 = "e43728a30a74a19e9e969cef3e94f3c0f9718a1c606bfa5306bd9b3c87ca8d31"
TRANSFORM500 = "66987dd265f8c4a4f4216703e9b89b465cdcac7299044303ac5229edf33ccd57"


def panel_ids():
    return sorted(sorted(range(500000, 550000), key=lambda i: (
        hashlib.sha256(f"k1-representation-v1:{i}".encode("ascii")).hexdigest(), i
    ))[:1024])


def _checked(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f"Input hash mismatch: {path.name}")


def _instrument(model, batch, std, mean):
    import torch
    captured, bonds, handles = {}, {}, []

    def capture(layer):
        def hook(module, inputs, output):
            # Introduce an activation leaf only where frozen parameters leave
            # the graph absent. This does not alter values or model weights.
            output.requires_grad_(True)
            captured[layer] = (inputs[0].detach(), output)
        return hook

    def capture_bond(layer):
        def hook(module, inputs, output):
            bonds[layer] = output.detach()
        return hook

    try:
        for layer in (3, 6, 9):
            handles.append(model.neural_atom_mixers[str(layer)].register_forward_hook(capture(layer)))
            handles.append(model.edge_updates[layer - 1].register_forward_hook(capture_bond(layer)))
        prediction = _forward(model, batch) * std + mean
        outputs = [captured[layer][1] for layer in (3, 6, 9)]
        isolated = torch.autograd.grad(prediction[0], outputs, retain_graph=True)
        for gradient in isolated:
            if gradient[batch.batch != 0].abs().max().item() > 1e-9:
                raise ValueError("Cross-molecule gradient coupling")
        gradients = torch.autograd.grad(prediction.sum(), outputs)
        arrays = {}
        for layer, gradient in zip((3, 6, 9), gradients):
            hidden, after = captured[layer]
            arrays[layer] = tuple(t.detach().cpu().numpy() for t in (
                hidden, after - hidden, gradient, bonds[layer]
            ))
        return prediction.detach(), arrays
    finally:
        for handle in handles:
            handle.remove()


def run(args):
    import numpy as np
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from .qm9_neural_atom import make_encoder
    from .qm9_neural_atom import _state_sha256

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Requires an allocated remote accelerator job")
    if len(args.source_commit) != 40 or len(args.source_archive_sha256) != 64:
        raise ValueError("Missing immutable source identity")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    ids = panel_ids()
    _atomic_json(out / "row_manifest.json", {"source_idx": ids, "selection": "sha256-label-independent-v1"})
    role_events, hashes, reproductions = [], {}, {}
    _atomic_json(out / "started.json", {"source_commit": args.source_commit,
        "source_archive_sha256": args.source_archive_sha256, "job_id": os.environ["SLURM_JOB_ID"],
        "training_executed": False, "model_inference_executed": True,
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False})

    for scale in (100, 500):
        if scale == 100:
            _checked(args.inputs100 / "target_transform.json", TRANSFORM_SHA256)
            transform = json.loads((args.inputs100 / "target_transform.json").read_text())
            model = _model("k1", args.inputs100 / "k1_best_model.pt")
            payload = _payload("k1", args.inputs100 / "k1_best_development_payload.pt")
            root, role, start = args.cache100, "original_100k", 100000
        else:
            for name, digest in (("best_model.pt", MODEL500), ("best_predictions.pt", PAYLOAD500),
                                 ("target_transform.json", TRANSFORM500)):
                _checked(args.inputs500 / name, digest)
            transform = json.loads((args.inputs500 / "target_transform.json").read_text())
            saved = torch.load(args.inputs500 / "best_model.pt", map_location="cpu", weights_only=False)
            if saved["mean"] != transform["mean"] or saved["std"] != transform["std"]:
                raise ValueError("500K checkpoint transform differs")
            model = make_encoder("neural_atom_k1").to("cuda").eval()
            model.load_state_dict(saved["model"], strict=True)
            payload = torch.load(args.inputs500 / "best_predictions.pt", map_location="cpu", weights_only=False)
            payload = {"source_idx": payload["source_idx"], "prediction_eV": payload["prediction"],
                       "target_eV": payload["target"]}
            root, role, start = args.cache500, "unseen_500k", 500000
        if sum(p.numel() for p in model.parameters()) != 3658817:
            raise ValueError("Architecture identity differs")
        model.requires_grad_(False)
        initial = _state_sha256(model)
        mean, std = float(transform["mean"]), float(transform["std"])
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError("Nonfinite transform")
        graphs = _accepted_development(root, role)
        role_events.append({"scale": scale, "role": role, "purpose": "prediction-reproduction", "rows": 50000})
        _atomic_json(out / "role_events.json", role_events)
        predicted, targets = [], []
        if not torch.equal(payload["source_idx"].view(-1).long(), torch.arange(start, start + 50000)):
            raise ValueError("Reference payload row identity mismatch")
        with torch.no_grad():
            for batch in DataLoader(graphs, batch_size=128, shuffle=False, num_workers=0):
                targets.append(batch.y.view(-1).float())
                predicted.append((_forward(model, batch.to("cuda")) * std + mean).cpu())
        pred, target = torch.cat(predicted), torch.cat(targets)
        reference = payload["prediction_eV"].view(-1).float()
        if not torch.equal(target, payload["target_eV"].view(-1).float()) or not torch.isfinite(pred).all():
            raise ValueError("Reference targets or predictions invalid")
        max_diff = float((pred - reference).abs().max())
        mae_diff = float(((pred-target).abs().mean() - (reference-target).abs().mean()).abs())
        if max_diff > .001 or mae_diff > .0001:
            raise ValueError("Reference reproduction failed")
        reproductions[str(scale)] = {"max_abs_eV": max_diff, "mae_difference_eV": mae_diff}
        _atomic_json(out / "reproduction.json", reproductions)
        del graphs
        graphs = _accepted_development(args.cache500, "unseen_500k")
        role_events.append({"scale": scale, "role": "unseen_500k", "purpose": "representation-panel", "rows": len(ids)})
        _atomic_json(out / "role_events.json", role_events)
        loader = DataLoader(Subset(graphs, [i-500000 for i in ids]), batch_size=128, shuffle=False, num_workers=0)
        for chunk, batch in enumerate(loader):
            descriptors = {k: v.tolist() for k, v in _descriptors(batch).items()}
            graph_index = batch.batch.numpy()
            edge_graph = graph_index[batch.edge_index[0].numpy()]
            batch = batch.to("cuda")
            with torch.no_grad():
                baseline = _forward(model, batch) * std + mean
            hooked, arrays = _instrument(model, batch, std, mean)
            if not torch.isfinite(hooked).all() or float((hooked-baseline).abs().max()) > 1e-6:
                raise ValueError("Hook changed predictions")
            rows = []
            for i, source_idx in enumerate(batch.source_idx.view(-1).tolist()):
                layers = {}
                for layer, (h, u, g, e) in arrays.items():
                    layers[str(layer)] = exchange_summary(h[graph_index == i], u[graph_index == i], g[graph_index == i])
                    edges = e[edge_graph == i]
                    layers[str(layer)]["bond_spectrum"] = spectrum_summary(edges) if len(edges) else None
                rows.append({"source_idx": source_idx, "target_eV": float(batch.y.view(-1)[i]),
                    "prediction_eV": float(hooked[i]), "descriptors": {k: v[i] for k,v in descriptors.items()}, "layers": layers})
            name = f"scale{scale}_chunk{chunk:02d}.json"
            _atomic_json(out / name, rows)
            hashes[name] = sha256_file(out / name)
            _atomic_json(out / "progress.json", {"complete": False, "artifact_sha256": hashes})
            print(f"scale={scale} chunk={chunk} rows={len(rows)}", flush=True)
        if _state_sha256(model) != initial:
            raise ValueError("Frozen model state mutated")
        del model, graphs
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    for name in ("row_manifest.json", "started.json", "role_events.json", "reproduction.json"):
        hashes[name] = sha256_file(out / name)
    _atomic_json(out / "completion_manifest.json", {
        "complete": True, "training_executed": False, "model_inference_executed": True,
        "experiment_purpose": "NO_TRAIN", "comparison_class": "CONTEXT_ONLY",
        "artifact_sha256": hashes, "source_commit": args.source_commit,
        "source_archive_sha256": args.source_archive_sha256, "panel_rows_per_scale": len(ids),
        "checkpoint_sha256": {"100": ARMS["k1"]["model_sha256"], "500": MODEL500},
        "wall_seconds": elapsed, "device_hours": elapsed/3600,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "device": torch.cuda.get_device_name(0), "physical_batch": 128,
        "precision": "fp32", "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False,
    })


def main():
    parser = argparse.ArgumentParser()
    for name in ("cache100", "cache500", "inputs100", "inputs500", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
