"""Frozen 500K PairToken intervention audit on the accepted internal development role.

No weights are updated. Small independently retrievable chunks make an interrupted
inference job resumable without trusting a transient worker filesystem.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .pcqm_k1_causal_audit import _atomic_json, _descriptors, _summarize
from .pcqm_k1_explainability import sha256_file
from .pcqm_k1_pair_token_audit import MODES, _forward
from .pcqm_k1_pair_token_500k import PARAMETERS, make_model
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from .pcqm_k1_scale_runner import find_cache
from .training_reproducibility import atomic_torch_save


CHECKPOINT_SHA256 = "cd428d17736b4c6af7595faac340e54f70b0506d6f8daa45405985c20e9ef105"
PAYLOAD_SHA256 = "35fdaa76f10e166ccb123b6005929a420f18b2678a286023d4aa62e48ec9ca01"
DEVELOPMENT_SHA256 = "1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1"
DEVELOPMENT_ROWS = 50_000
CHUNK_ROWS = 5_000
PHYSICAL_BATCH = 128


def _check_inputs(cache_root: Path, checkpoint: Path, payload: Path) -> tuple[dict, dict]:
    import torch

    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Frozen 500K PairToken checkpoint hash changed")
    if sha256_file(payload) != PAYLOAD_SHA256:
        raise RuntimeError("Frozen 500K PairToken prediction payload hash changed")
    found_root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
    if found_root.resolve() != cache_root.resolve():
        raise RuntimeError("Fixed 500K cache root is not the selected cache")
    development = [s for s in manifest["geometry_shards"] if s["role"] == "development"]
    if len(development) != 1 or development[0]["sha256"] != DEVELOPMENT_SHA256:
        raise RuntimeError("Development shard identity changed")
    shard = cache_root / development[0]["file"]
    if sha256_file(shard) != DEVELOPMENT_SHA256:
        raise RuntimeError("Development shard bytes changed")
    for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if manifest.get(role) is not False:
            raise RuntimeError(f"Protected role was read: {role}")
    frozen = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if frozen.get("mode") != "neural_atom_k1_pair_token" or frozen.get("epoch") != 48:
        raise RuntimeError("Unexpected checkpoint model or selected epoch")
    contract = frozen.get("contract", {})
    for key, expected in {
        "benchmark_id": "pcqm-fixed500k-dev50k-matched60-v4",
        "data_role_fingerprint": FIXED_500K_MANIFEST_SHA256,
        "seed": 42,
        "precision": "fp32",
        "physical_batch_per_device": PHYSICAL_BATCH,
        "sample_exposure": 29_998_080,
    }.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"Frozen checkpoint contract mismatch: {key}")
    return manifest, frozen


def _load_development(path: Path):
    import torch
    from torch_geometric.data import InMemoryDataset
    from .pcqm_wedge import WedgeData  # noqa: F401 -- pickled graph dependency

    class PackedGraphDataset(InMemoryDataset):
        def __init__(self, source: Path):
            super().__init__(root=None)
            self._data, self.slices = torch.load(source, map_location="cpu", weights_only=False)

    dataset = PackedGraphDataset(path)
    if len(dataset) != DEVELOPMENT_ROWS:
        raise RuntimeError("Development graph count changed")
    expected = torch.arange(500_000, 550_000, dtype=torch.long)
    if not torch.equal(dataset._data.source_idx.view(-1).long(), expected):
        raise RuntimeError("Development source indices changed")
    return dataset


def _identity() -> dict:
    return {
        "cache_manifest_sha256": FIXED_500K_MANIFEST_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "payload_sha256": PAYLOAD_SHA256,
        "development_shard_sha256": DEVELOPMENT_SHA256,
        "modes": list(MODES),
        "chunk_rows": CHUNK_ROWS,
    }


def _existing_chunks(output_root: Path) -> dict[str, str]:
    path = output_root / "progress.json"
    if not path.exists():
        if list((output_root / "chunks").glob("chunk_*.pt")):
            raise RuntimeError("Unrecorded audit chunks require inspection")
        return {}
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous.get("identity") != _identity():
        raise RuntimeError("Audit progress belongs to another input identity")
    hashes = previous.get("chunk_sha256", {})
    for name, expected_sha in hashes.items():
        if name not in {f"chunk_{i:02d}.pt" for i in range(DEVELOPMENT_ROWS // CHUNK_ROWS)}:
            raise RuntimeError("Unexpected chunk in audit progress")
        if sha256_file(output_root / "chunks" / name) != expected_sha:
            raise RuntimeError(f"Audit chunk changed: {name}")
    unexpected = {p.name for p in (output_root / "chunks").glob("chunk_*.pt")} - set(hashes)
    if unexpected:
        raise RuntimeError(f"Unrecorded audit chunks require inspection: {sorted(unexpected)}")
    return hashes


def run_audit(cache_root: Path, checkpoint: Path, payload: Path,
              output_root: Path, *, batch_size: int = PHYSICAL_BATCH,
              audit_source_commit: str, source_archive_sha256: str) -> dict:
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    if batch_size != PHYSICAL_BATCH:
        raise RuntimeError("Frozen audit requires physical batch 128")
    if len(audit_source_commit) != 40 or len(source_archive_sha256) != 64:
        raise RuntimeError("Audit source provenance is incomplete")
    manifest, frozen = _check_inputs(cache_root, checkpoint, payload)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = make_model().to("cuda").eval()
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise RuntimeError("Parameter count changed")
    model.load_state_dict(frozen["model"], strict=True)
    mean, std = float(frozen["mean"]), float(frozen["std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise RuntimeError("Invalid frozen target normalization")
    dev_item = next(s for s in manifest["geometry_shards"] if s["role"] == "development")
    development = _load_development(cache_root / dev_item["file"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "chunks").mkdir(exist_ok=True)
    chunk_hashes = _existing_chunks(output_root)
    started = time.monotonic()
    with torch.no_grad():
        for offset in range(0, DEVELOPMENT_ROWS, CHUNK_ROWS):
            name = f"chunk_{offset // CHUNK_ROWS:02d}.pt"
            if name in chunk_hashes:
                continue
            loader = DataLoader(
                Subset(development, range(offset, offset + CHUNK_ROWS)),
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0, pin_memory=True,
            )
            parts = {mode: [] for mode in MODES}
            targets, source_indices, descriptors = [], [], {}
            for graph_batch in loader:
                graph_batch = graph_batch.to("cuda", non_blocking=True)
                for key, value in _descriptors(graph_batch).items():
                    descriptors.setdefault(key, []).append(value.float().cpu())
                for mode in MODES:
                    normalized = _forward(model, graph_batch, mode)
                    parts[mode].append((normalized * std + mean).float().cpu())
                targets.append(graph_batch.y.view(-1).float().cpu())
                source_indices.append(graph_batch.source_idx.view(-1).long().cpu())
            rows = {
                "source_idx": torch.cat(source_indices),
                "target": torch.cat(targets),
                "predictions": {key: torch.cat(value) for key, value in parts.items()},
                "descriptors": {key: torch.cat(value) for key, value in descriptors.items()},
            }
            expected = torch.arange(500_000 + offset, 500_000 + offset + CHUNK_ROWS)
            if not torch.equal(rows["source_idx"], expected):
                raise RuntimeError(f"Chunk order changed: {name}")
            for prediction in rows["predictions"].values():
                if not bool(torch.isfinite(prediction).all()):
                    raise RuntimeError(f"Non-finite intervention prediction: {name}")
            chunk_path = output_root / "chunks" / name
            atomic_torch_save(chunk_path, rows)
            chunk_hashes[name] = sha256_file(chunk_path)
            _atomic_json(output_root / "progress.json", {
                "format": "molgap-pair-token-scale-audit-progress-v1",
                "identity": _identity(), "chunk_sha256": chunk_hashes,
                "complete": False,
            })
            print(f"audit {name} rows={offset + CHUNK_ROWS}/{DEVELOPMENT_ROWS}", flush=True)

    ordered = []
    for index in range(DEVELOPMENT_ROWS // CHUNK_ROWS):
        name = f"chunk_{index:02d}.pt"
        path = output_root / "chunks" / name
        if sha256_file(path) != chunk_hashes.get(name):
            raise RuntimeError(f"Chunk missing or corrupt: {name}")
        ordered.append(torch.load(path, map_location="cpu", weights_only=False))
    source_idx = torch.cat([row["source_idx"] for row in ordered])
    target = torch.cat([row["target"] for row in ordered])
    if not torch.equal(source_idx, torch.arange(500_000, 550_000)):
        raise RuntimeError("Final row order changed")
    reference = torch.load(payload, map_location="cpu", weights_only=False)
    if not torch.equal(source_idx, reference["source_idx"].view(-1).long()):
        raise RuntimeError("Accepted payload row identity changed")
    if not torch.equal(target, reference["target"].view(-1).float()):
        raise RuntimeError("Accepted payload targets changed")
    frozen_pred = reference["prediction"].view(-1).float().numpy().astype(np.float64)
    predictions = {
        mode: torch.cat([row["predictions"][mode] for row in ordered]).numpy().astype(np.float64)
        for mode in MODES
    }
    descriptors = {
        key: torch.cat([row["descriptors"][key] for row in ordered]).numpy().astype(np.float64)
        for key in ordered[0]["descriptors"]
    }
    target_np = target.numpy().astype(np.float64)
    difference = np.abs(predictions["baseline"] - frozen_pred)
    base_mae = float(np.abs(predictions["baseline"] - target_np).mean())
    frozen_mae = float(np.abs(frozen_pred - target_np).mean())
    if float(difference.max()) > 5e-4 or abs(base_mae - frozen_mae) > 5e-6:
        raise RuntimeError("Frozen baseline did not reproduce accepted 500K predictions")
    interventions, _ = _summarize(target_np, predictions, descriptors, {})
    result = {
        "format": "molgap-pcqm-k1-pair-token-scale-audit-v1",
        "complete": True,
        "training_executed": False,
        "model_inference_executed": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "development_rows": DEVELOPMENT_ROWS,
        "batch_size": batch_size,
        "parameter_count": PARAMETERS,
        "selected_epoch": frozen["epoch"],
        "identity": _identity(),
        "audit_source_commit": audit_source_commit,
        "source_archive_sha256": source_archive_sha256,
        "chunk_sha256": chunk_hashes,
        "seconds_this_attempt": time.monotonic() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "baseline_reproduction": {
            "frozen_mae_eV": frozen_mae,
            "audit_mae_eV": base_mae,
            "max_prediction_absolute_difference_eV": float(difference.max()),
        },
        "interventions": interventions,
    }
    _atomic_json(output_root / "causal_audit.json", result)
    _atomic_json(output_root / "completion_manifest.json", {
        "complete": True, "identity": _identity(),
        "causal_audit_sha256": sha256_file(output_root / "causal_audit.json"),
        "chunk_sha256": chunk_hashes,
    })
    _atomic_json(output_root / "progress.json", {
        "format": "molgap-pair-token-scale-audit-progress-v1",
        "identity": _identity(), "chunk_sha256": chunk_hashes,
        "complete": True,
    })
    return result
