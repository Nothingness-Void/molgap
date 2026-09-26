"""Post-training, strictly NO_TRAIN portability audit for two K1 arms.

This is a separate logical role/action from the 100K screen. Checkpoint
reproduction on the original development role precedes any 500K role read.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from .pcqm_k1_cross_scale_diagnostic import (
    ARMS as FROZEN_REFERENCE,
    DEVELOPMENT,
    MANIFESTS,
    _accepted_development,
)
from .pcqm_k1_variants_runner import FORBIDDEN_MODEL_FIELDS
from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file


MODES = ("neural_atom_k1_rwse_refresh", "neural_atom_k1_degree_balance")
CHUNK_ROWS = 5_000
BATCH = 128
MAX_REPRODUCTION_ABS_EV = 0.001
TRANSFORM_SHA256 = "20e6730d57080b0bec9901a26a931034aad162848e0075940fd1e1273bf084b3"


def _graphs(root: Path, role: str):
    graphs = _accepted_development(root, role)
    for field in FORBIDDEN_MODEL_FIELDS:
        if field in graphs._data:
            del graphs._data[field]
            graphs.slices.pop(field, None)
    return graphs


def _model(mode: str, checkpoint: Path, expected_sha256: str):
    import torch
    from .pcqm_k1_variants import make_encoder

    if sha256_file(checkpoint) != expected_sha256:
        raise RuntimeError(f"Checkpoint hash changed: {mode}")
    model = make_encoder(mode)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True
    )
    return model.to("cuda").eval()


def _payload(path: Path, expected_sha256: str, role: str):
    import torch

    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Original development payload hash changed: {role}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    start, stop, _ = DEVELOPMENT["original_100k"]
    if not torch.equal(payload["source_idx"].view(-1).long(), torch.arange(start, stop)):
        raise RuntimeError(f"Original development source indices changed: {role}")
    for field in ("target_eV", "prediction_eV"):
        if payload[field].numel() != stop - start or not torch.isfinite(payload[field]).all():
            raise RuntimeError(f"Original payload invalid: {role}/{field}")
    return payload


def _infer(graphs, model, *, start: int, output: Path, role: str, mode: str,
           mean: float, std: float):
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in range(0, len(graphs), CHUNK_ROWS):
        count = min(CHUNK_ROWS, len(graphs) - offset)
        loader = DataLoader(
            Subset(graphs, range(offset, offset + count)), batch_size=BATCH,
            shuffle=False, drop_last=False, num_workers=0, pin_memory=True,
        )
        source, target, prediction = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to("cuda", non_blocking=True)
                value = model(
                    batch.x, batch.edge_index, batch.edge_attr,
                    batch.batch, batch.random_walk_pe,
                ).view(-1)
                prediction.append((value.float() * std + mean).cpu())
                target.append(batch.y.view(-1).float().cpu())
                source.append(batch.source_idx.view(-1).long().cpu())
        row = {
            "source_idx": torch.cat(source),
            "target_eV": torch.cat(target),
            "prediction_eV": torch.cat(prediction),
        }
        if not torch.equal(
            row["source_idx"], torch.arange(start + offset, start + offset + count)
        ) or not all(torch.isfinite(row[k]).all() for k in ("target_eV", "prediction_eV")):
            raise RuntimeError(f"Invalid role rows or finite values: {role}/{mode}/{offset}")
        chunk = output / f"chunk_{offset // CHUNK_ROWS:02d}.pt"
        atomic_torch_save(chunk, row)
        rows.append({"file": chunk.name, "sha256": sha256_file(chunk), "rows": count})
        print(f"audit {role}/{mode} {offset + count}/{len(graphs)}", flush=True)
    return rows


def _joined(output: Path, chunks: list[dict]):
    import torch

    parts = []
    for chunk in chunks:
        path = output / chunk["file"]
        if sha256_file(path) != chunk["sha256"]:
            raise RuntimeError(f"Audit chunk hash changed: {path}")
        parts.append(torch.load(path, map_location="cpu", weights_only=False))
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def run(
    *, cache_100k: Path, cache_500k: Path, reference_model: Path,
    reference_payload: Path, transform_asset: Path, candidate_root: Path,
    output: Path, source_commit: str, source_archive_sha256: str,
    modes: tuple[str, ...] = MODES,
):
    import torch
    from .pcqm_k1_variants import make_encoder

    if len(source_commit) != 40 or len(source_archive_sha256) != 64:
        raise RuntimeError("Unfrozen source identity")
    if sha256_file(transform_asset) != TRANSFORM_SHA256:
        raise RuntimeError("100K training-only transform bytes changed")
    transform = json.loads(transform_asset.read_text(encoding="utf-8"))
    if transform.get("ddof") != 1:
        raise RuntimeError("Target transform convention changed")
    mean, std = float(transform["mean"]), float(transform["std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise RuntimeError("Invalid target transform")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() != 1:
        raise RuntimeError("NO_TRAIN audit requires one visible GPU")
    output.mkdir(parents=True, exist_ok=True)
    frozen = {
        "neural_atom_k1_v4": {
            "checkpoint": reference_model,
            "checkpoint_sha256": FROZEN_REFERENCE["k1"]["model_sha256"],
            "payload": reference_payload,
            "payload_sha256": FROZEN_REFERENCE["k1"]["payload_sha256"],
        }
    }
    if not modes or len(modes) != len(set(modes)) or "neural_atom_k1_v4" in modes:
        raise ValueError("Audit must bind explicit distinct candidate modes")
    for mode in modes:
        root = candidate_root / mode
        record = json.loads((root / "arm_record.json").read_text(encoding="utf-8"))
        if (
            record.get("complete") is not True
            or record.get("fixed_manifest_sha256") != MANIFESTS["original_100k"]
            or record.get("source_commit") != source_commit
            or record.get("contract", {}).get("source_archive_sha256") != source_archive_sha256
            or record.get("training", {}).get("epochs_completed") != 40
            or record.get("official_validation_role_read") is not False
            or record.get("test_dev_role_read") is not False
        ):
            raise RuntimeError(f"Training terminal not accepted for audit: {mode}")
        frozen[mode] = {
            "checkpoint": root / "best_model.pt",
            "checkpoint_sha256": record["training"]["best_model_sha256"],
            "payload": root / "best_development_payload.pt",
            "payload_sha256": record["training"]["payload_sha256"],
        }
    started = time.monotonic()
    original = _graphs(cache_100k, "original_100k")
    reproduction = {}
    for mode, row in frozen.items():
        model = _model(mode, row["checkpoint"], row["checkpoint_sha256"])
        saved = _payload(row["payload"], row["payload_sha256"], mode)
        target = output / "original_100k" / mode
        chunks = _infer(original, model, start=100_000, output=target,
                        role="original_100k", mode=mode, mean=mean, std=std)
        joined = _joined(target, chunks)
        if not torch.equal(joined["target_eV"], saved["target_eV"].view(-1).float()):
            raise RuntimeError(f"Original targets differ: {mode}")
        diff = float((joined["prediction_eV"] - saved["prediction_eV"].view(-1).float()).abs().max())
        if diff > MAX_REPRODUCTION_ABS_EV:
            raise RuntimeError(f"Checkpoint reproduction failed: {mode} {diff}")
        reproduction[mode] = {"max_abs_eV": diff, "chunks": chunks}
        del model
    del original
    # The disjoint fixed-500K development role opens only after all reproductions.
    unseen = _graphs(cache_500k, "unseen_500k")
    audit = {}
    for mode, row in frozen.items():
        model = _model(mode, row["checkpoint"], row["checkpoint_sha256"])
        target = output / "unseen_500k" / mode
        chunks = _infer(unseen, model, start=500_000, output=target,
                        role="unseen_500k", mode=mode, mean=mean, std=std)
        joined = _joined(target, chunks)
        prediction = joined["prediction_eV"]
        mae = float((prediction - joined["target_eV"]).abs().mean())
        if not math.isfinite(mae):
            raise RuntimeError(f"Nonfinite 500K audit MAE: {mode}")
        audit[mode] = {"mae_eV": mae, "rows": len(joined["target_eV"]), "chunks": chunks}
        del model
    terminal = {
        "format": "molgap-k1-post100k-portability-audit-v1",
        "complete": True,
        "experiment_purpose": "NO_TRAIN",
        "training_executed_in_audit_stage": False,
        "model_inference_executed": True,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "manifest_sha256": MANIFESTS,
        "target_transform_sha256": TRANSFORM_SHA256,
        "reference_bundle_id": "reference-k1-v4-100k-s42-v5-recovered",
        "checkpoint_sha256": {mode: row["checkpoint_sha256"] for mode, row in frozen.items()},
        "reproduction": reproduction,
        "unseen_500k": audit,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output / "terminal.json", terminal)
    return terminal
