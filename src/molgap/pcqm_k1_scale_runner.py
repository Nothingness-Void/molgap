"""Paired T4x2 training for the frozen K1 PCQM 500K scale bridge."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .pcqm_gap_data import sha256_file
from .pcqm_k1_scale import SCALE_TRAIN_ROWS, VALIDATION_ROWS
from .screen_policy import validate_paired_screen_contract, validate_screen_arm


SEED = 42
BATCH_SIZE = 128
EPOCHS = 40
MIN_GAIN_EV = 0.0047308445
EXPECTED_PARAMETERS = {"full_gps": 4_771_073, "neural_atom_k1": 3_658_817}
TASK_ID = "pcqm-k1-scale500k-s42-v1"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_cache(expected_sha256: str) -> tuple[Path, dict]:
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("format") == "molgap-pcqm-k1-scale500k-cache-v1":
            candidates.append((path.parent, payload))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one scale cache, found {candidates}")
    root, manifest = candidates[0]
    if manifest.get("aggregate_sha256") != expected_sha256:
        raise RuntimeError("Scale-cache aggregate identity changed")
    for key in ("official_validation_role_read", "test_dev_role_read", "shadow_labels_read"):
        if manifest.get(key) is not False:
            raise RuntimeError(f"Sealed role changed: {key}")
    return root, manifest


def load_roles(root: Path, manifest: dict) -> dict[str, list]:
    import torch

    roles = {"train": [], "validation": []}
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Scale shard changed: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if len(payload) != shard["graph_count"]:
            raise RuntimeError(f"Scale shard count changed: {path.name}")
        roles[shard["role"]].extend(payload)
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Scale-cache aggregate recomputation changed")
    if len(roles["train"]) != SCALE_TRAIN_ROWS or len(roles["validation"]) != VALIDATION_ROWS:
        raise RuntimeError("Scale role counts changed")
    return roles


def _contract(arm: str, accelerator: str, cache_sha256: str) -> dict:
    return {
        "arm": arm,
        "task_id": TASK_ID,
        "platform_id": "kaggle2",
        "accelerator": accelerator,
        "data_role_fingerprint": cache_sha256,
        "seed": SEED,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-lr4e-4-wd1e-5-clip1",
        "schedule_fingerprint": "cosine40-eta1e-6",
        "sample_exposure": "pcqm-official-train-derived500000-gap40",
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }


def run_worker(arm: str, output: Path, *, source_commit: str, cache_sha256: str) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from .qm9_gape import forward_gap, set_seed, train_gap
    from .qm9_neural_atom import make_encoder

    if arm not in EXPECTED_PARAMETERS:
        raise ValueError(arm)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each scale worker requires one visible T4")
    validate_screen_arm(physical_batch_per_device=BATCH_SIZE)
    torch.backends.cuda.matmul.allow_tf32 = False
    root, manifest = find_cache(cache_sha256)
    roles = load_roles(root, manifest)
    set_seed(SEED)
    model = make_encoder(arm).to("cuda")
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError(f"{arm} parameter count changed: {parameters}")
    torch.cuda.reset_peak_memory_stats()
    batch = next(iter(DataLoader(roles["train"][:BATCH_SIZE], batch_size=BATCH_SIZE))).to("cuda")
    loss = functional.l1_loss(forward_gap(model, batch, augmented=False), batch.y.view(-1))
    loss.backward()
    peak = int(torch.cuda.max_memory_allocated())
    total = int(torch.cuda.get_device_properties(0).total_memory)
    reserve = 1.0 - peak / total
    if not bool(torch.isfinite(loss)) or reserve < 0.15:
        raise RuntimeError(f"{arm} batch-128 preflight failed")
    del model, batch, loss
    torch.cuda.empty_cache()
    set_seed(SEED)
    training = train_gap(
        make_encoder(arm),
        roles,
        output / arm,
        augmented=False,
        source_commit=source_commit,
        cache_sha256=cache_sha256,
    )
    result = {
        "arm": arm,
        "preflight": {
            "physical_batch_per_device": BATCH_SIZE,
            "finite_forward_backward": True,
            "peak_memory_mib": peak / 1024**2,
            "total_memory_mib": total / 1024**2,
            "memory_reserve_fraction": reserve,
        },
        "training": training,
        "contract": _contract(arm, torch.cuda.get_device_name(0), cache_sha256),
    }
    atomic_json(output / f"{arm}_worker.json", result)
    return result


def aggregate(output: Path, *, source_commit: str, cache_sha256: str) -> dict:
    import numpy as np
    import torch

    from .pcqm_k1_shadow_audit import paired_bootstrap

    results = {
        arm: json.loads((output / f"{arm}_worker.json").read_text(encoding="utf-8"))
        for arm in EXPECTED_PARAMETERS
    }
    comparability = validate_paired_screen_contract([item["contract"] for item in results.values()])
    payloads = {
        arm: torch.load(output / arm / "best_validation_payload.pt", map_location="cpu", weights_only=False)
        for arm in EXPECTED_PARAMETERS
    }
    target = payloads["full_gps"]["target_eV"].contiguous()
    if not torch.equal(target, payloads["neural_atom_k1"]["target_eV"].contiguous()):
        raise RuntimeError("Paired validation targets changed")
    errors = {
        arm: (payload["prediction_eV"].contiguous() - target).abs()
        for arm, payload in payloads.items()
    }
    mae = {arm: float(value.mean()) for arm, value in errors.items()}
    delta = (errors["neural_atom_k1"] - errors["full_gps"]).numpy().astype(np.float64)
    ci = paired_bootstrap(delta)
    gain = mae["full_gps"] - mae["neural_atom_k1"]
    passed = gain >= MIN_GAIN_EV and ci[1] < 0.0
    summary = {
        "format": "molgap-pcqm-k1-scale500k-result-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "results": results,
        "comparability": comparability,
        "validation_gap_mae_eV": mae,
        "paired_gain_full_minus_k1_eV": gain,
        "paired_bootstrap_95_ci_k1_minus_full_eV": list(ci),
        "minimum_gain_eV": MIN_GAIN_EV,
        "scale_gate_passed": passed,
        "model_inference_executed_by_acceptance": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "shadow_labels_read": False,
    }
    atomic_json(output / "metrics.json", summary)
    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "completion_manifest.json"
    }
    atomic_json(output / "completion_manifest.json", {**summary, "artifact_sha256": artifacts})
    return summary
