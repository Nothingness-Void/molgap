"""Frozen 100K K1/PairToken inference on accepted 100K and 500K development roles.

This is a NO_TRAIN diagnostic. It never constructs an optimizer or reads a
training shard. Checkpoints are reproduced on their original role before the
non-overlapping 500K development role is opened.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .pcqm_k1_causal_audit import _atomic_json, _descriptors
from .pcqm_k1_explainability import sha256_file
from .training_reproducibility import atomic_torch_save


MANIFESTS = {
    "original_100k": "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d",
    "unseen_500k": "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751",
}
DEVELOPMENT = {
    "original_100k": (100_000, 150_000, "f8c0d054d4794ce9a8887e6c1799806d89533f6ed3aeb93f3e838b7319ac842a"),
    "unseen_500k": (500_000, 550_000, "1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1"),
}
ARMS = {
    "k1": {
        "model_sha256": "53f9118f34a95e02e3f0d798a56389af53ff55739753b4208c9f53c36d116d95",
        "payload_sha256": "966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91",
        "parameter_count": 3_658_817,
        "original_mae_eV": 0.1413736343383789,
    },
    "pair_token": {
        "model_sha256": "ac8b576a3c7e44d50012a885e13e4c5904ad17c3574319796b13518efe98d1cc",
        "payload_sha256": "b67be3f9ba0c67ca0eb1cf7dffa00ad535e9f04f5f8371aadad1e6a2711e0331",
        "parameter_count": 3_681_665,
        "original_mae_eV": 0.13833005726337433,
    },
}
TRANSFORM_SHA256 = "20e6730d57080b0bec9901a26a931034aad162848e0075940fd1e1273bf084b3"
TARGET_TRANSFORM_ID = "a5ebd05f82f481020f2059c77a7bc7b9dee84081fdb9806fdc18d93a6b8d89d7"
CHUNK_ROWS = 5_000
PHYSICAL_BATCH = 128
REPRODUCTION_MAX_ABS_EV = 0.001
REPRODUCTION_MAE_TOLERANCE_EV = 0.0001


def _accepted_development(cache_root: Path, role: str):
    import torch
    from torch_geometric.data import InMemoryDataset
    from .pcqm_wedge import WedgeData  # noqa: F401 -- required for trusted graph pickle

    manifest_path = cache_root / "manifest.json"
    if sha256_file(manifest_path) != MANIFESTS[role]:
        raise RuntimeError(f"Fixed {role} manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(manifest.get(key) is not False for key in (
        "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"
    )):
        raise RuntimeError("Protected role flag is not false")
    start, stop, shard_sha = DEVELOPMENT[role]
    expected_role = manifest.get("roles", {}).get("development", {})
    if (expected_role.get("source_idx_start"), expected_role.get("source_idx_stop"), expected_role.get("rows")) != (start, stop, stop - start):
        raise RuntimeError(f"{role} development row identity changed")
    shards = [item for item in manifest["geometry_shards"] if item["role"] == "development"]
    if len(shards) != 1 or shards[0]["sha256"] != shard_sha or shards[0]["rows"] != stop - start:
        raise RuntimeError(f"{role} development shard identity changed")
    shard = cache_root / shards[0]["file"]
    if sha256_file(shard) != shard_sha:
        raise RuntimeError(f"{role} development shard bytes changed")

    class PackedGraphDataset(InMemoryDataset):
        def __init__(self, source: Path):
            super().__init__(root=None)
            self._data, self.slices = torch.load(source, map_location="cpu", weights_only=False)

    graphs = PackedGraphDataset(shard)
    if len(graphs) != stop - start or not torch.equal(
        graphs._data.source_idx.view(-1).long(), torch.arange(start, stop, dtype=torch.long)
    ):
        raise RuntimeError(f"{role} source indices changed")
    return graphs


def _model(arm: str, checkpoint: Path):
    import torch
    from .k1_pair_token import make_encoder as make_pair
    from .qm9_neural_atom import make_encoder as make_k1

    model = (make_k1("neural_atom_k1") if arm == "k1" else make_pair("neural_atom_k1_pair_token"))
    if sum(parameter.numel() for parameter in model.parameters()) != ARMS[arm]["parameter_count"]:
        raise RuntimeError(f"{arm} architecture parameter count changed")
    if sha256_file(checkpoint) != ARMS[arm]["model_sha256"]:
        raise RuntimeError(f"{arm} checkpoint bytes changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to("cuda").eval()


def _payload(arm: str, path: Path):
    import torch

    if sha256_file(path) != ARMS[arm]["payload_sha256"]:
        raise RuntimeError(f"{arm} original prediction payload bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("source_idx", "target_eV", "prediction_eV"):
        if key not in payload or payload[key].numel() != 50_000:
            raise RuntimeError(f"{arm} original payload missing {key}")
    return payload


def _forward(model, batch):
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe).view(-1)


def _identity(source_commit: str, source_archive_sha256: str):
    return {
        "format": "molgap-k1-cross-scale-frozen-inputs-v1",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "manifest_sha256": MANIFESTS,
        "checkpoint_sha256": {arm: row["model_sha256"] for arm, row in ARMS.items()},
        "payload_sha256": {arm: row["payload_sha256"] for arm, row in ARMS.items()},
        "target_transform_sha256": TRANSFORM_SHA256,
        "physical_batch": PHYSICAL_BATCH,
        "chunk_rows": CHUNK_ROWS,
    }


def _progress(output_root: Path, identity: dict):
    path = output_root / "progress.json"
    if not path.exists():
        if list(output_root.glob("*/chunks/chunk_*.pt")):
            raise RuntimeError("Unrecorded chunks require inspection")
        return {}
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous.get("identity") != identity:
        raise RuntimeError("Progress input identity differs")
    hashes = previous.get("chunk_sha256", {})
    for name, expected in hashes.items():
        if ".." in Path(name).parts or sha256_file(output_root / name) != expected:
            raise RuntimeError(f"Corrupt or missing recorded chunk: {name}")
    seen = {path.relative_to(output_root).as_posix() for path in output_root.glob("*/chunks/chunk_*.pt")}
    if seen != set(hashes):
        raise RuntimeError("Unrecorded chunks require inspection")
    return hashes


def run_diagnostic(
    *, cache_100k: Path, cache_500k: Path, model_paths: dict[str, Path],
    payload_paths: dict[str, Path], transform_path: Path, output_root: Path,
    source_commit: str, source_archive_sha256: str, preflight_only: bool = False,
) -> dict:
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("An allocated DCU/CUDA device is required")
    if len(source_commit) != 40 or len(source_archive_sha256) != 64:
        raise RuntimeError("Incomplete source provenance")
    if sha256_file(transform_path) != TRANSFORM_SHA256:
        raise RuntimeError("Frozen 100K target transform bytes changed")
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    if transform.get("asset_id") != TARGET_TRANSFORM_ID or transform.get("ddof") != 1:
        raise RuntimeError("Frozen target transform identity changed")
    mean, std = float(transform["mean"]), float(transform["std"])
    if not (math.isfinite(mean) and math.isfinite(std) and std > 0):
        raise RuntimeError("Invalid target transform")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    identity = _identity(source_commit, source_archive_sha256)
    output_root.mkdir(parents=True, exist_ok=True)
    hashes = _progress(output_root, identity)
    started = time.monotonic()
    records = {}
    for role, root in (("original_100k", cache_100k), ("unseen_500k", cache_500k)):
        graphs = _accepted_development(root, role)
        start, stop, _ = DEVELOPMENT[role]
        for arm in ("k1", "pair_token"):
            model = _model(arm, model_paths[arm])
            payload = _payload(arm, payload_paths[arm])
            if role == "original_100k" and not torch.equal(
                payload["source_idx"].view(-1).long(), torch.arange(start, stop)
            ):
                raise RuntimeError(f"{arm} original payload source indices changed")
            offset_range = range(0, PHYSICAL_BATCH if preflight_only else stop - start, CHUNK_ROWS)
            arm_role = f"{role}/{arm}"
            rows = []
            with torch.no_grad():
                for offset in offset_range:
                    chunk_name = f"{arm_role}/chunks/chunk_{offset // CHUNK_ROWS:02d}.pt"
                    if chunk_name in hashes and not preflight_only:
                        rows.append(torch.load(output_root / chunk_name, map_location="cpu", weights_only=False))
                        continue
                    length = PHYSICAL_BATCH if preflight_only else CHUNK_ROWS
                    loader = DataLoader(
                        Subset(graphs, range(offset, offset + length)),
                        batch_size=PHYSICAL_BATCH, shuffle=False, drop_last=False,
                        num_workers=0, pin_memory=True,
                    )
                    parts = {key: [] for key in ("source_idx", "target_eV", "prediction_eV")}
                    descriptors = {}
                    for batch in loader:
                        for key, value in _descriptors(batch).items():
                            descriptors.setdefault(key, []).append(value.float())
                        batch = batch.to("cuda", non_blocking=True)
                        prediction = (_forward(model, batch) * std + mean).float().cpu()
                        parts["source_idx"].append(batch.source_idx.view(-1).long().cpu())
                        parts["target_eV"].append(batch.y.view(-1).float().cpu())
                        parts["prediction_eV"].append(prediction)
                    row = {key: torch.cat(value) for key, value in parts.items()}
                    row["descriptors"] = {key: torch.cat(value) for key, value in descriptors.items()}
                    if not torch.equal(row["source_idx"], torch.arange(start + offset, start + offset + length)):
                        raise RuntimeError(f"Row order changed: {chunk_name}")
                    if not bool(torch.isfinite(row["prediction_eV"]).all()):
                        raise RuntimeError(f"Nonfinite prediction: {chunk_name}")
                    rows.append(row)
                    if not preflight_only:
                        chunk_path = output_root / chunk_name
                        chunk_path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_torch_save(chunk_path, row)
                        hashes[chunk_name] = sha256_file(chunk_path)
                        _atomic_json(output_root / "progress.json", {
                            "identity": identity, "chunk_sha256": hashes, "complete": False,
                        })
                        print(f"{arm_role} {offset + CHUNK_ROWS}/{stop - start}", flush=True)
            source_idx = torch.cat([row["source_idx"] for row in rows])
            target = torch.cat([row["target_eV"] for row in rows])
            prediction = torch.cat([row["prediction_eV"] for row in rows])
            record = {"rows": len(target), "mae_eV": float((prediction - target).abs().mean())}
            if role == "original_100k":
                expected_target = payload["target_eV"].view(-1).float()[:len(target)]
                expected_pred = payload["prediction_eV"].view(-1).float()[:len(target)]
                if not torch.equal(target, expected_target):
                    raise RuntimeError(f"{arm} original target mismatch")
                max_diff = float((prediction - expected_pred).abs().max())
                record["reproduction_max_abs_eV"] = max_diff
                if max_diff > REPRODUCTION_MAX_ABS_EV:
                    raise RuntimeError(f"{arm} original checkpoint failed prediction reproduction: {max_diff}")
                if not preflight_only and abs(record["mae_eV"] - ARMS[arm]["original_mae_eV"]) > REPRODUCTION_MAE_TOLERANCE_EV:
                    raise RuntimeError(f"{arm} original checkpoint failed MAE reproduction")
            records[arm_role] = record
            del model
        del graphs
        if preflight_only:
            break
    terminal = {
        "format": "molgap-k1-cross-scale-frozen-diagnostic-v1",
        "complete": not preflight_only,
        "preflight_only": preflight_only,
        "training_executed": False,
        "model_inference_executed": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "identity": identity,
        "records": records,
        "chunk_sha256": hashes,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output_root / ("preflight.json" if preflight_only else "terminal.json"), terminal)
    if not preflight_only:
        _atomic_json(output_root / "progress.json", {
            "identity": identity, "chunk_sha256": hashes, "complete": True,
        })
    return terminal
