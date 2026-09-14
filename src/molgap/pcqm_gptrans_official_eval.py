"""Resumable standalone evaluation of the frozen full GPTrans bundle."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from .pcqm_gptrans_full_runner import TRAINING_CONTRACT, TRAINING_CONTRACT_SHA256, _gptrans_source_path
from .pcqm_k1_full_runner import _architecture_source_sha256
from .pcqm_k1_gptrans_fusion import (
    EXPECTED_VALID_ROWS, _accepted_valid_shards, _validate_fusion_contract,
    atomic_json, atomic_torch_save, calibration_mask, mae, sha256_file,
)
from .training_reproducibility import canonical_fingerprint, configure_fp32_determinism


class PackedGraphs(InMemoryDataset):
    def __init__(self, path):
        super().__init__(root=None)
        self.data, self.slices = torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def make_evaluation_model():
    from .gptrans import OGBGPTransTiny

    architecture = TRAINING_CONTRACT["architecture"]
    if _architecture_source_sha256(_gptrans_source_path()) != architecture["source_sha256_lf"]:
        raise RuntimeError("Frozen GPTrans architecture source changed")
    keys = ("node_channels", "pair_channels", "num_layers", "num_heads", "shortest_path_cap", "dropout", "drop_path", "layer_scale")
    model = OGBGPTransTiny(**{key: architecture[key] for key in keys}, n_targets=1)
    if sum(p.numel() for p in model.parameters()) != architecture["parameter_count"]:
        raise RuntimeError("GPTrans parameter count changed")
    # Inference replaces every parameter with the hash-verified trained state.
    return model


def check_part(part, *, identity, record, data):
    if part.get("identity") != identity or part.get("graph_sha256") != record["sha256"]:
        raise RuntimeError("Prediction identity changed")
    idx, target, pred = (part[k] for k in ("source_idx", "target_eV", "prediction_eV"))
    if any(t.ndim != 1 or len(t) != record["rows"] for t in (idx, target, pred)):
        raise RuntimeError("Prediction shape/count changed")
    if not torch.equal(idx, data.source_idx.view(-1).long()) or not torch.equal(target, data.y.view(-1).float()):
        raise RuntimeError("Prediction rows or targets differ from accepted graph")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise RuntimeError("Non-finite prediction/target")


def evaluate(*, model_root: Path, graph_root: Path, output: Path, accept_only=False, threads=8):
    torch.set_num_threads(threads)
    configure_fp32_determinism(42)
    base = json.loads((model_root / "acceptance.json").read_text())
    bundle_path = model_root / "model_bundle.pt"
    bundle_sha = sha256_file(bundle_path)
    if base.get("accepted") is not True or base.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256 or base.get("model_bundle_sha256") != bundle_sha:
        raise RuntimeError("An accepted matching GPTrans bundle is required")
    accepted, shards = _accepted_valid_shards(graph_root)
    graph_sha = sha256_file(graph_root / "acceptance.json")
    _validate_fusion_contract(accepted, graph_sha)
    identity = canonical_fingerprint({"bundle": bundle_sha, "graphs": graph_sha, "code": sha256_file(Path(__file__)), "precision": "fp32", "device": "cpu", "torch": torch.__version__, "threads": threads})
    output.mkdir(parents=True, exist_ok=True)
    model = None
    if not accept_only:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if bundle.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256:
            raise RuntimeError("Bundle contract changed")
        model = make_evaluation_model().eval()
        model.load_state_dict(bundle["state_dict"], strict=True)
        mean, std = (float(bundle["target_stats"][key]) for key in ("mean_eV", "sample_std_eV"))
    parts, hashes = [], {}
    started = time.monotonic()
    for record in shards:
        dataset = PackedGraphs(Path(record["path"]))
        part_path = output / f"valid_{record['shard_index']:04d}.pt"
        if part_path.exists():
            part = torch.load(part_path, map_location="cpu", weights_only=False)
        elif accept_only:
            raise FileNotFoundError(part_path)
        else:
            rows = {key: [] for key in ("source_idx", "target_eV", "prediction_eV")}
            with torch.inference_mode():
                for step, batch in enumerate(DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)):
                    prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1).float() * std + mean
                    rows["source_idx"].append(batch.source_idx.view(-1).long())
                    rows["target_eV"].append(batch.y.view(-1).float())
                    rows["prediction_eV"].append(prediction)
                    if step % 20 == 0:
                        progress = {"shard": record["shard_index"], "batch": step, "elapsed_seconds": time.monotonic()-started}
                        atomic_json(output / "progress.json", progress)
                        print(progress, flush=True)
            part = {"identity": identity, "graph_sha256": record["sha256"], **{k: torch.cat(v) for k, v in rows.items()}}
            check_part(part, identity=identity, record=record, data=dataset._data)
            atomic_torch_save(part_path, part)
        check_part(part, identity=identity, record=record, data=dataset._data)
        parts.append(part)
        hashes[part_path.name] = sha256_file(part_path)
        print(f"verified shard={record['shard_index']} rows={record['rows']}", flush=True)
    arrays = {k: torch.cat([p[k] for p in parts]).numpy() for k in ("source_idx", "target_eV", "prediction_eV")}
    calibration = calibration_mask(arrays["source_idx"])
    if len(arrays["source_idx"]) != EXPECTED_VALID_ROWS:
        raise RuntimeError("Official-valid coverage changed")
    result = {"accepted": bool(accept_only), "model_id": "gptrans_t_core", "model_bundle_sha256": bundle_sha, "graph_acceptance_sha256": graph_sha, "identity": identity, "official_valid_rows": EXPECTED_VALID_ROWS, "official_valid_gap_mae_eV": mae(arrays["prediction_eV"], arrays["target_eV"]), "calibration_gap_mae_eV": mae(arrays["prediction_eV"][calibration], arrays["target_eV"][calibration]), "holdout_gap_mae_eV": mae(arrays["prediction_eV"][~calibration], arrays["target_eV"][~calibration]), "prediction_sha256": hashes, "inference_device": "cpu", "precision": "fp32", "parameters_fitted": 0, "official_validation_role_read": True, "test_dev_role_read": False, "test_challenge_role_read": False, "model_inference_executed": not accept_only}
    if not accept_only:
        atomic_json(output / "metrics.json", result)
        atomic_json(output / "completion_manifest.json", {"complete": True, "identity": identity, "metrics_sha256": sha256_file(output / "metrics.json"), "prediction_sha256": hashes})
    else:
        manifest = json.loads((output / "completion_manifest.json").read_text())
        if manifest.get("identity") != identity or manifest.get("metrics_sha256") != sha256_file(output / "metrics.json") or manifest.get("prediction_sha256") != hashes:
            raise RuntimeError("Evaluation completion manifest changed")
        metrics = json.loads((output / "metrics.json").read_text())
        for key in ("official_valid_gap_mae_eV", "calibration_gap_mae_eV", "holdout_gap_mae_eV"):
            if result[key] != metrics.get(key):
                raise RuntimeError("Persisted metric differs from recomputed predictions")
        atomic_json(output / "acceptance.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ("model-root", "graph-root", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--accept-only", action="store_true")
    args = p.parse_args()
    evaluate(model_root=args.model_root, graph_root=args.graph_root, output=args.output, accept_only=args.accept_only)


if __name__ == "__main__":
    main()
