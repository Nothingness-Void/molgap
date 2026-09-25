"""Accept frozen local GPTrans checkpoints and compare paired internal roles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def paired_metrics(reference, candidate, start: int, stop: int) -> dict:
    import numpy as np
    import torch

    source_idx = torch.arange(start, stop, dtype=torch.long)
    for name, record in (("reference", reference), ("candidate", candidate)):
        if not torch.equal(record["source_idx"].view(-1).long(), source_idx):
            raise RuntimeError(f"{name} source-index order is not the frozen role")
        if len(record["prediction_eV"].view(-1)) != stop - start:
            raise RuntimeError(f"{name} prediction count changed")
        if not bool(torch.isfinite(record["prediction_eV"]).all()):
            raise RuntimeError(f"{name} predictions are non-finite")
        if not bool(torch.isfinite(record["target_eV"]).all()):
            raise RuntimeError(f"{name} targets are non-finite")
    if not torch.equal(reference["target_eV"].view(-1), candidate["target_eV"].view(-1)):
        raise RuntimeError("Paired targets differ")
    target = reference["target_eV"].view(-1).double().numpy()
    ref_error = np.abs(reference["prediction_eV"].view(-1).double().numpy() - target)
    candidate_error = np.abs(candidate["prediction_eV"].view(-1).double().numpy() - target)
    gain = ref_error - candidate_error
    rng = np.random.default_rng(42)
    bootstrap = np.empty(1000, dtype=np.float64)
    for index in range(len(bootstrap)):
        bootstrap[index] = gain[rng.integers(0, len(gain), len(gain))].mean()
    return {
        "rows": len(gain),
        "reference_mae_eV": float(ref_error.mean()),
        "candidate_mae_eV": float(candidate_error.mean()),
        "paired_gain_eV": float(gain.mean()),
        "row_bootstrap_95pct_eV": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "candidate_better_row_fraction": float((gain > 0).mean()),
        "material_3mev_point_gain": bool(gain.mean() >= 0.003),
    }


def transfer_predictions(cache_root: Path, model, mean: float, std: float) -> dict:
    import torch
    from molgap import pcqm_gptrans_v4 as v4

    manifest_path = cache_root / "manifest.json"
    expected_manifest = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
    if v4.sha256_file(manifest_path) != expected_manifest:
        raise RuntimeError("500K cache manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [item for item in manifest["geometry_shards"] if item["role"] == "development"]
    if len(entries) != 1 or (entries[0]["source_idx_min"], entries[0]["source_idx_max"]) != (500000, 549999):
        raise RuntimeError("500K internal-development role changed")
    path = cache_root / entries[0]["file"]
    if v4.sha256_file(path) != entries[0]["sha256"]:
        raise RuntimeError("500K development shard SHA256 changed")
    graphs, _ = v4._load_datasets((path,))
    predictions, targets, indices = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in v4._development_loader(graphs):
            batch = batch.to("cuda", non_blocking=True)
            predictions.append((v4._forward(model, batch) * std + mean).cpu().view(-1))
            targets.append(batch.y.view(-1).float().cpu())
            indices.append(batch.source_idx.view(-1).long().cpu())
    return {
        "prediction_eV": torch.cat(predictions),
        "target_eV": torch.cat(targets),
        "source_idx": torch.cat(indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cache-500k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from molgap import pcqm_gptrans_v4 as v4
    from molgap.noisy_nodes import make_noisy_nodes_pair_norm_model
    from molgap.v4_runtime import torch_load_compat

    ref_completion = json.loads((args.reference / "completion_manifest.json").read_text(encoding="utf-8"))
    candidate_completion = json.loads((args.candidate / "completion_manifest.json").read_text(encoding="utf-8"))
    if not (ref_completion.get("complete") and candidate_completion.get("complete")):
        raise RuntimeError("Both arms must have complete terminal manifests")
    if (ref_completion["epochs"], candidate_completion["epochs_completed"]) != (60, 60):
        raise RuntimeError("Both arms must finish 60 epochs")
    if (ref_completion["optimizer_steps"], candidate_completion["total_optimizer_steps"]) != (46860, 46860):
        raise RuntimeError("Both arms must finish 46,860 steps")
    paths = {
        "reference_model": args.reference / "best_model.pt",
        "reference_predictions": args.reference / "development_predictions.pt",
        "candidate_model": args.candidate / "best_model.pt",
        "candidate_predictions": args.candidate / "development_predictions.pt",
    }
    for name, path in paths.items():
        key = "best_model_sha256" if name.endswith("model") else "development_predictions_sha256"
        completion = ref_completion if name.startswith("reference") else candidate_completion
        if v4.sha256_file(path) != completion[key]:
            raise RuntimeError(f"{name} artifact SHA256 changed")
    ref_dev = torch_load_compat(paths["reference_predictions"], map_location="cpu", weights_only=False)
    candidate_dev = torch_load_compat(paths["candidate_predictions"], map_location="cpu", weights_only=False)
    development = paired_metrics(ref_dev, candidate_dev, 100000, 150000)

    ref_payload = torch_load_compat(paths["reference_model"], map_location="cpu", weights_only=False)
    if ref_payload["format"] != v4.RUN_FORMAT:
        raise RuntimeError("Reference model format changed")
    mean = float(ref_payload["target_stats"]["mean_eV"])
    std = float(ref_payload["target_stats"]["sample_std_eV"])
    ref_model = v4._make_model().to("cuda")
    ref_model.load_state_dict(ref_payload["model"], strict=True)
    ref_transfer = transfer_predictions(args.cache_500k, ref_model, mean, std)
    del ref_model
    torch.cuda.empty_cache()
    candidate_model = make_noisy_nodes_pair_norm_model().to("cuda")
    candidate_model.load_state_dict(torch_load_compat(paths["candidate_model"], map_location="cpu", weights_only=False), strict=True)
    candidate_transfer = transfer_predictions(args.cache_500k, candidate_model, mean, std)
    transfer = paired_metrics(ref_transfer, candidate_transfer, 500000, 550000)

    ref_trace = json.loads((args.reference / "trace.json").read_text(encoding="utf-8"))["rows"]
    candidate_trace = json.loads((args.candidate / "trace.json").read_text(encoding="utf-8"))["epochs"]
    if len(ref_trace) != 60 or len(candidate_trace) != 60:
        raise RuntimeError("Trace length changed")
    curve = [{
        "epoch": epoch,
        "reference_dev_mae_eV": float(ref_trace[epoch]["development_mae_eV"]),
        "candidate_dev_mae_eV": float(candidate_trace[epoch]["development_gap_mae_eV"]),
        "candidate_gain_eV": float(ref_trace[epoch]["development_mae_eV"] - candidate_trace[epoch]["development_gap_mae_eV"]),
    } for epoch in range(60)]
    result = {
        "format": "molgap-local-gptrans-100k-transfer-pair-analysis-v1",
        "status": "diagnostic_only",
        "source_idx_roles": {"development_100k": [100000, 150000], "development_500k_reused": [500000, 550000]},
        "reference_best_epoch": ref_completion["best_epoch"],
        "candidate_best_epoch": candidate_completion["best_epoch"],
        "development_100k": development,
        "frozen_transfer_500k_development": transfer,
        "curve": curve,
        "artifacts_sha256": {name: v4.sha256_file(path) for name, path in paths.items()},
        "limitations": ["one training seed", "500K development role already consumed for model selection", "row bootstrap does not measure training-seed variation"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("development_100k", "frozen_transfer_500k_development")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
