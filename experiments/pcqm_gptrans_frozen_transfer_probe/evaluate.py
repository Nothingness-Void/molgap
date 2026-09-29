"""Evaluate existing frozen 100K GPTrans weights on one fixed development shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from molgap.pcqm_gptrans_v4 import _forward, _load_datasets, _make_model


EXPECTED_HASHES = {
    "development": "1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1",
    "candidate": "e70c922d5ddd347551b1d378fc37e2105130a3f6d2bf4153344e8698dbfd82fa",
    "reference": "a06f1f163dc6fd7ebae45eec6bfcff75cf05da4ae077344a93926b7d498d0565",
}
START_SOURCE_IDX = 500_000
FULL_ROWS = 50_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path, variant: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "molgap-pcqm-gptrans-t-100k-reference-v4":
        raise ValueError(f"Unexpected checkpoint format: {path}")
    if payload.get("model_config", {}).get("variant") != variant:
        raise ValueError(f"Unexpected checkpoint variant: {path}")
    if payload.get("manifest_sha256") != "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d":
        raise ValueError(f"Unexpected 100K graph manifest: {path}")
    model = _make_model(variant=variant)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=FULL_ROWS)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > FULL_ROWS:
        raise ValueError("--limit must be within 1..50000")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    paths = {
        "development": args.development.resolve(),
        "candidate": args.candidate.resolve(),
        "reference": args.reference.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != EXPECTED_HASHES[name]:
            raise ValueError(f"Missing or identity-mismatched {name}: {path}")

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate, candidate_payload = load_model(paths["candidate"], "pair_update_norm", device)
    reference, reference_payload = load_model(paths["reference"], "reference", device)
    if candidate_payload["target_stats"] != reference_payload["target_stats"]:
        raise ValueError("Candidate and reference target transforms differ")
    stats = candidate_payload["target_stats"]

    graphs, _ = _load_datasets((paths["development"],))
    if len(graphs) != FULL_ROWS:
        raise ValueError(f"Expected {FULL_ROWS} development graphs, found {len(graphs)}")
    subset = torch.utils.data.Subset(graphs, range(args.limit))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    candidate_predictions: list[torch.Tensor] = []
    reference_predictions: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            rows.append(batch.source_idx.view(-1).cpu())
            targets.append(batch.y.view(-1).float().cpu())
            candidate_predictions.append(
                (_forward(candidate, batch) * stats["sample_std_eV"] + stats["mean_eV"]).cpu()
            )
            reference_predictions.append(
                (_forward(reference, batch) * stats["sample_std_eV"] + stats["mean_eV"]).cpu()
            )

    index = torch.cat(rows).numpy().astype(np.int64)
    target = torch.cat(targets).numpy().astype(np.float64)
    candidate_pred = torch.cat(candidate_predictions).numpy().astype(np.float64)
    reference_pred = torch.cat(reference_predictions).numpy().astype(np.float64)
    if not np.array_equal(index, np.arange(START_SOURCE_IDX, START_SOURCE_IDX + args.limit)):
        raise ValueError("Development source_idx order does not match the frozen role")
    if not all(np.isfinite(value).all() for value in (target, candidate_pred, reference_pred)):
        raise ValueError("Nonfinite targets or predictions")

    reference_error = np.abs(reference_pred - target)
    candidate_error = np.abs(candidate_pred - target)
    row_gain = reference_error - candidate_error
    rng = np.random.default_rng(42)
    bootstrap = np.empty(2_000, dtype=np.float64)
    for iteration in range(len(bootstrap)):
        bootstrap[iteration] = row_gain[rng.integers(0, len(row_gain), len(row_gain))].mean()

    args.output.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        source_idx=index,
        target_eV=target,
        candidate_prediction_eV=candidate_pred,
        reference_prediction_eV=reference_pred,
    )
    gain = float(row_gain.mean())
    summary = {
        "format": "molgap-gptrans-frozen-transfer-probe-v1",
        "scope": "full_500k_internal_development" if args.limit == FULL_ROWS else "partial_smoke_only",
        "rows": int(args.limit),
        "source_idx_start": int(index[0]),
        "source_idx_stop_exclusive": int(index[-1]) + 1,
        "inputs_sha256": EXPECTED_HASHES,
        "prediction_file": "predictions.npz",
        "prediction_sha256": sha256(prediction_path),
        "candidate_mae_eV": float(candidate_error.mean()),
        "reference_mae_eV": float(reference_error.mean()),
        "reference_minus_candidate_gain_eV": gain,
        "paired_row_bootstrap_95pct_eV": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "descriptive_gain_reaches_existing_0p003_eV_floor": bool(gain >= 0.003),
        "device": str(device),
        "batch_size": args.batch_size,
        "protected_roles_read": False,
    }
    target_path = args.output / "summary.json"
    temporary = target_path.with_name(target_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
