"""Verify the public repaired-2M 2D loader reproduces its accepted metrics.

This is a packaging gate, not a new experiment. It replays the already accepted
external prediction table through `molgap.inference` and checks that the public
path recovers the same average MAE per scope. It never reads a sealed set and
never retrains anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molgap.constants import EVALUATE_DIR, EXPERIMENTS_DIR, MODEL_REGISTRY, TARGET_COLS
from molgap.inference import (
    load_repaired_2m_2d,
    predict_smiles_batch_repaired_2m_2d,
)


TARGETS = tuple(TARGET_COLS)
SCOPES = ("all", "ood1000", "p8_targeted_hard")
# Column prefix in the accepted table for each registry preset.
PRESETS = {
    "repaired_2m_dense_2d": "repaired_2m_dense_2d",
    "repaired_2m_equal_2d": "repaired_2m_equal_2d",
}
ACCEPTED_PREDICTIONS = (
    EXPERIMENTS_DIR
    / "repaired_2m_scaling"
    / "results"
    / "hierarchical_dual_schnet_external"
    / "predictions.csv"
)
DEFAULT_OUTPUT = (
    EVALUATE_DIR
    / "project_freeze"
    / "public_inference_consistency"
    / "repaired_2m_public_inference.json"
)
# The accepted table was produced under CUDA autocast; the public path runs
# fp32. Per-row differences of that size are expected, aggregate MAE is not.
SCOPE_MAE_TOLERANCE_EV = 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checkpoint_identity(key: str) -> dict[str, object]:
    spec = MODEL_REGISTRY[key]
    experts = {
        expert: {
            "path": str(MODEL_REGISTRY[expert]["checkpoint"]),
            "sha256": sha256(Path(MODEL_REGISTRY[expert]["checkpoint"])),
        }
        for expert in spec["experts"]
    }
    identity: dict[str, object] = {
        "kind": spec["kind"],
        "encoder_passes": int(spec["encoder_passes"]),
        "experts": experts,
    }
    if spec.get("gates"):
        identity["gates"] = [
            {"path": str(path), "sha256": sha256(Path(path))}
            for path in spec["gates"]
        ]
    return identity


def evaluate_preset(
    key: str,
    prefix: str,
    frame: pd.DataFrame,
    *,
    batch_size: int,
    device: torch.device | None,
) -> dict[str, object]:
    models = load_repaired_2m_2d(device, key=key)
    valid_idx, predictions = predict_smiles_batch_repaired_2m_2d(
        frame.smiles.astype(str).tolist(),
        models=models,
        batch_size=batch_size,
    )
    if len(valid_idx) != len(frame):
        raise RuntimeError(
            f"{key}: public path kept {len(valid_idx):,} of {len(frame):,} accepted rows"
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{key}: public path produced non-finite predictions")

    accepted = frame.loc[:, [f"{prefix}_{t}" for t in TARGETS]].to_numpy(np.float64)
    truth = frame.loc[:, list(TARGETS)].to_numpy(np.float64)
    public = predictions.astype(np.float64)
    row_difference = np.abs(public - accepted)

    scopes: dict[str, object] = {}
    for scope in SCOPES:
        mask = (
            np.ones(len(frame), dtype=bool)
            if scope == "all"
            else frame.eval_set.eq(scope).to_numpy()
        )
        public_mae = float(np.abs(public[mask] - truth[mask]).mean())
        accepted_mae = float(np.abs(accepted[mask] - truth[mask]).mean())
        delta = public_mae - accepted_mae
        scopes[scope] = {
            "rows": int(mask.sum()),
            "public_average_mae_eV": public_mae,
            "accepted_average_mae_eV": accepted_mae,
            "delta_eV": delta,
            "within_tolerance": bool(abs(delta) <= SCOPE_MAE_TOLERANCE_EV),
        }

    failed = [scope for scope, block in scopes.items() if not block["within_tolerance"]]
    if failed:
        raise RuntimeError(
            f"{key}: public average MAE differs beyond {SCOPE_MAE_TOLERANCE_EV} eV "
            f"on {failed}"
        )
    return {
        "registry_key": key,
        "accepted_column_prefix": prefix,
        "rows": int(len(frame)),
        "row_difference_eV": {
            "max": float(row_difference.max()),
            "mean": float(row_difference.mean()),
        },
        "scopes": scopes,
        "artifacts": checkpoint_identity(key),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-predictions", type=Path, default=ACCEPTED_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.accepted_predictions)
    required = {"eval_set", "cid", "smiles", *TARGETS}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Accepted table misses {sorted(missing)}")
    device = None if args.device is None else torch.device(args.device)

    presets = {
        key: evaluate_preset(
            key, prefix, frame, batch_size=args.batch_size, device=device
        )
        for key, prefix in PRESETS.items()
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "check": "public_inference_reproduces_accepted_external_metrics",
        "note": (
            "The accepted table was produced under CUDA autocast; the public "
            "path runs fp32, so per-row differences are expected and only "
            "aggregate average MAE is gated."
        ),
        "scope_mae_tolerance_eV": SCOPE_MAE_TOLERANCE_EV,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(
                device or ("cuda" if torch.cuda.is_available() else "cpu")
            ),
        },
        "accepted_predictions": {
            "path": str(args.accepted_predictions),
            "sha256": sha256(args.accepted_predictions),
            "rows": int(len(frame)),
        },
        "presets": presets,
        "sealed_20k_used": False,
    }
    atomic_json(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
