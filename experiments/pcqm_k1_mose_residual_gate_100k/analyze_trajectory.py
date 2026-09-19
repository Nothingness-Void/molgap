"""No-inference trajectory and paired-payload audit for the MoSE residual gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load_payload(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"target_eV", "prediction_eV", "source_idx"}
    if set(payload) != required:
        raise RuntimeError(f"Unexpected payload keys in {path}: {sorted(payload)}")
    return {key: value.detach().view(-1).double() for key, value in payload.items()}


def _align(reference: dict, other: dict) -> torch.Tensor:
    lookup = {int(value): index for index, value in enumerate(other["source_idx"].tolist())}
    indices = torch.tensor(
        [lookup[int(value)] for value in reference["source_idx"].tolist()], dtype=torch.long
    )
    if not torch.equal(reference["source_idx"], other["source_idx"][indices]):
        raise RuntimeError("source_idx alignment failed")
    if not torch.equal(reference["target_eV"], other["target_eV"][indices]):
        raise RuntimeError("target alignment failed")
    return other["prediction_eV"][indices]


def _mae(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return float((target - prediction).abs().mean())


def _quintiles(score: torch.Tensor, target: torch.Tensor, predictions: dict) -> list[dict]:
    order = torch.argsort(score, stable=True)
    rows = []
    for number, indices in enumerate(torch.tensor_split(order, 5), start=1):
        reference = _mae(target[indices], predictions["k1"][indices])
        candidate = _mae(target[indices], predictions["residual_gate"][indices])
        mose = _mae(target[indices], predictions["mose_replacement"][indices])
        rows.append(
            {
                "quintile": number,
                "rows": int(indices.numel()),
                "score_min": float(score[indices].min()),
                "score_max": float(score[indices].max()),
                "k1_mae_eV": reference,
                "residual_gate_minus_k1_eV": candidate - reference,
                "mose_replacement_minus_k1_eV": mose - reference,
                "residual_gate_minus_mose_replacement_eV": candidate - mose,
            }
        )
    return rows


def _trace_summary(path: Path) -> dict:
    epochs = json.loads(path.read_text(encoding="utf-8"))["epochs"]
    values = torch.tensor([row["development_gap_mae_eV"] for row in epochs])
    best = int(torch.argmin(values))
    return {
        "epochs": len(epochs),
        "best_epoch": int(epochs[best]["epoch"]),
        "best_mae_eV": float(values[best]),
        "epoch_34_mae_eV": float(values[34]) if len(values) > 34 else None,
        "last_epoch_mae_eV": float(values[-1]),
        "epoch_34_to_last_improvement_eV": (
            float(values[34] - values[-1]) if len(values) > 34 else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-payload", type=Path, required=True)
    parser.add_argument("--candidate-payload", type=Path, required=True)
    parser.add_argument("--mose-payload", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--mose-trace", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = _load_payload(args.reference_payload)
    target = reference["target_eV"]
    predictions = {
        "k1": reference["prediction_eV"],
        "residual_gate": _align(reference, _load_payload(args.candidate_payload)),
        "mose_replacement": _align(reference, _load_payload(args.mose_payload)),
    }
    reference_error = (target - predictions["k1"]).abs()
    candidate_error = (target - predictions["residual_gate"]).abs()
    candidate_correction = predictions["residual_gate"] - predictions["k1"]
    mose_correction = predictions["mose_replacement"] - predictions["k1"]
    centered_x = mose_correction - mose_correction.mean()
    centered_y = candidate_correction - candidate_correction.mean()
    slope = float((centered_x * centered_y).sum() / centered_x.square().sum())
    intercept = float(candidate_correction.mean() - slope * mose_correction.mean())
    fitted = intercept + slope * mose_correction
    r2 = 1.0 - float(
        (candidate_correction - fitted).square().sum()
        / centered_y.square().sum()
    )

    state = torch.load(args.candidate_model, map_location="cpu", weights_only=True)
    module_norms = {
        name: float(value.double().norm())
        for name, value in state.items()
        if name.startswith("mose_residual.") or name.startswith("mose_gate.")
    }
    result = {
        "format": "molgap-k1-mose-residual-gate-trajectory-v1",
        "rows": int(target.numel()),
        "mae_eV": {name: _mae(target, prediction) for name, prediction in predictions.items()},
        "paired": {
            "residual_gate_minus_k1_eV": _mae(target, predictions["residual_gate"])
            - _mae(target, predictions["k1"]),
            "mose_replacement_minus_k1_eV": _mae(target, predictions["mose_replacement"])
            - _mae(target, predictions["k1"]),
            "residual_gate_minus_mose_replacement_eV": _mae(
                target, predictions["residual_gate"]
            )
            - _mae(target, predictions["mose_replacement"]),
            "residual_gate_row_improvement_fraction": float(
                (candidate_error < reference_error).double().mean()
            ),
            "residual_gate_row_tie_fraction": float(
                (candidate_error == reference_error).double().mean()
            ),
            "correction_moves_toward_target_fraction": float(
                ((target - predictions["k1"]) * candidate_correction > 0).double().mean()
            ),
            "mean_absolute_candidate_correction_eV": float(candidate_correction.abs().mean()),
        },
        "correction_relation_to_mose_replacement": {
            "pearson": float(torch.corrcoef(torch.stack((candidate_correction, mose_correction)))[0, 1]),
            "linear_slope": slope,
            "linear_intercept_eV": intercept,
            "linear_r2": r2,
        },
        "target_quintiles": _quintiles(target, target, predictions),
        "k1_absolute_error_quintiles": _quintiles(reference_error, target, predictions),
        "candidate_correction_magnitude_quintiles": _quintiles(
            candidate_correction.abs(), target, predictions
        ),
        "traces": {
            "k1": _trace_summary(args.reference_trace),
            "residual_gate": _trace_summary(args.candidate_trace),
            "mose_replacement": _trace_summary(args.mose_trace),
        },
        "candidate_module_tensor_norms": module_norms,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
