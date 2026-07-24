"""Build PPT-ready v1/v2/v3 metric tables.

Main comparison: Phase 7 full vs Phase 8 replacement300k vs expansion500k on the
same common-eval molecules. This is the fairest v1/v2/v3 table because all rows,
labels, conformer generation, and prediction columns are aligned.

Also records:
- PCQM4Mv2 valid proxy Gap MAE/R2 (leaderboard-style proxy, not OGB submission)
- internal train/test metrics from each run (not directly comparable across
  phases because the splits/datasets differ)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COMMON_CSV = Path("results/phase8/full_expansion500k_common_eval_predictions.csv")
PCQM_CSV = Path("results/phase8/pcqm4mv2_proxy_p7_v2_v3_predictions.csv")
OUT_JSON = Path("results/phase8/v1_v2_v3_ppt_metrics.json")
OUT_MD = Path("results/phase8/v1_v2_v3_ppt_metrics.md")
OUT_COMMON_CSV = Path("results/phase8/v1_v2_v3_common_eval_metrics.csv")
OUT_PCQM_CSV = Path("results/phase8/v1_v2_v3_pcqm_proxy_metrics.csv")
OUT_INTERNAL_CSV = Path("results/phase8/v1_v2_v3_internal_test_metrics.csv")

TARGETS = ("homo", "lumo", "gap")
DISPLAY_TARGETS = {"homo": "HOMO", "lumo": "LUMO", "gap": "Gap"}
MODELS = {
    "v1_phase7": "phase7_full",
    "v2_replacement300k": "replacement300k_full",
    "v3_expansion500k": "expansion500k_full",
}
COMPONENTS = ("gps_2d", "schnet_3d", "hybrid")


def _f(v: Any, digits: int = 6) -> float:
    return round(float(v), digits)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": _f(mean_absolute_error(y_true, y_pred)),
        "rmse": _f(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": _f(r2_score(y_true, y_pred)),
        "bias": _f(np.mean(y_pred - y_true)),
    }


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def common_eval_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    blocks = {"all": df}
    for eval_set in sorted(df["eval_set"].unique()):
        blocks[eval_set] = df[df["eval_set"] == eval_set]

    for block, sub in blocks.items():
        for model_name, prefix in MODELS.items():
            for component in COMPONENTS:
                row: dict[str, Any] = {
                    "scope": "common_eval",
                    "block": block,
                    "model": model_name,
                    "component": component,
                    "n": int(len(sub)),
                }
                mae_vals, rmse_vals, r2_vals = [], [], []
                for target in TARGETS:
                    metric = _metrics(
                        sub[target].to_numpy(dtype=np.float64),
                        sub[f"{prefix}_{component}_{target}"].to_numpy(dtype=np.float64),
                    )
                    disp = DISPLAY_TARGETS[target]
                    for key, value in metric.items():
                        row[f"{disp}_{key}"] = value
                    mae_vals.append(metric["mae"])
                    rmse_vals.append(metric["rmse"])
                    r2_vals.append(metric["r2"])
                row["average_mae"] = _f(np.mean(mae_vals))
                row["average_rmse"] = _f(np.mean(rmse_vals))
                row["average_r2"] = _f(np.mean(r2_vals))
                rows.append(row)
    return pd.DataFrame(rows)


def pcqm_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for model, prefix in {
        "v1_phase7": "p7",
        "v2_replacement300k": "p8",
        "v3_expansion500k": "v3",
    }.items():
        metric = _metrics(
            df["gap_true"].to_numpy(dtype=np.float64),
            df[f"{prefix}_gap_pred"].to_numpy(dtype=np.float64),
        )
        rows.append({
            "scope": "pcqm4mv2_valid_proxy",
            "model": model,
            "component": "hybrid",
            "n": int(len(df)),
            "Gap_mae": metric["mae"],
            "Gap_rmse": metric["rmse"],
            "Gap_r2": metric["r2"],
            "Gap_bias": metric["bias"],
            "Gap_median_abs_err": _f(df[f"{prefix}_gap_abs_err"].median()),
        })
    return pd.DataFrame(rows)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _internal_row_from_metrics(label: str, path: Path, nested: str | None) -> dict[str, Any] | None:
    data = _load_json(path)
    if data is None:
        return None
    if nested:
        data = data[nested]
    elif "test_metrics" in data:
        data = data["test_metrics"]

    row: dict[str, Any] = {
        "scope": "internal_test_not_cross_phase_comparable",
        "model": label,
        "component": "hybrid",
    }
    mae_vals, r2_vals = [], []
    for disp in ("HOMO", "LUMO", "Gap"):
        metric = data[disp]
        row[f"{disp}_mae"] = _f(metric["mae"])
        row[f"{disp}_r2"] = _f(metric["r2"])
        mae_vals.append(metric["mae"])
        r2_vals.append(metric["r2"])
    row["average_mae"] = _f(np.mean(mae_vals))
    row["average_r2"] = _f(np.mean(r2_vals))
    if "best_val_mae" in data:
        row["best_val_mae"] = _f(data["best_val_mae"])
    return row


def internal_metrics() -> pd.DataFrame:
    rows = [
        _internal_row_from_metrics(
            "v1_phase7",
            Path("results/phase7/fusion_optuna_metrics.json"),
            None,
        ),
        _internal_row_from_metrics(
            "v2_replacement300k",
            Path("results/phase8/fusion_replacement_300k_metrics.json"),
            "baseline",
        ),
        _internal_row_from_metrics(
            "v3_expansion500k",
            Path("results/phase8/fusion_expansion_500k_metrics.json"),
            "baseline",
        ),
    ]
    return pd.DataFrame([r for r in rows if r is not None])


def _fmt(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _md_table(df: pd.DataFrame, cols: list[str], digits: int = 4) -> str:
    view = df[cols].copy()
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def build_markdown(
    common: pd.DataFrame,
    pcqm: pd.DataFrame,
    internal: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    hybrid_all = common[(common["component"] == "hybrid") & (common["block"] == "all")]
    hybrid_ood = common[(common["component"] == "hybrid") & (common["block"] == "ood1000")]
    hybrid_hard = common[(common["component"] == "hybrid") & (common["block"] == "p8_targeted_hard")]
    component_all = common[common["block"] == "all"]

    cols_main = [
        "model", "n", "HOMO_mae", "HOMO_r2", "LUMO_mae", "LUMO_r2",
        "Gap_mae", "Gap_r2", "average_mae", "average_r2",
    ]
    cols_block = ["model", "n", "Gap_mae", "Gap_r2", "average_mae", "average_r2"]
    cols_component = ["model", "component", "Gap_mae", "Gap_r2", "average_mae", "average_r2"]

    pcqm_text = "PCQM4Mv2 proxy predictions not available."
    if not pcqm.empty:
        pcqm_text = _md_table(
            pcqm,
            ["model", "n", "Gap_mae", "Gap_rmse", "Gap_r2", "Gap_bias", "Gap_median_abs_err"],
        )

    return f"""# V1 / V2 / V3 PPT Metrics

Date: 2026-06-30

Inputs:

- common eval predictions: `{args.common_csv}`
- PCQM4Mv2 proxy predictions: `{args.pcqm_csv}`

## Recommended PPT Table

Same 1,977 common-eval molecules for all three models. This is the fairest
cross-version comparison.

{_md_table(hybrid_all, cols_main)}

## Common Eval By Slice

### OOD-1000

{_md_table(hybrid_ood, cols_block)}

### P8 Targeted Hard

{_md_table(hybrid_hard, cols_block)}

## Component-Level Common Eval

All common-eval molecules. Useful if a slide needs to show 2D, 3D, and fusion.

{_md_table(component_all, cols_component)}

## PCQM4Mv2 Valid Proxy

This is a leakage-filtered PCQM4Mv2 valid proxy, **not** an OGB submission.

{pcqm_text}

## Internal Test Metrics

These are training-run records and are **not** a fair cross-phase comparison,
because the datasets/splits differ. Use them only as provenance.

{_md_table(internal, ["model", "HOMO_mae", "HOMO_r2", "LUMO_mae", "LUMO_r2", "Gap_mae", "Gap_r2", "average_mae", "average_r2", "best_val_mae"])}

## Files

- JSON: `{args.out_json}`
- common eval CSV: `{args.out_common_csv}`
- PCQM proxy CSV: `{args.out_pcqm_csv}`
- internal CSV: `{args.out_internal_csv}`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-csv", type=Path, default=COMMON_CSV)
    parser.add_argument("--pcqm-csv", type=Path, default=PCQM_CSV)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-common-csv", type=Path, default=OUT_COMMON_CSV)
    parser.add_argument("--out-pcqm-csv", type=Path, default=OUT_PCQM_CSV)
    parser.add_argument("--out-internal-csv", type=Path, default=OUT_INTERNAL_CSV)
    args = parser.parse_args()

    common = common_eval_metrics(args.common_csv)
    pcqm = pcqm_metrics(args.pcqm_csv)
    internal = internal_metrics()

    for path in (args.out_json, args.out_md, args.out_common_csv, args.out_pcqm_csv, args.out_internal_csv):
        path.parent.mkdir(parents=True, exist_ok=True)

    common.to_csv(args.out_common_csv, index=False, encoding="utf-8")
    pcqm.to_csv(args.out_pcqm_csv, index=False, encoding="utf-8")
    internal.to_csv(args.out_internal_csv, index=False, encoding="utf-8")
    payload = {
        "common_eval": _records(common),
        "pcqm_proxy": _records(pcqm),
        "internal_test_not_cross_phase_comparable": _records(internal),
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(build_markdown(common, pcqm, internal, args), encoding="utf-8")

    hybrid_all = common[(common["component"] == "hybrid") & (common["block"] == "all")]
    print("Hybrid common-eval all:")
    for _, row in hybrid_all.iterrows():
        print(
            f"  {row['model']}: avg MAE={row['average_mae']:.5f}, "
            f"avg R2={row['average_r2']:.5f}, Gap MAE={row['Gap_mae']:.5f}, "
            f"Gap R2={row['Gap_r2']:.5f}"
        )
    print(f"Markdown -> {args.out_md}")
    print(f"JSON -> {args.out_json}")


if __name__ == "__main__":
    main()
