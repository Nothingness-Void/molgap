"""Diagnose why the archive-r01 learned Router failed external promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from molgap.constants import RESULTS_DIR


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r01-learned-router"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=OUT_DIR / "external_predictions.parquet")
    parser.add_argument("--schema", type=Path, default=OUT_DIR / "feature_schema.json")
    parser.add_argument("--gain-model", type=Path, default=OUT_DIR / "router_gain_model.txt")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "error_analysis.json")
    parser.add_argument("--decision-out", type=Path, default=OUT_DIR / "error_analysis.md")
    parser.add_argument("--deciles-out", type=Path, default=OUT_DIR / "error_deciles.csv")
    parser.add_argument("--importance-out", type=Path, default=OUT_DIR / "feature_importance.csv")
    return parser.parse_args()


def actual_gain(frame: pd.DataFrame) -> np.ndarray:
    if frame["dataset"].eq("pcqm_proxy").all():
        return (
            np.abs(frame["base_gap"] - frame["y_gap"])
            - np.abs(frame["expert_gap"] - frame["y_gap"])
        ).to_numpy()
    errors_base = np.column_stack([
        np.abs(frame[f"base_{target}"] - frame[f"y_{target}"])
        for target in ("homo", "lumo", "gap")
    ])
    errors_expert = np.column_stack([
        np.abs(frame[f"expert_{target}"] - frame[f"y_{target}"])
        for target in ("homo", "lumo", "gap")
    ])
    return (errors_base - errors_expert) @ np.array([0.25, 0.25, 0.50])


def block_metrics(frame: pd.DataFrame, win_delta: float = 0.001) -> dict[str, float]:
    gain = actual_gain(frame)
    wins = gain > win_delta
    probability = frame["p_expert_wins"].to_numpy()
    route = frame["learned_budget_matched_route"].astype(bool).to_numpy()
    fixed = frame["fixed_route"].astype(bool).to_numpy()
    return {
        "n": int(len(frame)),
        "actual_mean_gain": float(gain.mean()),
        "predicted_mean_gain": float(frame["predicted_gain"].mean()),
        "actual_win_rate": float(wins.mean()),
        "predicted_mean_probability": float(probability.mean()),
        "gain_spearman": float(spearmanr(gain, frame["predicted_gain"]).statistic),
        "win_roc_auc": float(roc_auc_score(wins, probability)),
        "win_average_precision": float(average_precision_score(wins, probability)),
        "learned_route_rate": float(route.mean()),
        "learned_precision": float(wins[route].mean()) if route.any() else None,
        "fixed_route_rate": float(fixed.mean()),
        "fixed_precision": float(wins[fixed].mean()) if fixed.any() else None,
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.predictions)
    blocks = {
        "common_all": frame.loc[frame["dataset"].eq("common_all")],
        "common_ood1000": frame.loc[
            frame["dataset"].eq("common_all") & frame["eval_set"].eq("ood1000")
        ],
        "common_p8_targeted_hard": frame.loc[
            frame["dataset"].eq("common_all")
            & frame["eval_set"].eq("p8_targeted_hard")
        ],
        "pcqm_proxy": frame.loc[frame["dataset"].eq("pcqm_proxy")],
    }
    metrics = {name: block_metrics(block) for name, block in blocks.items()}

    decile_rows = []
    for name, block in blocks.items():
        block = block.copy()
        block["actual_gain"] = actual_gain(block)
        block["score_decile"] = pd.qcut(
            block["predicted_gain"], q=10, labels=False, duplicates="drop"
        )
        for decile, part in block.groupby("score_decile"):
            decile_rows.append({
                "dataset": name,
                "score_decile": int(decile),
                "n": int(len(part)),
                "predicted_gain": float(part["predicted_gain"].mean()),
                "actual_gain": float(part["actual_gain"].mean()),
                "actual_win_rate": float((part["actual_gain"] > 0.001).mean()),
            })
    pd.DataFrame(decile_rows).to_csv(args.deciles_out, index=False)

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    booster = lgb.Booster(model_str=args.gain_model.read_text(encoding="utf-8"))
    importance = pd.DataFrame({
        "feature": schema["features"],
        "gain_importance": booster.feature_importance(importance_type="gain"),
        "split_importance": booster.feature_importance(importance_type="split"),
    }).sort_values("gain_importance", ascending=False)
    importance.to_csv(args.importance_out, index=False)

    result = {
        "win_delta_eV": 0.001,
        "datasets": metrics,
        "top_gain_features": importance.head(15).to_dict(orient="records"),
        "conclusion": (
            "The internal router ranking does not transfer to PCQM: predicted gain remains "
            "positive while actual expert gain and route precision fall. This is domain-prior "
            "shift, so external thresholds must not be tuned post hoc."
        ),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# archive-r01 Learned Router Error Analysis",
        "",
        "| evaluation | actual gain | predicted gain | actual win | predicted win | gain Spearman | learned precision | fixed precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        lines.append(
            f"| {name} | {row['actual_mean_gain']:+.6f} | "
            f"{row['predicted_mean_gain']:+.6f} | {row['actual_win_rate']:.1%} | "
            f"{row['predicted_mean_probability']:.1%} | {row['gain_spearman']:+.3f} | "
            f"{row['learned_precision']:.1%} | {row['fixed_precision']:.1%} |"
        )
    lines.extend([
        "",
        "**Conclusion:** the Router has weak useful ranking on the expansion500k held-out domain, "
        "but that ranking does not transfer to PCQM. It overestimates expert utility under the "
        "shifted expert-win prior. Do not tune on external tests; keep fixed routed-v4.",
    ])
    args.decision_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["conclusion"], indent=2), flush=True)


if __name__ == "__main__":
    main()
