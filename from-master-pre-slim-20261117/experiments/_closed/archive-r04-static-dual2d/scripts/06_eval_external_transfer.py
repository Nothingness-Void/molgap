"""Test frozen dual-2D static blends on public external evaluation sets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molgap.constants import RESULTS_DIR
from molgap.archive.phase8_r04_static_dual2d.external_eval import (
    build_external_graphs,
    evaluate_seed,
    predict_seed_experts,
)


PHASE8 = RESULTS_DIR / "phase8"
OUT_DIR = PHASE8 / "archive" / "archive-r04-static-dual2d"
SEEDS = (42, 43, 44)
TARGETS = ("homo", "lumo", "gap")


def routed_v4_common(frame: pd.DataFrame) -> np.ndarray:
    base = frame[[f"expansion500k_full_hybrid_{target}" for target in TARGETS]].to_numpy()
    dual = frame[[f"expansion500k_dualgps_hybrid_{target}" for target in TARGETS]].to_numpy()
    routed = base.copy()
    routed[base[:, 2] < 4.0] = dual[base[:, 2] < 4.0]
    return routed.astype(np.float64)


def routed_v4_pcqm(frame: pd.DataFrame) -> np.ndarray:
    base = frame["v3_gap_pred"].to_numpy(dtype=np.float64)
    dual = frame["dualgps_gap_pred"].to_numpy(dtype=np.float64)
    return np.where(base < 4.0, dual, base)


def v4_metrics(y: np.ndarray, prediction: np.ndarray, target_indices: tuple[int, ...]) -> dict:
    return {
        TARGETS[index]: {"mae": float(np.abs(prediction[:, index] - y[:, index]).mean())}
        for index in target_indices
    }


def evaluate_dataset(
    frame: pd.DataFrame,
    *,
    dataset: str,
    smiles_column: str,
    y: np.ndarray,
    target_indices: tuple[int, ...],
    v4_prediction: np.ndarray,
    frozen: dict,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    graphs, valid_positions = build_external_graphs(frame[smiles_column])
    if not graphs:
        raise RuntimeError(f"No valid 2D graphs for {dataset}")
    valid_frame = frame.iloc[valid_positions].reset_index(drop=True)
    valid_y = y[valid_positions]
    valid_v4 = v4_prediction[valid_positions]
    output = {
        "input_n": int(len(frame)),
        "valid_n": int(len(valid_frame)),
        "invalid_n": int(len(frame) - len(valid_frame)),
        "v4": v4_metrics(valid_y, valid_v4, target_indices),
        "seeds": {},
    }
    rows = []
    for seed in SEEDS:
        record = frozen["seeds"][str(seed)]
        expert_predictions, order = predict_seed_experts(
            graphs,
            checkpoint_dir=OUT_DIR,
            seed=seed,
            device=device,
        )
        if not np.array_equal(order, valid_positions):
            raise RuntimeError(f"Seed {seed} prediction rows do not match {dataset}")
        evaluation = evaluate_seed(
            valid_y,
            expert_predictions,
            np.asarray(record["static_weights"], dtype=np.float64),
            record["best_single"],
            target_indices=target_indices,
            bootstrap_seed=seed * 10 + len(dataset),
        )
        output["seeds"][str(seed)] = evaluation
        static = evaluation["methods"]["static_weights"]
        reference = evaluation["methods"][record["best_single"]]
        row = valid_frame.copy()
        row["dataset"] = dataset
        row["seed"] = seed
        row["reference_name"] = record["best_single"]
        for index, target in enumerate(TARGETS):
            if index not in target_indices:
                continue
            row[f"target_{target}"] = valid_y[:, index]
            row[f"v4_{target}"] = valid_v4[:, index]
            row[f"local_{target}"] = expert_predictions[:, index, 0]
            row[f"global_{target}"] = expert_predictions[:, index, 1]
            row[f"static_mae_{target}"] = static[target]["mae"]
            row[f"reference_mae_{target}"] = reference[target]["mae"]
            row[f"static_prediction_{target}"] = (
                expert_predictions[:, index, :] @ np.asarray(record["static_weights"])[index]
            )
        rows.extend(row.to_dict("records"))
    return output, rows


def decision_markdown(result: dict) -> str:
    lines = [
        "# Dual-2D Static Candidate External Transfer",
        "",
        "Frozen internal validation weights and each seed's predeclared internal "
        "best single expert were used unchanged. No sealed set was read. v4 is "
        "reported only as context and is not used to choose weights or references.",
        "",
        "| seed | set | reference | static Gap MAE | reference Gap MAE | improvement | 95% CI |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for dataset, block in result["datasets"].items():
        for seed in SEEDS:
            item = block["seeds"][str(seed)]
            gap = item["static_vs_reference"]["gap"]
            methods = item["methods"]
            reference = item["reference"]
            lines.append(
                f"| {seed} | {dataset} | {reference} | "
                f"{methods['static_weights']['gap']['mae']:.6f} | "
                f"{methods[reference]['gap']['mae']:.6f} | "
                f"{gap['improvement_eV']:+.6f} | "
                f"[{gap['ci95'][0]:+.6f}, {gap['ci95'][1]:+.6f}] |"
            )
    verdict = result["decision"]
    lines.extend([
        "",
        f"**Decision: {verdict['verdict']}.** {verdict['reason']}",
        "",
        "Production remains routed dual-GPS v4 regardless of this candidate gate.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = json.loads((OUT_DIR / "dual2d_three_seed_metrics.json").read_text())
    common = pd.read_csv(PHASE8 / "gps_arch_dualgps_common_eval_predictions.csv")
    common_y = common[list(TARGETS)].to_numpy(dtype=np.float64)
    common_v4 = routed_v4_common(common)
    pcqm = pd.read_csv(PHASE8 / "gps_arch_dualgps_pcqm_proxy_predictions.csv")
    pcqm_y = np.zeros((len(pcqm), 3), dtype=np.float64)
    pcqm_y[:, 2] = pcqm["gap_true"].to_numpy(dtype=np.float64)
    pcqm_v4 = np.zeros_like(pcqm_y)
    pcqm_v4[:, 2] = routed_v4_pcqm(pcqm)

    datasets, rows = {}, []
    common_blocks = {
        "common_all": np.ones(len(common), dtype=bool),
        "common_ood1000": common.eval_set.eq("ood1000").to_numpy(),
        "common_p8_targeted_hard": common.eval_set.eq("p8_targeted_hard").to_numpy(),
    }
    for name, mask in common_blocks.items():
        frame = common.loc[mask].reset_index(drop=True)
        datasets[name], prediction_rows = evaluate_dataset(
            frame,
            dataset=name,
            smiles_column="canonical_smiles",
            y=common_y[mask],
            target_indices=(0, 1, 2),
            v4_prediction=common_v4[mask],
            frozen=frozen,
            device=device,
        )
        rows.extend(prediction_rows)
    datasets["pcqm_proxy"], prediction_rows = evaluate_dataset(
        pcqm,
        dataset="pcqm_proxy",
        smiles_column="canonical_smiles",
        y=pcqm_y,
        target_indices=(2,),
        v4_prediction=pcqm_v4,
        frozen=frozen,
        device=device,
    )
    rows.extend(prediction_rows)

    gap_deltas = [
        datasets[name]["seeds"][str(seed)]["static_vs_reference"]["gap"]["delta"]
        for name in datasets for seed in SEEDS
    ]
    passes = bool(all(delta <= 0.0 for delta in gap_deltas))
    result = {
        "experiment": "dual-2D static candidate external transfer",
        "sealed_metrics_opened": False,
        "weight_selection": "frozen internal validation target-wise weights",
        "reference_selection": "frozen internal best single expert per seed",
        "datasets": datasets,
        "decision": {
            "gate": "all three seeds must not regress in Gap MAE on common, OOD, P8-hard, or PCQM proxy",
            "all_seed_all_set_gap_non_regression": passes,
            "verdict": "advance to scale feasibility" if passes else "stop static dual-2D candidate",
            "reason": (
                "Frozen static blending keeps the same direction across all external blocks."
                if passes else
                "At least one frozen seed/block regresses against its predeclared single-expert reference."
            ),
            "production_change": False,
        },
    }
    (OUT_DIR / "external_transfer_metrics.json").write_text(json.dumps(result, indent=2))
    (OUT_DIR / "external_transfer_decision.md").write_text(decision_markdown(result))
    pd.DataFrame(rows).to_csv(OUT_DIR / "external_transfer_predictions.csv", index=False)
    print(json.dumps({"decision": result["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
