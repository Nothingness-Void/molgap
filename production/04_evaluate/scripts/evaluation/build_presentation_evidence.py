"""Collect every number the interview deck quotes into one reproducible record.

One slide can only quote a number that resolves to a local file. This script
walks the accepted records, recomputes the derived statistics the deck needs
(R2 per scope, worst-decile residual share, parameter counts), and emits a
slide-indexed evidence pack.

It computes nothing new about model quality: R2 and residual shares are derived
from the already accepted paired external prediction table. Accuracy claims and
their boundaries stay in `project_freeze/track_a_final_decision.md`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from molgap.constants import (
    EVALUATE_DIR,
    EXPERIMENTS_DIR,
    MODEL_REGISTRY,
    PRODUCTION_DIR,
    REPO_ROOT,
    TARGET_COLS,
)


TARGETS = tuple(TARGET_COLS)
SCOPES = ("all", "ood1000", "p8_targeted_hard")
PAIRED_METHODS = {
    "routed_v4_500k": "routed-v4 500K (previous production baseline)",
    "repaired_2m_equal_2d": "repaired-2M GPS7/GPS9 equal",
    "repaired_2m_dense_2d": "repaired-2M three-GPS dense",
}
REPAIRED = EXPERIMENTS_DIR / "repaired_2m_scaling" / "results"
PAIRED_PREDICTIONS = REPAIRED / "hierarchical_dual_schnet_external" / "predictions.csv"
FREEZE = EVALUATE_DIR / "project_freeze"
DEFAULT_OUTPUT = FREEZE / "presentation_evidence" / "presentation_evidence.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def paired_accuracy() -> dict[str, object]:
    """Recompute R2 and residual concentration from the accepted paired table."""
    frame = pd.read_csv(PAIRED_PREDICTIONS)
    truth = frame[list(TARGETS)].to_numpy(float)
    scopes: dict[str, object] = {}
    for scope in SCOPES:
        mask = (
            np.ones(len(frame), dtype=bool)
            if scope == "all"
            else frame.eval_set.eq(scope).to_numpy()
        )
        block: dict[str, object] = {"rows": int(mask.sum()), "methods": {}}
        for prefix, label in PAIRED_METHODS.items():
            values = frame[[f"{prefix}_{t}" for t in TARGETS]].to_numpy(float)
            r2 = [
                float(r2_score(truth[mask, index], values[mask, index]))
                for index in range(len(TARGETS))
            ]
            errors = np.abs(values[mask] - truth[mask])
            row_error = errors.mean(axis=1)
            gap_error = errors[:, 2]
            count = int(np.ceil(0.10 * len(row_error)))
            block["methods"][prefix] = {
                "label": label,
                **{
                    f"{target}_r2": r2[index]
                    for index, target in enumerate(TARGETS)
                },
                "average_r2": float(np.mean(r2)),
                **{
                    f"{target}_mae_eV": float(errors[:, index].mean())
                    for index, target in enumerate(TARGETS)
                },
                "average_mae_eV": float(errors.mean()),
                "worst_decile_share_of_average_error": float(
                    np.sort(row_error)[::-1][:count].sum() / row_error.sum()
                ),
                "worst_decile_share_of_gap_error": float(
                    np.sort(gap_error)[::-1][:count].sum() / gap_error.sum()
                ),
            }
        scopes[scope] = block
    return {
        "source": relative(PAIRED_PREDICTIONS),
        "comparison": "same molecules, same accepted ETKDGv3+MMFF200 rows",
        "aligned_rows": int(len(frame)),
        "scopes": scopes,
    }


def architecture() -> dict[str, object]:
    """Report the exact shape and parameter count of each shipped preset."""
    from molgap.gps import GPSWrapper
    from molgap.multi2d_router_fusion import load_dense_gate_checkpoint

    experts: dict[str, object] = {}
    for key in ("repaired_2m_gps7", "repaired_2m_gps9", "repaired_2m_gps11_160"):
        spec = MODEL_REGISTRY[key]
        model = GPSWrapper(**spec["params"])
        experts[key] = {
            "hidden_channels": int(spec["params"]["hidden_channels"]),
            "num_layers": int(spec["params"]["num_layers"]),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "checkpoint": relative(Path(spec["checkpoint"])),
        }

    gate_spec = MODEL_REGISTRY["repaired_2m_dense_2d"]
    gate = load_dense_gate_checkpoint(Path(gate_spec["gates"][0]))
    gate_parameters = int(sum(p.numel() for p in gate.parameters()))

    presets: dict[str, object] = {}
    for key in ("repaired_2m_dense_2d", "repaired_2m_equal_2d"):
        spec = MODEL_REGISTRY[key]
        total = sum(experts[name]["parameters"] for name in spec["experts"])
        blend_parameters = gate_parameters * len(spec.get("gates", ()))
        presets[key] = {
            "kind": spec["kind"],
            "experts": list(spec["experts"]),
            "encoder_passes": int(spec["encoder_passes"]),
            "blend": (
                f"learned dense soft gate, {len(spec['gates'])} seeds averaged"
                if spec["kind"] == "multi2d_dense"
                else "fixed equal average, no learned parameters"
            ),
            "blend_parameters": blend_parameters,
            "total_parameters": total + blend_parameters,
            "uses_3d_conformer": False,
            "uses_schnet_branch": False,
        }
    return {
        "experts": experts,
        "dense_gate_parameters_per_seed": gate_parameters,
        # The gate consumes only the three experts' 3x3 predictions, so it is a
        # blend over predictions and not a fourth encoder.
        "dense_gate_input_features": 27,
        "presets": presets,
    }


def corpus() -> dict[str, object]:
    materialization = read_json(REPAIRED / "materialization_report.json")
    selection = read_json(REPAIRED / "selection_report.json")
    training = read_json(REPAIRED / "retention_d_seed42_raw" / "train_metrics.json")
    replay = training["replay_sampling"]
    return {
        "name": "repaired-2M",
        "rows": int(materialization["rows"]),
        "unique_cid": int(materialization["unique_cid"]),
        "unique_canonical_smiles": int(materialization["unique_canonical_smiles"]),
        "immutable_targeted_rows": int(selection["immutable_rows"]),
        "ledger_rows_reconciled": int(selection["ledger_rows_including_overlaps"]),
        "graphs_per_encoder": int(training["n_graphs"]),
        "train_split_old_rows": int(replay["old_train_rows"]),
        "train_split_new_rows": int(replay["new_train_rows"]),
        "targeted_replay_weight": float(replay["old_weight"]),
        "effective_targeted_draw_fraction": float(
            replay["expected_old_draw_fraction"]
        ),
        "level_of_theory": "B3LYP/6-31G*, gas phase, Kohn-Sham eigenvalues",
        "elements": "CHONSFCl",
        "molecular_weight_range": [200.0, 1000.0],
        "sources": [
            relative(REPAIRED / "materialization_report.json"),
            relative(REPAIRED / "selection_report.json"),
            relative(REPAIRED / "retention_d_seed42_raw" / "train_metrics.json"),
        ],
    }


def delta_and_uq() -> dict[str, object]:
    delta_path = (
        PRODUCTION_DIR / "05_delta_gw" / "results"
        / "delta_model_v3_desc_pred_metrics.json"
    )
    uq_path = PRODUCTION_DIR / "06_uq" / "results_v3" / "uq_ensemble_metrics.json"
    config_path = PRODUCTION_DIR / "06_uq" / "results_v3" / "feature_config.json"
    delta = read_json(delta_path)
    uq = read_json(uq_path)
    config = read_json(config_path)
    return {
        "bound_to_registry_key": config["hybrid_key"],
        "bound_to_recommended_model": config["hybrid_key"]
        in ("repaired_2m_dense_2d", "repaired_2m_equal_2d"),
        "reference": "OE62 / GW5000, G0W0@PBE0",
        "rows": int(delta["n"]),
        "scaffolds": int(delta["n_scaffolds"]),
        "delta_test_rows": int(delta["n_test"]),
        "delta": {
            target: {
                "uncorrected_b3lyp_mae_eV": float(delta[target]["mae_raw"]),
                "constant_shift_mae_eV": float(delta[target]["mae_const"]),
                "delta_model_mae_eV": float(delta[target]["mae_delta_model"]),
                "y_randomized_mae_eV": float(delta[target]["mae_yrand"]),
                "r2": float(delta[target]["r2_delta_model"]),
                "signal_real": bool(delta[target]["signal_real"]),
            }
            for target in TARGETS
        },
        "uncertainty": {
            "members": int(uq["n_members"]),
            "calibration_rows": int(uq["n_calib"]),
            "test_rows": int(uq["n_test"]),
            **{
                target: {
                    "mae_eV": float(uq[target]["mae"]),
                    "ence_before_calibration": float(uq[target]["ence_pre"]),
                    "ence_after_calibration": float(uq[target]["ence_post"]),
                    "coverage_1sigma": float(uq[target]["coverage_1sigma"]),
                    "coverage_2sigma": float(uq[target]["coverage_2sigma"]),
                }
                for target in TARGETS
            },
        },
        "sources": [relative(delta_path), relative(uq_path), relative(config_path)],
    }


def transferability() -> dict[str, object]:
    path = (
        EXPERIMENTS_DIR / "pcqm_route_b" / "results" / "official_valid_5k_fusion"
        / "metrics.json"
    )
    metrics = read_json(path)
    gap = metrics["gap_mae_eV"]
    return {
        "benchmark": "PCQM4Mv2",
        "split_read": "fixed official-validation subset",
        "rows": int(metrics["n_valid"]),
        "fusion_three_seed_gap_mae_eV": float(gap["fusion_equal_seed_ensemble"]),
        "gine_v7_single_model_gap_mae_eV": float(
            gap["gine_v7_fixed_valid_reference"]
        ),
        "official_test_used": bool(metrics["official_test_used"]),
        "sealed_20k_used": bool(metrics["sealed_20k_used"]),
        "leaderboard_submission": False,
        "ogb_compliant": False,
        "non_compliance_reason": (
            "Encoders were warm-started from PubChemQC checkpoints rather than "
            "trained from scratch on official PCQM4Mv2 data, and test-dev was "
            "never submitted. The number positions the architecture; it is not a "
            "rank."
        ),
        "warm_start_sources": [
            relative(REPAIRED / "gps9_seed42_raw" / "model.pt"),
            relative(REPAIRED / "gps11_160_seed42_raw" / "model.pt"),
        ],
        "sources": [relative(path)],
    }


def rejected_paths() -> list[dict[str, object]]:
    """Each closed branch, with the metric that closed it."""
    transplant = EXPERIMENTS_DIR / "_closed" / "archive-r07-exact2m-encoder-transplant"
    return [
        {
            "path": "exact-2M encoder transplant into routed-v4",
            "why_closed": (
                "All three paired seeds regressed: average MAE +0.005456 eV "
                "versus the paired 500K control, seed std 0.000121 eV"
            ),
            "record": relative(transplant / "decision.md"),
        },
        {
            "path": "30%-teacher distillation of the expert ensemble",
            "why_closed": (
                "Passed the internal exact-2M gate but failed fixed external "
                "retention: common +0.00482/+0.00570 eV, P8-hard "
                "+0.01187/+0.01481 eV"
            ),
            "record": relative(
                EXPERIMENTS_DIR / "distillation" / "distilled_2m_scnet" / "decision.md"
            ),
        },
        {
            "path": "pre-dispatch hard Router over the three GPS experts",
            "why_closed": (
                "Collapsed to GPS9 on every molecule, used GPS7 only for part of "
                "the LUMO target, and never selected GPS11-160, so it delivered "
                "neither specialization nor compute savings"
            ),
            "record": relative(
                REPAIRED / "three_gps_router_fusion" / "decision.md"
            ),
        },
        {
            "path": "bounded dual-SchNet residual head on the frozen 2D identity",
            "why_closed": (
                "Its internal gate passed, but on the same external molecules it "
                "regressed against its own 2D base by +0.023251 eV (equal) and "
                "+0.024239 eV (dense); the SchNet encoders remain accepted "
                "assets, only this coupling is rejected"
            ),
            "record": relative(
                REPAIRED / "hierarchical_dual_schnet_external" / "decision.md"
            ),
        },
    ]


def geometry_leverage() -> dict[str, object]:
    path = (
        EXPERIMENTS_DIR / "qm9_architecture" / "results" / "conformer_scaling"
        / "decision.md"
    )
    return {
        "question": "how much accuracy is left on the table by ETKDG geometries",
        "etkdg_to_dft_geometry_average_mae_gain_eV": -0.02428,
        "conformer_averaging_k1_to_k6_gain_eV": -0.00511,
        "interpretation": (
            "Averaging removes ETKDG's random conformer noise but not its "
            "systematic offset from a relaxed geometry: only about a fifth of "
            "the geometry gap is reachable by sampling more conformers. Geometry "
            "quality was never disproven as a lever, which is why the 3D "
            "direction is paused rather than closed."
        ),
        "scope_caveat": "measured on the QM9 architecture screen, not on PubChemQC",
        "sources": [relative(path)],
    }


def load_optional(path: Path) -> dict | None:
    return read_json(path) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cost = load_optional(FREEZE / "cost_comparison" / "dft_vs_ml_cost.json")
    experimental = load_optional(
        FREEZE / "experimental_offset" / "experimental_value_offset.json"
    )
    latency = {
        key: load_optional(FREEZE / "inference_latency" / f"{key}_local.json")
        for key in ("repaired_2m_dense_2d", "repaired_2m_equal_2d")
    }
    consistency = load_optional(
        FREEZE / "public_inference_consistency" / "repaired_2m_public_inference.json"
    )
    smoke = load_optional(
        FREEZE / "public_api_smoke_test" / "repaired_2m_smoke_test.json"
    )

    accuracy = paired_accuracy()
    result = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "slide-indexed evidence pack for the 2026-08-19 interview deck",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "recommended_registry_key": "repaired_2m_dense_2d",
        "lower_cost_registry_key": "repaired_2m_equal_2d",
        "corpus": corpus(),
        "architecture": architecture(),
        "accuracy": accuracy,
        "transferability": transferability(),
        "delta_and_uq": delta_and_uq(),
        "geometry_leverage": geometry_leverage(),
        "rejected_paths": rejected_paths(),
        "cost_vs_dft": cost,
        "experimental_offset": experimental,
        "latency": latency,
        "public_path_consistency": consistency,
        "smoke_test": smoke,
        "claim_boundaries": [
            "Targets are gas-phase B3LYP/6-31G* Kohn-Sham eigenvalues, not "
            "experimental values and not GW.",
            "Training covered CHONSFCl at MW 200-1000; anything outside is "
            "extrapolation.",
            "Delta and uncertainty are calibrated to the previous v3 base, not to "
            "the recommended model.",
            "The PCQM4Mv2 number is a fixed-validation proxy under a "
            "non-compliant warm start; it is not a leaderboard rank.",
            "The property database does not exist yet.",
            "The sealed 20K was never read.",
        ],
    }
    atomic_write(args.output, json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
