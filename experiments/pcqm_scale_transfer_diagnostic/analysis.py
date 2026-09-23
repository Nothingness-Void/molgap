"""Analysis script executing the 100K-to-500K scale-transfer diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

from molgap.scale_transfer_diagnostic import (
    compute_cross_scale_exposure_mapping,
    analyze_repository_mechanisms,
)


def run_diagnostic() -> None:
    root = Path(__file__).resolve().parent

    exposure_mapping = compute_cross_scale_exposure_mapping()

    mechanisms = [
        {
            "name": "K1 PairToken",
            "delta_100k_eV": 0.003044,
            "has_node_denoising": False,
            "has_pair_normalization": False,
            "tail_slope_stable": False,
        },
        {
            "name": "GPTrans Pair PreNorm",
            "delta_100k_eV": 0.003125,
            "has_node_denoising": False,
            "has_pair_normalization": False,
            "tail_slope_stable": False,
        },
        {
            "name": "GPTrans Noisy Nodes",
            "delta_100k_eV": 0.004840,
            "has_node_denoising": True,
            "has_pair_normalization": False,
            "tail_slope_stable": True,
        },
        {
            "name": "GPTrans Noisy Nodes + Pair Update Norm",
            "delta_100k_eV": 0.009382,
            "has_node_denoising": True,
            "has_pair_normalization": True,
            "tail_slope_stable": True,
        },
    ]

    analysis_results = analyze_repository_mechanisms(mechanisms)

    output_payload = {
        "format": "molgap-scale-transfer-diagnostic-result-v1",
        "analysis_date": "2026-09-22",
        "exposure_mapping": {
            "total_steps_100k_60ep": exposure_mapping.total_steps_100k_60ep,
            "total_steps_500k_60ep": exposure_mapping.total_steps_500k_60ep,
            "total_presentations_100k_60ep": exposure_mapping.total_presentations_100k_60ep,
            "total_presentations_500k_60ep": exposure_mapping.total_presentations_500k_60ep,
            "equivalent_500k_epochs_for_100k_end": round(exposure_mapping.equivalent_500k_epochs_for_100k_end, 3),
            "equivalent_500k_step_for_100k_end": exposure_mapping.equivalent_500k_step_for_100k_end,
        },
        "mechanisms_evaluated": analysis_results,
        "screening_rule": {
            "minimum_stqs_for_500k_authorization": 60.0,
            "highly_qualified_threshold": 80.0,
        },
    }

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "analysis.json"
    out_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Diagnostic completed successfully. Results written to {out_path}")


if __name__ == "__main__":
    run_diagnostic()
