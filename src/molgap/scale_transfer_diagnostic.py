"""Scale-transfer diagnostic and qualification engine for PCQM architectures.

Models cross-scale presentation coordinates, empirical margin retention frontiers,
and computes deterministic Scale-Transfer Qualification Scores (STQS) from
100K learning curve dynamics and regularization mechanisms.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CrossScaleExposureMapping:
    train_size_100k: int
    train_size_500k: int
    batch_size: int
    steps_per_epoch_100k: int
    steps_per_epoch_500k: int
    total_steps_100k_60ep: int
    total_steps_500k_60ep: int
    total_presentations_100k_60ep: int
    total_presentations_500k_60ep: int
    equivalent_500k_epochs_for_100k_end: float
    equivalent_500k_step_for_100k_end: int


def compute_cross_scale_exposure_mapping(
    train_size_100k: int = 100000,
    train_size_500k: int = 500000,
    batch_size: int = 128,
    epochs: int = 60,
) -> CrossScaleExposureMapping:
    """Compute exact optimizer step and sample presentation mapping across scales."""
    steps_100k = train_size_100k // batch_size
    steps_500k = train_size_500k // batch_size
    total_steps_100k = steps_100k * epochs
    total_steps_500k = steps_500k * epochs
    pres_100k = steps_100k * batch_size * epochs
    pres_500k = steps_500k * batch_size * epochs

    equiv_epochs = pres_100k / train_size_500k
    equiv_step = total_steps_100k

    return CrossScaleExposureMapping(
        train_size_100k=train_size_100k,
        train_size_500k=train_size_500k,
        batch_size=batch_size,
        steps_per_epoch_100k=steps_100k,
        steps_per_epoch_500k=steps_500k,
        total_steps_100k_60ep=total_steps_100k,
        total_steps_500k_60ep=total_steps_500k,
        total_presentations_100k_60ep=pres_100k,
        total_presentations_500k_60ep=pres_500k,
        equivalent_500k_epochs_for_100k_end=equiv_epochs,
        equivalent_500k_step_for_100k_end=equiv_step,
    )


def compute_margin_retention_projection(
    delta_100k_eV: float,
    has_anti_collapse_node_reg: bool = False,
    has_anti_collapse_pair_reg: bool = False,
    historical_unregularized_decay_rate: float = 0.817,
) -> dict[str, Any]:
    """Project 500K retained gain based on 100K gain and regularization strength."""
    threshold = 0.003
    margin_100k = delta_100k_eV - threshold

    # Unregularized structural bias experiences severe decay across 7.5x step horizon.
    unregularized_retention_fraction = 1.0 - historical_unregularized_decay_rate
    unregularized_expected_500k_gain = delta_100k_eV * unregularized_retention_fraction

    # Regularization (Noisy Nodes, Pair Norm) protects representation rank.
    # Mitigation factor: node denoising reduces decay by ~50%; joint reduces by ~75%.
    decay_reduction = 0.0
    if has_anti_collapse_node_reg:
        decay_reduction += 0.45
    if has_anti_collapse_pair_reg:
        decay_reduction += 0.35
    effective_decay_rate = max(0.10, historical_unregularized_decay_rate * (1.0 - decay_reduction))
    regularized_retention_fraction = 1.0 - effective_decay_rate
    regularized_expected_500k_gain = delta_100k_eV * regularized_retention_fraction

    # Critical 100K gain required to clear 0.003 eV gate under respective regimes:
    critical_unregularized_100k_gain = threshold / unregularized_retention_fraction
    critical_regularized_100k_gain = threshold / regularized_retention_fraction

    is_qualified = regularized_expected_500k_gain >= threshold

    return {
        "delta_100k_eV": delta_100k_eV,
        "margin_above_100k_gate_eV": margin_100k,
        "unregularized_decay_rate": historical_unregularized_decay_rate,
        "effective_decay_rate": effective_decay_rate,
        "projected_500k_gain_unregularized_eV": unregularized_expected_500k_gain,
        "projected_500k_gain_regularized_eV": regularized_expected_500k_gain,
        "critical_100k_gain_unregularized_eV": critical_unregularized_100k_gain,
        "critical_100k_gain_regularized_eV": critical_regularized_100k_gain,
        "margin_buffer_over_critical_regularized_eV": delta_100k_eV - critical_regularized_100k_gain,
        "is_qualified_for_500k": is_qualified,
    }


def compute_scale_transfer_qualification_score(
    mechanism_name: str,
    delta_100k_eV: float,
    has_node_denoising: bool,
    has_pair_normalization: bool,
    tail_slope_stable: bool = True,
    historical_decay_rate: float = 0.817,
) -> dict[str, Any]:
    """Compute the deterministic Scale-Transfer Qualification Score (STQS) [0 - 100]."""
    threshold = 0.003

    # Margin subscore (0 - 50 pts)
    # 0.00300 eV -> 20 pts; 0.00484 eV -> 35 pts; >= 0.00800 eV -> 50 pts
    if delta_100k_eV < threshold:
        margin_score = 0.0
    else:
        excess = delta_100k_eV - threshold
        margin_score = min(50.0, 20.0 + (excess / 0.005) * 30.0)

    # Node anti-collapse subscore (0 - 25 pts)
    node_score = 25.0 if has_node_denoising else 0.0

    # Relation variance normalization subscore (0 - 20 pts)
    pair_score = 20.0 if has_pair_normalization else 0.0

    # Tail stability subscore (0 - 5 pts)
    tail_score = 5.0 if tail_slope_stable else 0.0

    total_score = margin_score + node_score + pair_score + tail_score

    # Disposition decision
    if total_score >= 60.0:
        recommendation = "QUALIFIED_FOR_500K"
    elif total_score >= 40.0:
        recommendation = "PROVISIONAL_NEEDS_REPLICATION"
    else:
        recommendation = "DISQUALIFIED_HIGH_DECAY_RISK"

    projection = compute_margin_retention_projection(
        delta_100k_eV,
        has_anti_collapse_node_reg=has_node_denoising,
        has_anti_collapse_pair_reg=has_pair_normalization,
        historical_unregularized_decay_rate=historical_decay_rate,
    )

    return {
        "mechanism_name": mechanism_name,
        "stqs_total_score": round(total_score, 2),
        "recommendation": recommendation,
        "score_components": {
            "margin_score": round(margin_score, 2),
            "node_anti_collapse_score": node_score,
            "pair_normalization_score": pair_score,
            "tail_stability_score": tail_score,
        },
        "projection": projection,
    }


def analyze_repository_mechanisms(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate STQS scores across historical and candidate mechanisms."""
    results = []
    for c in cases:
        score = compute_scale_transfer_qualification_score(
            mechanism_name=c["name"],
            delta_100k_eV=c["delta_100k_eV"],
            has_node_denoising=c.get("has_node_denoising", False),
            has_pair_normalization=c.get("has_pair_normalization", False),
            tail_slope_stable=c.get("tail_slope_stable", True),
        )
        results.append(score)
    return results
