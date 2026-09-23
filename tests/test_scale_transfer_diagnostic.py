from __future__ import annotations

from molgap.scale_transfer_diagnostic import (
    compute_cross_scale_exposure_mapping,
    compute_margin_retention_projection,
    compute_scale_transfer_qualification_score,
    analyze_repository_mechanisms,
)


def test_cross_scale_exposure_mapping():
    m = compute_cross_scale_exposure_mapping()
    assert m.steps_per_epoch_100k == 781
    assert m.steps_per_epoch_500k == 3906
    assert m.total_steps_100k_60ep == 46860
    assert m.total_steps_500k_60ep == 234360
    assert m.total_presentations_100k_60ep == 5998080
    assert m.total_presentations_500k_60ep == 29998080
    assert abs(m.equivalent_500k_epochs_for_100k_end - 11.996) < 0.01


def test_margin_retention_projection():
    # Historical unregularized case: barely crossing 0.003 eV gate
    res_pairtoken = compute_margin_retention_projection(
        delta_100k_eV=0.003044,
        has_anti_collapse_node_reg=False,
        has_anti_collapse_pair_reg=False,
    )
    assert not res_pairtoken["is_qualified_for_500k"]
    assert res_pairtoken["projected_500k_gain_regularized_eV"] < 0.001

    # Highly regularized case: Noisy Nodes + Pair Update Norm (0.00938 eV)
    res_joint = compute_margin_retention_projection(
        delta_100k_eV=0.009382,
        has_anti_collapse_node_reg=True,
        has_anti_collapse_pair_reg=True,
    )
    assert res_joint["is_qualified_for_500k"]
    assert res_joint["projected_500k_gain_regularized_eV"] > 0.006


def test_stqs_scores_historical_and_candidates():
    cases = [
        {"name": "K1 PairToken", "delta_100k_eV": 0.003044, "has_node_denoising": False, "has_pair_normalization": False},
        {"name": "GPTrans Pair PreNorm", "delta_100k_eV": 0.003125, "has_node_denoising": False, "has_pair_normalization": False},
        {"name": "GPTrans Noisy Nodes", "delta_100k_eV": 0.004840, "has_node_denoising": True, "has_pair_normalization": False},
        {"name": "GPTrans Noisy Nodes + Pair Update Norm", "delta_100k_eV": 0.009382, "has_node_denoising": True, "has_pair_normalization": True},
    ]
    results = analyze_repository_mechanisms(cases)
    scores = {r["mechanism_name"]: r for r in results}

    assert scores["K1 PairToken"]["recommendation"] == "DISQUALIFIED_HIGH_DECAY_RISK"
    assert scores["GPTrans Pair PreNorm"]["recommendation"] == "DISQUALIFIED_HIGH_DECAY_RISK"
    assert scores["GPTrans Noisy Nodes"]["recommendation"] == "QUALIFIED_FOR_500K"
    assert scores["GPTrans Noisy Nodes + Pair Update Norm"]["recommendation"] == "QUALIFIED_FOR_500K"
    assert scores["GPTrans Noisy Nodes + Pair Update Norm"]["stqs_total_score"] >= 95.0
