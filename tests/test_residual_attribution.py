import pandas as pd

from molgap.residual_attribution import analyze_comparison


def test_analyze_comparison_reports_paired_delta():
    frame = pd.DataFrame(
        {
            "smiles": ["CCO"] * 20,
            "homo": [0.0] * 20,
            "lumo": [1.0] * 20,
            "gap": [1.0] * 20,
            "base_homo": [0.2] * 20,
            "base_lumo": [1.2] * 20,
            "base_gap": [1.2] * 20,
            "candidate_homo": [0.1] * 20,
            "candidate_lumo": [1.1] * 20,
            "candidate_gap": [1.1] * 20,
        }
    )
    report, strata = analyze_comparison(
        frame, baseline="base", candidates=["candidate"]
    )
    metrics = report["candidates"]["candidate"]["scopes"]["all"]
    assert metrics["average"]["delta_mae_eV"] < 0
    assert metrics["average"]["candidate_win_rate"] == 1.0
    assert not strata.empty
