from molgap.model_reporting import write_comparison


def test_report_preserves_pending_cells(tmp_path):
    rows = [
        {
            "model": "pending",
            "role": "candidate",
            "common_eval_rows": None,
            "common_average_mae_eV": None,
            "ood_average_mae_eV": None,
            "p8_hard_average_mae_eV": None,
            "pcqm_gap_mae_eV": None,
            "pubchemqc100k_average_mae_eV": None,
            "pubchemqc100k_gap_mae_eV": None,
            "encoder_passes": None,
        }
    ]
    write_comparison(rows, tmp_path / "report.csv", tmp_path / "report.md")
    assert "| pending | candidate |  |  |  |  |  |  |  |  |" in (
        tmp_path / "report.md"
    ).read_text()


def test_report_marks_row_counts_so_scopes_are_not_mixed(tmp_path):
    rows = [
        {
            "model": "paired",
            "role": "production",
            "common_eval_rows": 1973,
            "common_average_mae_eV": 0.097638,
            "ood_average_mae_eV": 0.106655,
            "p8_hard_average_mae_eV": 0.088407,
            "pcqm_gap_mae_eV": 0.302120,
            "pubchemqc100k_average_mae_eV": None,
            "pubchemqc100k_gap_mae_eV": None,
            "encoder_passes": 3.0,
        }
    ]
    write_comparison(rows, tmp_path / "report.csv", tmp_path / "report.md")
    document = (tmp_path / "report.md").read_text()
    assert "| paired | production | 1,973 |" in document
    assert "only comparable within the same `Rows` value" in document
