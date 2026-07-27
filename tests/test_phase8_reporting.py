from molgap.phase8_reporting import write_comparison


def test_report_preserves_pending_cells(tmp_path):
    rows = [
        {
            "model": "pending",
            "role": "candidate",
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
    assert "| pending | candidate |  |  |  |  |  |  |  |" in (
        tmp_path / "report.md"
    ).read_text()
