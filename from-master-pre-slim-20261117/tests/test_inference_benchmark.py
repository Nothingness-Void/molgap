from __future__ import annotations

import pytest

from molgap.inference_benchmark import benchmark_markdown, benchmark_smiles


def test_benchmark_smiles_repeats_source_to_exact_rows() -> None:
    assert benchmark_smiles(["CCO", "c1ccccc1"], 5) == [
        "CCO",
        "c1ccccc1",
        "CCO",
        "c1ccccc1",
        "CCO",
    ]


@pytest.mark.parametrize("source, rows", [([], 1), (["CCO"], 0)])
def test_benchmark_smiles_rejects_empty_or_nonpositive_input(source, rows) -> None:
    with pytest.raises(ValueError):
        benchmark_smiles(source, rows)


def test_benchmark_markdown_labels_new_smiles_scope() -> None:
    document = benchmark_markdown(
        {
            "model": "routed_gps7_gps9_schnet_500k_v4",
            "hardware": {"device": "cpu"},
            "measurement": {"model_load_s": 1.0, "timed_repeats_per_size": 3},
            "results": [
                {
                    "input_rows": 1,
                    "median_batch_s": 0.1,
                    "p95_batch_s": 0.1,
                    "median_ms_per_molecule": 100.0,
                    "median_molecules_per_s": 10.0,
                    "routed_fraction": 0.0,
                }
            ],
        }
    )
    assert "not a precomputed-catalog lookup" in document
