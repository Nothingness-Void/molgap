import pytest
import torch

from molgap.reproducibility_audit import (
    accept_short_training_retests,
    install_sorted_segment_scatter,
    summarize_repetitions,
)


def test_summarize_repetitions_accepts_identical_forward_and_gradient() -> None:
    rows = [
        {
            "error": None,
            "forward_sha256": "forward",
            "gradient_sha256": "gradient",
            "forward_max_abs_diff": 0.0,
            "gradient_max_abs_diff": 0.0,
        }
        for _ in range(3)
    ]

    result = summarize_repetitions(rows)

    assert result["successful_repetitions"] == 3
    assert result["all_forward_bitwise_equal"] is True
    assert result["all_gradient_bitwise_equal"] is True


def test_summarize_repetitions_rejects_hash_drift_and_errors() -> None:
    rows = [
        {
            "error": None,
            "forward_sha256": "a",
            "gradient_sha256": "same",
            "forward_max_abs_diff": 0.0,
            "gradient_max_abs_diff": 0.0,
        },
        {
            "error": None,
            "forward_sha256": "b",
            "gradient_sha256": "same",
            "forward_max_abs_diff": 1.0e-6,
            "gradient_max_abs_diff": 0.0,
        },
        {"error": "RuntimeError: unsupported"},
    ]

    result = summarize_repetitions(rows)

    assert result["successful_repetitions"] == 2
    assert result["all_forward_bitwise_equal"] is False
    assert result["all_gradient_bitwise_equal"] is True
    assert result["errors"] == ["RuntimeError: unsupported"]


def test_sorted_segment_scatter_matches_cpu_reference() -> None:
    torch_scatter = pytest.importorskip("torch_scatter")

    source = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]])
    index = torch.tensor([1, 0, 1, 0])
    reference_sum = torch_scatter.scatter(source, index, dim=0, dim_size=3, reduce="sum")
    reference_mean = torch_scatter.scatter(source, index, dim=0, dim_size=3, reduce="mean")
    reference_max = torch_scatter.scatter(source, index, dim=0, dim_size=3, reduce="max")

    install_sorted_segment_scatter()

    assert torch.equal(
        torch_scatter.scatter(source, index, dim=0, dim_size=3, reduce="sum"),
        reference_sum,
    )
    assert torch.equal(
        torch_scatter.scatter(source, index, dim=0, dim_size=3, reduce="mean"),
        reference_mean,
    )
    assert torch.equal(
        torch_scatter.scatter(source, index, dim=0, dim_size=3, reduce="max"),
        reference_max,
    )


def test_accept_short_training_retests_requires_exact_replay(tmp_path) -> None:
    common_metrics = {
        "candidate": "candidate",
        "model_source_commit": "source",
        "cache_aggregate_sha256": "cache",
        "seed": 42,
        "precision": "float32",
        "batch_size": 48,
        "optimizer": "AdamW",
        "learning_rate": 0.00016,
        "weight_decay": 0.000001,
        "target": "gap",
        "parameter_count": 10,
        "device": "Z100SM",
        "epochs": [
            {"epoch": 0, "train_mae_eV": 0.2, "validation_mae_eV": 0.15},
            {"epoch": 1, "train_mae_eV": 0.1, "validation_mae_eV": 0.12},
        ],
    }
    common_runtime = {
        "complete": True,
        "rocblas_default_atomics_mode": "0",
        "deterministic_algorithms_enabled": True,
        "scatter_implementation": (
            "unique-group-key+argsort+searchsorted+segment_csr"
        ),
        "data_loader_generator_seed": 42,
        "initialization": {"model_state_sha256": "initial"},
        "first_batch": {"prediction_sha256": "prediction", "loss": 1.0},
    }
    state = {"weight": torch.tensor([1.0, 2.0])}
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        (root / "metrics.json").write_text(
            __import__("json").dumps(common_metrics), encoding="utf-8"
        )
        (root / "deterministic_runtime.json").write_text(
            __import__("json").dumps(common_runtime), encoding="utf-8"
        )
        torch.save(state, root / "best_model.pt")
        torch.save({"model_state": state}, root / "last_checkpoint.pt")

    output = tmp_path / "acceptance.json"
    result = accept_short_training_retests(
        tmp_path / "first", tmp_path / "second", output
    )

    assert result["accepted"] is True
    assert result["epoch_metrics_bitwise_equal"] is True
    assert output.exists()
