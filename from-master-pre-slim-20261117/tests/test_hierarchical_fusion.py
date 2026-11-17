from __future__ import annotations

import numpy as np
import torch

from molgap.hierarchical_fusion import (
    ConservativeFusionConfig,
    ConservativeHierarchicalResidualHead,
    HierarchicalFusionConfig,
    fit_conservative_hierarchical_fusion,
    fit_hierarchical_fusion,
    hierarchical_context,
    predict_conservative_hierarchical_fusion,
    predict_hierarchical_fusion,
)


def test_hierarchical_fusion_preserves_bound_and_shapes():
    rng = np.random.default_rng(8)
    rows = 500
    targets = rng.normal(size=(rows, 3)).astype(np.float32)
    experts = np.stack(
        [
            targets + rng.normal(0, scale, size=targets.shape)
            for scale in (0.12, 0.10, 0.15)
        ],
        axis=1,
    ).astype(np.float32)
    weights = rng.dirichlet([2.0, 2.0, 1.0], size=(rows, 3)).astype(np.float32)
    base = np.sum(weights * experts.transpose(0, 2, 1), axis=-1)
    primary = rng.normal(size=(rows, 12)).astype(np.float32)
    augmented = rng.normal(size=(rows, 10)).astype(np.float32)
    context = hierarchical_context(experts, weights, primary, augmented)
    config = HierarchicalFusionConfig(
        hidden_channels=16,
        correction_scale_eV=0.10,
        batch_size=128,
        epochs=4,
        patience=3,
    )
    model, report = fit_hierarchical_fusion(
        base,
        context,
        targets,
        np.arange(0, 350),
        np.arange(350, 425),
        config=config,
    )
    prediction, correction = predict_hierarchical_fusion(
        model,
        base[425:],
        context[425:],
    )
    assert prediction.shape == (75, 3)
    assert correction.shape == (75, 3)
    assert np.max(np.abs(correction)) <= 0.100001
    assert np.isfinite(prediction).all()
    assert report["best_epoch"] >= 0


def test_conservative_head_starts_as_exact_identity() -> None:
    config = ConservativeFusionConfig(gate_init=0.1, correction_scale_eV=0.03)
    model = ConservativeHierarchicalResidualHead(
        7,
        np.zeros(7, dtype=np.float32),
        np.ones(7, dtype=np.float32),
        config,
    )
    base = torch.randn(5, 3)
    context = torch.randn(5, 7) * 1_000
    prediction, correction, confidence = model(base, context)
    torch.testing.assert_close(prediction, base)
    torch.testing.assert_close(correction, torch.zeros_like(correction))
    torch.testing.assert_close(confidence, torch.full_like(confidence, 0.1))


def test_conservative_fit_can_retain_identity(tmp_path) -> None:
    rng = np.random.default_rng(22)
    rows = 160
    targets = rng.normal(size=(rows, 3)).astype(np.float32)
    base = targets.copy()
    context = rng.normal(size=(rows, 9)).astype(np.float32)
    config = ConservativeFusionConfig(
        hidden_channels=8,
        batch_size=64,
        epochs=3,
        patience=2,
        minimum_validation_improvement_eV=0.0001,
    )
    model, report = fit_conservative_hierarchical_fusion(
        base,
        context,
        targets,
        np.arange(0, 100),
        np.arange(100, 130),
        config=config,
        checkpoint_path=tmp_path / "checkpoint.pt",
        progress_path=tmp_path / "progress.json",
        contract_id="identity-smoke",
    )
    prediction, correction, confidence = predict_conservative_hierarchical_fusion(
        model,
        base[130:],
        context[130:],
    )
    assert report["selected_identity"] is True
    np.testing.assert_array_equal(prediction, base[130:])
    np.testing.assert_array_equal(correction, np.zeros_like(correction))
    assert np.isfinite(confidence).all()

    resumed, resumed_report = fit_conservative_hierarchical_fusion(
        base,
        context,
        targets,
        np.arange(0, 100),
        np.arange(100, 130),
        config=config,
        checkpoint_path=tmp_path / "checkpoint.pt",
        progress_path=tmp_path / "progress.json",
        resume=True,
        contract_id="identity-smoke",
    )
    resumed_prediction = predict_conservative_hierarchical_fusion(
        resumed, base[130:], context[130:]
    )[0]
    assert resumed_report["selected_identity"] is True
    np.testing.assert_array_equal(resumed_prediction, base[130:])
