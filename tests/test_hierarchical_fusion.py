from __future__ import annotations

import numpy as np

from molgap.hierarchical_fusion import (
    HierarchicalFusionConfig,
    fit_hierarchical_fusion,
    hierarchical_context,
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
