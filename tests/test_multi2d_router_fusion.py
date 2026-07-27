from __future__ import annotations

import numpy as np
import torch

from molgap.multi2d_router_fusion import (
    GateTrainingConfig,
    dense_gate_features,
    fit_dense_soft_gate,
    fit_predispatch_router,
    load_dense_gate_checkpoint,
    predict_dense_gate,
    predict_hard_route,
    predispatch_features,
    route_cost,
)


def synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = 600
    descriptors = rng.normal(size=(rows, 5)).astype(np.float32)
    targets = rng.normal(size=(rows, 3)).astype(np.float32)
    predictions = np.stack(
        (
            targets + rng.normal(0, 0.12, size=targets.shape),
            targets + rng.normal(0, 0.10, size=targets.shape),
            targets + rng.normal(0, 0.14, size=targets.shape),
        ),
        axis=1,
    ).astype(np.float32)
    predictions[descriptors[:, 0] > 0.8, 2] = (
        targets[descriptors[:, 0] > 0.8]
        + rng.normal(0, 0.02, size=(np.count_nonzero(descriptors[:, 0] > 0.8), 3))
    )
    return predictions, targets, descriptors


def test_feature_shapes_and_finite_values():
    predictions, _, descriptors = synthetic()
    dense = dense_gate_features(predictions)
    predispatch = predispatch_features(predictions[:, 0], descriptors)
    assert dense.shape == (600, 27)
    assert predispatch.shape == (600, 9)
    assert np.isfinite(dense).all()
    assert np.isfinite(predispatch).all()


def test_dense_and_predispatch_models_train_and_predict():
    predictions, targets, descriptors = synthetic()
    train = np.arange(0, 400)
    validation = np.arange(400, 500)
    config = GateTrainingConfig(
        hidden_channels=16,
        batch_size=128,
        epochs=5,
        patience=3,
        seed=4,
    )
    dense, dense_report = fit_dense_soft_gate(
        predictions,
        targets,
        train,
        validation,
        config=config,
    )
    fused, weights = predict_dense_gate(dense, predictions[500:])
    assert fused.shape == (100, 3)
    assert weights.shape == (100, 3, 3)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
    assert dense_report["best_epoch"] >= 0

    router, router_report = fit_predispatch_router(
        predictions,
        targets,
        descriptors,
        train,
        validation,
        config=config,
    )
    routed, selected, probability = predict_hard_route(
        router,
        predictions[500:],
        descriptors[500:],
    )
    assert routed.shape == (100, 3)
    assert selected.shape == (100, 3)
    assert probability.shape == (100, 3, 3)
    assert np.isfinite(routed).all()
    assert router_report["best_epoch"] >= 0
    cost = route_cost(selected)
    assert 1.0 <= cost["expected_encoder_passes"] <= 3.0


def test_checkpoint_state_is_self_contained():
    predictions, targets, _ = synthetic()
    config = GateTrainingConfig(
        hidden_channels=8,
        batch_size=128,
        epochs=2,
        patience=2,
    )
    model, _ = fit_dense_soft_gate(
        predictions,
        targets,
        np.arange(0, 400),
        np.arange(400, 500),
        config=config,
    )
    state = model.state_dict()
    assert "feature_mean" in state
    assert "feature_std" in state
    assert all(torch.isfinite(value).all() for value in state.values())


def test_dense_checkpoint_round_trip(tmp_path):
    predictions, targets, _ = synthetic()
    config = GateTrainingConfig(
        hidden_channels=8,
        batch_size=128,
        epochs=2,
        patience=2,
    )
    model, report = fit_dense_soft_gate(
        predictions,
        targets,
        np.arange(0, 400),
        np.arange(400, 500),
        config=config,
    )
    path = tmp_path / "dense.pt"
    torch.save(
        {
            "kind": "three_gps_dense_soft_gate",
            "experts": ("gps7", "gps9", "gps11_160"),
            "targets": ("homo", "lumo", "gap"),
            "config": report["config"],
            "state_dict": model.state_dict(),
        },
        path,
    )
    loaded = load_dense_gate_checkpoint(path)
    expected, _ = predict_dense_gate(model, predictions[500:])
    actual, _ = predict_dense_gate(loaded, predictions[500:])
    assert np.allclose(expected, actual)
