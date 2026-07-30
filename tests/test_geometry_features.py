import torch

from molgap.geometry_features import (
    ANGLE_FEATURE_DIM,
    FULL_FEATURE_DIM,
    local_geometry_features,
    select_geometry_features,
)


def _chain_geometry():
    z = torch.tensor([6, 6, 6, 6, 1, 1], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [1.52, 0.00, 0.00],
            [2.08, 1.42, 0.00],
            [3.55, 1.55, 0.45],
            [-0.55, 0.90, 0.00],
            [4.05, 0.70, 0.70],
        ],
        dtype=torch.float32,
    )
    return z, pos


def test_local_geometry_features_are_finite_and_have_expected_dimensions():
    z, pos = _chain_geometry()
    features = local_geometry_features(z, pos)

    assert features.shape == (len(z), FULL_FEATURE_DIM)
    assert torch.isfinite(features).all()
    assert select_geometry_features(features, "angle").shape[1] == ANGLE_FEATURE_DIM
    assert select_geometry_features(features, "angle_dihedral").shape[1] == FULL_FEATURE_DIM


def test_local_geometry_features_are_rigid_transform_invariant():
    z, pos = _chain_geometry()
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transformed = pos @ rotation.T + torch.tensor([3.0, -2.0, 1.5])

    original = local_geometry_features(z, pos)
    moved = local_geometry_features(z, transformed)

    torch.testing.assert_close(original, moved, atol=1e-5, rtol=1e-5)
