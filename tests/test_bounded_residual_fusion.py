import torch

from molgap.fusion import BoundedResidualFusionHead


def test_bounded_residual_starts_as_exact_identity():
    model = BoundedResidualFusionHead(dim_2d=8, dim_3d=5, hidden=16)
    h2 = torch.randn(4, 8)
    h3 = torch.randn(4, 5)
    baseline = torch.randn(4, 3)

    torch.testing.assert_close(model(h2, h3, baseline), baseline)


def test_bounded_residual_never_exceeds_limit():
    model = BoundedResidualFusionHead(
        dim_2d=8,
        dim_3d=5,
        hidden=16,
        max_abs_correction_eV=0.05,
    )
    with torch.no_grad():
        model.delta_head[-1].weight.fill_(100.0)
        model.delta_head[-1].bias.fill_(100.0)
    correction = model.correction(
        torch.randn(4, 8),
        torch.randn(4, 5),
        torch.randn(4, 3),
    )

    assert torch.all(correction.abs() <= 0.05)
