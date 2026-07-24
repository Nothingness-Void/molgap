import torch

from molgap.egnn import EGNNWrapper


def test_egnn_prediction_is_rigid_transform_invariant():
    torch.manual_seed(7)
    model = EGNNWrapper(hidden_channels=32, num_layers=2, num_rbf=8, dropout=0.0)
    model.eval()
    z = torch.tensor([6, 7, 8, 1, 6, 1], dtype=torch.long)
    pos = torch.randn(6, 3)
    batch = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transformed = pos @ rotation.T + torch.tensor([2.0, -3.0, 1.5])

    with torch.no_grad():
        original = model(z, pos, batch)
        rigid_transform = model(z, transformed, batch)

    torch.testing.assert_close(original, rigid_transform, atol=1e-6, rtol=1e-6)
