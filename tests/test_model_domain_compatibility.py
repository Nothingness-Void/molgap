"""Synthetic compatibility coverage for the first model-domain migration."""
import importlib
import io

import pytest
import torch

from test_experiment_spec import payload


CASES = [
    ("schnet", "SchNetWrapper", dict(hidden_channels=8, num_filters=8, num_interactions=1, num_gaussians=5, cutoff=3., dropout=0.)),
    ("gine", "GINEWrapper", dict(in_channels=3, edge_dim=2, hidden_channels=8, num_layers=1, dropout=0.)),
    ("egnn", "EGNNWrapper", dict(hidden_channels=8, num_layers=1, num_rbf=4, dropout=0.)),
    ("tensornet", "TensorNetWrapper", dict(hidden_channels=8, num_layers=1, num_rbf=4, dropout=0.)),
    ("fusion", "FusionHead", dict(hidden=8, dim_2d=3, dim_3d=4, dropout=0.)),
]


def inputs(name):
    if name == "fusion":
        return (torch.randn(2, 3), torch.randn(2, 4))
    if name == "gine":
        return (torch.randn(4, 3), torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
                torch.randn(4, 2), torch.tensor([0, 0, 1, 1]))
    return (torch.tensor([6, 1, 8, 1]),
            torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [0., 1., 0.]]),
            torch.tensor([0, 0, 1, 1]))


@pytest.mark.parametrize("name,class_name,kwargs", CASES)
def test_old_import_state_forward_gradient_and_pickle(name, class_name, kwargs):
    old = importlib.import_module(f"molgap.{name}")
    new = importlib.import_module(f"molgap.models.{name}")
    cls = getattr(old, class_name)
    assert cls is getattr(new, class_name)
    assert cls.__module__ == f"molgap.{name}"
    torch.manual_seed(31)
    model = cls(**kwargs).eval()
    args = inputs(name)
    output = model(*args)
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g).item() for g in grads)
    restored = getattr(new, class_name)(**kwargs).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.state_dict().keys() == model.state_dict().keys()
    torch.testing.assert_close(restored(*args), output.detach(), rtol=0, atol=0)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=False)
    assert type(loaded) is cls
    torch.testing.assert_close(loaded(*args), output.detach(), rtol=0, atol=0)


def test_all_nested_serialized_model_classes_keep_historical_names():
    for name, _, _ in CASES:
        old = importlib.import_module(f"molgap.{name}")
        new = importlib.import_module(f"molgap.models.{name}")
        for symbol in old.__all__:
            item = getattr(old, symbol)
            assert item is getattr(new, symbol)
            assert item.__module__ == f"molgap.{name}"


def test_static_package_closure_includes_model_implementations():
    from pathlib import Path
    from molgap.source_dependencies import discover_source_dependencies

    root = Path(__file__).resolve().parents[1]
    roots = [f"src/molgap/{name}.py" for name, _, _ in CASES]
    result = discover_source_dependencies(root, roots, require_static=True)
    assert "src/molgap/models/__init__.py" in result.source_files
    for name, _, _ in CASES:
        assert f"src/molgap/models/{name}.py" in result.source_files


def test_registered_package_includes_selected_model_shim_dependencies(payload):
    from molgap.experiment_source_inventory import registered_source_files
    from molgap.experiment_spec import ExperimentSpec

    payload["arms"] = [arm for arm in payload["arms"] if arm["family"]["name"] == "gptrans_t"]
    selected = registered_source_files(
        ExperimentSpec(payload),
        extras=["src/molgap/schnet.py", "src/molgap/fusion.py"],
    )
    assert "src/molgap/models/__init__.py" in selected
    assert "src/molgap/models/schnet.py" in selected
    assert "src/molgap/models/fusion.py" in selected
    assert "src/molgap/models/tensornet.py" not in selected
