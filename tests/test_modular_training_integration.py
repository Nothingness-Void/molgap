"""Optional Clean models use the owning forward without changing Live state."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from molgap.family_training_kernel import FamilyKernel


@pytest.mark.parametrize("family", ("neural_atom_k1", "gptrans_t"))
def test_explicit_model_forward_is_independent_of_live_weights(monkeypatch, family):
    kernel = FamilyKernel.__new__(FamilyKernel)
    kernel.recipe = SimpleNamespace(family=family)
    kernel.device = torch.device("cpu")
    kernel.model = torch.nn.Linear(1, 1, bias=False)
    clean = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        kernel.model.weight.fill_(1)
        clean.weight.fill_(2)
    seen = []
    if family == "neural_atom_k1":
        import molgap.k1_execution_layout as layout
        import molgap.family_model_ops as owning
        @contextmanager
        def bound(model, batch, device):
            seen.append(model)
            yield
        monkeypatch.setattr(layout, "cpu_layout_context", bound)
        monkeypatch.setattr(owning, "k1_batch_to_device", lambda batch, *args, **kwargs: batch)
        monkeypatch.setattr(owning, "k1_forward", lambda model, batch: model(batch.x))
    else:
        import molgap.gptrans_execution_backend as owning
        kernel._gptrans_batch = lambda batch, **kwargs: batch
        def forward(model, batch):
            seen.append(model)
            return model(batch.x)
        monkeypatch.setattr(owning, "prepared_forward", forward)
    batch = SimpleNamespace(x=torch.ones(2, 1))
    assert torch.equal(kernel.forward(batch), torch.ones(2, 1))
    assert torch.equal(kernel.forward(batch, model=clean), torch.full((2, 1), 2.))
    assert seen == [kernel.model, clean]
    assert torch.equal(kernel.model.weight, torch.ones(1, 1))
