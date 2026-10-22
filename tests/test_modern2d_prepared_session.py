"""Actual shared Session on tiny generated CPU inputs, never native training."""
from copy import deepcopy

import pytest
import torch

from molgap.modern2d_execution import prepared_kernel_execution
import test_modern2d_integration_session as preview_owner
from test_modern2d_integration_session import make_preview_session
from test_family_training_session import assert_equal


@pytest.mark.parametrize("family,version", [("ppgt_small", "2"), ("esa_small", "3")])
def test_prepared_shared_session_exact_midpass_resume_and_cleanup(tmp_path, monkeypatch, family, version):
    if family == "ppgt_small":
        original_config = preview_owner.preview_config
        # Exercise the actual batched-RRWP candidate, not its already covered
        # stochastic graphwise fallback. Only the generated fixture is changed.
        monkeypatch.setattr(preview_owner, "preview_config", lambda name: {
            **original_config(name), "attention_dropout": 0., "stochastic_depth": 0.})
    complete, _, _, _ = make_preview_session(tmp_path, family, version, [])
    before = complete.kernel._model_forward
    with prepared_kernel_execution(complete.kernel) as report:
        assert report["native_qualified"] is False
        with pytest.raises(RuntimeError, match="nested"):
            with prepared_kernel_execution(complete.kernel): pass
        final = complete.run()
        endpoint = deepcopy(complete.state_dict())
    assert complete.kernel._model_forward == before
    assert not hasattr(complete.kernel, "_prepared_execution_active")
    complete.close()
    retained = []
    first, _, _, _ = make_preview_session(tmp_path, family, version, retained)
    with prepared_kernel_execution(first.kernel):
        first.run(should_pause=lambda: True)
    first.close()
    from molgap.training_reproducibility import atomic_torch_save
    checkpoint = tmp_path / (family + "-prepared-paused.pt")
    atomic_torch_save(checkpoint, retained[-1])
    serialized = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored, _, _, _ = make_preview_session(tmp_path, family, version, [])
    restored.load_state_dict(serialized)
    with prepared_kernel_execution(restored.kernel):
        assert restored.run() == final
        assert_equal(endpoint, restored.state_dict())
    restored.close()
    assert final.optimizer_step == 4 and final.sample_presentations == 512


def test_prepared_kernel_does_not_admit_real_cpu_or_scientific_cuda_training(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    session, _, _, _ = make_preview_session(tmp_path, "esa_small", "3", [])
    kernel = session.kernel
    original = kernel.recipe
    try:
        # Do not allocate CUDA; this checks only the admission boundary.
        kernel.device = torch.device("cuda")
        kernel.recipe = SimpleNamespace(optimizer_steps=4, dataset=SimpleNamespace(development=True),
            development_rows=3, selection="best-development-live", formal_interface=False)
        with pytest.raises(ValueError, match="unqualified"):
            with prepared_kernel_execution(kernel): pass
        kernel.device = torch.device("cpu")
        kernel.recipe = replace(original, dataset=replace(original.dataset, manifest="synthetic/not-real.json"))
        # An actual data descriptor, not a name alone, must still satisfy the
        # existing Session/loader admission. This scope itself adds no access.
        kernel.recipe = SimpleNamespace(dataset=SimpleNamespace(manifest="roles/train100k.json"))
        with pytest.raises(ValueError, match="unqualified"):
            with prepared_kernel_execution(kernel): pass
    finally:
        kernel.recipe, kernel.device = original, torch.device("cpu")
        session.close()
