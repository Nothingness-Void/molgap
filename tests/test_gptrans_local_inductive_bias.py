"""Synthetic-only checks for the two local inductive-bias additions."""
import pytest
import torch
import json
import hashlib

from molgap.gptrans import OGBGPTransTiny
from molgap.gptrans_local_inductive_bias import apply_local_inductive_bias
from molgap.pcqm_gptrans_v4 import _load_observed_trace
from molgap.v4_runtime import model_state_sha256


def _inputs():
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    return {
        "x": torch.zeros((5, 9), dtype=torch.long),
        "edge_index": edges,
        "edge_attr": torch.zeros((edges.shape[1], 3), dtype=torch.long),
        "batch": torch.tensor([0, 0, 0, 1, 1]),
        "random_walk_pe": torch.arange(80, dtype=torch.float32).reshape(5, 16) / 80,
    }


@pytest.mark.parametrize("mode", ["rwse16", "rwse16_local_edge"])
def test_frozen_random_core_and_initial_output_are_preserved(mode):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        base = OGBGPTransTiny()
        core_sha = model_state_sha256(base)
        rng_before = torch.get_rng_state().clone()
        candidate = apply_local_inductive_bias(base, mode)
        assert torch.equal(torch.get_rng_state(), rng_before)
    assert core_sha == "8988db8659c6c7e2b27401312f43684215c34cd8d69309aed7ce946ee9cb1ec6"
    assert model_state_sha256(candidate.base) == core_sha
    candidate.eval()
    candidate.base.eval()
    inputs = _inputs()
    with torch.no_grad():
        expected = candidate.base(**inputs)
        observed = candidate(**inputs)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["rwse16", "rwse16_local_edge"])
def test_new_paths_have_finite_first_step_gradients(mode):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        candidate = apply_local_inductive_bias(OGBGPTransTiny(), mode)
    candidate.eval()
    loss = candidate(**_inputs()).square().sum()
    loss.backward()
    assert torch.isfinite(candidate.rwse_to_node.weight.grad).all()
    assert candidate.rwse_to_node.weight.grad.abs().sum() > 0
    if mode == "rwse16_local_edge":
        gradients = [layer.weight.grad for layer in candidate.edge_to_node]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_rwse_and_real_bond_alignment_fail_closed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        candidate = apply_local_inductive_bias(OGBGPTransTiny(), "rwse16_local_edge")
    data = _inputs()
    with pytest.raises(ValueError, match="RWSE16"):
        candidate(**{**data, "random_walk_pe": None})
    with pytest.raises(ValueError, match="RWSE16"):
        candidate(**{**data, "random_walk_pe": torch.full((5, 16), float("nan"))})
    crossing = data["edge_index"].clone()
    crossing[1, 0] = 3
    with pytest.raises(ValueError, match="crosses graph boundaries"):
        candidate(**{**data, "edge_index": crossing})


def test_observed_trace_binds_saved_checkpoint_and_reconciles_one_crash_gap(tmp_path):
    checkpoint = tmp_path / "last_checkpoint.pt"
    checkpoint.write_bytes(b"observed checkpoint bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    raw = [{
        "epoch": 0,
        "optimizer_steps_cumulative": 781,
        "sample_presentations_cumulative": 99_968,
        "learning_rate": 2.5e-4,
        "train_mae_eV": 0.3,
        "live_development_mae_eV": 0.28,
        "development_mae_eV": 0.27,
        "elapsed_seconds": 10.0,
    }]
    observed = _load_observed_trace(tmp_path, "rwse16", raw, checkpoint)
    assert observed[0]["optimizer_step"] == 781
    assert observed[0]["sample_presentations"] == 99_968
    assert observed[0]["live_dev_mae_eV"] == 0.28
    assert observed[0]["checkpoint_identity"] == digest
    assert json.loads((tmp_path / "observed_trace.json").read_text())["rows"] == observed
    assert _load_observed_trace(tmp_path, "rwse16", raw, checkpoint) == observed
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="checkpoint hash changed"):
        _load_observed_trace(tmp_path, "rwse16", raw, checkpoint)
