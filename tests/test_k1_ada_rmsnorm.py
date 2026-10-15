"""Tests for K1 AdaRMSNorm model and sandbox execution."""
import io
import torch
from molgap.k1_ada_rmsnorm import AdaRMSNorm, make_k1_ada_rmsnorm
from molgap.fast_sandbox import run_sandbox, synthetic_batch
from molgap.k1_screen_training import _forward


def test_ada_rmsnorm_identity():
    norm = AdaRMSNorm(192)
    x = torch.randn(10, 192)
    out = norm(x)
    # At alpha=0, beta=1, out should be standard RMSNorm(x)
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = x * rms
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_k1_ada_rmsnorm_architecture():
    model = make_k1_ada_rmsnorm()
    assert sum(p.numel() for p in model.parameters()) == 3_658_817
    assert len(list(model.buffers())) == 0
    assert len(model.local_blocks) == 9
    for block in model.local_blocks:
        assert isinstance(block.norm1, AdaRMSNorm)
        assert isinstance(block.norm3, AdaRMSNorm)
        assert not block.norm_with_batch


def test_k1_ada_rmsnorm_sandbox_execution():
    def factory():
        return make_k1_ada_rmsnorm()

    result = run_sandbox(factory, forward=_forward, seed=42, threads=1)
    assert result["status"] == "PASSED"
    assert result["checks"]["forward"]["status"] == "PASSED"
    assert result["checks"]["backbone_gradients"]["status"] == "PASSED"
    assert result["checks"]["state_dict_roundtrip"]["status"] == "PASSED"
    assert "error" not in result
