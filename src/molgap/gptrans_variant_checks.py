"""Tiny remote-only model checks before the train-role runtime preflight."""
from pathlib import Path


def run_checks(initial_state: Path, variant: str) -> dict:
    import torch
    from .gptrans_variants import center_valid_logits, normalize_pair
    from .pcqm_gptrans_v4 import _make_model, _verify_model_identity
    from .training_reproducibility import configure_fp32_determinism

    configure_fp32_determinism(42)
    model = _make_model(initial_state, variant).cuda()
    parameters, _ = _verify_model_identity(model)
    mask = torch.tensor([[[[False, False, False, True]]]], device="cuda")
    logits = torch.randn(1, 8, 4, 4, device="cuda")
    centered = center_valid_logits(logits, mask)
    assert torch.allclose(centered, center_valid_logits(logits + 7.0, mask), atol=2e-6, rtol=1e-5)
    padded = logits.clone()
    padded[..., -1] = 1234
    assert torch.allclose(centered[..., :-1], center_valid_logits(padded, mask)[..., :-1], atol=1e-6)
    pair = torch.randn(1, 32, 4, 4, device="cuda", requires_grad=True)
    norm = normalize_pair(pair)
    assert torch.allclose(norm.mean(1), torch.zeros_like(norm[:, 0]), atol=1e-6)
    norm.square().sum().backward()
    assert bool(torch.isfinite(pair.grad).all())
    if variant in ("memory_value", "memory_message"):
        from .gptrans_memory import readback
        delta = torch.randn_like(pair)
        observed = readback(pair, delta, mask, variant)
        changed_padding = pair.detach().clone()
        changed_padding[..., -1] = 1000
        assert torch.allclose(observed, readback(changed_padding, delta, mask, variant), atol=1e-6)
        assert not torch.allclose(observed, readback(pair + 1, delta, mask, variant))
        # With zero history both definitions reduce to the original direct message.
        original = (torch.softmax(delta.masked_fill(mask, float("-inf")), dim=-1) * delta).sum(-1).transpose(1, 2)
        assert torch.allclose(original, readback(torch.zeros_like(pair), delta, mask, variant), atol=1e-6)
        readback(pair, delta, mask, variant).sum().backward()
        assert bool(torch.isfinite(pair.grad).all())
    x = torch.zeros(4, 9, dtype=torch.long, device="cuda")
    edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], device="cuda")
    bonds = torch.zeros(4, 3, dtype=torch.long, device="cuda")
    batch = torch.tensor([0, 0, 1, 1], device="cuda")
    prediction = model(x, edges, bonds, batch)
    assert prediction.shape == (2, 1) and bool(torch.isfinite(prediction).all())
    prediction.square().mean().backward()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    return {"accepted": True, "variant": variant, "parameters": parameters,
            "initial_state_matches_reference": True, "finite_backward": True,
            "padding_and_offset_checks_passed": True, "synthetic_data_only": True}
