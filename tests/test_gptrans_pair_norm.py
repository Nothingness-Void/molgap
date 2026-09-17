from __future__ import annotations

import torch

from molgap.gptrans import GPTransBlock
from molgap.gptrans_variants import MODES, PairStateNormalizationBlock, normalize_pair


def test_pair_normalization_modes_are_registered() -> None:
    assert {"pair_update_norm", "pair_post_norm"}.issubset(MODES)


def test_parameter_free_block_preserves_parameter_identity() -> None:
    reference = GPTransBlock(32, 8, 4, 0.0, 0.0, 1.0)
    candidate = PairStateNormalizationBlock(
        32, 8, 4, 0.0, 0.0, 1.0, variant="pair_update_norm"
    )
    candidate.load_state_dict(reference.state_dict(), strict=True)
    assert list(reference.state_dict()) == list(candidate.state_dict())
    assert sum(p.numel() for p in reference.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )


def test_pair_normalization_is_per_pair_across_channels() -> None:
    pair = torch.arange(2 * 4 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3)
    normalized = normalize_pair(pair)
    channel_mean = normalized.permute(0, 2, 3, 1).mean(dim=-1)
    assert torch.allclose(channel_mean, torch.zeros_like(channel_mean), atol=1e-6)
