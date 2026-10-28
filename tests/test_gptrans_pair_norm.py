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


def test_bond_local_pair_norm_supports_proportional_profile_width(tmp_path) -> None:
    from dataclasses import replace
    from molgap.family_training_kernel import load_family_model
    from molgap.gptrans_bond_local import APPROVED_SHAPES
    from molgap.v4_runtime import state_dict_sha256
    from test_family_training_recipe import recipe
    from molgap.gptrans_bond_local import make_bond_local_model
    reference = make_bond_local_model(pair_update_norm=True)
    assert sum(p.numel() for p in reference.parameters()) == 5_871_201
    assert state_dict_sha256(make_bond_local_model().state_dict()) == (
        "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b")
    for mode, (layers, nodes, pairs, heads) in APPROVED_SHAPES.items():
        candidate = make_bond_local_model(num_layers=layers, node_channels=nodes,
            pair_channels=pairs, num_heads=heads, pair_update_norm=True)
        digest = state_dict_sha256(candidate.state_dict())
        initial = tmp_path / (mode + ".pt")
        torch.save(candidate.state_dict(), initial)
        frozen = replace(recipe("gptrans_t"), mode=mode, initialization_sha256=digest)
        loaded, identity = load_family_model(frozen, initial)
        assert identity["state_sha256"] == state_dict_sha256(loaded.state_dict()) == digest
        assert sum(p.numel() for p in loaded.parameters()) == (
            12_478_993 if nodes == 384 else 7_700_481)
        assert len(loaded.blocks) == layers
        for block in loaded.blocks:
            assert block.norm.normalized_shape == (nodes,)
            assert block.core.variant == "pair_update_norm"
            assert block.core.attention.head_channels == 32
