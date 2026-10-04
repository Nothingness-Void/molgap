"""Constructor and frozen state tests only: no data, forward or optimization."""
import json
from pathlib import Path

import pytest
import torch

from molgap.gptrans import OGBGPTransTiny
from molgap.gptrans_author_variants import apply_author_variant
from molgap.gptrans_capacity import construct, load_initial, configuration, architecture_identity, PARAMETER_CAP
from molgap.pcqm_gptrans_v4 import _state_sha256, _ema_decay, _verify_model_identity
from molgap.training_reproducibility import atomic_torch_save

MODES = ("degree_node352_ema999", "degree_ffn2_ema999", "degree_bond_local_ema999")
COUNTS = (9678273, 6822753, 5871201)


@pytest.mark.parametrize("mode,count", list(zip(MODES, COUNTS)))
def test_full_frozen_factory_roundtrip_and_rng(mode, count, tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        base = apply_author_variant(OGBGPTransTiny(), "degree_scale").state_dict()
    before = torch.get_rng_state().clone()
    model = construct(mode, base)
    assert torch.equal(torch.get_rng_state(), before)
    assert sum(p.numel() for p in model.parameters()) == count <= PARAMETER_CAP
    path = tmp_path / "initial.pt"
    atomic_torch_save(path, {"format": "molgap-gptrans-capacity-initial-v1", "variant": mode,
        "architecture_identity": architecture_identity(mode), "parameters": count,
        "state_sha256": _state_sha256(model), "model_state": model.state_dict()})
    restored = load_initial(mode, path)
    assert torch.equal(torch.get_rng_state(), before)
    assert _state_sha256(restored) == _state_sha256(model)
    assert _verify_model_identity(restored) == (count, architecture_identity(mode))
    assert _ema_decay(mode) == .999
    if mode == MODES[1]:
        assert torch.equal(model.blocks[0].ffn[0].weight[:256], base["blocks.0.ffn.0.weight"])
        assert not torch.count_nonzero(model.blocks[0].ffn[3].weight[:, 256:])
    if mode == MODES[2]:
        assert torch.equal(model.blocks[0].core.ffn[0].weight, base["blocks.0.ffn.0.weight"])
        assert not torch.count_nonzero(model.blocks[0].output.weight)


@pytest.mark.parametrize("mode", MODES)
def test_rejects_partial_factory(mode):
    with pytest.raises(ValueError, match="complete frozen"):
        apply_author_variant(OGBGPTransTiny(), mode)


def test_parameter_and_mount_authority():
    assert PARAMETER_CAP == 10493634
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "experiments/pcqm_gptrans_capacity_relations_100k/gpu/kernel-metadata.json").read_text())
    assert "kaseichou/pcqm4mv2-ogb-fixed-500k-scnet-v1" in metadata["dataset_sources"]
    assert all(s.startswith("kaseichou/") for s in metadata["dataset_sources"])


def test_scale_sampler_equal_updates_and_cycle_boundary():
    from molgap.gptrans_scale_ema import rung_indices, TOTAL_STEPS, RUNG_STEPS, BATCH
    assert TOTAL_STEPS == 60 * RUNG_STEPS == 46860 and BATCH == 128
    with pytest.raises((ValueError, IndexError)):
        rung_indices(-1)
    for rung in (0, 4, 5, 59):
        ids = rung_indices(rung)
        assert len(ids) == RUNG_STEPS * BATCH
        assert min(ids) >= 0 and max(ids) < 500000
        assert ids == rung_indices(rung)
