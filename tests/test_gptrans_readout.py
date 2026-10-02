"""Synthetic tensor/dispatch checks only; no molecular model execution."""
import ast
from pathlib import Path

import pytest
import torch

from molgap.gptrans_readout import READOUT_MODES, apply_readout_variant, select_readout
from molgap.gptrans_author_variants import MODES, PATH_MODES, SCALED_MODES
from molgap.pcqm_gptrans_v4 import AUTHOR_MODES, _ema_decay, _scientific_fields
from molgap.experiment_spec import ADDONS
from molgap.gptrans_author_screen import validate_arm_allocation


@pytest.fixture
def states():
    node = torch.tensor([[[9.], [2.], [4.], [999.]], [[8.], [3.], [999.], [999.]]])
    pair = torch.ones(2, 1, 4, 4) * 999
    pair[:, :, 0, 0] = torch.tensor([[7.], [6.]])
    pair[0, :, 1, 2], pair[0, :, 2, 1] = 10., 20.
    mask = torch.tensor([[True, True, False], [True, False, False]])
    bonds = torch.zeros(2, 3, 3, dtype=torch.bool)
    bonds[0, 0, 1] = bonds[0, 1, 0] = True
    return node, pair, mask, bonds


def test_atom_mean_excludes_padding_and_keeps_virtual_pair(states):
    assert torch.equal(select_readout(*states, READOUT_MODES[0]), torch.tensor([[3., 7.], [3., 6.]]))


def test_bond_mean_excludes_virtual_nonbonded_and_uses_bondless_fallback(states):
    assert torch.equal(select_readout(*states, READOUT_MODES[1]), torch.tensor([[9., 15.], [8., 6.]]))


def test_permutation_invariance(states):
    node, pair, mask, bonds = states
    p = torch.tensor([0, 2, 1, 3]); real = torch.tensor([1, 0, 2])
    for mode in READOUT_MODES:
        assert torch.equal(select_readout(*states, mode), select_readout(node[:, p], pair[:, :, p][:, :, :, p],
            mask[:, real], bonds[:, real][:, :, real], mode))


def test_masks_fail_closed(states):
    node, pair, mask, bonds = states
    bonds[0, 0, 2] = True
    with pytest.raises(ValueError, match="padded"):
        select_readout(node, pair, mask, bonds, READOUT_MODES[1])


def test_readout_preserves_recipe_and_dispatch():
    for mode in READOUT_MODES:
        assert mode in set(MODES) & SCALED_MODES & set(AUTHOR_MODES)
        assert mode not in PATH_MODES
        assert (mode, "1") in ADDONS
        assert _ema_decay(mode) == .999
        assert _scientific_fields(mode) == _scientific_fields("degree_scale_ema999")
    assert validate_arm_allocation({"arms": dict.fromkeys(READOUT_MODES)}) == READOUT_MODES


def test_binding_adds_no_tensor_or_rng_use():
    model = torch.nn.Module(); model.register_parameter("retained", torch.nn.Parameter(torch.ones(2)))
    # Bind only; never call the model's forward or construct a molecular model.
    before = model.state_dict(); rng = torch.random.get_rng_state().clone()
    apply_readout_variant(model, READOUT_MODES[0])
    assert set(before) == set(model.state_dict())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_terminal_and_implementation_bindings_present():
    root = Path(__file__).parents[1]
    for name in ("gptrans_readout", "gptrans_author_acceptance", "gptrans_author_terminal", "gptrans_followup_release"):
        ast.parse((root / "src/molgap" / (name + ".py")).read_text())
    assert 'experiments/pcqm_gptrans_readout_100k' in (root / "src/molgap/gptrans_author_terminal.py").read_text()
    assert '"readout_module": preflight["readout_implementation_sha256"]' in (root / "src/molgap/pcqm_gptrans_v4.py").read_text()
