"""Small synthetic checks for the read-only residual grouping."""

import numpy as np
import torch

from experiments.pcqm_k1_cross_scale_frozen.analyze_structural_residuals import effect, groups


def test_structure_boundaries_are_fixed_and_input_only():
    labels = groups({
        "atom_count": torch.tensor([12, 13, 15, 16, 17]),
        "conjugated_bond_fraction": torch.tensor([.25, .5, .75, .8, .1]),
    })
    assert labels["atoms_le_12"].tolist() == [True, False, False, False, False]
    assert labels["atoms_13_to_15"].tolist() == [False, True, True, False, False]
    assert labels["atoms_ge_16_and_conjugation_gt_075"].tolist() == [False, False, False, True, False]


def test_gain_is_reference_absolute_error_minus_candidate_absolute_error():
    answer = effect(
        np.array([1., 3.]), np.array([.5, 4.]), np.array([0., 2.]),
        np.array([True, True]),
    )
    assert answer == {"rows": 2, "gain_eV": -.25, "candidate_win_fraction": .5}
