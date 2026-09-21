from molgap.gptrans_feature_denoising import (
    ARMS,
    ATOM_FEATURE_DIMS,
    BOND_FEATURE_DIMS,
    EXPECTED_PARAMETERS,
    _scientific_fields,
)


def test_feature_denoising_parameter_budgets_are_frozen():
    assert ARMS == ("full_atom", "full_atom_bond")
    assert ATOM_FEATURE_DIMS == (119, 5, 12, 12, 10, 6, 6, 2, 2)
    assert BOND_FEATURE_DIMS == (5, 6, 2)
    assert EXPECTED_PARAMETERS == {
        "full_atom": 5_291_535,
        "full_atom_bond": 5_291_964,
    }


def test_feature_denoising_contracts_keep_v4_training_identity():
    for arm in ARMS:
        contract = _scientific_fields(arm)
        assert contract["physical_batch_per_device"] == 128
        assert contract["precision"] == "fp32"
        assert contract["tf32_enabled"] is False
        assert contract["deterministic_algorithms"] is True
        assert contract["epochs"] == 60
        assert contract["sample_presentations"] == 5_998_080
        assert contract["pair_update_norm"] is True
        assert contract["auxiliary_weight"] == 0.10


def test_bond_arm_changes_only_denoising_scope():
    atom = _scientific_fields("full_atom")
    atom_bond = _scientific_fields("full_atom_bond")
    changed = {key for key in atom if atom[key] != atom_bond[key]}
    assert changed == {"model_id", "loss_fingerprint"}
