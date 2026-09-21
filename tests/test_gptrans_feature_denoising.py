from molgap.gptrans_feature_denoising import (
    ARMS,
    ATOM_FEATURE_DIMS,
    BOND_FEATURE_DIMS,
    EXPECTED_PARAMETERS,
    REFERENCE_DEVELOPMENT_MAE_EV,
    REFERENCE_MODEL_SHA256,
    _scientific_fields,
)


def test_fixed_shard_loader_accepts_pyg_inmemory_payload(tmp_path):
    import torch
    from torch_geometric.data import Data, InMemoryDataset

    from molgap.pcqm_gptrans_v4 import _load_datasets

    graphs = [
        Data(
            x=torch.tensor([[1], [2]], dtype=torch.long),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.zeros((1, 3), dtype=torch.long),
            y=torch.tensor([0.1]),
        )
        for _ in range(2)
    ]
    data, slices = InMemoryDataset.collate(graphs)
    shard = tmp_path / "graphs.pt"
    torch.save((data, slices), shard)

    dataset, shards = _load_datasets((shard,))
    assert len(dataset) == 2
    assert len(shards) == 1
    assert shards[0][0].x.shape == (2, 1)


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


def test_strict_reference_identity_is_frozen():
    assert REFERENCE_MODEL_SHA256 == (
        "c841cdee799daa7a874e0f112ce6dea2932fe0f640f434812bac15b83684b092"
    )
    assert REFERENCE_DEVELOPMENT_MAE_EV == 0.14724504947662354
