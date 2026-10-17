"""PPGT/2 CPU ABI, invariance, frozen bounds and meta-only parameter inventory."""
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch

from modern2d_fixtures import graph
from molgap.ppgt_author import CONFIG_TYPE, PPGTAuthor, PPGTAuthorConfig, make_model
from test_ppgt_author_fidelity import fixed_weights, small_config, varied_batch


def test_versioned_factory_role_abi_ignores_y_rwse_without_batch_mutation():
    cfg = small_config()
    from dataclasses import asdict
    model = make_model({"model_config": asdict(cfg)}).eval()
    assert CONFIG_TYPE is PPGTAuthorConfig
    fixed_weights(model)
    batch = varied_batch()
    batch.role_name = ["train_100k", "common_dev"]
    before = {name: value.clone() if torch.is_tensor(value) else list(value)
              for name, value in batch.to_dict().items()}
    first = model(batch)
    assert first.shape == (2,) and torch.isfinite(first).all()
    assert before.keys() == batch.to_dict().keys()
    for name, value in before.items():
        if torch.is_tensor(value):
            assert torch.equal(value, batch[name])
        else:
            assert value == batch[name]
    batch.y.fill_(float("nan"))
    batch.random_walk_pe.fill_(float("inf"))
    assert torch.equal(first, model(batch))


def test_whole_output_node_edge_permutation_and_companion_invariance():
    model = PPGTAuthor(small_config()).eval()
    fixed_weights(model)
    a, b = graph(5, 3), graph(2, 7)
    a.edge_attr[:, 0] = torch.arange(a.edge_index.shape[1]) % 5
    order = torch.tensor([4, 1, 0, 3, 2])
    inverse = torch.argsort(order)
    changed = a.clone()
    changed.x = a.x[order]
    changed.edge_index = inverse[a.edge_index]
    edge_order = torch.arange(changed.edge_index.shape[1]).flip(0)
    changed.edge_index = changed.edge_index[:, edge_order]
    changed.edge_attr = changed.edge_attr[edge_order]
    original = model(Batch.from_data_list([a, b]))
    permuted = model(Batch.from_data_list([changed, b]))
    alone = model(Batch.from_data_list([a]))
    torch.testing.assert_close(original, permuted, atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(original[:1], alone, atol=3e-6, rtol=2e-5)


@pytest.mark.parametrize("field,value", [("max_nodes", 1025), ("max_edges", 4097),
    ("max_nodes", 0), ("max_edges", True), ("epsilon", float("nan")),
    ("layer_scale", 0), ("use_phi", False), ("stochastic_depth", 1.0)])
def test_invalid_configuration_and_unreviewed_bounds_rejected(field, value):
    with pytest.raises(ValueError):
        replace(small_config(), **{field: value})


def test_all_member_bounds_and_protected_geometry_roles_fail_closed():
    batch = varied_batch()
    with pytest.raises(ValueError, match="safety cap"):
        PPGTAuthor(small_config(max_nodes=4))(batch)
    with pytest.raises(ValueError, match="safety cap"):
        PPGTAuthor(small_config(max_edges=3))(batch)
    model = PPGTAuthor(small_config())
    batch.role_name = ["train", "independent_audit"]
    with pytest.raises(ValueError, match="ROLE_BLOCKED"):
        model(batch)
    del batch.role_name
    batch.pos = torch.zeros(batch.num_nodes, 3)
    with pytest.raises(ValueError, match="geometry"):
        model(batch)


def test_initialization_norm_scale_bias_and_uniform_depth_contract():
    cfg = small_config(stochastic_depth=0.2)
    model = PPGTAuthor(cfg)
    assert all(block.drop_path == 0.2 for block in model.blocks)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert torch.count_nonzero(module.bias) == 0
    for block in model.blocks:
        assert torch.equal(block.attention_scale, torch.full((cfg.hidden,), 1e-4))
        assert torch.equal(block.ffn_scale, torch.full((cfg.hidden,), 1e-4))
        assert block.norm1_h.epsilon == 1e-6
        assert torch.equal(block.norm1_h.alpha, torch.zeros(cfg.hidden))
        assert torch.equal(block.norm1_h.beta, torch.ones(cfg.hidden))
    assert [model.head[i].out_features for i in (0, 3, 6)] == [cfg.hidden, cfg.hidden, 1]


def test_meta_only_budget_base_and_table_small_inventory(record_property):
    # Do not materialize the budget/base/paper-sized tensors on this machine.
    reports = {}
    configs = {"molgap_budget": PPGTAuthorConfig(),
               "author_pcqm_base": PPGTAuthorConfig.author_pcqm_base(),
               "table_small_dense_structure": replace(PPGTAuthorConfig(), hidden=512, heads=16)}
    for name, cfg in configs.items():
        with torch.device("meta"):
            model = PPGTAuthor(cfg)
        inventory = {key: sum(p.numel() for p in child.parameters())
                     for key, child in model.named_children()}
        reports[name] = {"parameters": sum(inventory.values()), "inventory": inventory}
        assert len({id(block) for block in model.blocks}) == cfg.blocks
        assert all(p.device.type == "meta" for p in model.parameters())
    base = configs["author_pcqm_base"]
    assert (base.blocks, base.hidden, base.heads, base.expansion, base.stem_expansion,
            base.rrwp_steps, base.stem_blocks, base.spe_bases, base.pooling) == (12, 768, 12, 4, 4, 8, 2, 5, "mean")
    assert 16 * (4 * 512**2 + 2 * 2 * 512**2) == 33_554_432 > 17_600_000
    assert reports["table_small_dense_structure"]["parameters"] > 33_554_432
    record_property("ppgt_author_meta_parameter_inventory", str(reports))
    print("PPGT_AUTHOR_META_INVENTORY", reports)
