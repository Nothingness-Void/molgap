"""External author equations and full-sized reference configuration, CPU only."""
from dataclasses import replace
import ast

import pytest
import torch

from molgap.esa_model import ESA, ESAConfig
from molgap.esa_features import edge_tokens, shared_endpoint_mask
from molgap.modern2d_inputs import categorical_one_hot, topology_views
from molgap.ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS
from torch_geometric.data import Batch
from modern2d_fixtures import graph
from esa_reference_harness import author_estimator, ROOT


def test_forward_slice_is_ast_identical_to_pinned_upstream_if_download_retained():
    from pathlib import Path
    source = Path(__file__).parents[1] / ("data/cache/modern2d_reference/"
        "edge-set-attention-861a0b12d445ae80cc69b3948af623b4331321d4/esa/models.py")
    if not source.exists():
        pytest.skip("Optional raw download not retained; committed slice has pinned manifest")
    cls = next(n for n in ast.parse(source.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == "Estimator")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    frozen = ast.parse((ROOT / "estimator_forward.py.txt").read_bytes()).body[0]
    assert ast.dump(method) == ast.dump(frozen)


@pytest.mark.parametrize("nodes", [3, 17])
def test_full_sized_author_pre_ln_reference_every_layer(nodes, record_property):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    # README widths/heads/MLP/seeds, with author's supported pre-LN FP32 branch.
    # Not a smaller surrogate for the full configuration; 17 nodes gives 32
    # tokens and additionally exercises the author's conditional PMA residual.
    cfg = ESAConfig(hidden=256, heads=4, layer_types=("M", "M", "S", "P"),
                    seeds=32, graph_dim=512, mlp_hidden=512, include_self=False)
    model = ESA(cfg).eval()
    batch = Batch.from_data_list([graph(nodes, 1)])
    g = topology_views(batch)[0]
    author, ns = author_estimator(cfg, model, g.edge_index.shape[1])
    author.st_fast.eval(); author.node_edge_mlp.eval(); author.output_mlp.eval()
    observed, expected, handles = {}, {}, []
    pairs = [("token", model.token_mlp, author.node_edge_mlp),
        *[(f"encoder{i}", a, b) for i, (a,b) in enumerate(zip(model.encoder, author.st_fast.encoder))],
        ("PMA", model.pool, author.st_fast.decoder[0]),
        ("projection", model.graph_projection, author.st_fast.decoder_linear),
        ("output", model.head, author.output_mlp)]
    def hook(target, name):
        def save(module, args, out): target[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return save
    for name, a, b in pairs:
        handles += [a.register_forward_hook(hook(observed, name)), b.register_forward_hook(hook(expected, name))]
    x = categorical_one_hot(g.x, ATOM_FEATURE_DIMS)
    bonds = categorical_one_hot(g.edge_attr, BOND_FEATURE_DIMS)
    with torch.no_grad():
        got = model(batch)
        want = author.forward(x, g.edge_index, torch.zeros(nodes, dtype=torch.long), bonds,
                              g.edge_index.shape[1], None)
    for handle in handles: handle.remove()
    assert torch.equal(edge_tokens(g), torch.cat((x[g.edge_index[0]], x[g.edge_index[1]], bonds), -1))
    assert torch.equal(shared_endpoint_mask(g.edge_index, include_self=False),
        ns["create_edge_adjacency_mask"](g.edge_index, g.edge_index.shape[1]))
    errors = {}
    for name, _, _ in pairs:
        errors[name] = float((observed[name].reshape(-1) - expected[name].reshape(-1)).abs().max())
        torch.testing.assert_close(observed[name].reshape(-1), expected[name].reshape(-1), rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(got, want, rtol=2e-5, atol=3e-6)
    errors["prediction"] = float((got - want).abs().max())
    record_property("author_fp32_max_abs_errors", str(errors))
    print(f"ESA_REFERENCE nodes={nodes} errors={errors}")
    # Certification scope must not conceal the deliberate safe self-mask delta.
    assert not model.config.include_self and replace(cfg, include_self=True) != cfg
