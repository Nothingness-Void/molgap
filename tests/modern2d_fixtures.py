"""Generated topology only; no real cache/labels/network."""
import torch
from torch_geometric.data import Data, Batch


def graph(nodes=3, source=0, edges=None):
    if edges is None:
        pairs = [(i, i + 1) for i in range(nodes - 1)]
        pairs += [(j, i) for i, j in pairs]
        edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()
    x = torch.zeros(nodes, 9, dtype=torch.long)
    x[:, 0] = torch.arange(nodes) + 5
    return Data(x=x, edge_index=edges, edge_attr=torch.zeros(edges.shape[1], 3, dtype=torch.long),
                random_walk_pe=torch.zeros(nodes, 16), source_idx=torch.tensor([source]), y=torch.tensor([2.0]))


def batch():
    return Batch.from_data_list([graph(3, 0), graph(2, 1)])


def model_config(family):
    if family == "ppgt_small":
        return dict(hidden=16, heads=2, blocks=1, pair_dim=8, pe_mlp_hidden=16,
                    stem_blocks=1, rrwp_steps=3, spe_bases=1)
    return dict(hidden=16, heads=2, layer_types=("M", "S", "P"), graph_dim=16,
                mlp_hidden=32, seeds=3)


def factory(family, config=None, version="1"):
    from molgap.ppgt_model import make_model as ppgt
    from molgap.esa_model import make_model as esa
    if family == "esa_small" and version == "2":
        from molgap.esa_invariant import make_model as esa
    return (ppgt if family == "ppgt_small" else esa)({"model_config": config or model_config(family)})
