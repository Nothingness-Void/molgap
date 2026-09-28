"""A 2D multi-hop MetaGIN-family backbone for the fixed PCQM screen.

This is an independent implementation of the high-level MetaGIN information
flow, not a port of the paper authors' code or training recipe.  In particular,
all features and labels come from the accepted OGB graph cache and the model
predicts normalized Gap directly under the frozen V5 screen contract.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import scatter


WIDTH = 256
DEPTH = 4
HOPS = 3


class _HopMessage(nn.Module):
    def __init__(self, width: int, *, real_bond: bool) -> None:
        super().__init__()
        self.sender = nn.Linear(width, width, bias=False)
        self.receiver = nn.Linear(width, width, bias=False)
        if real_bond:
            from ogb.graphproppred.mol_encoder import BondEncoder

            self.edge_encoder = BondEncoder(width)
        else:
            # 0 is unused; 1, 2 and 3+ encode path multiplicity.
            self.edge_encoder = nn.Embedding(4, width)
        self.gate = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, nodes, edge_index, edge_features):
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Hop edge_index must have shape [2, E]")
        source, target = edge_index
        encoded = self.edge_encoder(edge_features)
        message = self.sender(nodes[source]) + self.receiver(nodes[target]) + encoded
        message = torch.nn.functional.silu(message) * torch.sigmoid(self.gate(message))
        pooled = scatter(message, target, dim=0, dim_size=nodes.shape[0], reduce="sum")
        return self.output(pooled)


class _MetaHopBlock(nn.Module):
    def __init__(self, width: int, *, virtual: bool) -> None:
        super().__init__()
        self.hops = nn.ModuleList(
            [_HopMessage(width, real_bond=hop == 0) for hop in range(HOPS)]
        )
        self.local_norm = nn.LayerNorm(width)
        self.virtual = virtual
        if virtual:
            self.virtual_update = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU()
            )
            self.virtual_return = nn.Linear(width, width, bias=False)
        self.return_scale = nn.Parameter(torch.full((width,), 0.1))
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_gate = nn.Linear(width, width * 2)
        self.ffn_value = nn.Linear(width, width * 2)
        self.ffn_return = nn.Linear(width * 2, width)
        self.ffn_scale = nn.Parameter(torch.full((width,), 0.1))

    def forward(self, nodes, virtual_state, batch, edges):
        # Sequential propagation makes the three edge types a distinct 2D
        # backbone rather than three independent feature additions to K1.
        propagated = nodes
        update = torch.zeros_like(nodes)
        for hop, (edge_index, attributes) in zip(self.hops, edges, strict=True):
            propagated = hop(propagated, edge_index, attributes)
            update = update + propagated
        if self.virtual:
            pooled = global_mean_pool(nodes, batch)
            virtual_state = virtual_state + self.virtual_update(pooled)
            update = update + self.virtual_return(virtual_state)[batch]
        nodes = nodes + self.return_scale * self.local_norm(update)
        normalized = self.ffn_norm(nodes)
        gated = torch.nn.functional.silu(self.ffn_gate(normalized))
        nodes = nodes + self.ffn_scale * self.ffn_return(
            gated * self.ffn_value(normalized)
        )
        return nodes, virtual_state


class MetaGIN2D(nn.Module):
    """Four-stage 1/2/3-hop, virtual-state, gated-MetaFormer Gap encoder."""

    def __init__(self, width: int = WIDTH) -> None:
        super().__init__()
        from ogb.graphproppred.mol_encoder import AtomEncoder

        self.atom_encoder = AtomEncoder(width)
        self.rwse_encoder = nn.Sequential(nn.Linear(16, width), nn.SiLU())
        self.blocks = nn.ModuleList(
            [_MetaHopBlock(width, virtual=layer > 0) for layer in range(DEPTH)]
        )
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def forward(self, graph):
        required = (
            "x", "edge_index", "edge_attr", "batch", "random_walk_pe",
            "hop2_edge_index", "hop2_count", "hop3_edge_index", "hop3_count",
        )
        if any(not hasattr(graph, name) for name in required):
            raise ValueError("MetaGIN2D requires accepted fixed-graph and hop sidecar fields")
        if graph.x.ndim != 2 or graph.x.shape[1] != 9:
            raise ValueError("Expected OGB nine-column atom categories")
        if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 3:
            raise ValueError("Expected OGB three-column real-bond categories")
        if graph.random_walk_pe.shape != (graph.num_nodes, 16):
            raise ValueError("Expected RWSE16 from the fixed graph cache")
        for hop in (2, 3):
            edge_index = getattr(graph, f"hop{hop}_edge_index")
            count = getattr(graph, f"hop{hop}_count")
            if edge_index.shape != (2, count.numel()):
                raise ValueError(f"Hop {hop} sidecar is not aligned")
        edges = (
            (graph.edge_index, graph.edge_attr.long()),
            (graph.hop2_edge_index, graph.hop2_count.long().clamp(1, 3)),
            (graph.hop3_edge_index, graph.hop3_count.long().clamp(1, 3)),
        )
        nodes = self.atom_encoder(graph.x.long()) + self.rwse_encoder(
            graph.random_walk_pe.float()
        )
        virtual = nodes.new_zeros((int(graph.num_graphs), nodes.shape[-1]))
        for block in self.blocks:
            nodes, virtual = block(nodes, virtual, graph.batch, edges)
        return self.head(global_add_pool(self.output_norm(nodes), graph.batch)).view(-1)


def model_parameters() -> int:
    return sum(parameter.numel() for parameter in MetaGIN2D().parameters())
