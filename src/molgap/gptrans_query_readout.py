"""Separate full BondLocal propagation from dynamic molecular-query readout."""
import torch
from torch import nn
from torch.nn import functional as F

from .gptrans_bond_local import BondLocalGPTrans, _flat_bond_targets
from .gptrans_bond_pair_norm import construct_family_model
from .gptrans_variants import normalize_pair


class QueryReadoutBlock(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, query, pair, padding, memory):
        core, attention = self.core, self.core.attention
        channels = query.shape[-1]
        clean_query = core.node_norm1(query)
        clean_memory = core.node_norm1(memory)
        weight, bias = attention.qkv.weight, attention.qkv.bias
        q = F.linear(clean_query, weight[:channels], bias[:channels])
        kv = F.linear(clean_memory, weight[channels:], bias[channels:])
        size, keys = memory.shape[:2]
        q = q.reshape(size, 1, attention.num_heads, attention.head_channels).transpose(1, 2)
        k, v = kv.reshape(size, keys, 2, attention.num_heads,
                          attention.head_channels).permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q @ k.transpose(-2, -1)) * attention.scale
        logits = logits + attention.pair_to_attention(pair)
        weights = attention.attention_dropout(
            torch.softmax(logits.masked_fill(padding, float("-inf")), dim=-1))
        node_update = (weights @ v).transpose(1, 2).reshape(size, 1, channels)
        pair_update = attention.attention_to_pair(weights + logits)
        pair_weights = torch.softmax(
            pair_update.masked_fill(padding, float("-inf")), dim=-1)
        pair_message = (pair_weights * pair_update).sum(-1).transpose(1, 2)
        node_update = node_update + attention.pair_to_node(pair_message)
        node_update = attention.output_dropout(attention.output(node_update))
        pair = pair + normalize_pair(pair_update)
        query = query + core.drop_path(core.attention_scale * node_update)
        query = query + core.drop_path(core.ffn_scale * core.ffn(core.node_norm2(query)))
        return query, pair


class QueryReadoutGPTrans(BondLocalGPTrans):
    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        node, pair, padding = self._dense_inputs(x, edge_index, edge_attr, batch)
        graph, source, target = self._local_edges(edge_index, batch, len(x))
        edges = (graph, source + 1, target + 1)
        flat_target = _flat_bond_targets(graph, edges[2], node.shape[1])
        for block in self.blocks[:self.full_layers]:
            node, pair = block(node, pair, padding, edges, flat_target=flat_target)
        # Memory includes the split-point graph token. It remains differentiable;
        # its column zero is not replaced by the evolving query.
        memory, query, row = node, node[:, :1], pair[:, :, :1, :]
        for block in self.blocks[self.full_layers:]:
            query, row = block(query, row, padding, memory)
        return self.readout(torch.cat((query[:, 0], row[:, :, 0, 0]), -1))


def mount(recipe, model):
    from .model_substrate import ModelSubstrate
    full_layers, query_layers = {
        "bond_pairnorm_full6": (6, 0),
        "bond_pairnorm_query6": (6, 6),
        "bond_pairnorm_query8": (8, 4),
    }[recipe.mode]
    blocks = list(model.blocks[:full_layers])
    blocks.extend(QueryReadoutBlock(block.core)
                  for block in model.blocks[full_layers:full_layers + query_layers])
    owner = ModelSubstrate(model, "gptrans_t")
    owner.replace_submodule("query_readout", "blocks", nn.ModuleList(blocks))
    model.__class__ = QueryReadoutGPTrans
    model.full_layers = full_layers
    model._molgap_substrate = owner
    return model


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode not in {"bond_pairnorm_full6", "bond_pairnorm_query6", "bond_pairnorm_query8"}:
        raise ValueError("Unknown propagation/readout split")
    return FamilyModePolicy(construct=construct_family_model, mount=mount,
                            objective_name="degree_bond_local_ema999")
