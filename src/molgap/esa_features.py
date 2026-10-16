"""Directed edge tokens and explicit allowed-key masks, no labels/geometry."""
import torch

from .modern2d_inputs import categorical_one_hot
from .ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS


def edge_tokens(graph):
    if graph.edge_index.shape[1] == 0:
        raise ValueError("ESA empty-edge graphs are unsupported; no virtual bond is inserted")
    nodes = categorical_one_hot(graph.x, ATOM_FEATURE_DIMS)
    bonds = categorical_one_hot(graph.edge_attr, BOND_FEATURE_DIMS)
    i, j = graph.edge_index
    return torch.cat((nodes[i], nodes[j], bonds), -1)


def shared_endpoint_mask(edge_index, *, include_self=True):
    if type(include_self) is not bool:
        raise TypeError("Self-mask policy must be explicit bool")
    i, j = edge_index
    allowed = ((i[:, None] == i[None, :]) | (i[:, None] == j[None, :]) |
               (j[:, None] == i[None, :]) | (j[:, None] == j[None, :]))
    allowed.fill_diagonal_(include_self)
    return allowed


def feature_cost(edges):
    if type(edges) is not int or edges < 1:
        raise ValueError("Positive directed-edge count required")
    return {"allowed_mask_bytes_bool": edges ** 2,
            "one_hot_token_bytes_fp32": 4 * edges * (2 * sum(ATOM_FEATURE_DIMS) + sum(BOND_FEATURE_DIMS)),
            "scope": "FORMULA_NOT_NATIVE_MEMORY_MEASUREMENT"}
