"""RRWP and SPE from topology only; independent Eq6/Eq9 CPU implementation."""
import math

import torch

from .modern2d_inputs import GraphView


def rrwp(graph: GraphView, steps: int):
    if type(steps) is not int or steps < 1:
        raise ValueError("Positive RRWP channel count required")
    n = len(graph.x)
    adjacency = torch.zeros(n, n, dtype=torch.float32)
    seen = {}
    for e, (i, j) in enumerate(graph.edge_index.t().tolist()):
        attr = graph.edge_attr[e]
        if (i, j) in seen and not torch.equal(seen[i, j], attr):
            raise ValueError("Conflicting parallel bond features")
        seen[i, j] = attr
        adjacency[i, j] = 1
    degree = adjacency.sum(-1)
    walk = adjacency / degree.clamp_min(1)[:, None]
    power = torch.eye(n)
    channels = [power]
    for _ in range(1, steps):
        power = power @ walk
        channels.append(power)
    return torch.stack(channels, -1), degree


def sinusoidal_pe(values, bases: int):
    if type(bases) is not int or bases < 0:
        raise ValueError("Nonnegative SPE bases required")
    channels = [values.unsqueeze(-1)]
    for i in range(bases):
        phase = (2 ** i) * math.pi * values
        channels.extend((phase.sin().unsqueeze(-1), phase.cos().unsqueeze(-1)))
    return torch.cat(channels, -1).flatten(-2)


def positional_inputs(graph, steps):
    probabilities, degree = rrwp(graph, steps)
    n = len(graph.x)
    # Finite zero-degree convention is explicit, not an added chemistry edge.
    reciprocal = torch.where(degree > 0, degree.clamp_min(1).reciprocal(), 0)
    extensions = torch.stack((reciprocal[:, None].expand(n, n),
                              reciprocal[None, :].expand(n, n), torch.full((n, n), 1 / n)), -1)
    node = torch.stack((degree.clamp_min(1).log(), torch.full((n,), math.log(n))), -1)
    return probabilities, torch.cat((probabilities, extensions), -1), node


def feature_cost(nodes, *, steps=24, bases=10):
    if type(nodes) is not int or nodes < 1:
        raise ValueError("Positive graph size required")
    return {"rrwp_bytes_fp32": 4 * nodes * nodes * steps,
            "extended_spe_bytes_fp32": 4 * nodes * nodes * (steps + 3) * (1 + 2 * bases),
            "walk_dense_matmul_macs": (steps - 1) * nodes ** 3,
            "scope": "FORMULA_NOT_MEASURED_DATASET_DISTRIBUTION"}
