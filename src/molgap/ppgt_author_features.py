"""Bounded device-local topology features for PPGT author-compatible version 2.

Conventions follow LiamMa/PPGT at 62cafd3 (MIT; retained in
tests/vendor/ppgt_reference). ``steps`` counts transition powers, so identity
plus powers 1..steps has steps+1 channels. No cached RWSE or label is consumed.
This module is a code path, not a real-data or accelerator qualification.
"""
import math

import torch


@torch.no_grad()
def rrwp(graph, steps, *, sym_norm=True, max_nodes=256, max_edges=1024):
    """Match add_full_rrwp's unweighted adjacency and identity convention.

    Parallel directed edges are summed, as in PyG to_dense_adj. The author's
    asymmetric-adjacency branch takes the binary undirected union. Isolates
    retain their identity channel and have zero transition channels.
    """
    if type(steps) is not int or steps < 1 or type(sym_norm) is not bool:
        raise ValueError("Positive author walk length and explicit normalization required")
    if any(type(v) is not int or v < 1 for v in (max_nodes, max_edges)):
        raise ValueError("Positive topology bounds required")
    if max_nodes > 1024 or max_edges > 4096:
        raise ValueError("PPGT author bounds exceed reviewed ceiling 1024 nodes/4096 edges")
    n, e = len(graph.x), graph.edge_index.shape[1]
    if not 0 < n <= max_nodes or e > max_edges:
        raise ValueError("PPGT topology safety cap exceeded; no member may be dropped")
    device = graph.x.device
    if graph.edge_index.device != device or graph.edge_attr.device != device:
        raise ValueError("Topology must occupy one device")
    # Autocast must not change the frozen feature transform's FP32 arithmetic.
    with torch.autocast(device_type=device.type, enabled=False):
        adjacency = torch.zeros((n, n), dtype=torch.float32, device=device)
        adjacency.index_put_(tuple(graph.edge_index.long()),
                             torch.ones(e, device=device), accumulate=True)
        if bool((adjacency != adjacency.t()).any()):
            adjacency = ((adjacency + adjacency.t()) > 0).float()
        degree = adjacency.sum(1)
        inverse = torch.where(degree > 0, degree.clamp_min(1).reciprocal(), 0)
        if sym_norm:
            inverse = inverse.sqrt()
            transition = adjacency * inverse[:, None] * inverse[None, :]
        else:
            transition = adjacency * inverse[:, None]
        powers = [torch.eye(n, device=device), transition]
        for _ in range(1, steps):
            powers.append(powers[-1] @ transition)
        return torch.stack(powers, -1), degree


def sinusoidal_pe(values, bases):
    """Author order: all raw channels, then feature/frequency/(sin,cos).

    v1 interleaves the raw channel inside each feature's Fourier expansion;
    that different parameter/input mapping is deliberately not reused.
    """
    if type(bases) is not int or bases < 0:
        raise ValueError("Nonnegative Fourier basis count required")
    if bases == 0:
        return values
    frequencies = (2 ** torch.arange(bases, device=values.device)) * math.pi
    phase = values.unsqueeze(-1) * frequencies
    periodic = torch.stack((phase.sin(), phase.cos()), -1).flatten(-3)
    return torch.cat((values, periodic), -1)


def positional_inputs(graph, steps, *, sym_norm=True, max_nodes=256, max_edges=1024):
    """Return RRWP, pair (RRWP+1/N), and node log1p(N), in query/key order.

    The released PCQM YAML does not append inverse degrees or log degree.
    Relative sparse RRWP is transposed once when written and again when read
    by attention; the resulting dense query/key tensor is the original walk.
    """
    walk, _ = rrwp(graph, steps, sym_norm=sym_norm,
                   max_nodes=max_nodes, max_edges=max_edges)
    n = len(graph.x)
    inv_order = walk.new_full((n, n, 1), 1 / n)
    log_order = walk.new_full((n, 1), math.log1p(n))
    return walk, torch.cat((walk, inv_order), -1), log_order


@torch.no_grad()
def batched_positional_inputs(graphs, steps, *, sym_norm=True, max_nodes=256, max_edges=1024):
    """Equal-size FP32 walks without per-molecule host reads or tiny GEMMs.

    This opt-in execution candidate retains parallel-edge counts and the
    author's per-graph asymmetric union. It is not a native parity certificate:
    batched GEMM may round differently from the historical per-graph GEMM.
    """
    if (type(steps) is not int or steps < 1 or type(sym_norm) is not bool or
            any(type(v) is not int or v < 1 for v in (max_nodes, max_edges)) or
            max_nodes > 1024 or max_edges > 4096 or not graphs):
        raise ValueError("Positive bounded author walk configuration required")
    n, device = len(graphs[0].x), graphs[0].x.device
    if not 0 < n <= max_nodes or any(len(g.x) != n or g.x.device != device or
            g.edge_index.device != device or g.edge_attr.device != device or
            g.edge_index.shape[1] > max_edges for g in graphs):
        raise ValueError("Equal-size device-local bounded graphs required")
    b = len(graphs)
    with torch.autocast(device_type=device.type, enabled=False):
        adjacency = torch.zeros((b, n, n), dtype=torch.float32, device=device)
        sizes = [g.edge_index.shape[1] for g in graphs]
        total = sum(sizes)
        membership = torch.repeat_interleave(torch.arange(b, device=device),
            torch.tensor(sizes, device=device), output_size=total)
        edges = torch.cat([g.edge_index for g in graphs], 1)
        adjacency.index_put_((membership, edges[0], edges[1]),
            torch.ones(total, dtype=torch.float32, device=device), accumulate=True)
        transpose = adjacency.transpose(1, 2)
        asymmetric = (adjacency != transpose).any(dim=(1, 2), keepdim=True)
        adjacency = torch.where(asymmetric, ((adjacency + transpose) > 0).float(), adjacency)
        degree = adjacency.sum(-1)
        inverse = torch.where(degree > 0, degree.clamp_min(1).reciprocal(), 0)
        if sym_norm:
            inverse = inverse.sqrt()
            transition = adjacency * inverse[:, :, None] * inverse[:, None, :]
        else:
            transition = adjacency * inverse[:, :, None]
        powers = [torch.eye(n, device=device).expand(b, n, n), transition]
        for _ in range(1, steps):
            powers.append(torch.bmm(powers[-1], transition))
        walk = torch.stack(powers, -1)
        pair = torch.cat((walk, walk.new_full((b, n, n, 1), 1 / n)), -1)
        return walk, pair, walk.new_full((b, n, 1), math.log1p(n))


def feature_cost(nodes, *, steps=24, bases=10):
    """Formula only: transient FP32 feature tensors for one bounded graph."""
    if any(type(v) is not int or v < 1 for v in (nodes, steps)) or type(bases) is not int or bases < 0:
        raise ValueError("Invalid feature dimensions")
    return {"rrwp_bytes_fp32": 4 * nodes * nodes * (steps + 1),
            "spe_bytes_fp32": 4 * nodes * nodes * (steps + 2) * (1 + 2 * bases),
            "walk_dense_matmul_macs": (steps - 1) * nodes ** 3,
            "scope": "PER_GRAPH_FORMULA_NOT_NATIVE_MEMORY_QUALIFICATION"}
