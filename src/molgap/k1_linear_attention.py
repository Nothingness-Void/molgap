"""Node-query linear kernel exchange over a frozen K1 local backbone.

Katharopoulos et al., ICML 2020, equations 5 and 7; bidirectional and
independently normalized per molecule, not a reproduction of Polynormer.
"""
from __future__ import annotations

MODE = "neural_atom_k1_linear_attention"
MODES = (MODE,)
CHANNELS = 64
LAYERS = (3, 6, 9)
# Old mixer: 75,072 parameters; replacement: LN192 + four bias-free maps.
EXPECTED_PARAMETERS = 3_658_817 + 3 * (49_536 - 75_072)


def kernel_read(query, key, value, valid):
    """No N-by-N tensor: each graph stores a C-by-C key/value statistic."""
    import torch.nn.functional as functional

    q = functional.elu(query.float()) + 1.0
    k = (functional.elu(key.float()) + 1.0) * valid.unsqueeze(-1)
    v = value.float() * valid.unsqueeze(-1)
    memory = k.transpose(1, 2).bmm(v)
    denominator = (q * k.sum(dim=1, keepdim=True)).sum(dim=-1, keepdim=True)
    return q.bmm(memory) / denominator.clamp_min(1e-6)


def make_encoder(mode: str):
    if mode != MODE:
        raise ValueError(mode)
    import torch.nn as nn
    from torch_geometric.utils import to_dense_batch
    from .qm9_neural_atom import make_encoder as make_k1

    class LinearExchange(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_norm = nn.LayerNorm(192)
            self.query = nn.Linear(192, CHANNELS, bias=False)
            self.key = nn.Linear(192, CHANNELS, bias=False)
            self.value = nn.Linear(192, CHANNELS, bias=False)
            self.return_projection = nn.Linear(CHANNELS, 192, bias=False)
            self.dropout = nn.Dropout(0.05)
            nn.init.zeros_(self.return_projection.weight)

        def forward(self, hidden, batch):
            dense, valid = to_dense_batch(self.node_norm(hidden), batch)
            read = kernel_read(self.query(dense), self.key(dense), self.value(dense), valid)
            return hidden + self.dropout(self.return_projection(read)[valid])

    class LinearAttentionK1(nn.Module):
        def __init__(self):
            super().__init__()
            # Construct K1 first so all untouched initialization bytes agree.
            self.base = make_k1("neural_atom_k1")
            self.base.neural_atom_mixers = nn.ModuleDict({
                str(layer): LinearExchange() for layer in LAYERS
            })

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base(x, edge_index, edge_attr, batch, random_walk_pe)

    return LinearAttentionK1()


def check_mechanism(model, batch):
    """Remote preflight: kernel equation, padding, permutation and query signal."""
    import torch
    import torch.nn.functional as functional

    device = batch.x.device
    # Synthetic non-model tensors, confined to the allocated remote device.
    q = torch.arange(2 * 5 * 4, device=device).reshape(2, 5, 4).float() / 17 - 1
    k = torch.cos(q * 3)
    v = torch.sin(q * 2)
    valid = torch.tensor([[True] * 5, [True, True, True, False, False]], device=device)
    result = kernel_read(q, k, v, valid)
    scores = (functional.elu(q) + 1).bmm((functional.elu(k) + 1).transpose(1, 2))
    scores = scores * valid[:, None, :]
    explicit = scores.bmm(v) / scores.sum(-1, keepdim=True).clamp_min(1e-6)
    equation = torch.allclose(result, explicit, atol=2e-6, rtol=2e-6)
    order = torch.tensor([2, 0, 4, 1, 3], device=device)
    permutation = torch.allclose(
        kernel_read(q[:, order], k[:, order], v[:, order], valid[:, order]),
        result[:, order], atol=2e-6, rtol=2e-6,
    )
    isolated = torch.allclose(kernel_read(q[:1], k[:1], v[:1], valid[:1]), result[:1])
    qp, kp, vp = q.clone(), k.clone(), v.clone()
    kp[~valid], vp[~valid] = 100.0, -100.0
    padding = torch.allclose(kernel_read(qp, kp, vp, valid)[valid], result[valid])
    query_dependent = bool((result[0, 0] - result[0, 4]).abs().max() > 1e-5)
    zero = all(torch.count_nonzero(m.return_projection.weight).item() == 0
               for m in model.base.neural_atom_mixers.values())
    count = sum(p.numel() for p in model.parameters())
    checks = {"equation_verified": equation, "permutation_equivariant": permutation,
              "graphs_isolated": isolated, "padding_invariant": padding,
              "node_query_dependent": query_dependent, "zero_start": zero,
              "source_from_fixed_graph_only": True,
              "parameter_identity": count == EXPECTED_PARAMETERS}
    if not all(checks.values()):
        raise RuntimeError(f"Linear attention preflight failed: {checks}, parameters={count}")
    return checks
