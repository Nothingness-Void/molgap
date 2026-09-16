"""One pre-normalized all-pair relation token beside frozen K1."""
from __future__ import annotations

import math


MODE = "neural_atom_k1_pair_token"
MODES = (MODE,)
HIDDEN_CHANNELS = 192
PAIR_CHANNELS = 32
TARGET_LAYER = 6
ADDED_PARAMETERS = 22_784
PARAMETERS = {MODE: 3_681_601}


class _PairTokenFactory:
    @staticmethod
    def make():
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch_geometric.utils import to_dense_batch

        class PairToken(nn.Module):
            def __init__(self):
                super().__init__()
                self.source = nn.Linear(HIDDEN_CHANNELS, PAIR_CHANNELS)
                self.target = nn.Linear(HIDDEN_CHANNELS, PAIR_CHANNELS)
                self.pair_norm = nn.LayerNorm(PAIR_CHANNELS)
                self.query = nn.Parameter(torch.empty(PAIR_CHANNELS))
                nn.init.normal_(self.query, std=PAIR_CHANNELS ** -0.5)
                self.token_norm = nn.LayerNorm(PAIR_CHANNELS)
                self.token_ffn = nn.Sequential(
                    nn.Linear(PAIR_CHANNELS, 2 * PAIR_CHANNELS),
                    nn.SiLU(),
                    nn.Linear(2 * PAIR_CHANNELS, PAIR_CHANNELS),
                )
                self.return_projection = nn.Linear(
                    PAIR_CHANNELS, HIDDEN_CHANNELS, bias=False
                )
                nn.init.zeros_(self.return_projection.weight)

            def compute_update(self, hidden, batch):
                dense, valid = to_dense_batch(hidden, batch)
                source = self.source(dense).unsqueeze(2)
                target = self.target(dense).unsqueeze(1)
                pair = self.pair_norm(functional.silu(source + target))
                pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
                logits = torch.einsum("bijd,d->bij", pair, self.query)
                logits = logits / math.sqrt(PAIR_CHANNELS)
                logits = logits.masked_fill(~pair_valid, float("-inf"))
                assignment = torch.softmax(logits.flatten(1), dim=-1).reshape_as(logits)
                token = torch.einsum("bij,bijd->bd", assignment, pair)
                token = self.token_norm(token + self.token_ffn(token))
                graph_update = self.return_projection(token)
                update = graph_update[batch]
                return update, {
                    "assignment": assignment,
                    "pair_valid": pair_valid,
                    "valid": valid,
                    "token": token,
                }

            def forward(self, hidden, batch):
                update, _ = self.compute_update(hidden, batch)
                return hidden + update

        return PairToken()


def make_encoder(mode: str):
    if mode != MODE:
        raise ValueError(mode)

    import torch.nn as nn

    from .qm9_neural_atom import make_encoder as make_k1

    class K1PairToken(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = make_k1("neural_atom_k1")
            self.relation_token = _PairTokenFactory.make()

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base.head(
                self.encode(x, edge_index, edge_attr, batch, random_walk_pe)
            )

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
            expected = (x.shape[0], self.base.rwse_dim)
            if tuple(random_walk_pe.shape) != expected:
                raise ValueError(
                    f"random_walk_pe must have shape {expected}, "
                    f"got {tuple(random_walk_pe.shape)}"
                )
            hidden = self.base._embed_nodes(x)
            hidden = hidden + self.base.rwse_encoder(random_walk_pe.float())
            edge_state = self.base._embed_edges(edge_attr)
            for layer, (edge_update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                edge_state = edge_update(hidden, edge_index, edge_state)
                hidden = block(
                    hidden, edge_index, batch, edge_attr=edge_state
                )
                if str(layer) in self.base.neural_atom_mixers:
                    hidden = self.base.neural_atom_mixers[str(layer)](
                        hidden, batch
                    )
                if layer == TARGET_LAYER:
                    hidden = self.relation_token(hidden, batch)
            return self.base._pool(hidden, batch)

    return K1PairToken()


def check_mechanism(model, batch) -> dict:
    import torch

    probe = torch.linspace(
        -1.0,
        1.0,
        steps=int(batch.num_nodes) * HIDDEN_CHANNELS,
        device=batch.batch.device,
    ).reshape(int(batch.num_nodes), HIDDEN_CHANNELS)
    update, diagnostics = model.relation_token.compute_update(probe, batch.batch)
    assignment = diagnostics["assignment"]
    pair_valid = diagnostics["pair_valid"]
    valid = diagnostics["valid"]
    valid_nodes = valid.sum(dim=-1)
    checks = {
        "target_layer": TARGET_LAYER,
        "pair_channels": PAIR_CHANNELS,
        "relation_tokens": 1,
        "pair_source": "all-ordered-pairs-of-current-layer6-node-states",
        "pair_normalization": "per-pair-across-channels",
        "dense_atom_to_atom_attention": False,
        "valid_pair_count_exact": bool(
            torch.equal(pair_valid.sum(dim=(1, 2)), valid_nodes.square())
        ),
        "assignment_mass_one": bool(
            torch.allclose(
                assignment.sum(dim=(1, 2)),
                torch.ones_like(assignment.sum(dim=(1, 2))),
                atol=1e-6,
                rtol=0,
            )
        ),
        "padding_mass_zero": bool(
            assignment.masked_select(~pair_valid).abs().sum().item() == 0.0
        ),
        "zero_return_projection": bool(
            torch.count_nonzero(model.relation_token.return_projection.weight).item()
            == 0
        ),
        "zero_initial_update_exact": bool(
            torch.count_nonzero(update).item() == 0
        ),
    }
    required = {
        "valid_pair_count_exact",
        "assignment_mass_one",
        "padding_mass_zero",
        "zero_return_projection",
        "zero_initial_update_exact",
    }
    if not all(checks[name] is True for name in required):
        raise RuntimeError(f"Pair-token invariant failed: {checks}")
    return checks
