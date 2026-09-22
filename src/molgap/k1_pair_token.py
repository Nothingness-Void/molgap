"""Value-decoupled pair token on the existing K1; provenance in K1_ADAPTER.md.

Only the model path is migrated. No experimental registry or runner is imported.
"""
from __future__ import annotations

import math

VALUE_DECOUPLED_MODE = "neural_atom_k1_pair_token_value_decoupled"
HIDDEN_CHANNELS = 192
PAIR_CHANNELS = 32
TARGET_LAYER = 6


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
                self.mode = VALUE_DECOUPLED_MODE
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
                    PAIR_CHANNELS, HIDDEN_CHANNELS, bias=False,
                )
                nn.init.zeros_(self.return_projection.weight)
                # Identity preserves the coupled pair-token function initially;
                # keys and transmitted values can specialize independently later.
                self.value_projection = nn.Linear(
                    PAIR_CHANNELS, PAIR_CHANNELS, bias=False,
                )
                nn.init.eye_(self.value_projection.weight)

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
                value = self.value_projection(pair)
                token = torch.einsum("bij,bijd->bd", assignment, value)
                token = self.token_norm(token + self.token_ffn(token))
                return_features = token[batch]
                update = self.return_projection(return_features)
                return update, {
                    "assignment": assignment,
                    "pair_valid": pair_valid,
                    "valid": valid,
                    "token": token,
                    "return_features": return_features,
                }

            def forward(self, hidden, batch):
                update, _ = self.compute_update(hidden, batch)
                return hidden + update

        return PairToken()


def make_encoder(mode: str):
    if type(mode) is not str or mode != VALUE_DECOUPLED_MODE:
        raise ValueError("Only the K1 value-decoupled pair-token mode is supported")

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
                hidden = block(hidden, edge_index, batch, edge_attr=edge_state)
                if str(layer) in self.base.neural_atom_mixers:
                    hidden = self.base.neural_atom_mixers[str(layer)](hidden, batch)
                if layer == TARGET_LAYER:
                    hidden = self.relation_token(hidden, batch)
            return self.base._pool(hidden, batch)

    return K1PairToken()
