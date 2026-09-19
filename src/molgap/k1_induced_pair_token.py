"""A bounded induced-pair relation token beside frozen K1."""
from __future__ import annotations

import math


MODE = "neural_atom_k1_induced_pair_token"
MODES = (MODE,)
HIDDEN_CHANNELS = 192
PAIR_CHANNELS = 32
INDUCING_SLOTS = 4
TARGET_LAYER = 6
ADDED_PARAMETERS = 23_104
PARAMETERS = {MODE: 3_681_921}


class _InducedPairTokenFactory:
    @staticmethod
    def make():
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch_geometric.utils import to_dense_batch

        class InducedPairToken(nn.Module):
            def __init__(self):
                super().__init__()
                self.source = nn.Linear(HIDDEN_CHANNELS, PAIR_CHANNELS)
                self.target = nn.Linear(HIDDEN_CHANNELS, PAIR_CHANNELS)
                self.source_queries = nn.Parameter(
                    torch.empty(INDUCING_SLOTS, PAIR_CHANNELS)
                )
                self.target_queries = nn.Parameter(
                    torch.empty(INDUCING_SLOTS, PAIR_CHANNELS)
                )
                nn.init.normal_(self.source_queries, std=PAIR_CHANNELS ** -0.5)
                nn.init.normal_(self.target_queries, std=PAIR_CHANNELS ** -0.5)
                self.pair_norm = nn.LayerNorm(PAIR_CHANNELS)
                self.pair_query = nn.Parameter(torch.empty(PAIR_CHANNELS))
                nn.init.normal_(self.pair_query, std=PAIR_CHANNELS ** -0.5)
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

            @staticmethod
            def _assignment(projected, queries, valid):
                logits = torch.einsum("bnd,kd->bkn", projected, queries)
                logits = logits / math.sqrt(PAIR_CHANNELS)
                logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
                return torch.softmax(logits, dim=-1)

            def compute_update(self, hidden, batch):
                dense, valid = to_dense_batch(hidden, batch)
                source = self.source(dense)
                target = self.target(dense)
                source_assignment = self._assignment(
                    source, self.source_queries, valid
                )
                target_assignment = self._assignment(
                    target, self.target_queries, valid
                )
                source_slots = torch.einsum("bkn,bnd->bkd", source_assignment, source)
                target_slots = torch.einsum("bkn,bnd->bkd", target_assignment, target)
                pair = self.pair_norm(
                    functional.silu(
                        source_slots.unsqueeze(2) + target_slots.unsqueeze(1)
                    )
                )
                logits = torch.einsum("bkld,d->bkl", pair, self.pair_query)
                logits = logits / math.sqrt(PAIR_CHANNELS)
                pair_assignment = torch.softmax(logits.flatten(1), dim=-1).reshape_as(logits)
                token = torch.einsum("bkl,bkld->bd", pair_assignment, pair)
                token = self.token_norm(token + self.token_ffn(token))
                update = self.return_projection(token)[batch]
                return update, {
                    "valid": valid,
                    "source_assignment": source_assignment,
                    "target_assignment": target_assignment,
                    "pair_assignment": pair_assignment,
                    "token": token,
                }

            def forward(self, hidden, batch):
                update, _ = self.compute_update(hidden, batch)
                return hidden + update

        return InducedPairToken()


def make_encoder(mode: str):
    if mode != MODE:
        raise ValueError(mode)

    import torch.nn as nn

    from .qm9_neural_atom import make_encoder as make_k1

    class K1InducedPairToken(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = make_k1("neural_atom_k1")
            self.relation_token = _InducedPairTokenFactory.make()

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

    return K1InducedPairToken()


def check_mechanism(model, batch) -> dict:
    import torch

    probe = torch.linspace(
        -1.0,
        1.0,
        steps=int(batch.num_nodes) * HIDDEN_CHANNELS,
        device=batch.batch.device,
    ).reshape(int(batch.num_nodes), HIDDEN_CHANNELS)
    update, diagnostics = model.relation_token.compute_update(probe, batch.batch)
    valid = diagnostics["valid"]
    source_assignment = diagnostics["source_assignment"]
    target_assignment = diagnostics["target_assignment"]
    pair_assignment = diagnostics["pair_assignment"]
    checks = {
        "target_layer": TARGET_LAYER,
        "pair_channels": PAIR_CHANNELS,
        "inducing_slots": INDUCING_SLOTS,
        "latent_ordered_pairs": INDUCING_SLOTS ** 2,
        "pair_source": "learned-source-and-target-induced-node-summaries",
        "pair_normalization": "per-latent-pair-across-channels",
        "dense_atom_to_atom_attention": False,
        "source_assignment_mass_one": bool(
            torch.allclose(
                source_assignment.sum(dim=-1),
                torch.ones_like(source_assignment.sum(dim=-1)),
                atol=1e-6,
                rtol=0,
            )
        ),
        "target_assignment_mass_one": bool(
            torch.allclose(
                target_assignment.sum(dim=-1),
                torch.ones_like(target_assignment.sum(dim=-1)),
                atol=1e-6,
                rtol=0,
            )
        ),
        "padding_mass_zero": bool(
            source_assignment.masked_select(~valid.unsqueeze(1)).abs().sum().item()
            == 0.0
            and target_assignment.masked_select(~valid.unsqueeze(1)).abs().sum().item()
            == 0.0
        ),
        "pair_assignment_mass_one": bool(
            torch.allclose(
                pair_assignment.sum(dim=(1, 2)),
                torch.ones_like(pair_assignment.sum(dim=(1, 2))),
                atol=1e-6,
                rtol=0,
            )
        ),
        "zero_return_projection": bool(
            torch.count_nonzero(model.relation_token.return_projection.weight).item()
            == 0
        ),
        "zero_initial_update_exact": bool(torch.count_nonzero(update).item() == 0),
    }
    required = {
        "source_assignment_mass_one",
        "target_assignment_mass_one",
        "padding_mass_zero",
        "pair_assignment_mass_one",
        "zero_return_projection",
        "zero_initial_update_exact",
    }
    if not all(checks[name] is True for name in required):
        raise RuntimeError(f"Induced-pair-token invariant failed: {checks}")
    return checks
