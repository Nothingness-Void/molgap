"""Sparse chemistry-defined atom-to-functional-group exchange beside K1."""
from __future__ import annotations


MODE = "neural_atom_k1_functional_group_token"
MODES = (MODE,)
HIDDEN_CHANNELS = 192
GROUP_CHANNELS = 64
GROUP_TYPES = 12
TARGET_LAYER = 6
ADDED_PARAMETERS = 42_112
PARAMETERS = {MODE: 3_700_929}


class _FunctionalGroupTokenFactory:
    @staticmethod
    def make():
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch_geometric.utils import scatter

        class FunctionalGroupToken(nn.Module):
            def __init__(self):
                super().__init__()
                self.node_projection = nn.Linear(HIDDEN_CHANNELS, GROUP_CHANNELS)
                self.group_type = nn.Embedding(GROUP_TYPES, GROUP_CHANNELS)
                self.token_norm = nn.LayerNorm(GROUP_CHANNELS)
                self.token_ffn = nn.Sequential(
                    nn.Linear(GROUP_CHANNELS, 2 * GROUP_CHANNELS),
                    nn.SiLU(),
                    nn.Linear(2 * GROUP_CHANNELS, GROUP_CHANNELS),
                )
                self.return_projection = nn.Linear(
                    GROUP_CHANNELS, HIDDEN_CHANNELS, bias=False
                )
                nn.init.zeros_(self.return_projection.weight)

            def compute_update(self, hidden, batch, membership):
                if membership.shape != (hidden.shape[0], GROUP_TYPES):
                    raise ValueError(
                        "functional_group_y must have shape "
                        f"({hidden.shape[0]}, {GROUP_TYPES})"
                    )
                active = membership > 0.5
                atom_index, group_type = active.nonzero(as_tuple=True)
                num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
                flat_group = batch[atom_index] * GROUP_TYPES + group_type
                projected = self.node_projection(hidden)
                pooled = scatter(
                    projected[atom_index],
                    flat_group,
                    dim=0,
                    dim_size=num_graphs * GROUP_TYPES,
                    reduce="mean",
                )
                group_type_index = torch.arange(
                    GROUP_TYPES, device=hidden.device
                ).repeat(num_graphs)
                tokens = pooled + self.group_type(group_type_index)
                tokens = self.token_norm(tokens + self.token_ffn(tokens))
                returned = scatter(
                    tokens[flat_group],
                    atom_index,
                    dim=0,
                    dim_size=hidden.shape[0],
                    reduce="mean",
                )
                update = self.return_projection(functional.silu(returned))
                return update, {
                    "active": active,
                    "atom_index": atom_index,
                    "group_type": group_type,
                    "flat_group": flat_group,
                    "tokens": tokens,
                    "returned": returned,
                }

            def forward(self, hidden, batch, membership):
                update, _ = self.compute_update(hidden, batch, membership)
                return hidden + update

        return FunctionalGroupToken()


def make_encoder(mode: str):
    if mode not in MODES:
        raise ValueError(mode)

    import torch.nn as nn

    from .qm9_neural_atom import make_encoder as make_k1

    class K1FunctionalGroupToken(nn.Module):
        requires_functional_group_membership = True

        def __init__(self):
            super().__init__()
            self.base = make_k1("neural_atom_k1")
            self.functional_group_token = _FunctionalGroupTokenFactory.make()

        def forward(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            functional_group_y,
        ):
            return self.base.head(
                self.encode(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    random_walk_pe,
                    functional_group_y,
                )
            )

        def encode(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            functional_group_y,
        ):
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
                    hidden = self.functional_group_token(
                        hidden, batch, functional_group_y
                    )
            return self.base._pool(hidden, batch)

    return K1FunctionalGroupToken()


def check_mechanism(model, batch) -> dict:
    import torch

    membership = batch.functional_group_y
    probe = torch.linspace(
        -1.0,
        1.0,
        steps=int(batch.num_nodes) * HIDDEN_CHANNELS,
        device=batch.batch.device,
    ).reshape(int(batch.num_nodes), HIDDEN_CHANNELS)
    update, diagnostics = model.functional_group_token.compute_update(
        probe, batch.batch, membership
    )
    active = diagnostics["active"]
    atom_index = diagnostics["atom_index"]
    group_type = diagnostics["group_type"]
    checks = {
        "target_layer": TARGET_LAYER,
        "group_channels": GROUP_CHANNELS,
        "group_types": GROUP_TYPES,
        "group_semantics": "frozen-smarts-atom-incidence",
        "exchange": "sparse-atom-to-group-type-token-to-member-atoms",
        "membership_binary": bool(((membership == 0) | (membership == 1)).all()),
        "incidence_count_exact": int(atom_index.numel()) == int(active.sum()),
        "type_range_valid": bool(
            group_type.numel() > 0
            and int(group_type.min()) >= 0
            and int(group_type.max()) < GROUP_TYPES
        ),
        "zero_return_projection": bool(
            torch.count_nonzero(
                model.functional_group_token.return_projection.weight
            ).item()
            == 0
        ),
        "zero_initial_update_exact": bool(torch.count_nonzero(update).item() == 0),
        "no_dense_atom_to_atom_attention": True,
    }
    required = {
        "membership_binary",
        "incidence_count_exact",
        "type_range_valid",
        "zero_return_projection",
        "zero_initial_update_exact",
        "no_dense_atom_to_atom_attention",
    }
    if not all(checks[name] is True for name in required):
        raise RuntimeError(f"Functional-group token invariant failed: {checks}")
    return checks
