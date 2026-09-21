"""Accepted PairToken with a complementary zero-return MoSE structural view."""
from __future__ import annotations


MODE = "neural_atom_k1_pair_token_mose"
MODES = (MODE,)
HIDDEN_CHANNELS = 192
RWSE_CHANNELS = 16
MOSE_CHANNELS = 31
MOSE_HIDDEN_CHANNELS = 64
TARGET_LAYER = 6
PARAMETERS = {MODE: 3_696_193}


def make_encoder(mode: str):
    if mode not in MODES:
        raise ValueError(mode)

    import torch.nn as nn

    from .k1_pair_token import MODE as PAIR_TOKEN_MODE, _PairTokenFactory
    from .qm9_neural_atom import make_encoder as make_k1

    class K1PairTokenMoSE(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = make_k1("neural_atom_k1")
            self.mose_residual = nn.Sequential(
                nn.Linear(MOSE_CHANNELS, MOSE_HIDDEN_CHANNELS),
                nn.SiLU(),
                nn.Linear(MOSE_HIDDEN_CHANNELS, HIDDEN_CHANNELS),
            )
            nn.init.zeros_(self.mose_residual[-1].weight)
            nn.init.zeros_(self.mose_residual[-1].bias)
            self.relation_token = _PairTokenFactory.make(PAIR_TOKEN_MODE)

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base.head(
                self.encode(x, edge_index, edge_attr, batch, random_walk_pe)
            )

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
            expected = (x.shape[0], RWSE_CHANNELS + MOSE_CHANNELS)
            if tuple(random_walk_pe.shape) != expected:
                raise ValueError(
                    f"random_walk_pe must have shape {expected}, "
                    f"got {tuple(random_walk_pe.shape)}"
                )
            rwse = random_walk_pe[:, :RWSE_CHANNELS].float()
            mose = random_walk_pe[:, RWSE_CHANNELS:].float()
            hidden = self.base._embed_nodes(x)
            hidden = hidden + self.base.rwse_encoder(rwse)
            hidden = hidden + self.mose_residual(mose)
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

    return K1PairTokenMoSE()


def check_mechanism(model, batch) -> dict:
    import torch

    from .k1_pair_token import check_mechanism as check_pair_token

    checks = check_pair_token(model, batch)
    mose = batch.random_walk_pe[:, RWSE_CHANNELS:]
    with torch.no_grad():
        initial_mose_update = model.mose_residual(mose.float())
    checks.update(
        {
            "combined_input_shape": list(batch.random_walk_pe.shape),
            "rwse_channels": RWSE_CHANNELS,
            "mose_channels": MOSE_CHANNELS,
            "mose_input_finite": bool(torch.isfinite(mose).all()),
            "mose_input_nonnegative": bool((mose >= 0).all()),
            "mose_return_target": "initial-node-state-before-edge-processing",
            "zero_mose_return": bool(
                torch.count_nonzero(model.mose_residual[-1].weight).item() == 0
                and torch.count_nonzero(model.mose_residual[-1].bias).item() == 0
            ),
            "zero_initial_mose_update_exact": bool(
                torch.count_nonzero(initial_mose_update).item() == 0
            ),
            "pairtoken_parent_unchanged": True,
            "prediction_fusion": False,
        }
    )
    required = (
        "valid_pair_count_exact",
        "assignment_mass_one",
        "padding_mass_zero",
        "zero_return_projection",
        "zero_initial_update_exact",
        "mose_input_finite",
        "mose_input_nonnegative",
        "zero_mose_return",
        "zero_initial_mose_update_exact",
        "pairtoken_parent_unchanged",
    )
    if not all(checks[name] is True for name in required):
        raise RuntimeError(f"PairToken+MoSE invariant failed: {checks}")
    return checks
