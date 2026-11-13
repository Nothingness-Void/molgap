"""Model construction by read-only reuse of src/molgap (no new model classes).

k1   : molgap.qm9_neural_atom.make_encoder('neural_atom_k1'); binhyb/laplace only swap the last Linear's out_features.
gine : molgap.gine.GINEWrapper (hidden / layers configurable, 2D categorical inputs as in the shipped wrapper).
fpmlp: plain nn.Sequential over [ECFP4 2048 bits | standardised RDKit 2D descriptors].
"""
from __future__ import annotations

import torch
import torch.nn as nn

K1_PARAMS = 3_658_817


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def build(cfg: dict, n_out: int, desc_dim: int = 0) -> nn.Module:
    fam = cfg["family"]
    if fam == "k1":
        from molgap.qm9_neural_atom import make_encoder
        m = make_encoder("neural_atom_k1")
        if n_out != 1:
            last = m.head[-1]
            m.head[-1] = nn.Linear(last.in_features, n_out)
        expect = K1_PARAMS + (m.head[-1].in_features + 1) * (n_out - 1)
        assert n_params(m) == expect, (n_params(m), expect)
        return m
    if fam == "gine":
        from molgap.gine import GINEWrapper
        return GINEWrapper(in_channels=9, edge_dim=3, hidden_channels=int(cfg["hidden"]),
                           num_layers=int(cfg["layers"]), dropout=0.05, n_targets=n_out)
    if fam == "fpmlp":
        d_in = 2048 + desc_dim
        h = int(cfg.get("hidden", 1024))
        return nn.Sequential(nn.Linear(d_in, h), nn.SiLU(), nn.Dropout(0.1), nn.Linear(h, h // 2), nn.SiLU(),
                             nn.Dropout(0.1), nn.Linear(h // 2, h // 4), nn.SiLU(), nn.Linear(h // 4, n_out))
    raise SystemExit(f"unknown family {fam}")


def forward(model: nn.Module, fam: str, b) -> torch.Tensor:
    """Raw output [B, n_out] (normalised-target units)."""
    if fam == "k1":
        return model(b.x, b.edge_index, b.edge_attr, b.batch, b.random_walk_pe)
    if fam == "gine":
        return model(b.x, b.edge_index, b.edge_attr, b.batch)
    return model(torch.cat([b.ecfp.float(), b.desc], dim=1))
