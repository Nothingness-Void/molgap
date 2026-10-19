"""Unweighted effective-resistance pair residual over reviewed BondLocalGPTrans.

These are topology descriptors, not a conjugation model. The family loader owns
the exact frozen initialization; attach only after that load and backend bind.
"""
from dataclasses import replace
import weakref

import torch
from torch import nn

from .gptrans_bond_local import BondLocalGPTrans, construct_family_model
from .model_substrate import ModelSubstrate, SlotBinding, SlotName


@torch.no_grad()
def resistance_features(adjacency, real_mask, *, dtype=torch.float32):
    """Return detached B,4,N,N channels: R, diag sum/difference, (1-R)+ A.

    ``real_mask`` excludes the virtual node and padding. Disconnected pairs are
    explicitly zeroed: a finite pseudoinverse does not define their resistance.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Resistance pinv requires float32 or float64")
    if (adjacency.ndim != 3 or adjacency.shape[-2] != adjacency.shape[-1]
            or real_mask.shape != adjacency.shape[:2]
            or adjacency.dtype != torch.bool or real_mask.dtype != torch.bool
            or adjacency.device != real_mask.device):
        raise ValueError("Expected Boolean B,N,N adjacency and B,N real mask")
    with torch.autocast(device_type=adjacency.device.type, enabled=False):
        valid = real_mask[:, :, None] & real_mask[:, None, :]
        size = adjacency.shape[-1]
        diagonal = torch.eye(size, dtype=torch.bool, device=adjacency.device)[None]
        bonds = (adjacency | adjacency.transpose(-1, -2)) & valid & ~diagonal
        weighted = bonds.to(dtype)
        laplacian = torch.diag_embed(weighted.sum(-1)) - weighted
        inverse = torch.linalg.pinv(laplacian, hermitian=True)
        diag = inverse.diagonal(dim1=-2, dim2=-1)
        diag_sum = diag[:, :, None] + diag[:, None, :]
        resistance = (diag_sum - 2 * inverse).clamp_min(0)
        resistance.masked_fill_(diagonal, 0)
        # Repeated squaring finds every component without relying on numerical
        # nullspace thresholds, shortest-path caps, or pinv cross-component values.
        connected = bonds | (diagonal & valid)
        for _ in range(max(0, (size - 1).bit_length())):
            connected = torch.bmm(connected.to(dtype), connected.to(dtype)) > 0
        connected &= valid
        features = torch.stack((resistance, diag_sum,
            (diag[:, :, None] - diag[:, None, :]).abs(),
            (1 - resistance).clamp_min(0) * weighted), dim=1)
        return features.masked_fill(~connected[:, None], 0).detach()


class ResistanceBias(nn.Module):
    """One first-attention injection; no per-layer work or retained batch cache."""

    def __init__(self, model):
        super().__init__()
        self._model_ref = weakref.ref(model)
        reference = next(model.parameters())
        # Linear initialization is discarded, but must not advance the training RNG.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(4, model.pair_channels, bias=False,
                                        dtype=reference.dtype, device="cpu")
            nn.init.zeros_(self.projection.weight)

    def forward(self, payload):
        model = self._model_ref()
        if model is None or not hasattr(model, "_execution_layout"):
            raise ValueError("Resistance addon requires the prepared execution layout")
        pair = payload.pair
        layout = model._execution_layout
        if (layout["batch_size"] != pair.shape[0]
                or layout["max_nodes"] + 1 != pair.shape[-1]):
            raise ValueError("Resistance layout differs from dense pair shape")
        real = ~payload.key_padding_mask[:, 0, 0, :].clone()
        real[:, 0] = False
        adjacency = torch.zeros((pair.shape[0], pair.shape[-1], pair.shape[-1]),
                                dtype=torch.bool, device=pair.device)
        graph, source, target = layout["edges"]
        adjacency[graph, source + 1, target + 1] = True
        features = resistance_features(adjacency, real)
        residual = self.projection(features.permute(0, 2, 3, 1).to(
            self.projection.weight.dtype)).permute(0, 3, 1, 2).to(pair.dtype)
        valid = real[:, None, :, None] & real[:, None, None, :]
        return replace(payload, pair=pair + residual.masked_fill(~valid, 0))


def apply_addon(model, config=None):
    """Attach the fixed residual after exact initial load and accelerated binding."""
    from .gptrans_execution_backend import dense_inputs
    if config is not None and config != {}:
        raise ValueError("Resistance addon has no configurable scientific fields")
    if not isinstance(model, BondLocalGPTrans):
        raise ValueError("Resistance addon requires reviewed BondLocalGPTrans")
    if getattr(model._dense_inputs, "__func__", None) is not dense_inputs:
        raise ValueError("Bind the prepared GPTrans backend before the resistance addon")
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "gptrans_t")
    if owner.family != "gptrans_t":
        raise ValueError("Resistance addon requires the GPTrans substrate")
    owner.attach("resistance", ResistanceBias(model),
                 SlotBinding("gptrans_t", SlotName.PRE_MESSAGE, 1))
    return model


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode != "degree_bond_local_resistance":
        raise ValueError("Unknown resistance mode")
    return FamilyModePolicy(construct=construct_family_model, objective_name="degree_bond_local_ema999")
