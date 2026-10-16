"""Conjugated-component controls for the bounded PCQM GraphState screen."""
from __future__ import annotations

import torch
import torch.nn as nn

from .pcqm_gap_architecture import (
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper,
    _LowRankGatedProjection,
)


BASELINE_ID = "ogb_distance_angle_triangle_edge_state_graph_state9"
DESCRIPTOR_ID = (
    "ogb_distance_angle_triangle_edge_state_graph_state9_conjugated_descriptor"
)
COMPONENT_STATE_ID = (
    "ogb_distance_angle_triangle_edge_state_graph_state9_conjugated_component"
)
BASELINE_PARAMETERS = 3_665_809
DESCRIPTOR_CHANNELS = 32
COMPONENT_CHANNELS = 32
DESCRIPTOR_PARAMETER_DELTA = (
    2 * 8 + (8 * DESCRIPTOR_CHANNELS + DESCRIPTOR_CHANNELS)
    + DESCRIPTOR_CHANNELS * 192
)
COMPONENT_PARAMETER_DELTA = (
    192 * COMPONENT_CHANNELS
    + 2 * (2 * COMPONENT_CHANNELS)
    + ((2 * COMPONENT_CHANNELS) * COMPONENT_CHANNELS + COMPONENT_CHANNELS)
    + ((2 * COMPONENT_CHANNELS) * (2 * COMPONENT_CHANNELS) + 2 * COMPONENT_CHANNELS)
    + ((2 * COMPONENT_CHANNELS) * COMPONENT_CHANNELS + COMPONENT_CHANNELS)
    + 2 * COMPONENT_CHANNELS
    + 2 * COMPONENT_CHANNELS
    + (COMPONENT_CHANNELS * 16 + 16)
    + 2 * (16 * 192 + 192)
)
DESCRIPTOR_PARAMETERS = BASELINE_PARAMETERS + DESCRIPTOR_PARAMETER_DELTA
COMPONENT_STATE_PARAMETERS = DESCRIPTOR_PARAMETERS + COMPONENT_PARAMETER_DELTA
PARAMETER_BUDGET = 4_000_000
COMPONENT_BLOCKS = (3, 6, 9)

COMMON_KWARGS = {
    "in_channels": 9,
    "edge_dim": 3,
    "hidden_channels": 192,
    "num_layers": 9,
    "num_heads": 4,
    "dropout": 0.1,
    "n_targets": 1,
    "pooling": "mean",
    "rwse_dim": 16,
    "edge_state_channels": 64,
    "wedge_channels": 16,
    "geometry_basis_channels": 16,
    "graph_state_channels": 64,
    "graph_exchange_rank": 32,
}


class OGBConjugatedDescriptorGraphStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """GraphState9 plus one repeated per-atom conjugated-system descriptor."""

    def __init__(self, *args, descriptor_channels=DESCRIPTOR_CHANNELS, **kwargs):
        if int(descriptor_channels) != DESCRIPTOR_CHANNELS:
            raise ValueError("descriptor_channels must remain 32")
        super().__init__(*args, global_mode="graph_state", **kwargs)
        hidden_channels = self.head[0].in_features
        self.conjugated_descriptor = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, DESCRIPTOR_CHANNELS),
            nn.SiLU(),
        )
        self.descriptor_to_atom = nn.Linear(
            DESCRIPTOR_CHANNELS, hidden_channels, bias=False
        )
        nn.init.zeros_(self.descriptor_to_atom.weight)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        conjugated_component_id,
        conjugated_component_count,
        conjugated_features,
    ):
        embedding = self.encode(
            x, edge_index, edge_attr, batch, random_walk_pe, wedge_edge_ids,
            edge_distance, wedge_angle_cos, geometry_valid,
            conjugated_component_id, conjugated_component_count,
            conjugated_features,
        )
        return self.head(embedding)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        conjugated_component_id,
        conjugated_component_count,
        conjugated_features,
    ):
        payload = (
            conjugated_component_id,
            conjugated_component_count,
            conjugated_features,
        )
        return self._encode_geometry(
            x, edge_index, edge_attr, batch, random_walk_pe, wedge_edge_ids,
            edge_distance, wedge_angle_cos, geometry_valid,
            auxiliary_payload=payload,
        )

    def _initialize_geometry_auxiliary(self, h, batch, auxiliary_payload):
        if not isinstance(auxiliary_payload, tuple) or len(auxiliary_payload) != 3:
            raise ValueError("conjugated payload must contain id, count, features")
        component_id, component_count, features = auxiliary_payload
        component_id = component_id.reshape(-1).long()
        component_count = component_count.reshape(-1).long()
        if component_id.shape[0] != h.shape[0] or features.shape != (h.shape[0], 8):
            raise ValueError("conjugated atom payload is not aligned")
        if component_count.shape[0] <= int(batch.max()):
            raise ValueError("conjugated component counts do not cover the batch")
        if not torch.isfinite(features).all() or int(component_count.min()) < 0:
            raise ValueError("conjugated payload is invalid")
        valid = component_id >= 0
        if valid.any() and bool(
            (component_id[valid] >= component_count[batch[valid]]).any()
        ):
            raise ValueError("local conjugated component id exceeds graph count")
        offsets = torch.cumsum(component_count, dim=0) - component_count
        global_id = component_id.clone()
        global_id[valid] += offsets[batch[valid]]
        descriptor = self.conjugated_descriptor(features.float())
        graph_state = super()._initialize_geometry_auxiliary(h, batch, None)
        return {
            "graph_state": graph_state,
            "descriptor": descriptor,
            "global_component_id": global_id,
            "component_count": int(component_count.sum()),
            "valid": valid,
        }

    def _update_geometry_auxiliary(
        self, layer, h, edge_index, edge_state, batch, auxiliary_state
    ):
        h, edge_state, graph_state = super()._update_geometry_auxiliary(
            layer, h, edge_index, edge_state, batch,
            auxiliary_state["graph_state"],
        )
        auxiliary_state["graph_state"] = graph_state
        if layer + 1 == 2:
            h = h + self.descriptor_to_atom(auxiliary_state["descriptor"])
        return h, edge_state, auxiliary_state


class OGBConjugatedComponentStateGraphStateWrapper(
    OGBConjugatedDescriptorGraphStateWrapper
):
    """Descriptor control plus one persistent conjugated-component state."""

    COMPONENT_BLOCKS = COMPONENT_BLOCKS

    def __init__(self, *args, component_channels=COMPONENT_CHANNELS, **kwargs):
        if int(component_channels) != COMPONENT_CHANNELS:
            raise ValueError("component_channels must remain 32")
        super().__init__(*args, **kwargs)
        hidden_channels = self.head[0].in_features
        self.component_atom_projection = nn.Linear(
            hidden_channels, COMPONENT_CHANNELS, bias=False
        )
        self.component_update_norm = nn.LayerNorm(2 * COMPONENT_CHANNELS)
        self.component_update_gate = nn.Linear(
            2 * COMPONENT_CHANNELS, COMPONENT_CHANNELS
        )
        self.component_update_value = nn.Sequential(
            nn.Linear(2 * COMPONENT_CHANNELS, 2 * COMPONENT_CHANNELS),
            nn.SiLU(),
            nn.Linear(2 * COMPONENT_CHANNELS, COMPONENT_CHANNELS),
        )
        self.component_output_norm = nn.LayerNorm(COMPONENT_CHANNELS)
        self.component_to_atom = _LowRankGatedProjection(
            COMPONENT_CHANNELS, hidden_channels, rank=16
        )

    @staticmethod
    def _mean_by_component(values, component_ids, valid, component_count):
        output = values.new_zeros((component_count, values.shape[1]))
        counts = values.new_zeros((component_count, 1))
        if valid.any():
            ids = component_ids[valid]
            output.index_add_(0, ids, values[valid])
            counts.index_add_(0, ids, values.new_ones((ids.shape[0], 1)))
        return output / counts.clamp_min_(1.0)

    def _initialize_geometry_auxiliary(self, h, batch, auxiliary_payload):
        state = super()._initialize_geometry_auxiliary(h, batch, auxiliary_payload)
        state["component_state"] = self._mean_by_component(
            state["descriptor"],
            state["global_component_id"],
            state["valid"],
            state["component_count"],
        )
        return state

    def _update_geometry_auxiliary(
        self, layer, h, edge_index, edge_state, batch, auxiliary_state
    ):
        h, edge_state, auxiliary_state = super()._update_geometry_auxiliary(
            layer, h, edge_index, edge_state, batch, auxiliary_state
        )
        if layer + 1 not in self.COMPONENT_BLOCKS:
            return h, edge_state, auxiliary_state
        atom_summary = self._mean_by_component(
            self.component_atom_projection(h),
            auxiliary_state["global_component_id"],
            auxiliary_state["valid"],
            auxiliary_state["component_count"],
        )
        component_state = auxiliary_state["component_state"]
        combined = self.component_update_norm(
            torch.cat([component_state, atom_summary], dim=-1)
        )
        gate = torch.sigmoid(self.component_update_gate(combined))
        proposal = self.component_update_value(combined)
        component_state = self.component_output_norm(
            component_state + gate * proposal
        )
        auxiliary_state["component_state"] = component_state
        valid = auxiliary_state["valid"]
        if valid.any():
            update = h.new_zeros(h.shape)
            update[valid] = self.component_to_atom(
                component_state[auxiliary_state["global_component_id"][valid]]
            )
            h = h + update
        return h, edge_state, auxiliary_state


def make_conjugated_encoder(candidate: str):
    if candidate == DESCRIPTOR_ID:
        return OGBConjugatedDescriptorGraphStateWrapper(**COMMON_KWARGS)
    if candidate == COMPONENT_STATE_ID:
        return OGBConjugatedComponentStateGraphStateWrapper(**COMMON_KWARGS)
    raise ValueError(f"unknown conjugated candidate: {candidate}")


__all__ = [
    "BASELINE_ID",
    "DESCRIPTOR_ID",
    "COMPONENT_STATE_ID",
    "BASELINE_PARAMETERS",
    "DESCRIPTOR_PARAMETERS",
    "COMPONENT_STATE_PARAMETERS",
    "PARAMETER_BUDGET",
    "make_conjugated_encoder",
]
