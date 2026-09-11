"""Distance/angle bottom fusion for the accepted K1 and GPTrans-T cores."""
from __future__ import annotations

import torch
import torch.nn as nn

from .gptrans import OGBGPTransTiny
from .pcqm_gap_architecture import (
    OGBGeometrySparseTriangleEdgeStateGPSWrapper,
    _FixedGaussianBasis,
)
from .qm9_neural_atom import (
    LATENT_CHANNELS,
    MAX_SLOTS,
    MIXER_LAYERS,
    RWSE_DIM,
    _LocalGPSBlockFactory,
    _NeuralAtomMixerFactory,
)


GEOMETRY_GPTRANS_T = "geometry_gptrans_t"
GEOMETRY_NEURAL_ATOM_K1 = "geometry_neural_atom_k1"
MODEL_IDS = (GEOMETRY_GPTRANS_T, GEOMETRY_NEURAL_ATOM_K1)


class GeometryGPTransTiny(OGBGPTransTiny):
    """GPTrans-T with zero-start bond-distance and bond-angle channels."""

    def __init__(self) -> None:
        super().__init__(
            node_channels=256,
            pair_channels=32,
            num_layers=12,
            num_heads=8,
            shortest_path_cap=20,
            dropout=0.1,
            drop_path=0.1,
            layer_scale=1.0,
            n_targets=1,
        )
        basis_channels = 16
        self.distance_basis = _FixedGaussianBasis(0.75, 2.25, basis_channels)
        self.angle_basis = _FixedGaussianBasis(-1.0, 1.0, basis_channels)
        self.distance_to_pair = nn.Linear(
            basis_channels, self.pair_channels, bias=False
        )
        self.angle_to_node = nn.Linear(
            basis_channels, self.node_channels, bias=False
        )
        nn.init.zeros_(self.distance_to_pair.weight)
        nn.init.zeros_(self.angle_to_node.weight)

    @staticmethod
    def _validate_geometry(
        edge_index,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        batch,
    ) -> None:
        if tuple(edge_distance.shape) != (edge_index.shape[1], 1):
            raise ValueError("edge_distance is not aligned to directed bonds")
        if wedge_edge_ids.ndim != 2 or wedge_edge_ids.shape[1] != 2:
            raise ValueError("wedge_edge_ids must have shape [W, 2]")
        if tuple(wedge_angle_cos.shape) != (wedge_edge_ids.shape[0], 1):
            raise ValueError("wedge_angle_cos is not aligned to wedges")
        if wedge_edge_ids.numel() and (
            int(wedge_edge_ids.min()) < 0
            or int(wedge_edge_ids.max()) >= edge_index.shape[1]
        ):
            raise ValueError("wedge_edge_ids reference an invalid directed bond")
        graph_count = int(batch.max()) + 1 if batch.numel() else 0
        if geometry_valid.numel() != graph_count:
            raise ValueError("geometry_valid does not cover the graph batch")
        if not torch.isfinite(edge_distance).all() or not torch.isfinite(
            wedge_angle_cos
        ).all():
            raise ValueError("geometry contains non-finite values")

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
    ):
        del random_walk_pe
        self._validate_geometry(
            edge_index,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            batch,
        )
        node, pair, key_padding_mask = self._dense_inputs(
            x, edge_index, edge_attr, batch
        )
        edge_batch, edge_source, edge_target = self._local_edges(
            edge_index, batch, int(x.shape[0])
        )
        valid = geometry_valid.reshape(-1).float()
        distance = self.distance_basis(edge_distance.float())
        distance = distance * valid[edge_batch].view(-1, 1)
        pair[
            edge_batch,
            :,
            edge_source + 1,
            edge_target + 1,
        ] += self.distance_to_pair(distance)

        if wedge_edge_ids.shape[0]:
            first = wedge_edge_ids[:, 0].long()
            centers = edge_index[1, first]
            center_graph = batch[centers]
            angles = self.angle_basis(wedge_angle_cos.float())
            angles = angles * valid[center_graph].view(-1, 1)
            updates = self.angle_to_node(angles)
            summed = updates.new_zeros((x.shape[0], self.node_channels))
            counts = updates.new_zeros((x.shape[0], 1))
            summed.index_add_(0, centers, updates)
            counts.index_add_(0, centers, updates.new_ones((updates.shape[0], 1)))
            averaged = summed / counts.clamp_min_(1.0)
            graph_count = int(valid.numel())
            node_counts = torch.bincount(batch, minlength=graph_count)
            offsets = torch.cat(
                (node_counts.new_zeros(1), node_counts.cumsum(0)[:-1])
            )
            local = torch.arange(x.shape[0], device=x.device) - offsets[batch]
            node[batch, local + 1] += averaged

        for block in self.blocks:
            node, pair = block(node, pair, key_padding_mask)
        graph_state = torch.cat((node[:, 0], pair[:, :, 0, 0]), dim=-1)
        return self.readout(graph_state)


class GeometryNeuralAtomK1(OGBGeometrySparseTriangleEdgeStateGPSWrapper):
    """K1 global allocation on the accepted distance-angle local backbone."""

    def __init__(self) -> None:
        super().__init__(
            in_channels=9,
            edge_dim=3,
            hidden_channels=192,
            num_layers=9,
            num_heads=4,
            dropout=0.05,
            n_targets=1,
            pooling="mean",
            rwse_dim=RWSE_DIM,
            edge_state_channels=64,
            wedge_channels=16,
            geometry_mode="distance_angle",
            geometry_basis_channels=16,
        )
        self.convs = nn.ModuleList(
            [_LocalGPSBlockFactory.make(block) for block in self.convs]
        )
        # The sparse wedge path is new to K1. Zero its only return projections
        # so the added relation state cannot perturb the accepted K1 function
        # before optimization learns to use it.
        for projection in self.wedge_to_edge:
            nn.init.zeros_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        for projection in self.wedge_to_node:
            nn.init.zeros_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        self.neural_atom_mixers = nn.ModuleDict(
            {
                str(layer): _NeuralAtomMixerFactory.make(
                    hidden_channels=192,
                    latent_channels=LATENT_CHANNELS,
                    num_heads=4,
                    max_slots=MAX_SLOTS,
                    active_slots=1,
                    dropout=0.05,
                )
                for layer in MIXER_LAYERS
            }
        )

    def _update_geometry_auxiliary(
        self,
        layer,
        h,
        edge_index,
        edge_state,
        batch,
        auxiliary_state,
    ):
        del edge_index
        one_based_layer = int(layer) + 1
        if one_based_layer in MIXER_LAYERS:
            h = self.neural_atom_mixers[str(one_based_layer)](h, batch)
        return h, edge_state, auxiliary_state


def make_geometry_transfer_model(model_id: str):
    if model_id == GEOMETRY_GPTRANS_T:
        return GeometryGPTransTiny()
    if model_id == GEOMETRY_NEURAL_ATOM_K1:
        return GeometryNeuralAtomK1()
    raise ValueError(f"unknown geometry transfer model: {model_id}")


def forward_geometry_transfer(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
        batch.wedge_edge_ids,
        batch.edge_distance,
        batch.wedge_angle_cos,
        batch.geometry_valid,
    ).view(-1)
